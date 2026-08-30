from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from bdse.config import load_config
from bdse.data.nuplan_dataset import PreprocessedBDSEDataset
from bdse.external_baselines.model_factory import load_model_for_config
from bdse.planner.nuplan_planner import BDSEPlannerCore
from bdse.planner.tournament import _evidence_action_potential_cost
from bdse.utils import configure_torch_for_device, resolve_torch_device


def _quantile(values: np.ndarray, alpha: float) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64)[np.isfinite(values)])
    if values.size == 0:
        return float("nan")
    rank = int(math.ceil((values.size + 1) * (1.0 - float(alpha))))
    return float(values[min(max(rank, 1), values.size) - 1])


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_residual_action_variance(pred: dict[str, Any]) -> np.ndarray:
    """Return the direct residual variance using the runtime output contract.

    Calibration must fail loudly if the key changes; silently substituting zero
    variance would invalidate the residual certificate.
    """
    if "residual_action_var" not in pred:
        raise KeyError(
            "V61 residual calibration requires model output 'residual_action_var'; "
            "refusing to silently calibrate with zero uncertainty"
        )
    value = np.asarray(pred["residual_action_var"], dtype=np.float32)
    if value.ndim != 2:
        raise ValueError(f"residual_action_var must be [E,K], got {value.shape}")
    return value

def _load_provenance(path: str | None) -> tuple[dict[str, Any] | None, bool]:
    if not path:
        return None, False
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    independent = bool(
        data.get("group_disjoint", False)
        and data.get("no_group_overlap", False)
        and str(data.get("calibration_role", "")) == "calibration_only"
    )
    if not independent:
        raise ValueError("provenance does not establish a group-disjoint calibration-only split")
    return data, True


def _collect(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    device = resolve_torch_device(args.device, context="V61 dual-certificate calibration")
    configure_torch_for_device(device)
    model = load_model_for_config(args.checkpoint, cfg, device)
    core = BDSEPlannerCore(model=model, cfg=cfg)
    dataset = PreprocessedBDSEDataset(args.preprocessed_dir, split=args.split, max_scenarios=args.max_scenarios)

    evidence_scores: list[np.ndarray] = []
    residual_scores: list[float] = []
    residual_all_rival_scores: list[float] = []
    residual_proposal_scores: list[float] = []
    residual_raw_errors: list[float] = []
    residual_sigmas: list[float] = []
    evidence_raw_errors: list[np.ndarray] = []
    family_scores: dict[int, list[float]] = defaultdict(list)
    scene_count = 0
    proposal_count = 0
    residual_policy_scene_count = 0
    beta = max(float(args.beta), 0.0)
    prior_radius = max(float(args.prior_radius), 0.0)

    for sample in tqdm(dataset.iter_samples(), total=len(dataset)):
        if sample.teacher is None:
            continue
        pred, _selection, tournament, _ = core._run_certificate_stage(
            sample.runtime, sample.candidates, sample.evidence_bank, cfg
        )
        pairs = np.asarray(pred.get("pair_indices", []), dtype=np.int64)
        pairs = pairs.reshape(-1, 2) if pairs.size else np.zeros((0, 2), dtype=np.int64)
        d_hat = np.asarray(pred.get("certificate_pair_atom_delta", pred.get("pair_atom_delta", [])), dtype=np.float32)
        if pairs.size and d_hat.ndim == 2 and d_hat.shape[1] == pairs.shape[0]:
            E = min(d_hat.shape[0], sample.teacher.g_evid.shape[0], sample.evidence_bank.E)
            a, b = pairs[:, 0], pairs[:, 1]
            ok = (a >= 0) & (b >= 0) & (a < sample.teacher.g_evid.shape[1]) & (b < sample.teacher.g_evid.shape[1])
            if bool(ok.any()):
                d_use = d_hat[:E, ok]
                p_use = pairs[ok]
                scale = float(pred.get("pair_margin_scale", 1.0)) if bool(cfg.get("model", {}).get("pair_margin_normalized", True)) else 1.0
                scale = max(scale, 1.0e-6)
                truth = (sample.teacher.g_evid[:E, p_use[:, 1]] - sample.teacher.g_evid[:E, p_use[:, 0]]) / scale
                var = np.asarray(pred.get("certificate_pair_atom_var", np.zeros_like(d_hat)), dtype=np.float32)
                if var.shape != d_hat.shape:
                    var = np.zeros_like(d_hat)
                var = var[:E, ok]
                sigma = np.sqrt(np.maximum(var, 0.0) + prior_radius**2)
                scores = d_use - truth - beta * sigma
                active = np.asarray(sample.evidence_bank.active_mask[:E], dtype=bool)
                finite = active[:, None] & np.isfinite(scores) & np.isfinite(truth) & np.isfinite(d_use)
                evidence_scores.append(scores[finite].astype(np.float64))
                evidence_raw_errors.append((d_use - truth)[finite].astype(np.float64))
                fam = np.asarray(pred.get("family_ids", np.zeros((E,), dtype=np.int64)), dtype=np.int64)[:E]
                for fid in np.unique(fam[active]):
                    family_scores[int(fid)].extend(scores[finite & (fam[:, None] == int(fid))].astype(np.float64).tolist())

        diag = tournament.diagnostics or {}
        raw_anchor = int(diag.get("pair_action_anchor_raw_anchor_action", diag.get("pair_action_anchor_pre_structural_action", tournament.action_index)))
        raw_proposed = int(diag.get("pair_action_anchor_raw_proposed_action", diag.get("pair_action_anchor_proposed_action", tournament.action_index)))
        teacher_cost = np.asarray(sample.teacher.J_T, dtype=np.float64).reshape(-1)
        valid = np.asarray(sample.candidates.valid_mask, dtype=bool).reshape(-1)
        selected = list(getattr(_selection, "selected", []))
        scale = max(float(pred.get("rival_pair_margin_scale", pred.get("pair_margin_scale", 1.0))), 1.0e-6)
        residual_action_var = _require_residual_action_variance(pred)
        anchor_cost, corrected_cost, sigma_matrix, _set_diag = _evidence_action_potential_cost(
            pred["J0"],
            pred.get("g", None),
            pred.get("residual_action_potential", None),
            selected,
            valid,
            residual_action_variance=residual_action_var,
            residual_set_atom_factors=pred.get("residual_set_atom_factors", None),
            residual_set_action_factors=pred.get("residual_set_action_factors", None),
            set_residual_scale=float((cfg.get("runtime", {}) or {}).get("set_conditioned_residual_scale", 1.0)),
            normalize_margins=bool((cfg.get("model", {}) or {}).get("pair_margin_normalized", True)),
            margin_scale=scale,
        )
        if corrected_cost.size != teacher_cost.size:
            raise ValueError(
                f"residual calibration action count mismatch: corrected={corrected_cost.size} "
                f"teacher={teacher_cost.size}"
            )
        if sigma_matrix is None or np.asarray(sigma_matrix).shape != (valid.size, valid.size):
            raise ValueError(
                "V61 residual calibration requires a full action-pair sigma matrix from residual_action_var"
            )

        # V61 policy-aligned split-conformal score.  Deployment deterministically
        # compares the selected-local anchor with the frozen model's lowest-cost
        # rival.  Calibrating that selected rival on every scene is valid because
        # the rival-selection rule uses only frozen model outputs and inputs, not
        # teacher labels.  The old max-over-all-rivals bound remains a diagnostic
        # but is no longer used to suppress every residual flip.
        if corrected_cost.size == teacher_cost.size and valid.size == teacher_cost.size:
            finite_anchor = valid & np.isfinite(anchor_cost)
            if bool(finite_anchor.any()):
                anchor_idx = int(np.flatnonzero(finite_anchor)[np.argmin(anchor_cost[finite_anchor])])
                rival_idx = np.flatnonzero(valid & (np.arange(valid.size) != anchor_idx) & np.isfinite(corrected_cost))
                if rival_idx.size:
                    policy_pos = int(np.argmin(corrected_cost[rival_idx]))
                    policy_rival = int(rival_idx[policy_pos])
                    predicted_margin = float((corrected_cost[anchor_idx] - corrected_cost[policy_rival]) / scale)
                    true_margin = float((teacher_cost[anchor_idx] - teacher_cost[policy_rival]) / scale)
                    sm = np.asarray(sigma_matrix, dtype=np.float64)
                    policy_sigma = float(sm[policy_rival, anchor_idx])
                    policy_score = predicted_margin - true_margin - beta * policy_sigma
                    if np.isfinite(policy_score) and np.isfinite(policy_sigma):
                        residual_scores.append(float(policy_score))
                        residual_raw_errors.append(float(predicted_margin - true_margin))
                        residual_sigmas.append(float(policy_sigma))
                        residual_policy_scene_count += 1

                    predicted_all = (float(corrected_cost[anchor_idx]) - corrected_cost[rival_idx].astype(np.float64)) / scale
                    truth_all = (float(teacher_cost[anchor_idx]) - teacher_cost[rival_idx]) / scale
                    sigma_all = sm[rival_idx, anchor_idx]
                    scores_all = predicted_all - truth_all - beta * sigma_all
                    finite_all = np.isfinite(scores_all) & np.isfinite(sigma_all)
                    if bool(finite_all.any()):
                        residual_all_rival_scores.append(float(np.max(scores_all[finite_all])))

        # Keep the original proposal-conditional score as a diagnostic only.
        if raw_anchor != raw_proposed and 0 <= raw_anchor < teacher_cost.size and 0 <= raw_proposed < teacher_cost.size:
            true_margin = float((teacher_cost[raw_anchor] - teacher_cost[raw_proposed]) / scale)
            predicted_margin = float(diag.get("pair_action_anchor_raw_margin", float("nan")))
            sigma = max(float(diag.get("pair_action_anchor_residual_sigma", 0.0)), 0.0)
            if np.isfinite(true_margin) and np.isfinite(predicted_margin) and np.isfinite(sigma):
                residual_proposal_scores.append(predicted_margin - true_margin - beta * sigma)
                proposal_count += 1
        scene_count += 1

    evidence = np.concatenate(evidence_scores) if evidence_scores else np.zeros((0,), dtype=np.float64)
    evidence_error = np.concatenate(evidence_raw_errors) if evidence_raw_errors else np.zeros((0,), dtype=np.float64)
    np.savez_compressed(
        args.raw_output,
        evidence_scores=evidence,
        evidence_raw_errors=evidence_error,
        residual_scores=np.asarray(residual_scores, dtype=np.float64),
        residual_all_rival_scores=np.asarray(residual_all_rival_scores, dtype=np.float64),
        residual_proposal_scores=np.asarray(residual_proposal_scores, dtype=np.float64),
        residual_raw_errors=np.asarray(residual_raw_errors, dtype=np.float64),
        residual_sigmas=np.asarray(residual_sigmas, dtype=np.float64),
        scene_count=np.asarray([scene_count], dtype=np.int64),
        proposal_count=np.asarray([proposal_count], dtype=np.int64),
        residual_policy_scene_count=np.asarray([residual_policy_scene_count], dtype=np.int64),
        family_ids=np.asarray(sorted(family_scores), dtype=np.int64),
        family_scores_json=np.asarray([json.dumps({str(k): v for k, v in family_scores.items()})]),
        calibration_beta=np.asarray([float(beta)], dtype=np.float64),
        calibration_prior_radius=np.asarray([float(prior_radius)], dtype=np.float64),
        source_checkpoint_sha256=np.asarray([_sha256_file(args.checkpoint)]),
        source_config_sha256=np.asarray([_sha256_file(args.config)]),
    )
    print(json.dumps({"raw_output": str(args.raw_output), "scene_count": scene_count, "residual_policy_scene_count": residual_policy_scene_count, "residual_proposal_count": proposal_count, "evidence_score_count": int(evidence.size)}, indent=2))


def _merge(args: argparse.Namespace) -> None:
    evidence_parts: list[np.ndarray] = []
    evidence_err_parts: list[np.ndarray] = []
    residual_parts: list[np.ndarray] = []
    residual_all_rival_parts: list[np.ndarray] = []
    residual_proposal_parts: list[np.ndarray] = []
    residual_err_parts: list[np.ndarray] = []
    sigma_parts: list[np.ndarray] = []
    family: dict[str, list[float]] = defaultdict(list)
    raw_betas: list[float] = []
    raw_prior_radii: list[float] = []
    raw_checkpoint_hashes: list[str] = []
    raw_config_hashes: list[str] = []
    scenes = proposals = policy_scenes = 0
    for path_text in args.merge_raw:
        data = np.load(path_text, allow_pickle=False)
        evidence_parts.append(data["evidence_scores"])
        evidence_err_parts.append(data["evidence_raw_errors"])
        residual_parts.append(data["residual_scores"])
        residual_all_rival_parts.append(data["residual_all_rival_scores"] if "residual_all_rival_scores" in data.files else np.zeros((0,), dtype=np.float64))
        residual_proposal_parts.append(data["residual_proposal_scores"] if "residual_proposal_scores" in data.files else np.zeros((0,), dtype=np.float64))
        residual_err_parts.append(data["residual_raw_errors"])
        sigma_parts.append(data["residual_sigmas"])
        scenes += int(data["scene_count"][0])
        proposals += int(data["proposal_count"][0])
        policy_scenes += int(data["residual_policy_scene_count"][0]) if "residual_policy_scene_count" in data.files else int(data["residual_scores"].size)
        raw_betas.append(float(data["calibration_beta"][0]) if "calibration_beta" in data.files else float(args.beta))
        raw_prior_radii.append(
            float(data["calibration_prior_radius"][0])
            if "calibration_prior_radius" in data.files
            else float(args.prior_radius)
        )
        if "source_checkpoint_sha256" not in data.files or "source_config_sha256" not in data.files:
            raise ValueError(f"Calibration shard {path_text} lacks V64.3.1 source SHA provenance; recompute it")
        raw_checkpoint_hashes.append(str(data["source_checkpoint_sha256"][0]))
        raw_config_hashes.append(str(data["source_config_sha256"][0]))
        family_json = json.loads(str(data["family_scores_json"][0]))
        for key, values in family_json.items():
            family[key].extend(float(v) for v in values)
    if raw_betas and (max(raw_betas) - min(raw_betas) > 1.0e-12):
        raise ValueError(f"Calibration shards disagree on beta: {raw_betas}")
    if raw_prior_radii and (max(raw_prior_radii) - min(raw_prior_radii) > 1.0e-12):
        raise ValueError(f"Calibration shards disagree on prior_radius: {raw_prior_radii}")
    if len(set(raw_checkpoint_hashes)) != 1:
        raise ValueError(f"Calibration shards were collected from different checkpoints: {raw_checkpoint_hashes}")
    if len(set(raw_config_hashes)) != 1:
        raise ValueError(f"Calibration shards were collected from different configs: {raw_config_hashes}")
    effective_beta = float(raw_betas[0]) if raw_betas else float(args.beta)
    effective_prior_radius = float(raw_prior_radii[0]) if raw_prior_radii else float(args.prior_radius)
    evidence = np.concatenate(evidence_parts) if evidence_parts else np.zeros((0,), dtype=np.float64)
    evidence_err = np.concatenate(evidence_err_parts) if evidence_err_parts else np.zeros((0,), dtype=np.float64)
    residual = np.concatenate(residual_parts) if residual_parts else np.zeros((0,), dtype=np.float64)
    residual_all_rival = np.concatenate(residual_all_rival_parts) if residual_all_rival_parts else np.zeros((0,), dtype=np.float64)
    residual_proposal = np.concatenate(residual_proposal_parts) if residual_proposal_parts else np.zeros((0,), dtype=np.float64)
    residual_err = np.concatenate(residual_err_parts) if residual_err_parts else np.zeros((0,), dtype=np.float64)
    sigmas = np.concatenate(sigma_parts) if sigma_parts else np.zeros((0,), dtype=np.float64)
    evidence_eps = max(0.0, _quantile(evidence, args.alpha)) if evidence.size else 0.0
    residual_eps = max(0.0, _quantile(residual, args.alpha)) if residual.size else float(args.residual_epsilon_fallback)
    provenance, independent = _load_provenance(args.provenance_json)
    family_out = {}
    for key, values in sorted(family.items(), key=lambda kv: int(kv[0])):
        arr = np.asarray(values, dtype=np.float64)
        eps = max(0.0, _quantile(arr, args.alpha)) if arr.size else 0.0
        family_out[key] = {"count": int(arr.size), "epsilon": eps, "empirical_violation_rate": float(np.mean(arr > eps)) if arr.size else float("nan")}
    output = {
        "method": "V64.3 calibration-consistent policy-selected-top-rival split-conformal dual certificate calibration",
        "alpha": float(args.alpha),
        "beta": effective_beta,
        "prior_radius": effective_prior_radius,
        "source_checkpoint_sha256": raw_checkpoint_hashes[0] if raw_checkpoint_hashes else None,
        "source_config_sha256": raw_config_hashes[0] if raw_config_hashes else None,
        "scene_count": int(scenes),
        "evidence_score_count": int(evidence.size),
        "residual_policy_scene_count": int(policy_scenes),
        "residual_all_rival_diagnostic_epsilon": float(max(0.0, _quantile(residual_all_rival, args.alpha))) if residual_all_rival.size else float("nan"),
        "residual_proposal_count": int(proposals),
        "residual_proposal_diagnostic_epsilon": float(max(0.0, _quantile(residual_proposal, args.alpha))) if residual_proposal.size else float("nan"),
        "recommended_adverse_certificate_epsilon": float(evidence_eps),
        "recommended_residual_flip_epsilon": float(residual_eps),
        "evidence_empirical_violation_rate": float(np.mean(evidence > evidence_eps)) if evidence.size else float("nan"),
        "residual_empirical_violation_rate": float(np.mean(residual > residual_eps)) if residual.size else float("nan"),
        "evidence_raw_error_mae": float(np.mean(np.abs(evidence_err))) if evidence_err.size else float("nan"),
        "residual_raw_error_mae": float(np.mean(np.abs(residual_err))) if residual_err.size else float("nan"),
        "residual_sigma_mean": float(np.mean(sigmas)) if sigmas.size else float("nan"),
        "family": family_out,
        "independent_calibration": independent,
        "provenance": provenance,
        "warning": (
            "Residual epsilon is calibrated on the frozen policy-selected top rival for every calibration scene; the all-rival maximum is diagnostic only."
            if residual.size
            else "No valid residual rival scores occurred in calibration; conservative fallback epsilon was used."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--raw-output")
    mode.add_argument("--merge-raw", nargs="+")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--preprocessed-dir")
    parser.add_argument("--split", nargs="+", default=["val_calib"])
    parser.add_argument("--max-scenarios", type=int, default=5000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--prior-radius", type=float, default=0.10)
    parser.add_argument("--residual-epsilon-fallback", type=float, default=0.05)
    parser.add_argument("--provenance-json")
    parser.add_argument("--output")
    args = parser.parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be in (0,1)")
    if args.raw_output:
        for name in ("config", "checkpoint", "preprocessed_dir"):
            if not getattr(args, name):
                raise ValueError(f"--{name.replace('_', '-')} is required in collection mode")
        _collect(args)
    else:
        if not args.output:
            raise ValueError("--output is required in merge mode")
        _merge(args)


if __name__ == "__main__":
    main()
