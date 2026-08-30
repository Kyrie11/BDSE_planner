from __future__ import annotations

import argparse
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




def _require_residual_action_variance(pred: dict[str, Any]) -> np.ndarray:
    """Return the direct residual variance using the runtime output contract.

    Calibration must fail loudly if the key changes; silently substituting zero
    variance would invalidate the residual certificate.
    """
    if "residual_action_var" not in pred:
        raise KeyError(
            "V59 residual calibration requires model output 'residual_action_var'; "
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
    device = resolve_torch_device(args.device, context="V59 dual-certificate calibration")
    configure_torch_for_device(device)
    model = load_model_for_config(args.checkpoint, cfg, device)
    core = BDSEPlannerCore(model=model, cfg=cfg)
    dataset = PreprocessedBDSEDataset(args.preprocessed_dir, split=args.split, max_scenarios=args.max_scenarios)

    evidence_scores: list[np.ndarray] = []
    residual_scores: list[float] = []
    residual_proposal_scores: list[float] = []
    residual_raw_errors: list[float] = []
    residual_sigmas: list[float] = []
    evidence_raw_errors: list[np.ndarray] = []
    family_scores: dict[int, list[float]] = defaultdict(list)
    scene_count = 0
    proposal_count = 0
    residual_uniform_scene_count = 0
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
                "V59 residual calibration requires a full action-pair sigma matrix from residual_action_var"
            )

        # Uniform scene-wise conformal score.  It upper-bounds the residual
        # margin error for every valid rival of the frozen selected-local anchor,
        # so deployment may safely choose any rival without relying on the very
        # sparse subset of calibration scenes that already produced a flip.
        if (
            0 <= raw_anchor < teacher_cost.size
            and corrected_cost.size == teacher_cost.size
            and valid.size == teacher_cost.size
        ):
            rival_idx = np.flatnonzero(valid & (np.arange(valid.size) != raw_anchor))
            if rival_idx.size:
                predicted = (float(corrected_cost[raw_anchor]) - corrected_cost[rival_idx].astype(np.float64)) / scale
                truth = (float(teacher_cost[raw_anchor]) - teacher_cost[rival_idx]) / scale
                if sigma_matrix is None:
                    rival_sigma = np.zeros_like(predicted)
                else:
                    sm = np.asarray(sigma_matrix, dtype=np.float64)
                    rival_sigma = sm[rival_idx, raw_anchor] if sm.shape == (valid.size, valid.size) else np.zeros_like(predicted)
                scores_all = predicted - truth - beta * rival_sigma
                finite_all = np.isfinite(scores_all) & np.isfinite(predicted) & np.isfinite(truth) & np.isfinite(rival_sigma)
                if bool(finite_all.any()):
                    best = int(np.argmax(np.where(finite_all, scores_all, -np.inf)))
                    residual_scores.append(float(scores_all[best]))
                    residual_raw_errors.append(float(predicted[best] - truth[best]))
                    residual_sigmas.append(float(rival_sigma[best]))
                    residual_uniform_scene_count += 1

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
        residual_proposal_scores=np.asarray(residual_proposal_scores, dtype=np.float64),
        residual_raw_errors=np.asarray(residual_raw_errors, dtype=np.float64),
        residual_sigmas=np.asarray(residual_sigmas, dtype=np.float64),
        scene_count=np.asarray([scene_count], dtype=np.int64),
        proposal_count=np.asarray([proposal_count], dtype=np.int64),
        residual_uniform_scene_count=np.asarray([residual_uniform_scene_count], dtype=np.int64),
        family_ids=np.asarray(sorted(family_scores), dtype=np.int64),
        family_scores_json=np.asarray([json.dumps({str(k): v for k, v in family_scores.items()})]),
    )
    print(json.dumps({"raw_output": str(args.raw_output), "scene_count": scene_count, "residual_uniform_scene_count": residual_uniform_scene_count, "residual_proposal_count": proposal_count, "evidence_score_count": int(evidence.size)}, indent=2))


def _merge(args: argparse.Namespace) -> None:
    evidence_parts: list[np.ndarray] = []
    evidence_err_parts: list[np.ndarray] = []
    residual_parts: list[np.ndarray] = []
    residual_proposal_parts: list[np.ndarray] = []
    residual_err_parts: list[np.ndarray] = []
    sigma_parts: list[np.ndarray] = []
    family: dict[str, list[float]] = defaultdict(list)
    scenes = proposals = uniform_scenes = 0
    for path_text in args.merge_raw:
        data = np.load(path_text, allow_pickle=False)
        evidence_parts.append(data["evidence_scores"])
        evidence_err_parts.append(data["evidence_raw_errors"])
        residual_parts.append(data["residual_scores"])
        residual_proposal_parts.append(data["residual_proposal_scores"] if "residual_proposal_scores" in data.files else np.zeros((0,), dtype=np.float64))
        residual_err_parts.append(data["residual_raw_errors"])
        sigma_parts.append(data["residual_sigmas"])
        scenes += int(data["scene_count"][0])
        proposals += int(data["proposal_count"][0])
        uniform_scenes += int(data["residual_uniform_scene_count"][0]) if "residual_uniform_scene_count" in data.files else int(data["residual_scores"].size)
        family_json = json.loads(str(data["family_scores_json"][0]))
        for key, values in family_json.items():
            family[key].extend(float(v) for v in values)
    evidence = np.concatenate(evidence_parts) if evidence_parts else np.zeros((0,), dtype=np.float64)
    evidence_err = np.concatenate(evidence_err_parts) if evidence_err_parts else np.zeros((0,), dtype=np.float64)
    residual = np.concatenate(residual_parts) if residual_parts else np.zeros((0,), dtype=np.float64)
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
        "method": "V59 scene-uniform split-conformal dual certificate calibration",
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "scene_count": int(scenes),
        "evidence_score_count": int(evidence.size),
        "residual_uniform_scene_count": int(uniform_scenes),
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
            "Residual epsilon is a scene-wise uniform bound over all valid rivals and must be frozen before test evaluation."
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
    parser.add_argument("--beta", type=float, default=1.0)
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
