from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import bdse.tools.fit_v64_3_25_eaf_icer_drc as v25
import bdse.tools.fit_v64_3_22_eaf_icer_tcr as v22

FOLDS = v25.FOLDS
FOLD_SEED = v25.FOLD_SEED
KS = v25.KS
MIN_EDGES = v25.MIN_EDGES
MIN_SCENES = v25.MIN_SCENES
MIN_FOLD_SCENES = v25.MIN_FOLD_SCENES
MIN_SELECTED_PER_FOLD = v25.MIN_SELECTED
MAIN_MIN_SELECTED = 64
EXPECTED_TRAIN_SCENES = v25.EXPECTED_TRAIN_SCENES
CATASTROPHIC_DELTA_THRESHOLD = -0.5
_BASE_NAMES = list(v22._ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
_OPERATOR_NAME = "operator::min_support_scalar_dominance_logit"


def _with_operator_margin(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    margin = np.minimum(
        np.asarray(data["support"], dtype=np.float64),
        np.asarray(data["scalar"], dtype=np.float64),
    ).reshape(-1, 1)
    out["X"] = np.concatenate([np.asarray(data["X"], dtype=np.float64), margin], axis=1)
    return out


def _score(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, certificate: str) -> np.ndarray:
    z, mean, std, metric_weight = v25._memory(train_x)
    q = ((query_x - mean[None, :]) / std[None, :]) * np.sqrt(metric_weight[None, :])
    d2 = np.maximum(
        np.sum(q * q, axis=1)[:, None]
        + np.sum(z * z, axis=1)[None, :]
        - 2.0 * (q @ z.T),
        0.0,
    )
    rows = np.arange(len(q))[:, None]
    bounds: list[np.ndarray] = []
    for k in KS:
        kk = min(int(k), len(z))
        nbr = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        dist = np.sqrt(d2[rows, nbr])
        w = 1.0 / np.maximum(dist, 1.0e-6)
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1.0e-12)
        y = train_y[nbr]
        mu = np.sum(w * y, axis=1)
        if certificate == "downside_rms":
            downside = np.minimum(y, 0.0)
            bound = mu - np.sqrt(np.sum(w * downside * downside, axis=1))
        elif certificate == "catastrophic_excess_rms":
            catastrophic_excess = np.minimum(y - CATASTROPHIC_DELTA_THRESHOLD, 0.0)
            bound = mu - np.sqrt(np.sum(w * catastrophic_excess * catastrophic_excess, axis=1))
        else:
            raise ValueError(f"unknown certificate={certificate}")
        bounds.append(bound)
    return np.min(np.stack(bounds, axis=1), axis=1)


def _crossfit(data: dict[str, Any], certificate: str, *, mode: str) -> dict[str, Any]:
    x = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["delta"], dtype=np.float64)
    toks = np.asarray(data["tok"], dtype=object)
    unique = sorted(set(map(str, toks)))
    if len(x) < MIN_EDGES or len(unique) < MIN_SCENES:
        raise SystemExit(f"STOP TRAIN SUPPORT {mode}/{certificate}: edges={len(x)} scenes={len(unique)}")

    folds: list[dict[str, Any]] = []
    for fold in range(FOLDS):
        hold = {t for t in unique if v25._fold(t) == fold}
        if len(hold) < MIN_FOLD_SCENES:
            raise SystemExit(f"STOP TRAIN SPLIT: fold too small {fold}: {len(hold)}")
        hold_mask = np.asarray([str(t) in hold for t in toks], dtype=bool)
        score = np.full(len(x), np.nan, dtype=np.float64)
        score[hold_mask] = _score(x[~hold_mask], y[~hold_mask], x[hold_mask], certificate)
        metrics = v25._selection(data, score, hold)
        metrics["fold"] = int(fold)
        metrics["hold_scenes"] = int(len(hold))
        worst = float(metrics["worst"])
        catastrophe_count = int(
            0 if metrics["count"] == 0 else 1 if np.isfinite(worst) and worst <= CATASTROPHIC_DELTA_THRESHOLD else 0
        )
        # Exact count is recovered by re-running the selected-path operator so the
        # report remains correct even if a future fold contains >1 catastrophe.
        idx_hold = np.flatnonzero(hold_mask)
        selected_delta: list[float] = []
        for token in sorted(hold):
            idx = idx_hold[np.asarray([str(toks[i]) == token for i in idx_hold], dtype=bool)]
            if not len(idx):
                continue
            accepted = idx[
                (np.asarray(data["support"])[idx] > 0.0)
                & (np.asarray(data["scalar"])[idx] > 0.0)
                & (score[idx] > 0.0)
            ]
            if len(accepted):
                j = sorted(
                    accepted.tolist(),
                    key=lambda q: (-float(data["scalar"][q]), -float(score[q]), int(data["action"][q])),
                )[0]
                selected_delta.append(float(y[j]))
        if selected_delta:
            catastrophe_count = int(np.sum(np.asarray(selected_delta) <= CATASTROPHIC_DELTA_THRESHOLD))
        metrics["catastrophe_count"] = catastrophe_count
        metrics["catastrophe_free"] = bool(catastrophe_count == 0)
        metrics["path_safe"] = bool(
            metrics["count"] >= MIN_SELECTED_PER_FOLD
            and metrics["sum"] >= -1.0e-9
            and metrics["catastrophe_free"]
        )
        folds.append(metrics)

    selected_count = int(sum(f["count"] for f in folds))
    return {
        "mode": mode,
        "certificate": certificate,
        "folds": folds,
        "all_folds_path_safe": bool(all(f["path_safe"] for f in folds)),
        "all_folds_catastrophe_free": bool(all(f["catastrophe_free"] for f in folds)),
        "fold_pass_count": int(sum(bool(f["path_safe"]) for f in folds)),
        "selected_count": selected_count,
        "teacher_improvement_sum": float(sum(f["sum"] for f in folds)),
        "mean_precision": float(np.nanmean([f["precision"] for f in folds])),
        "mean_capture": float(np.nanmean([f["capture"] for f in folds])),
        "mean_negative_rms": float(np.nanmean([f["negative_rms"] for f in folds])),
        "catastrophe_count": int(sum(f["catastrophe_count"] for f in folds)),
    }


def _save_memory(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    z, mean, std, metric_weight = v25._memory(np.asarray(data["X"], dtype=np.float64))
    names = [f"evidence::{name}" for name in _BASE_NAMES] + [_OPERATOR_NAME]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        memory_metric_z=z.astype("f4"),
        teacher_improvement=np.asarray(data["delta"], dtype=np.float32),
        feature_mean=mean.astype("f4"),
        feature_std=std.astype("f4"),
        feature_names=np.asarray(names, dtype="U128"),
        feature_metric_weight=metric_weight.astype("f4"),
        neighbor_k_values=np.asarray(KS, dtype="i4"),
        se_multiplier=np.asarray([1.0], dtype="f4"),
        certificate_kind=np.asarray(["mean_minus_catastrophic_excess_rms"], dtype="U64"),
        downside_multiplier=np.asarray([1.0], dtype="f4"),
        catastrophic_delta_threshold=np.asarray([CATASTROPHIC_DELTA_THRESHOLD], dtype="f4"),
    )
    return {
        "path": str(path),
        "sha256": v25._sha256_file(path),
        "row_count": int(len(data["delta"])),
        "feature_count": int(data["X"].shape[1]),
        "feature_names": names,
        "mode": "operator_margin_evidence",
        "certificate": "catastrophic_excess_rms",
        "catastrophic_delta_threshold": CATASTROPHIC_DELTA_THRESHOLD,
    }


def _cfg(base: dict[str, Any], memory: dict[str, Any]) -> dict[str, Any]:
    cfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
    # V31 explicitly removes the failed evidence-rebinding branches.  Their
    # absence, not merely disabled flags, is checked by the contract auditor.
    selector = cfg.setdefault("selector", {})
    selector.pop("proposal_conditioned_witness_rebinding", None)
    selector.pop("frontier_contrast_rebinding", None)

    ic = v25._icer(cfg)
    ic.update(
        {
            "model_type": "frozen_support_scalar_dominance_plus_operator_margin_catastrophic_excess_regret_certificate",
            "dominance_policy": "scalar_only",
            "incumbent_retention_policy": "preserve_admissible_incumbent",
            "regret_risk_enabled": True,
            "retention_regret_risk_enabled": False,
            "replacement_regret_risk_enabled": True,
            "regret_risk_model_type": "local_multiscale_catastrophic_excess_regret_certificate",
            "regret_risk_feature_mode": "operator_margin_evidence",
            "replacement_local_regret_memory_path": memory["path"],
            "replacement_local_regret_memory_sha256": memory["sha256"],
            "replacement_local_regret_neighbor_k_values": list(KS),
            "replacement_local_regret_certificate": "mean_minus_catastrophic_excess_rms",
            "replacement_local_regret_catastrophic_delta_threshold": CATASTROPHIC_DELTA_THRESHOLD,
            "replacement_regret_training_population": "TRAIN_only_final_guard_admissible_support_positive_scalar_dominance_positive_alternatives",
            "replacement_operator": (
                "preserve admissible incumbent by default; compute OMCER risk admissibility for every deployment-admissible "
                "support>0 AND scalar_dominance>0 alternative before extremization; keep only certificate>0 alternatives; "
                "choose exactly one by frozen scalar dominance; no post-extremal rerank, no second-best fallback after proposal formation"
            ),
            "operator_conditioning_statistic": "min(support_logit,scalar_dominance_logit)",
            "catastrophic_excess_definition": "min(teacher_improvement-(-0.5),0)",
            "all_flagged_policy": "preserve_legacy_for_structural_guard",
        }
    )
    version = "V64.3.31-EAF-ICER-OMCER"
    cfg.setdefault("metadata", {})["algorithm_version"] = version
    cfg.setdefault("provenance", {})["algorithm_version"] = version
    exp = cfg.setdefault("experiment", {})
    exp["name"] = "v64_3_31_eaf_icer_omcer"
    exp["algorithm"] = (
        "V64.3.31 EAF-ICER-OMCER: Operator-Margin Catastrophic-Excess Regret Certification"
    )
    exp["mechanism_chain"] = (
        "fixed bounded B=16/M=24 interface -> attributed deployment-admissible complete frontier -> "
        "frozen support/scalar eligibility -> operator-margin-conditioned catastrophic-tail risk admissibility before extremization -> "
        "one incumbent-contrastive extremal proposal -> incumbent-default monotone intervention -> unchanged structural deployment guard"
    )
    return cfg


def _write(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit V64.3.31 OMCER on the frozen 3000 TRAIN population only.")
    ap.add_argument("--train-frontier-edges", required=True)
    ap.add_argument("--base-v20-dual-config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--output-train-token-file", required=True)
    ap.add_argument("--output-report", required=True)
    args = ap.parse_args()

    edge_path = Path(args.train_frontier_edges)
    if not edge_path.is_file() or edge_path.stat().st_size <= 0:
        raise SystemExit(f"STOP TRAIN DATA: missing frontier provenance {edge_path}")
    by, frontier_row_count = v25._load_minimal_scenes(edge_path)
    if len(by) != EXPECTED_TRAIN_SCENES:
        raise SystemExit(f"STOP TRAIN DATA: expected exactly {EXPECTED_TRAIN_SCENES} frozen TRAIN scenes, got {len(by)}")
    data18 = v25._build(by)
    data19 = _with_operator_margin(data18)

    crossfit = {
        "evidence_downside_control": _crossfit(data18, "downside_rms", mode="evidence_only"),
        "evidence_catastrophic_excess_ablation": _crossfit(data18, "catastrophic_excess_rms", mode="evidence_only"),
        "operator_margin_downside_ablation": _crossfit(data19, "downside_rms", mode="operator_margin_evidence"),
        "omcer_main": _crossfit(data19, "catastrophic_excess_rms", mode="operator_margin_evidence"),
    }
    control = crossfit["evidence_downside_control"]
    main_cf = crossfit["omcer_main"]

    # The control must reproduce V25 exactly.  This guards the causal 2x2 against
    # accidental fold/population drift before the new statistic is interpreted.
    historical_control_exact = bool(
        control["selected_count"] == 71
        and abs(control["teacher_improvement_sum"] - 5.527642325753739) <= 1.0e-8
    )
    coverage_gain = float(main_cf["mean_capture"] - control["mean_capture"])
    selected_gain = int(main_cf["selected_count"] - control["selected_count"])
    gate = bool(
        historical_control_exact
        and main_cf["all_folds_path_safe"]
        and main_cf["all_folds_catastrophe_free"]
        and main_cf["selected_count"] >= MAIN_MIN_SELECTED
        and main_cf["teacher_improvement_sum"] >= -1.0e-9
        and selected_gain >= 5
        and coverage_gain >= 0.03
        and main_cf["mean_negative_rms"] <= control["mean_negative_rms"] + 1.0e-12
    )

    tokens = sorted(by)
    token_path = Path(args.output_train_token_file)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")

    report: dict[str, Any] = {
        "audit": "v64_3_31_eaf_icer_omcer_train_fit",
        "algorithm": "V64.3.31 EAF-ICER-OMCER",
        "train_scene_count": int(len(by)),
        "frontier_row_count": int(frontier_row_count),
        "replacement_edges": int(len(data18["delta"])),
        "replacement_scenes": int(data18["replacement_scene_count"]),
        "fold_seed": FOLD_SEED,
        "neighbor_k_values": list(KS),
        "downside_multiplier": 1.0,
        "decision_boundary": 0.0,
        "catastrophic_delta_threshold": CATASTROPHIC_DELTA_THRESHOLD,
        "operator_conditioning_statistic": "min(support_logit,scalar_dominance_logit)",
        "crossfit": crossfit,
        "historical_V25_control_exact": historical_control_exact,
        "main_selected_gain_over_V25_control": selected_gain,
        "main_capture_gain_over_V25_control": coverage_gain,
        "train_gate_pass": gate,
        "gate_contract": {
            "main": "omcer_main",
            "historical_V25_control_exact": True,
            "all_5_scene_folds_selected_path_nonharmful_and_catastrophe_free": True,
            "selected_count_min": MAIN_MIN_SELECTED,
            "selected_gain_over_V25_min": 5,
            "mean_capture_gain_over_V25_min": 0.03,
            "mean_negative_rms_noninferior_to_V25": True,
            "teacher_improvement_sum_min": 0.0,
            "fresh_validation_must_not_be_used_on_train_gate_fail": True,
        },
        "input_frontier": {
            "path": str(edge_path),
            "bytes": int(edge_path.stat().st_size),
            "sha256": v25._sha256_file(edge_path),
        },
        "train_token_manifest": {
            "path": str(token_path),
            "count": int(len(tokens)),
            "sha256": v25._sha256_file(token_path),
        },
        "fresh_validation_used": False,
        "diagnostic_interpretation": {
            "evidence_catastrophic_excess_ablation": "coverage-only relaxation is unsafe if operator state is omitted",
            "operator_margin_downside_ablation": "operator conditioning stabilizes tail but all-negative downside remains too conservative",
            "omcer_main": "the two factors are complementary on frozen TRAIN and must be tested together on untouched A/B",
        },
        "memory": {},
        "config": None,
    }
    _write(Path(args.output_report), report)
    if not gate:
        raise SystemExit(
            "STOP TRAIN OMCER: pre-registered 2x2/main gate failed; do not spend fresh GPU and do not tune K, -0.5 tail boundary, multiplier, zero boundary, or operator-margin definition"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    memory_path = out_dir / "v64_3_31_omcer_memory.npz"
    memory = _save_memory(memory_path, data19)
    base = yaml.safe_load(Path(args.base_v20_dual_config).read_text(encoding="utf-8"))
    config_path = out_dir / "v64_3_31_omcer.yaml"
    config_path.write_text(yaml.safe_dump(_cfg(base, memory), sort_keys=False), encoding="utf-8")
    report["memory"] = memory
    report["config"] = str(config_path)
    _write(Path(args.output_report), report)
    print(json.dumps({
        "pass": True,
        "train_gate_pass": True,
        "main_count": main_cf["selected_count"],
        "main_sum": main_cf["teacher_improvement_sum"],
        "main_capture": main_cf["mean_capture"],
        "main_catastrophe_count": main_cf["catastrophe_count"],
        "selected_gain_over_V25": selected_gain,
        "capture_gain_over_V25": coverage_gain,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
