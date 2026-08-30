from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import bdse.tools.fit_v64_3_22_eaf_icer_tcr as v22

_FIXED_FOLDS = 5
_FIXED_FOLD_SEED = "v64.3.23-eaf-icer-rcr-scene-crossfit-v1"
_FIXED_NEIGHBOR_K_VALUES = (32, 64)
_FIXED_SE_MULTIPLIER = 1.0
_MIN_TOTAL_REPLACEMENT_EDGES = 1024
_MIN_TOTAL_REPLACEMENT_SCENES = 256
_MIN_FOLD_SCENES = 40
_MIN_SELECTED_PER_FOLD = 8


def _icer(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]


def _fold_id(token: str) -> int:
    h = hashlib.sha256((_FIXED_FOLD_SEED + "::" + str(token)).encode()).digest()
    return int.from_bytes(h[:8], "big") % _FIXED_FOLDS


def _feature_metric_weight(mode: str) -> np.ndarray:
    evidence_n = len(v22._runtime_feature_names("evidence_only"))
    if mode == "evidence_only":
        return np.full(evidence_n, 1.0 / float(evidence_n), dtype=np.float64)
    if mode != "transition_conditioned":
        raise ValueError(mode)
    names = v22._runtime_feature_names(mode)
    transition_n = len(names) - evidence_n
    # The transition schema is intentionally split into auditable planner
    # semantics and geometry.  This prevents the 41 transition dimensions from
    # overwhelming the 18 frozen evidence dimensions merely by dimensionality.
    semantic_n = 21
    geometry_n = transition_n - semantic_n
    if geometry_n <= 0:
        raise RuntimeError("unexpected transition feature schema")
    return np.concatenate([
        np.full(evidence_n, 1.0 / float(evidence_n), dtype=np.float64),
        np.full(semantic_n, 1.0 / float(semantic_n), dtype=np.float64),
        np.full(geometry_n, 1.0 / float(geometry_n), dtype=np.float64),
    ])


def _standardized_metric_memory(train_x: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = np.maximum(train_x.std(axis=0), 1.0e-6)
    metric_weight = _feature_metric_weight(mode)
    if len(metric_weight) != train_x.shape[1]:
        raise RuntimeError("V64.3.23 metric/schema mismatch")
    memory = ((train_x - mean[None, :]) / std[None, :]) * np.sqrt(metric_weight[None, :])
    return memory, mean, std, metric_weight


def _local_regret_lower_bound(
    train_x: np.ndarray,
    train_delta: np.ndarray,
    query_x: np.ndarray,
    mode: str,
) -> np.ndarray:
    memory, mean, std, metric_weight = _standardized_metric_memory(train_x, mode)
    query = ((query_x - mean[None, :]) / std[None, :]) * np.sqrt(metric_weight[None, :])
    q2 = np.sum(query * query, axis=1, keepdims=True)
    m2 = np.sum(memory * memory, axis=1, keepdims=True).T
    d2 = np.maximum(q2 + m2 - 2.0 * (query @ memory.T), 0.0)
    rows = np.arange(len(query))[:, None]
    bounds: list[np.ndarray] = []
    for k in _FIXED_NEIGHBOR_K_VALUES:
        kk = min(int(k), memory.shape[0])
        nbr = np.argpartition(d2, kth=kk - 1, axis=1)[:, :kk]
        dist = np.sqrt(d2[rows, nbr])
        weight = 1.0 / np.maximum(dist, 1.0e-6)
        weight = weight / np.maximum(np.sum(weight, axis=1, keepdims=True), 1.0e-12)
        vals = train_delta[nbr]
        local_mean = np.sum(weight * vals, axis=1)
        local_var = np.sum(weight * (vals - local_mean[:, None]) ** 2, axis=1)
        effective_n = 1.0 / np.maximum(np.sum(weight * weight, axis=1), 1.0e-12)
        local_se = np.sqrt(local_var / np.maximum(effective_n, 1.0))
        bounds.append(local_mean - _FIXED_SE_MULTIPLIER * local_se)
    return np.min(np.stack(bounds, axis=1), axis=1).astype(np.float64)


def _selection_metrics(
    data: dict[str, Any],
    score: np.ndarray,
    hold_tokens: set[str],
    policy: str,
) -> dict[str, float]:
    toks = np.asarray(data["rep_tok"], dtype=object)
    support = np.asarray(data["rep_support"], dtype=np.float64)
    scalar = np.asarray(data["rep_scalar_dom"], dtype=np.float64)
    dual = np.asarray(data["rep_dual_dom"], dtype=np.float64)
    delta = np.asarray(data["rep_delta"], dtype=np.float64)
    selected: list[float] = []
    opportunities = captures = scenes = 0
    for tok in sorted(hold_tokens):
        idx = np.flatnonzero(toks == tok)
        if idx.size == 0:
            continue
        scenes += 1
        opportunities += int(np.any(delta[idx] > 0.0))
        mask = (support[idx] > 0.0) & (scalar[idx] > 0.0) & (score[idx] > 0.0)
        if policy == "scalar":
            rank = scalar
        elif policy == "dual_rank_only":
            rank = dual
        elif policy == "self_consistent_dual":
            # Unlike V21 per-view consensus, profile dominance may be negative.
            # The only extra condition is semantic consistency: the equal-mean
            # score actually used to rank/trigger the replacement must itself be
            # positive rather than selecting a negative-ranked extremum.
            mask = mask & (dual[idx] > 0.0)
            rank = dual
        else:
            raise ValueError(policy)
        eligible = idx[mask]
        if eligible.size:
            j = sorted(
                eligible.tolist(),
                key=lambda q: (-float(rank[q]), -float(score[q]), int(data["rep_action"][q])),
            )[0]
            selected.append(float(delta[j]))
            captures += int(float(delta[j]) > 0.0)
    arr = np.asarray(selected, dtype=np.float64)
    return {
        "holdout_scene_count": float(scenes),
        "selected_replacement_count": float(arr.size),
        "selected_replacement_precision": float(np.mean(arr > 0.0)) if arr.size else float("nan"),
        "selected_replacement_teacher_improvement_sum": float(arr.sum()) if arr.size else 0.0,
        "selected_replacement_teacher_improvement_mean": float(arr.mean()) if arr.size else float("nan"),
        "selected_replacement_worst_teacher_improvement": float(arr.min()) if arr.size else float("nan"),
        "opportunity_count": float(opportunities),
        "opportunity_capture_rate": float(captures / opportunities) if opportunities else float("nan"),
    }


def _crossfit_local(data: dict[str, Any], mode: str) -> dict[str, Any]:
    x = np.asarray(data["rep_X"], dtype=np.float64)
    delta = np.asarray(data["rep_delta"], dtype=np.float64)
    toks = np.asarray(data["rep_tok"], dtype=object)
    unique = sorted(set(str(t) for t in toks))
    if len(x) < _MIN_TOTAL_REPLACEMENT_EDGES or len(unique) < _MIN_TOTAL_REPLACEMENT_SCENES:
        raise SystemExit(
            f"insufficient TRAIN replacement support for V64.3.23/{mode}: edges={len(x)} scenes={len(unique)}"
        )
    folds: list[dict[str, Any]] = []
    all_selected_sum = 0.0
    all_selected_count = 0
    for fold in range(_FIXED_FOLDS):
        hold_tokens = {t for t in unique if _fold_id(t) == fold}
        if len(hold_tokens) < _MIN_FOLD_SCENES:
            raise SystemExit(
                f"TRAIN scene-level crossfit fold too small for V64.3.23/{mode}: fold={fold} scenes={len(hold_tokens)}"
            )
        hold = np.asarray([str(t) in hold_tokens for t in toks], dtype=bool)
        fit = ~hold
        score_hold = _local_regret_lower_bound(x[fit], delta[fit], x[hold], mode)
        score_all = np.full(len(x), np.nan, dtype=np.float64)
        score_all[hold] = score_hold
        edge = v22._risk_metrics(delta[hold], score_hold)
        scalar = _selection_metrics(data, score_all, hold_tokens, "scalar")
        dual_rank = _selection_metrics(data, score_all, hold_tokens, "dual_rank_only")
        scr = _selection_metrics(data, score_all, hold_tokens, "self_consistent_dual")
        safe = bool(
            scr["selected_replacement_count"] >= _MIN_SELECTED_PER_FOLD
            and scr["selected_replacement_teacher_improvement_sum"] >= -1.0e-9
        )
        all_selected_sum += float(scr["selected_replacement_teacher_improvement_sum"])
        all_selected_count += int(scr["selected_replacement_count"])
        folds.append({
            "fold": fold,
            "holdout_unique_scene_count": len(hold_tokens),
            "holdout_edge_count": int(hold.sum()),
            "edge_local_regret_lower_bound": edge,
            "scalar": scalar,
            "dual_rank_only": dual_rank,
            "self_consistent_dual": scr,
            "self_consistent_path_safe": safe,
        })
    return {
        "mode": mode,
        "fold_seed": _FIXED_FOLD_SEED,
        "neighbor_k_values": list(_FIXED_NEIGHBOR_K_VALUES),
        "se_multiplier": _FIXED_SE_MULTIPLIER,
        "metric_group_policy": "equal_group_average_squared_distance",
        "feature_names": v22._runtime_feature_names(mode),
        "folds": folds,
        "all_folds_self_consistent_path_safe": bool(all(f["self_consistent_path_safe"] for f in folds)),
        "self_consistent_fold_pass_count": int(sum(bool(f["self_consistent_path_safe"]) for f in folds)),
        "self_consistent_teacher_improvement_sum": float(all_selected_sum),
        "self_consistent_selected_replacement_count": int(all_selected_count),
        "dual_rank_only_teacher_improvement_sum": float(sum(f["dual_rank_only"]["selected_replacement_teacher_improvement_sum"] for f in folds)),
        "scalar_teacher_improvement_sum": float(sum(f["scalar"]["selected_replacement_teacher_improvement_sum"] for f in folds)),
        "mean_self_consistent_precision": float(np.mean([f["self_consistent_dual"]["selected_replacement_precision"] for f in folds])),
        "mean_self_consistent_capture": float(np.mean([f["self_consistent_dual"]["opportunity_capture_rate"] for f in folds])),
        "mean_edge_auc": float(np.nanmean([f["edge_local_regret_lower_bound"]["auc_positive_teacher_improvement"] for f in folds])),
    }


def _save_memory(path: Path, data: dict[str, Any], mode: str) -> dict[str, Any]:
    x = np.asarray(data["rep_X"], dtype=np.float64)
    delta = np.asarray(data["rep_delta"], dtype=np.float64)
    memory, mean, std, metric_weight = _standardized_metric_memory(x, mode)
    names = np.asarray(v22._runtime_feature_names(mode), dtype="U128")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        memory_metric_z=memory.astype(np.float32),
        teacher_improvement=delta.astype(np.float32),
        feature_mean=mean.astype(np.float32),
        feature_std=std.astype(np.float32),
        feature_names=names,
        feature_metric_weight=metric_weight.astype(np.float32),
        neighbor_k_values=np.asarray(_FIXED_NEIGHBOR_K_VALUES, dtype=np.int32),
        se_multiplier=np.asarray([_FIXED_SE_MULTIPLIER], dtype=np.float32),
    )
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "sha256": sha,
        "row_count": int(len(delta)),
        "feature_count": int(x.shape[1]),
        "neighbor_k_values": list(_FIXED_NEIGHBOR_K_VALUES),
        "se_multiplier": _FIXED_SE_MULTIPLIER,
        "metric_group_policy": "equal_group_average_squared_distance",
    }


def _make_cfg(
    base: dict[str, Any],
    mode: str,
    memory: dict[str, Any],
    dominance_policy: str,
    tag: str,
) -> dict[str, Any]:
    cfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
    ic = _icer(cfg)
    if not bool(ic.get("enabled", False)):
        raise SystemExit("base V20 ICER config must be enabled")
    ic.update({
        "model_type": "frozen_support_dominance_plus_local_multiscale_regret_coherence",
        "dominance_policy": dominance_policy,
        "incumbent_retention_policy": "preserve_admissible_incumbent",
        "regret_risk_enabled": True,
        "retention_regret_risk_enabled": False,
        "replacement_regret_risk_enabled": True,
        "regret_risk_model_type": "local_multiscale_regret_lower_bound",
        "regret_risk_feature_mode": mode,
        "replacement_local_regret_memory_path": memory["path"],
        "replacement_local_regret_memory_sha256": memory["sha256"],
        "replacement_local_regret_neighbor_k_values": list(memory["neighbor_k_values"]),
        "replacement_local_regret_se_multiplier": float(memory["se_multiplier"]),
        "replacement_local_regret_metric_group_policy": memory["metric_group_policy"],
        "replacement_regret_target": "teacher_margin_candidate_minus_teacher_margin_raw_incumbent",
        "regret_risk_threshold_policy": "fixed_zero_multiscale_local_lower_bound_no_validation_sweep",
        "regret_risk_training_population": "TRAIN_only_final_guard_admissible_raw_incumbents_and_frozen_support_positive_scalar_dominance_positive_alternatives",
        "regret_risk_crossfit_contract": "5fold_scene_level_fixed_K32_K64_oneSE_self_consistent_selected_path_nonharmful",
        "retention_operator": "final_guard_admissible_incumbent_is_preserved_by_default; no learned incumbent_to_anchor veto",
        "replacement_operator": (
            "anchor_support>0 AND scalar_dominance>0 AND multiscale_local_regret_lower_bound>0; "
            "scalar ranks scalar; dual-rank ranks equal-mean; RCR main also requires equal-mean>0"
        ),
        "all_flagged_policy": "preserve_legacy_for_structural_guard",
        "train_crossfit_replacement_path_safe": True,
    })
    version = "V64.3.23-EAF-ICER-RCR-DARM-DBR"
    cfg.setdefault("metadata", {})["algorithm_version"] = version
    cfg.setdefault("provenance", {})["algorithm_version"] = version
    exp = cfg.setdefault("experiment", {})
    exp["name"] = f"v64_3_23_eaf_icer_rcr_{tag}"
    exp["algorithm"] = "V64.3.23 EAF-ICER-RCR: Evidence-Attributed Incumbent-Contrastive Extremal Recovery with Regret-Coherent Local Reliability"
    exp["mechanism_chain"] = (
        "fixed B<=16 -> frozen EAF complete frontier + exact selected-evidence attribution -> deployment-complete admissible frontier -> "
        "frozen support/dominance -> TRAIN-only group-balanced multiscale local regret lower bound -> self-consistent extremal replacement with incumbent-default preservation -> unchanged final/structural guards"
    )
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit V64.3.23 regret-coherent local replacement memories and configs.")
    ap.add_argument("--train-frontier-edges", required=True)
    ap.add_argument("--base-v20-dual-config", required=True)
    ap.add_argument("--output-evidence-memory", required=True)
    ap.add_argument("--output-transition-memory", required=True)
    ap.add_argument("--output-evidence-scalar-config", required=True)
    ap.add_argument("--output-evidence-rcr-config", required=True)
    ap.add_argument("--output-transition-rcr-config", required=True)
    ap.add_argument("--output-train-token-file", required=True)
    ap.add_argument("--output-report", required=True)
    a = ap.parse_args()

    by = v22._load_scenes(Path(a.train_frontier_edges))
    evidence = v22._build_data(by, "evidence_only")
    transition = v22._build_data(by, "transition_conditioned")
    if not np.isfinite(float(transition.get("transition_nonzero_fraction", np.nan))) or float(transition["transition_nonzero_fraction"]) < 0.95:
        raise SystemExit(f"STOP TRAIN INSTRUMENTATION: transition feature coverage too low: {transition.get('transition_nonzero_fraction')}")

    crossfit = {
        "evidence_only": _crossfit_local(evidence, "evidence_only"),
        "transition_conditioned": _crossfit_local(transition, "transition_conditioned"),
    }
    # The paper/main mechanism is evidence-local RCR.  It is the only local
    # view that is path-safe in all five fixed scene folds on the frozen TRAIN
    # frontier.  Planner-transition conditioning is emitted only as an
    # independently audited ablation and cannot rescue a failed evidence-local
    # main mechanism.
    main_cf = crossfit["evidence_only"]
    train_gate_pass = bool(
        main_cf["self_consistent_selected_replacement_count"] >= 64
        and main_cf["self_consistent_teacher_improvement_sum"] >= -1.0e-9
        and main_cf["all_folds_self_consistent_path_safe"]
    )
    if not train_gate_pass:
        raise SystemExit(
            "STOP TRAIN RCR: evidence-local scene-level out-of-fold selected replacement path is not safe in all fixed folds; do not spend fresh validation GPU"
        )

    evidence_memory = _save_memory(Path(a.output_evidence_memory), evidence, "evidence_only")
    transition_memory = _save_memory(Path(a.output_transition_memory), transition, "transition_conditioned")
    base = yaml.safe_load(Path(a.base_v20_dual_config).read_text(encoding="utf-8"))
    bic = _icer(base)
    if str(bic.get("dominance_policy", "")) != "dual_equal_mean" or str(bic.get("all_flagged_policy", "")) != "preserve_legacy_for_structural_guard":
        raise SystemExit("base config must be frozen V20 dual_equal_mean deployment-complete ICER")

    configs = [
        (a.output_evidence_scalar_config, _make_cfg(base, "evidence_only", evidence_memory, "scalar_only", "evidence_local_scalar")),
        (a.output_evidence_rcr_config, _make_cfg(base, "evidence_only", evidence_memory, "scalar_positive_dual_mean_positive", "evidence_local_rcr")),
        (a.output_transition_rcr_config, _make_cfg(base, "transition_conditioned", transition_memory, "scalar_positive_dual_mean_positive", "transition_local_rcr_ablation")),
    ]
    # Only the evidence-local configs carry the strict TRAIN path-safe claim.
    # The transition-conditioned config is an ablation because its fixed-fold
    # path sign is not uniformly stable on TRAIN.
    for _, cfg in configs:
        ic = _icer(cfg)
        if str(ic.get("regret_risk_feature_mode", "")) == "transition_conditioned":
            ic["train_crossfit_replacement_path_safe"] = bool(crossfit["transition_conditioned"]["all_folds_self_consistent_path_safe"])
            ic["transition_conditioning_role"] = "controlled_ablation_not_required_for_main_promotion"
        else:
            ic["train_crossfit_replacement_path_safe"] = True
            ic["transition_conditioning_role"] = "not_used"
    for path, cfg in configs:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    report = {
        "audit": "v64_3_23_eaf_icer_rcr_train_only_fit",
        "algorithm": "V64.3.23 EAF-ICER-RCR",
        "official_v22_fresh_status": "not_run_v22_stopped_inside_train_only_fitter_before_fresh_selection",
        "train_frontier_scene_count": int(len(by)),
        "replacement_population": {
            "evidence_edge_count": int(len(evidence["rep_delta"])),
            "evidence_unique_scene_count": int(len(set(str(t) for t in evidence["rep_tok"]))),
            "transition_edge_count": int(len(transition["rep_delta"])),
            "transition_unique_scene_count": int(len(set(str(t) for t in transition["rep_tok"]))),
            "transition_nonzero_fraction": float(transition["transition_nonzero_fraction"]),
        },
        "fixed_policy": {
            "fold_count": _FIXED_FOLDS,
            "fold_seed": _FIXED_FOLD_SEED,
            "neighbor_k_values": list(_FIXED_NEIGHBOR_K_VALUES),
            "se_multiplier": _FIXED_SE_MULTIPLIER,
            "metric": "equal_group_average_squared_distance_then_inverse_distance_weighted_local_mean_minus_one_standard_error; minimum_across_K32_K64",
            "incumbent_retention": "preserve_final_guard_admissible_incumbent",
            "main_dominance": "evidence-local scalar_positive_and_equal_mean_positive; profile need not be individually positive",
            "transition_conditioning_role": "controlled_ablation_only_unless_two_independent_fresh_blocks_show_incremental_gain",
        },
        "crossfit": crossfit,
        "memories": {"evidence_only": evidence_memory, "transition_conditioned": transition_memory},
        "train_gate_pass": train_gate_pass,
        "fresh_validation_was_used_for_fit_or_policy_selection": False,
    }
    train_tokens = sorted(str(t) for t in by)
    train_token_path = Path(a.output_train_token_file)
    train_token_path.parent.mkdir(parents=True, exist_ok=True)
    train_token_path.write_text("\n".join(train_tokens) + "\n", encoding="utf-8")
    report["train_frontier_token_manifest"] = {
        "path": str(train_token_path),
        "count": len(train_tokens),
        "sha256": hashlib.sha256(train_token_path.read_bytes()).hexdigest(),
    }
    out_report = Path(a.output_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "pass": True,
        "train_gate_pass": train_gate_pass,
        "evidence_crossfit_sum": main_cf["self_consistent_teacher_improvement_sum"],
        "evidence_crossfit_precision": main_cf["mean_self_consistent_precision"],
        "evidence_crossfit_capture": main_cf["mean_self_consistent_capture"],
        "transition_crossfit_sum_diagnostic": crossfit["transition_conditioned"]["self_consistent_teacher_improvement_sum"],
        "transition_crossfit_fold_pass_count_diagnostic": crossfit["transition_conditioned"]["self_consistent_fold_pass_count"],
        "next_action": "run_two_independent_fresh_500_scene_blocks_without_threshold_or_neighbor_tuning",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
