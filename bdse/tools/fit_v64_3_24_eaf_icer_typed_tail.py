from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import bdse.tools.fit_v64_3_22_eaf_icer_tcr as v22
from bdse.planner.tournament import (
    _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES,
    _ICER_TYPED_EVIDENCE_FEATURE_NAMES,
)

# Pre-registered from V23 so this iteration tests one mechanism change rather
# than re-opening the neighborhood/uncertainty search space.
_FIXED_FOLDS = 5
_FIXED_FOLD_SEED = "v64.3.24-eaf-icer-typed-tail-scene-crossfit-v1"
_FIXED_NEIGHBOR_K_VALUES = (32, 64)
_FIXED_SE_MULTIPLIER = 1.0
_FIXED_TAIL_SE_MULTIPLIER = 1.0
_MIN_TOTAL_REPLACEMENT_EDGES = 1024
_MIN_TOTAL_REPLACEMENT_SCENES = 256
_MIN_FOLD_SCENES = 40
_MIN_SELECTED_PER_FOLD = 8


def _icer(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]


def _fold_id(token: str) -> int:
    h = hashlib.sha256((_FIXED_FOLD_SEED + "::" + str(token)).encode()).digest()
    return int.from_bytes(h[:8], "big") % _FIXED_FOLDS


def _runtime_feature_names(mode: str) -> list[str]:
    base = [f"evidence::{n}" for n in _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES]
    if mode == "evidence_only":
        return base
    if mode == "typed_interaction":
        return base + [f"typed::{n}" for n in _ICER_TYPED_EVIDENCE_FEATURE_NAMES]
    raise ValueError(mode)


def _row_feature(r: dict[str, Any], mode: str) -> list[float] | None:
    vals: list[float] = []
    for name in _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES:
        try:
            v = float(r.get(f"icer_feature_{name}", np.nan))
        except Exception:
            return None
        if not np.isfinite(v):
            return None
        vals.append(v)
    if mode == "typed_interaction":
        for name in _ICER_TYPED_EVIDENCE_FEATURE_NAMES:
            try:
                v = float(r.get(f"icer_typed_incumbent_{name}", np.nan))
            except Exception:
                return None
            if not np.isfinite(v):
                return None
            vals.append(v)
    return vals


def _build_replacement_data(by: dict[str, list[dict[str, Any]]], mode: str) -> dict[str, Any]:
    x: list[list[float]] = []
    delta: list[float] = []
    tok_out: list[str] = []
    support: list[float] = []
    scalar_dom: list[float] = []
    action: list[int] = []
    legacy_out: list[int] = []
    proposal_scenes = admissible_inc_scenes = multi_alt_scenes = 0
    typed_expected = typed_present = 0
    for tok, rows in by.items():
        if not rows:
            continue
        anchor = int(rows[0].get("anchor_action", -1))
        legacy = int(rows[0].get("raw_top_action", -1))
        if legacy < 0 or legacy == anchor:
            continue
        proposal_scenes += 1
        inc = next((r for r in rows if int(r.get("challenger_action", -999)) == legacy), None)
        if inc is None or float(inc.get("icer_admissible", inc.get("dacer_admissible", 0.0))) < 0.5:
            continue
        try:
            inc_tm = float(inc.get("teacher_margin", np.nan))
        except Exception:
            continue
        if not np.isfinite(inc_tm):
            continue
        admissible_inc_scenes += 1
        alt_count = 0
        for r in rows:
            ch = int(r.get("challenger_action", -1))
            if ch < 0 or ch in {anchor, legacy}:
                continue
            if float(r.get("icer_admissible", r.get("dacer_admissible", 0.0))) < 0.5:
                continue
            try:
                tm = float(r.get("teacher_margin", np.nan))
                sup = float(r.get("icer_support_logit", np.nan))
                sdom = float(r.get("icer_scalar_dominance_logit", np.nan))
            except Exception:
                continue
            if not all(np.isfinite(v) for v in [tm, sup, sdom]):
                continue
            # Match the frozen V23 deployment population exactly before learning
            # any recovery memory.  This prevents the risk estimator from being
            # trained on edges the operator can never choose.
            if not (sup > 0.0 and sdom > 0.0):
                continue
            if mode == "typed_interaction":
                typed_expected += 1
                typed_present += int(all(f"icer_typed_incumbent_{n}" in r for n in _ICER_TYPED_EVIDENCE_FEATURE_NAMES))
            xv = _row_feature(r, mode)
            if xv is None:
                continue
            x.append(xv)
            delta.append(tm - inc_tm)
            tok_out.append(str(tok))
            support.append(sup)
            scalar_dom.append(sdom)
            action.append(ch)
            legacy_out.append(legacy)
            alt_count += 1
        multi_alt_scenes += int(alt_count >= 2)
    if len(x) < _MIN_TOTAL_REPLACEMENT_EDGES or len(set(tok_out)) < _MIN_TOTAL_REPLACEMENT_SCENES:
        raise SystemExit(
            f"insufficient TRAIN replacement support for V64.3.24/{mode}: edges={len(x)} scenes={len(set(tok_out))}"
        )
    coverage = float(typed_present / typed_expected) if typed_expected else float("nan")
    if mode == "typed_interaction" and (not np.isfinite(coverage) or coverage < 0.99):
        raise SystemExit(f"STOP TRAIN INSTRUMENTATION: typed selected-evidence coverage too low: {coverage}")
    return {
        "rep_X": np.asarray(x, dtype=np.float64),
        "rep_delta": np.asarray(delta, dtype=np.float64),
        "rep_tok": np.asarray(tok_out, dtype=object),
        "rep_support": np.asarray(support, dtype=np.float64),
        "rep_scalar_dom": np.asarray(scalar_dom, dtype=np.float64),
        "rep_action": np.asarray(action, dtype=np.int64),
        "rep_legacy": np.asarray(legacy_out, dtype=np.int64),
        "proposal_scene_count": int(proposal_scenes),
        "admissible_incumbent_scene_count": int(admissible_inc_scenes),
        "multi_alternative_scene_count": int(multi_alt_scenes),
        "typed_feature_row_coverage": coverage,
    }


def _feature_metric_weight(mode: str) -> np.ndarray:
    e = len(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
    if mode == "evidence_only":
        return np.full(e, 1.0 / float(e), dtype=np.float64)
    if mode == "typed_interaction":
        t = len(_ICER_TYPED_EVIDENCE_FEATURE_NAMES)
        # Two auditable views get equal total distance mass.  This is fixed by
        # representation semantics, not selected from validation performance.
        return np.concatenate([
            np.full(e, 1.0 / float(e), dtype=np.float64),
            np.full(t, 1.0 / float(t), dtype=np.float64),
        ])
    raise ValueError(mode)


def _standardized_metric_memory(train_x: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = np.maximum(train_x.std(axis=0), 1.0e-6)
    metric_weight = _feature_metric_weight(mode)
    if len(metric_weight) != train_x.shape[1]:
        raise RuntimeError("V64.3.24 metric/schema mismatch")
    memory = ((train_x - mean[None, :]) / std[None, :]) * np.sqrt(metric_weight[None, :])
    return memory, mean, std, metric_weight


def _local_score(train_x: np.ndarray, train_delta: np.ndarray, query_x: np.ndarray, mode: str, tau: float, tail: bool) -> np.ndarray:
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
        w = 1.0 / np.maximum(dist, 1.0e-6)
        w = w / np.maximum(np.sum(w, axis=1, keepdims=True), 1.0e-12)
        y = train_delta[nbr]
        local_mean = np.sum(w * y, axis=1)
        local_var = np.sum(w * (y - local_mean[:, None]) ** 2, axis=1)
        effective_n = 1.0 / np.maximum(np.sum(w * w, axis=1), 1.0e-12)
        local_se = np.sqrt(local_var / np.maximum(effective_n, 1.0))
        score = local_mean - _FIXED_SE_MULTIPLIER * local_se
        if tail:
            downside = np.maximum(-y - tau, 0.0)
            dmean = np.sum(w * downside, axis=1)
            dvar = np.sum(w * (downside - dmean[:, None]) ** 2, axis=1)
            dse = np.sqrt(dvar / np.maximum(effective_n, 1.0))
            score = score - (dmean + _FIXED_TAIL_SE_MULTIPLIER * dse)
        bounds.append(score)
    return np.min(np.stack(bounds, axis=1), axis=1).astype(np.float64)


def _selection_metrics(data: dict[str, Any], score: np.ndarray, hold_tokens: set[str], rank_policy: str, tau: float) -> dict[str, float]:
    toks = np.asarray(data["rep_tok"], dtype=object)
    scalar = np.asarray(data["rep_scalar_dom"], dtype=np.float64)
    delta = np.asarray(data["rep_delta"], dtype=np.float64)
    selected: list[float] = []
    opportunities = captures = scenes = 0
    for tok in sorted(hold_tokens):
        idx = np.flatnonzero(toks == tok)
        if idx.size == 0:
            continue
        scenes += 1
        opportunities += int(np.any(delta[idx] > 0.0))
        eligible = idx[np.isfinite(score[idx]) & (score[idx] > 0.0)]
        if eligible.size:
            if rank_policy == "dominance_first":
                key = lambda q: (-float(scalar[q]), -float(score[q]), int(data["rep_action"][q]))
            elif rank_policy == "regret_risk_first":
                key = lambda q: (-float(score[q]), -float(scalar[q]), int(data["rep_action"][q]))
            else:
                raise ValueError(rank_policy)
            j = sorted(eligible.tolist(), key=key)[0]
            selected.append(float(delta[j]))
            captures += int(float(delta[j]) > 0.0)
    arr = np.asarray(selected, dtype=np.float64)
    material = arr < -float(tau)
    return {
        "holdout_scene_count": float(scenes),
        "selected_replacement_count": float(arr.size),
        "selected_replacement_precision": float(np.mean(arr > 0.0)) if arr.size else float("nan"),
        "selected_replacement_teacher_improvement_sum": float(arr.sum()) if arr.size else 0.0,
        "selected_replacement_teacher_improvement_mean": float(arr.mean()) if arr.size else float("nan"),
        "selected_replacement_worst_teacher_improvement": float(arr.min()) if arr.size else float("nan"),
        "selected_material_negative_count": float(material.sum()) if arr.size else 0.0,
        "selected_material_negative_rate": float(material.mean()) if arr.size else 0.0,
        "selected_material_negative_excess_sum": float(np.maximum(-arr[material] - tau, 0.0).sum()) if np.any(material) else 0.0,
        "opportunity_count": float(opportunities),
        "opportunity_capture_rate": float(captures / opportunities) if opportunities else float("nan"),
    }


def _crossfit(data: dict[str, Any], mode: str, tau: float, tail: bool, rank_policy: str) -> dict[str, Any]:
    x = np.asarray(data["rep_X"], dtype=np.float64)
    delta = np.asarray(data["rep_delta"], dtype=np.float64)
    toks = np.asarray(data["rep_tok"], dtype=object)
    unique = sorted(set(str(t) for t in toks))
    folds: list[dict[str, Any]] = []
    for fold in range(_FIXED_FOLDS):
        hold_tokens = {t for t in unique if _fold_id(t) == fold}
        if len(hold_tokens) < _MIN_FOLD_SCENES:
            raise SystemExit(f"TRAIN scene-level crossfit fold too small: fold={fold} scenes={len(hold_tokens)}")
        hold = np.asarray([str(t) in hold_tokens for t in toks], dtype=bool)
        fit = ~hold
        score_all = np.full(len(x), np.nan, dtype=np.float64)
        score_all[hold] = _local_score(x[fit], delta[fit], x[hold], mode, tau, tail)
        m = _selection_metrics(data, score_all, hold_tokens, rank_policy, tau)
        safe = bool(
            m["selected_replacement_count"] >= _MIN_SELECTED_PER_FOLD
            and m["selected_replacement_teacher_improvement_sum"] >= -1.0e-9
            and m["selected_material_negative_count"] == 0.0
        ) if tail else bool(
            m["selected_replacement_count"] >= _MIN_SELECTED_PER_FOLD
            and m["selected_replacement_teacher_improvement_sum"] >= -1.0e-9
        )
        folds.append({
            "fold": fold,
            "holdout_unique_scene_count": len(hold_tokens),
            "holdout_edge_count": int(hold.sum()),
            "selected_path": m,
            "path_safe": safe,
        })
    return {
        "mode": mode,
        "tail_objective": bool(tail),
        "rank_policy": rank_policy,
        "fold_seed": _FIXED_FOLD_SEED,
        "neighbor_k_values": list(_FIXED_NEIGHBOR_K_VALUES),
        "se_multiplier": _FIXED_SE_MULTIPLIER,
        "tail_se_multiplier": _FIXED_TAIL_SE_MULTIPLIER if tail else None,
        "material_delta_threshold": float(tau),
        "feature_names": _runtime_feature_names(mode),
        "folds": folds,
        "all_folds_path_safe": bool(all(f["path_safe"] for f in folds)),
        "fold_pass_count": int(sum(bool(f["path_safe"]) for f in folds)),
        "selected_replacement_count": int(sum(f["selected_path"]["selected_replacement_count"] for f in folds)),
        "selected_teacher_improvement_sum": float(sum(f["selected_path"]["selected_replacement_teacher_improvement_sum"] for f in folds)),
        "selected_material_negative_count": int(sum(f["selected_path"]["selected_material_negative_count"] for f in folds)),
        "mean_precision": float(np.nanmean([f["selected_path"]["selected_replacement_precision"] for f in folds])),
        "mean_capture": float(np.nanmean([f["selected_path"]["opportunity_capture_rate"] for f in folds])),
    }


def _save_memory(path: Path, data: dict[str, Any], mode: str, tau: float, include_tail: bool) -> dict[str, Any]:
    x = np.asarray(data["rep_X"], dtype=np.float64)
    delta = np.asarray(data["rep_delta"], dtype=np.float64)
    memory, mean, std, metric_weight = _standardized_metric_memory(x, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "memory_metric_z": memory.astype(np.float32),
        "teacher_improvement": delta.astype(np.float32),
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "feature_names": np.asarray(_runtime_feature_names(mode), dtype="U128"),
        "feature_metric_weight": metric_weight.astype(np.float32),
        "neighbor_k_values": np.asarray(_FIXED_NEIGHBOR_K_VALUES, dtype=np.int32),
        "se_multiplier": np.asarray([_FIXED_SE_MULTIPLIER], dtype=np.float32),
    }
    if include_tail:
        kwargs["material_delta_threshold"] = np.asarray([tau], dtype=np.float32)
        kwargs["tail_se_multiplier"] = np.asarray([_FIXED_TAIL_SE_MULTIPLIER], dtype=np.float32)
    np.savez_compressed(path, **kwargs)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path), "sha256": sha, "row_count": int(len(delta)), "feature_count": int(x.shape[1]),
        "neighbor_k_values": list(_FIXED_NEIGHBOR_K_VALUES), "se_multiplier": _FIXED_SE_MULTIPLIER,
        "tail_se_multiplier": _FIXED_TAIL_SE_MULTIPLIER if include_tail else None,
        "material_delta_threshold": float(tau) if include_tail else None,
        "metric_group_policy": "equal_evidence_and_typed_view_average_squared_distance" if mode == "typed_interaction" else "evidence_average_squared_distance",
    }


def _make_cfg(base: dict[str, Any], mode: str, memory: dict[str, Any], tag: str, tail: bool, rank_policy: str, train_safe: bool) -> dict[str, Any]:
    cfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
    ic = _icer(cfg)
    if not bool(ic.get("enabled", False)):
        raise SystemExit("base V20 ICER config must be enabled")
    ic.update({
        "model_type": "frozen_support_dominance_plus_typed_tail_coherent_local_regret_recovery" if tail else "frozen_support_dominance_plus_typed_local_regret_recovery_ablation",
        "dominance_policy": "scalar_only",
        "incumbent_retention_policy": "preserve_admissible_incumbent",
        "regret_risk_enabled": True,
        "retention_regret_risk_enabled": False,
        "replacement_regret_risk_enabled": True,
        "regret_risk_model_type": "local_multiscale_tail_coherence" if tail else "local_multiscale_regret_lower_bound",
        "regret_risk_feature_mode": mode,
        "replacement_rank_policy": rank_policy,
        "replacement_local_regret_memory_path": memory["path"],
        "replacement_local_regret_memory_sha256": memory["sha256"],
        "replacement_local_regret_neighbor_k_values": list(memory["neighbor_k_values"]),
        "replacement_local_regret_se_multiplier": float(memory["se_multiplier"]),
        "replacement_local_regret_metric_group_policy": memory["metric_group_policy"],
        "replacement_regret_target": "teacher_margin_candidate_minus_teacher_margin_raw_incumbent",
        "regret_risk_training_population": "TRAIN_only_final_guard_admissible_incumbents_then_frozen_support_positive_scalar_dominance_positive_alternatives",
        "regret_risk_threshold_policy": "fixed_zero_no_validation_sweep",
        "typed_evidence_contract": "selected_B_atom_type_and_predicted_g_candidate_minus_incumbent_only_no_new_query",
        "all_flagged_policy": "preserve_legacy_for_structural_guard",
        "train_crossfit_replacement_path_safe": bool(train_safe),
    })
    if tail:
        ic.update({
            "material_delta_threshold": float(memory["material_delta_threshold"]),
            "material_delta_threshold_source": "frozen_fallback_tau_delta_normalized",
            "tail_se_multiplier": float(memory["tail_se_multiplier"]),
            "tail_objective": "multiscale_mean_lower_bound_minus_upper_confidence_lower_partial_moment_beyond_material_delta",
            "train_crossfit_material_negative_selected_count": 0,
        })
    version = "V64.3.24-EAF-ICER-TTCR-DARM-DBR"
    cfg.setdefault("metadata", {})["algorithm_version"] = version
    cfg.setdefault("provenance", {})["algorithm_version"] = version
    exp = cfg.setdefault("experiment", {})
    exp["name"] = f"v64_3_24_eaf_icer_ttcr_{tag}"
    exp["algorithm"] = "V64.3.24 EAF-ICER-TTCR: Typed Tail-Coherent Incumbent-Contrastive Recovery"
    exp["mechanism_chain"] = (
        "fixed B<=16 -> frozen EAF complete frontier + exact selected-evidence attribution -> typed selected-evidence incumbent contrasts -> "
        "frozen support/scalar dominance -> TRAIN-only multiscale mean-and-material-downside coherence -> risk-first replacement with incumbent default -> unchanged final/structural guards"
    )
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit V64.3.24 typed tail-coherent replacement memories/configs from TRAIN frontier only.")
    ap.add_argument("--train-frontier-edges", required=True)
    ap.add_argument("--base-v20-dual-config", required=True)
    ap.add_argument("--output-evidence-memory", required=True)
    ap.add_argument("--output-typed-memory", required=True)
    ap.add_argument("--output-evidence-baseline-config", required=True)
    ap.add_argument("--output-evidence-tail-config", required=True)
    ap.add_argument("--output-typed-lcb-config", required=True)
    ap.add_argument("--output-typed-tail-dominance-config", required=True)
    ap.add_argument("--output-typed-tail-config", required=True)
    ap.add_argument("--output-train-token-file", required=True)
    ap.add_argument("--output-report", required=True)
    a = ap.parse_args()

    by = v22._load_scenes(Path(a.train_frontier_edges))
    evidence = _build_replacement_data(by, "evidence_only")
    typed = _build_replacement_data(by, "typed_interaction")
    # Both builders traverse the exact same frozen eligible population.  Any
    # row mismatch is an instrumentation bug, not an algorithmic result.
    if len(evidence["rep_delta"]) != len(typed["rep_delta"]) or not np.array_equal(evidence["rep_tok"], typed["rep_tok"]) or not np.allclose(evidence["rep_delta"], typed["rep_delta"]):
        raise SystemExit("STOP TRAIN INSTRUMENTATION: evidence and typed replacement populations are not identical")

    base = yaml.safe_load(Path(a.base_v20_dual_config).read_text(encoding="utf-8"))
    bic = _icer(base)
    if str(bic.get("dominance_policy", "")) != "dual_equal_mean" or str(bic.get("all_flagged_policy", "")) != "preserve_legacy_for_structural_guard":
        raise SystemExit("base config must be frozen V20 dual deployment-complete ICER")
    tau = float(base.get("fallback", {}).get("tau_delta_normalized", np.nan))
    if not np.isfinite(tau) or tau <= 0.0:
        raise SystemExit("frozen V20 fallback.tau_delta_normalized is missing/invalid")

    crossfit = {
        "v23_evidence_lcb_scalar": _crossfit(evidence, "evidence_only", tau, tail=False, rank_policy="dominance_first"),
        "evidence_tail_ablation": _crossfit(evidence, "evidence_only", tau, tail=True, rank_policy="dominance_first"),
        "typed_lcb_ablation": _crossfit(typed, "typed_interaction", tau, tail=False, rank_policy="dominance_first"),
        "typed_tail_dominance_ablation": _crossfit(typed, "typed_interaction", tau, tail=True, rank_policy="dominance_first"),
        "typed_tail_main": _crossfit(typed, "typed_interaction", tau, tail=True, rank_policy="regret_risk_first"),
    }
    main_cf = crossfit["typed_tail_main"]
    train_gate_pass = bool(
        main_cf["all_folds_path_safe"]
        and main_cf["selected_replacement_count"] >= _FIXED_FOLDS * _MIN_SELECTED_PER_FOLD
        and main_cf["selected_teacher_improvement_sum"] >= -1.0e-9
        and main_cf["selected_material_negative_count"] == 0
    )

    report = {
        "audit": "v64_3_24_eaf_icer_typed_tail_train_only_fit",
        "algorithm": "V64.3.24 EAF-ICER-TTCR",
        "train_frontier_scene_count": int(len(by)),
        "replacement_population": {
            "edge_count": int(len(typed["rep_delta"])),
            "unique_scene_count": int(len(set(str(t) for t in typed["rep_tok"]))),
            "typed_feature_row_coverage": float(typed["typed_feature_row_coverage"]),
            "feature_count_evidence": int(evidence["rep_X"].shape[1]),
            "feature_count_typed": int(typed["rep_X"].shape[1]),
        },
        "fixed_policy": {
            "fold_count": _FIXED_FOLDS,
            "fold_seed": _FIXED_FOLD_SEED,
            "neighbor_k_values": list(_FIXED_NEIGHBOR_K_VALUES),
            "mean_se_multiplier": _FIXED_SE_MULTIPLIER,
            "tail_se_multiplier": _FIXED_TAIL_SE_MULTIPLIER,
            "material_delta_threshold": tau,
            "material_delta_threshold_source": "frozen V20 fallback.tau_delta_normalized",
            "main_rank_policy": "regret_risk_first",
            "no_validation_tuning": True,
        },
        "crossfit": crossfit,
        "train_gate_pass": train_gate_pass,
        "fresh_validation_was_used_for_fit_or_policy_selection": False,
    }
    out_report = Path(a.output_report); out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    train_tokens = sorted(str(t) for t in by)
    token_path = Path(a.output_train_token_file); token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text("\n".join(train_tokens) + "\n", encoding="utf-8")

    if not train_gate_pass:
        raise SystemExit(
            "STOP TRAIN TTCR: typed tail-coherent selected path is not material-tail-safe in all five fixed folds; do not spend fresh validation GPU"
        )

    evidence_memory = _save_memory(Path(a.output_evidence_memory), evidence, "evidence_only", tau, include_tail=True)
    typed_memory = _save_memory(Path(a.output_typed_memory), typed, "typed_interaction", tau, include_tail=True)
    configs = [
        (a.output_evidence_baseline_config, _make_cfg(base, "evidence_only", evidence_memory, "v23_evidence_lcb_scalar", tail=False, rank_policy="dominance_first", train_safe=bool(crossfit["v23_evidence_lcb_scalar"]["all_folds_path_safe"]))),
        (a.output_evidence_tail_config, _make_cfg(base, "evidence_only", evidence_memory, "evidence_tail_ablation", tail=True, rank_policy="dominance_first", train_safe=bool(crossfit["evidence_tail_ablation"]["all_folds_path_safe"]))),
        (a.output_typed_lcb_config, _make_cfg(base, "typed_interaction", typed_memory, "typed_lcb_ablation", tail=False, rank_policy="dominance_first", train_safe=bool(crossfit["typed_lcb_ablation"]["all_folds_path_safe"]))),
        (a.output_typed_tail_dominance_config, _make_cfg(base, "typed_interaction", typed_memory, "typed_tail_dominance_ablation", tail=True, rank_policy="dominance_first", train_safe=bool(crossfit["typed_tail_dominance_ablation"]["all_folds_path_safe"]))),
        (a.output_typed_tail_config, _make_cfg(base, "typed_interaction", typed_memory, "typed_tail_main", tail=True, rank_policy="regret_risk_first", train_safe=True)),
    ]
    # The typed LCB ablation uses the same memory file.  Old-LCB loading ignores
    # the appended tail metadata; this makes representation the only change.
    for path, cfg in configs:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    report["memories"] = {"evidence_only": evidence_memory, "typed_interaction": typed_memory}
    report["configs"] = [str(x[0]) for x in configs]
    out_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
