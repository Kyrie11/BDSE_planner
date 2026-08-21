from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import bdse.tools.fit_v64_3_27_eaf_icer_trcc as v27

FOLDS = v27.FOLDS
FOLD_SEED = v27.FOLD_SEED
KS = v27.KS
MIN_SELECTED = v27.MIN_SELECTED
MAIN_MIN_SELECTED = v27.MAIN_MIN_SELECTED
EXPECTED_TRAIN_SCENES = v27.EXPECTED_TRAIN_SCENES
# Frozen identity from the independently audited V27 TRAIN instrumentation.
EXPECTED_TRAIN_TOKEN_SHA256 = "b36a847e7a3d7caa3c785ac96b6789ddefed071fae050170482108d950447da4"
EXPECTED_FRONTIER_ROWS = 75133
EXPECTED_REPLACEMENT_EDGES = 1455
EXPECTED_REPLACEMENT_SCENES = 310
EPS = 1.0e-9

# V28 freezes a tail *mode*, not a validation-tuned threshold.  -0.5 lies in
# the V27 TRAIN empty interval [-0.545756..., -0.477867...] and is frozen
# before any V28 fresh scene is selected.
CATASTROPHIC_DELTA_THRESHOLD = -0.5
# Proposal preservation is explicit: calibrate confirmation so that 95% of
# teacher-positive TRAIN aggregate proposals lie in the confirmed region.
POSITIVE_PROPOSAL_COVERAGE = 0.95
VAR_EPS = 1.0e-6

_BASE_NAMES = list(v27._BASE_NAMES)
_TYPE_NAMES = list(v27._TYPE_NAMES)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _icer(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]


def _fit_diag_tail_model(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    catastrophic = y <= CATASTROPHIC_DELTA_THRESHOLD
    if int(catastrophic.sum()) < 32 or int((~catastrophic).sum()) < 128:
        raise SystemExit(
            f"STOP TRAIN TAIL SUPPORT: catastrophic={int(catastrophic.sum())} benign={int((~catastrophic).sum())}"
        )
    mean = X.mean(axis=0)
    std = np.maximum(X.std(axis=0), VAR_EPS)
    Z = (X - mean[None, :]) / std[None, :]
    C = Z[catastrophic]
    B = Z[~catastrophic]
    return {
        "feature_mean": mean,
        "feature_std": std,
        "catastrophic_mean": C.mean(axis=0),
        "catastrophic_var": np.maximum(C.var(axis=0), VAR_EPS),
        "benign_mean": B.mean(axis=0),
        "benign_var": np.maximum(B.var(axis=0), VAR_EPS),
        "catastrophic_count": int(catastrophic.sum()),
        "benign_count": int((~catastrophic).sum()),
    }


def _tail_risk(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    z = (X - model["feature_mean"][None, :]) / model["feature_std"][None, :]
    cm = model["catastrophic_mean"]; cv = model["catastrophic_var"]
    bm = model["benign_mean"]; bv = model["benign_var"]
    log_cat = -0.5 * np.sum(np.log(cv)[None, :] + ((z - cm[None, :]) ** 2) / cv[None, :], axis=1)
    log_benign = -0.5 * np.sum(np.log(bv)[None, :] + ((z - bm[None, :]) ** 2) / bv[None, :], axis=1)
    return log_cat - log_benign


def _higher_quantile(x: np.ndarray, coverage: float) -> float:
    a = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    if not len(a):
        raise SystemExit("STOP TRAIN CALIBRATION: empty positive-proposal risk set")
    # Smallest order statistic with empirical CDF >= coverage.
    k = max(0, min(len(a) - 1, int(np.ceil(float(coverage) * len(a))) - 1))
    return float(a[k])


def _proposal_indices(data: dict[str, Any], aggregate_score: np.ndarray, hold: set[str]) -> list[int]:
    toks, support, scalar, action = data["tok"], data["support"], data["scalar"], data["action"]
    out: list[int] = []
    for token in sorted(hold):
        idx = np.flatnonzero(toks == token)
        accepted = idx[(support[idx] > 0.0) & (scalar[idx] > 0.0) & (aggregate_score[idx] > 0.0)]
        if len(accepted):
            j = sorted(
                accepted.tolist(),
                key=lambda q: (-float(scalar[q]), -float(aggregate_score[q]), int(action[q])),
            )[0]
            out.append(int(j))
    return out


def _pack(values: np.ndarray, opportunities: int, scene_count: int) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    neg = np.minimum(arr, 0.0)
    return {
        "scene_count": int(scene_count),
        "count": int(len(arr)),
        "precision": float(np.mean(arr > 0.0)) if len(arr) else float("nan"),
        "sum": float(arr.sum()) if len(arr) else 0.0,
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "worst": float(arr.min()) if len(arr) else float("nan"),
        "negative_rms": float(np.sqrt(np.mean(neg * neg))) if len(arr) else float("nan"),
        "negative_sq_sum": float(np.sum(neg * neg)) if len(arr) else 0.0,
        "opportunities": int(opportunities),
        "capture": float(np.sum(arr > 0.0) / opportunities) if opportunities else float("nan"),
    }


def _crossfit(data: dict[str, Any]) -> dict[str, Any]:
    A, T, y, toks = data["X_aggregate"], data["X_type"], data["delta"], data["tok"]
    unique = sorted(set(map(str, toks)))

    agg_scores: dict[int, np.ndarray] = {}
    proposals: dict[int, list[int]] = {}
    type_knn_scores: dict[int, np.ndarray] = {}
    for f in range(FOLDS):
        hold = {t for t in unique if v27._fold(t) == f}
        hm = np.asarray([str(t) in hold for t in toks], dtype=bool)
        fit = ~hm
        ag = np.full(len(y), np.nan, dtype=np.float64)
        ty = np.full(len(y), np.nan, dtype=np.float64)
        ag[hm] = v27._score(A[fit], y[fit], A[hm])
        ty[hm] = v27._score(T[fit], y[fit], T[hm])
        agg_scores[f] = ag
        type_knn_scores[f] = ty
        proposals[f] = _proposal_indices(data, ag, hold)

    control_folds: list[dict[str, Any]] = []
    type_knn_folds: list[dict[str, Any]] = []
    global_agg_folds: list[dict[str, Any]] = []
    global_concat_folds: list[dict[str, Any]] = []
    main_folds: list[dict[str, Any]] = []
    main_subset_ok = True
    veto_rows: list[dict[str, Any]] = []

    for f in range(FOLDS):
        hold = {t for t in unique if v27._fold(t) == f}
        hm = np.asarray([str(t) in hold for t in toks], dtype=bool)
        fit = ~hm
        pidx = np.asarray(proposals[f], dtype=np.int64)
        train_pidx = np.asarray([j for g in range(FOLDS) if g != f for j in proposals[g]], dtype=np.int64)
        if len(train_pidx) < 40 or np.sum(y[train_pidx] > 0.0) < 24:
            raise SystemExit(f"STOP TRAIN CALIBRATION SUPPORT: fold={f} train_proposals={len(train_pidx)} positive={int(np.sum(y[train_pidx] > 0.0))}")

        opportunities = 0
        for token in hold:
            idx = np.flatnonzero(toks == token)
            opportunities += int(np.any(y[idx] > 0.0))

        ctrl_vals = y[pidx]
        ctrl = _pack(ctrl_vals, opportunities, len(hold))
        ctrl["fold"] = int(f); ctrl["hold_scenes"] = int(len(hold))
        ctrl["path_safe"] = bool(ctrl["count"] >= MIN_SELECTED and ctrl["sum"] >= -EPS)
        control_folds.append(ctrl)

        # Historical V27 local type-KNN confirmation, retained as a diagnostic
        # using the exact same representation.  It isolates estimator choice.
        knn_keep = np.asarray([type_knn_scores[f][j] > 0.0 for j in pidx], dtype=bool)
        knn = _pack(y[pidx[knn_keep]], opportunities, len(hold))
        knn.update({"fold": int(f), "hold_scenes": int(len(hold)), "path_safe": bool(knn["count"] >= MIN_SELECTED and knn["sum"] >= -EPS)})
        type_knn_folds.append(knn)

        # Same global tail-mode estimator on aggregate features: representation
        # control.  It must not be mistaken for the main mechanism.
        agg_model = _fit_diag_tail_model(A[fit], y[fit])
        agg_train_risk = _tail_risk(agg_model, A[train_pidx])
        agg_thr = _higher_quantile(agg_train_risk[y[train_pidx] > 0.0], POSITIVE_PROPOSAL_COVERAGE)
        agg_hold_risk = _tail_risk(agg_model, A[pidx])
        agg_keep = agg_hold_risk <= agg_thr
        gad = _pack(y[pidx[agg_keep]], opportunities, len(hold))
        gad.update({"fold": int(f), "hold_scenes": int(len(hold)), "risk_threshold": float(agg_thr), "path_safe": bool(gad["count"] >= MIN_SELECTED and gad["sum"] >= -EPS)})
        global_agg_folds.append(gad)

        # Same global estimator on a naive aggregate+type concatenation.  This
        # diagnostic explicitly tests whether V28 works merely because it has
        # more coordinates; it must stay separate from the main mechanism.
        AT = np.concatenate([A, T], axis=1)
        concat_model = _fit_diag_tail_model(AT[fit], y[fit])
        concat_train_risk = _tail_risk(concat_model, AT[train_pidx])
        concat_thr = _higher_quantile(concat_train_risk[y[train_pidx] > 0.0], POSITIVE_PROPOSAL_COVERAGE)
        concat_hold_risk = _tail_risk(concat_model, AT[pidx])
        concat_keep = concat_hold_risk <= concat_thr
        gcd = _pack(y[pidx[concat_keep]], opportunities, len(hold))
        gcd.update({"fold": int(f), "hold_scenes": int(len(hold)), "risk_threshold": float(concat_thr), "path_safe": bool(gcd["count"] >= MIN_SELECTED and gcd["sum"] >= -EPS)})
        global_concat_folds.append(gcd)

        # V28 main: global type-resolved catastrophic-mode contrast.  The same
        # aggregate proposal is the only candidate; confirmation cannot re-rank.
        type_model = _fit_diag_tail_model(T[fit], y[fit])
        train_risk = _tail_risk(type_model, T[train_pidx])
        pos_train = y[train_pidx] > 0.0
        threshold = _higher_quantile(train_risk[pos_train], POSITIVE_PROPOSAL_COVERAGE)
        hold_risk = _tail_risk(type_model, T[pidx])
        keep = hold_risk <= threshold
        kept_idx = pidx[keep]
        vals = y[kept_idx]
        main = _pack(vals, opportunities, len(hold))
        main.update({
            "fold": int(f), "hold_scenes": int(len(hold)),
            "aggregate_proposal_count": int(len(pidx)),
            "tail_mode_veto_count": int(np.sum(~keep)),
            "tail_mode_veto_teacher_positive": int(np.sum(y[pidx[~keep]] > 0.0)),
            "tail_mode_veto_teacher_nonpositive": int(np.sum(y[pidx[~keep]] <= 0.0)),
            "risk_threshold": float(threshold),
            "calibration_positive_proposal_count": int(np.sum(pos_train)),
            "calibration_positive_proposal_coverage": float(POSITIVE_PROPOSAL_COVERAGE),
            "selected_subset_of_aggregate_control": True,
            "path_safe": bool(main["count"] >= MIN_SELECTED and main["sum"] >= -EPS),
        })
        main_folds.append(main)
        for j, risk in zip(pidx.tolist(), hold_risk.tolist()):
            if risk > threshold:
                veto_rows.append({
                    "fold": int(f), "scenario_token": str(toks[j]), "action": int(data["action"][j]),
                    "teacher_improvement": float(y[j]), "tail_log_likelihood_ratio": float(risk),
                    "risk_threshold": float(threshold),
                })

    def combine(mode: str, folds: list[dict[str, Any]]) -> dict[str, Any]:
        n = int(sum(x["count"] for x in folds))
        neg_sq = float(sum(x["negative_sq_sum"] for x in folds))
        finite_worst = [x["worst"] for x in folds if np.isfinite(x["worst"])]
        return {
            "mode": mode,
            "folds": folds,
            "all_folds_path_safe": bool(all(x["path_safe"] for x in folds)),
            "fold_pass_count": int(sum(x["path_safe"] for x in folds)),
            "selected_count": n,
            "teacher_improvement_sum": float(sum(x["sum"] for x in folds)),
            "selected_negative_rms": float(np.sqrt(neg_sq / n)) if n else float("nan"),
            "selected_worst": float(min(finite_worst)) if finite_worst else float("nan"),
            "mean_precision": float(np.nanmean([x["precision"] for x in folds])),
            "mean_capture": float(np.nanmean([x["capture"] for x in folds])),
        }

    return {
        "v25_aggregate_downside_control": combine("aggregate_local_downside", control_folds),
        "v27_type_knn_confirmation_diagnostic": combine("type_local_knn_confirmation", type_knn_folds),
        "global_aggregate_tail_mode_diagnostic": combine("global_aggregate_tail_mode_confirmation", global_agg_folds),
        "global_concat_tail_mode_diagnostic": combine("global_aggregate_plus_type_tail_mode_confirmation", global_concat_folds),
        "proposal_conditioned_type_tail_mode_main": {
            **combine("aggregate_propose_global_type_tail_mode_confirm_no_fallback", main_folds),
            "selected_subset_of_aggregate_control_all_folds": bool(main_subset_ok),
            "aggregate_proposal_count": int(sum(len(proposals[f]) for f in range(FOLDS))),
            "tail_mode_veto_count": int(sum(x["tail_mode_veto_count"] for x in main_folds)),
            "tail_mode_veto_teacher_positive": int(sum(x["tail_mode_veto_teacher_positive"] for x in main_folds)),
            "tail_mode_veto_teacher_nonpositive": int(sum(x["tail_mode_veto_teacher_nonpositive"] for x in main_folds)),
            "veto_rows": veto_rows,
        },
        "proposal_indices": proposals,
    }


def _save_tail_model(path: Path, X: np.ndarray, y: np.ndarray, proposal_indices: list[int], names: list[str]) -> dict[str, Any]:
    model = _fit_diag_tail_model(X, y)
    pidx = np.asarray(proposal_indices, dtype=np.int64)
    risk = _tail_risk(model, X[pidx])
    pos = y[pidx] > 0.0
    threshold = _higher_quantile(risk[pos], POSITIVE_PROPOSAL_COVERAGE)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        feature_mean=np.asarray(model["feature_mean"], dtype="f4"),
        feature_std=np.asarray(model["feature_std"], dtype="f4"),
        feature_names=np.asarray(names, dtype="U192"),
        catastrophic_mean=np.asarray(model["catastrophic_mean"], dtype="f4"),
        catastrophic_var=np.asarray(model["catastrophic_var"], dtype="f4"),
        benign_mean=np.asarray(model["benign_mean"], dtype="f4"),
        benign_var=np.asarray(model["benign_var"], dtype="f4"),
        risk_threshold=np.asarray([threshold], dtype="f4"),
        catastrophic_delta_threshold=np.asarray([CATASTROPHIC_DELTA_THRESHOLD], dtype="f4"),
        positive_proposal_coverage=np.asarray([POSITIVE_PROPOSAL_COVERAGE], dtype="f4"),
        catastrophic_count=np.asarray([model["catastrophic_count"]], dtype="i4"),
        benign_count=np.asarray([model["benign_count"]], dtype="i4"),
        calibration_proposal_count=np.asarray([len(pidx)], dtype="i4"),
        calibration_positive_proposal_count=np.asarray([int(pos.sum())], dtype="i4"),
    )
    return {
        "path": str(path), "sha256": _sha256_file(path), "feature_count": int(X.shape[1]),
        "feature_names": names, "catastrophic_delta_threshold": float(CATASTROPHIC_DELTA_THRESHOLD),
        "positive_proposal_coverage": float(POSITIVE_PROPOSAL_COVERAGE), "risk_threshold": float(threshold),
        "catastrophic_count": int(model["catastrophic_count"]), "benign_count": int(model["benign_count"]),
        "calibration_proposal_count": int(len(pidx)), "calibration_positive_proposal_count": int(pos.sum()),
    }


def _cfg_main(base: dict[str, Any], aggregate_memory: dict[str, Any], tail_model: dict[str, Any]) -> dict[str, Any]:
    cfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
    ic = _icer(cfg)
    ic.update({
        "model_type": "frozen_support_scalar_dominance_plus_aggregate_downside_proposal_global_type_tail_mode_confirmation",
        "dominance_policy": "scalar_only",
        "incumbent_retention_policy": "preserve_admissible_incumbent",
        "regret_risk_enabled": True,
        "retention_regret_risk_enabled": False,
        "replacement_regret_risk_enabled": True,
        "regret_risk_model_type": "local_multiscale_downside_regret_with_global_type_tail_confirmation",
        "regret_risk_feature_mode": "evidence_only",
        "replacement_local_regret_memory_path": aggregate_memory["path"],
        "replacement_local_regret_memory_sha256": aggregate_memory["sha256"],
        "replacement_local_regret_neighbor_k_values": list(KS),
        "replacement_local_regret_certificate": "mean_minus_downside_rms",
        "replacement_confirmation_regret_risk_feature_mode": "semantic_type_only",
        "replacement_confirmation_tail_mode_model_path": tail_model["path"],
        "replacement_confirmation_tail_mode_model_sha256": tail_model["sha256"],
        "replacement_confirmation_tail_mode_label_threshold": float(CATASTROPHIC_DELTA_THRESHOLD),
        "replacement_confirmation_positive_proposal_coverage": float(POSITIVE_PROPOSAL_COVERAGE),
        "replacement_regret_training_population": "TRAIN_only_final_guard_admissible_support_positive_scalar_dominance_positive_alternatives",
        "replacement_operator": (
            "preserve admissible incumbent by default; frozen V25 aggregate local-downside DRC proposes exactly one extremal alternative; "
            "a global type-resolved catastrophic-mode likelihood-ratio model confirms only that same candidate under a TRAIN-calibrated 95% teacher-positive proposal coverage contract; "
            "failed confirmation returns directly to incumbent with NO fallback/reselection"
        ),
        "replacement_selection_monotonicity": "selected_replacements_are_subset_of_V25_aggregate_DRC_selected_replacements_no_fallback",
        "all_flagged_policy": "preserve_legacy_for_structural_guard",
    })
    version = "V64.3.28-EAF-ICER-PTMC-DARM-DBR"
    cfg.setdefault("metadata", {})["algorithm_version"] = version
    cfg.setdefault("provenance", {})["algorithm_version"] = version
    exp = cfg.setdefault("experiment", {})
    exp["name"] = "v64_3_28_proposal_conditioned_tail_mode_confirmation"
    exp["algorithm"] = "V64.3.28 EAF-ICER-PTMC: Proposal-Conditioned Tail-Mode Confirmation"
    exp["mechanism_chain"] = (
        "fixed B<=16 -> frozen EAF complete DARM-anchor frontier -> deployment-admissible frontier -> frozen support/scalar dominance -> "
        "aggregate downside proposal -> global type-resolved catastrophic-mode confirmation of the same candidate -> no-fallback incumbent preservation -> "
        "unchanged one-sided/evidence certificate -> unchanged structural-risk guard -> final decision preservation"
    )
    return cfg


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-frontier-edges", required=True)
    ap.add_argument("--base-v20-dual-config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--output-train-token-file", required=True)
    ap.add_argument("--output-report", required=True)
    args = ap.parse_args()

    edge_path = Path(args.train_frontier_edges)
    if not edge_path.is_file() or edge_path.stat().st_size <= 0:
        raise SystemExit(f"STOP TRAIN DATA: missing V27 type-resolved TRAIN frontier provenance {edge_path}")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    by, frontier_row_count = v27._load_minimal_scenes(edge_path)
    if len(by) != EXPECTED_TRAIN_SCENES:
        raise SystemExit(f"STOP TRAIN DATA: expected exactly {EXPECTED_TRAIN_SCENES} frozen TRAIN scenes, got {len(by)}")
    tokens = sorted(by)
    token_bytes = ("\n".join(tokens) + "\n").encode("utf-8")
    token_sha = hashlib.sha256(token_bytes).hexdigest()
    if token_sha != EXPECTED_TRAIN_TOKEN_SHA256:
        raise SystemExit(f"STOP TRAIN DATA: frozen TRAIN token identity mismatch sha256={token_sha}")
    if int(frontier_row_count) != EXPECTED_FRONTIER_ROWS:
        raise SystemExit(f"STOP TRAIN DATA: frozen frontier row count drifted {frontier_row_count} != {EXPECTED_FRONTIER_ROWS}")
    data = v27._build(by)
    if len(data["delta"]) != EXPECTED_REPLACEMENT_EDGES or int(data["replacement_scene_count"]) != EXPECTED_REPLACEMENT_SCENES:
        raise SystemExit(
            f"STOP TRAIN DATA: frozen replacement population drifted edges={len(data['delta'])} scenes={data['replacement_scene_count']}"
        )
    cf = _crossfit(data)
    control = cf["v25_aggregate_downside_control"]
    main_cf = cf["proposal_conditioned_type_tail_mode_main"]

    tail_nonworse = bool(
        main_cf["selected_negative_rms"] <= control["selected_negative_rms"] + EPS
        and main_cf["selected_worst"] >= control["selected_worst"] - EPS
    )
    tail_strict = bool(
        main_cf["selected_negative_rms"] < control["selected_negative_rms"] - EPS
        or main_cf["selected_worst"] > control["selected_worst"] + EPS
    )
    gate = bool(
        main_cf["all_folds_path_safe"]
        and main_cf["selected_count"] >= MAIN_MIN_SELECTED
        and main_cf["teacher_improvement_sum"] >= -EPS
        and main_cf["selected_subset_of_aggregate_control_all_folds"]
        and tail_nonworse and tail_strict
    )

    token_path = Path(args.output_train_token_file); token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_bytes(token_bytes)
    y = data["delta"]
    report: dict[str, Any] = {
        "audit": "v64_3_28_eaf_icer_ptmc_train_fit",
        "algorithm": "V64.3.28 EAF-ICER-PTMC",
        "train_scene_count": int(len(by)), "frontier_row_count": int(frontier_row_count),
        "replacement_edges": int(len(y)), "replacement_scenes": int(data["replacement_scene_count"]),
        "aggregate_feature_count": int(data["X_aggregate"].shape[1]), "semantic_type_feature_count": int(data["X_type"].shape[1]),
        "semantic_type_feature_names": _TYPE_NAMES,
        "population_teacher_positive_fraction": float(np.mean(y > 0.0)),
        "population_teacher_improvement_sum": float(y.sum()), "population_teacher_improvement_mean": float(y.mean()), "population_teacher_improvement_worst": float(y.min()),
        "catastrophic_delta_threshold": float(CATASTROPHIC_DELTA_THRESHOLD),
        "catastrophic_edge_count": int(np.sum(y <= CATASTROPHIC_DELTA_THRESHOLD)),
        "nearest_below_catastrophic_threshold": float(np.max(y[y <= CATASTROPHIC_DELTA_THRESHOLD])),
        "nearest_above_catastrophic_threshold": float(np.min(y[y > CATASTROPHIC_DELTA_THRESHOLD])),
        "positive_proposal_coverage": float(POSITIVE_PROPOSAL_COVERAGE),
        "fold_seed": FOLD_SEED, "neighbor_k_values": list(KS), "downside_multiplier": 1.0, "decision_boundary": 0.0,
        "crossfit": {k:v for k,v in cf.items() if k != "proposal_indices"},
        "tail_mode_confirmation_incremental_on_train": bool(tail_nonworse and tail_strict),
        "train_gate_pass": bool(gate),
        "gate_contract": {
            "main": "aggregate_propose_global_type_tail_mode_confirm_no_fallback",
            "all_5_scene_folds_selected_path_nonharmful": True,
            "selected_count_min": MAIN_MIN_SELECTED,
            "teacher_improvement_sum_min": 0.0,
            "selected_replacements_must_be_subset_of_aggregate_control": True,
            "selected_negative_rms_nonworse_than_v25_aggregate": True,
            "selected_worst_nonworse_than_v25_aggregate": True,
            "at_least_one_tail_metric_strictly_better": True,
            "fresh_validation_must_not_be_used_on_train_gate_fail": True,
        },
        "input_frontier": {"path": str(edge_path), "bytes": int(edge_path.stat().st_size), "sha256": _sha256_file(edge_path)},
        "train_token_manifest": {"path": str(token_path), "count": int(len(tokens)), "sha256": _sha256_file(token_path), "expected_sha256": EXPECTED_TRAIN_TOKEN_SHA256},
        "frozen_population_contract": {"frontier_rows": EXPECTED_FRONTIER_ROWS, "replacement_edges": EXPECTED_REPLACEMENT_EDGES, "replacement_scenes": EXPECTED_REPLACEMENT_SCENES},
        "fresh_validation_used": False, "memories": {}, "models": {}, "configs": {},
        "scientific_note": (
            "V28 was designed after inspection of V27 TRAIN and therefore this TRAIN crossfit is a fail-closed implementation/design check, not independent paper evidence. "
            "Only new untouched double-fresh A/B can support the V28 mechanism claim. V28 changes the estimator, not the type representation: V27 local type-KNN is replaced by a global class-conditional catastrophic-mode contrast calibrated for 95% teacher-positive proposal coverage."
        ),
    }
    _write_report(Path(args.output_report), report)
    if not gate:
        raise SystemExit(
            "STOP TRAIN PTMC: global proposal-conditioned type tail-mode confirmation did not provide 5/5 path safety, >=64 retained replacements, subset monotonicity, and strictly non-worse tail over V25 aggregate DRC; do not tune catastrophic threshold/coverage/K/downside/support/dominance on validation"
        )

    base = yaml.safe_load(Path(args.base_v20_dual_config).read_text(encoding="utf-8"))
    agg_names = [f"evidence::{name}" for name in _BASE_NAMES]
    type_names = [f"semantic_type::{name}" for name in _TYPE_NAMES]
    aggregate_memory = v27._save_memory(out_dir / "v64_3_28_aggregate_downside_memory.npz", data["X_aggregate"], y, agg_names, "evidence_only")
    proposal_indices = [j for f in range(FOLDS) for j in cf["proposal_indices"][f]]
    tail_model = _save_tail_model(out_dir / "v64_3_28_type_tail_mode_model.npz", data["X_type"], y, proposal_indices, type_names)
    agg_cfg = out_dir / "v64_3_28_aggregate_downside.yaml"
    main_cfg = out_dir / "v64_3_28_tail_mode_confirmed.yaml"
    agg_cfg.write_text(yaml.safe_dump(v27._cfg_aggregate(base, aggregate_memory), sort_keys=False), encoding="utf-8")
    main_cfg.write_text(yaml.safe_dump(_cfg_main(base, aggregate_memory, tail_model), sort_keys=False), encoding="utf-8")
    report["memories"] = {"aggregate_downside": aggregate_memory}
    report["models"] = {"type_tail_mode": tail_model}
    report["configs"] = {"aggregate_downside": str(agg_cfg), "tail_mode_confirmed": str(main_cfg)}
    _write_report(Path(args.output_report), report)
    print(json.dumps({
        "pass": True, "train_gate_pass": True,
        "main_fold_pass_count": main_cf["fold_pass_count"], "main_count": main_cf["selected_count"],
        "main_sum": main_cf["teacher_improvement_sum"], "main_negative_rms": main_cf["selected_negative_rms"], "main_worst": main_cf["selected_worst"],
        "tail_mode_confirmation_incremental_on_train": bool(tail_nonworse and tail_strict),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
