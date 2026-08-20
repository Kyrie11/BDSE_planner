from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import bdse.tools.fit_v64_3_22_eaf_icer_tcr as v22
from bdse.planner.tournament import _ICER_SEMANTIC_FAMILY_FEATURE_NAMES

FOLDS = 5
# Keep the same scene-fold assignment used since V23.  Changing it after seeing
# V25 fresh would be an implicit TRAIN split search.
FOLD_SEED = "v64.3.23-eaf-icer-rcr-scene-crossfit-v1"
KS = (32, 64)
MIN_EDGES = 1024
MIN_SCENES = 256
MIN_FOLD_SCENES = 40
MIN_SELECTED = 8
MAIN_MIN_SELECTED = 64
EXPECTED_TRAIN_SCENES = 3000
EPS = 1.0e-9

_BASE_NAMES = list(v22._ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
_SEMANTIC_NAMES = list(_ICER_SEMANTIC_FAMILY_FEATURE_NAMES)
_KEEP_KEYS = {
    "scenario_token", "anchor_action", "raw_top_action", "challenger_action",
    "icer_admissible", "dacer_admissible", "teacher_margin",
    "icer_support_logit", "icer_scalar_dominance_logit",
} | {f"icer_feature_{name}" for name in _BASE_NAMES} | {
    f"icer_semantic_family_{name}" for name in _SEMANTIC_NAMES
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _icer(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]


def _fold(token: str) -> int:
    h = hashlib.sha256((FOLD_SEED + "::" + token).encode()).digest()
    return int.from_bytes(h[:8], "big") % FOLDS


def _load_minimal_scenes(path: Path) -> tuple[dict[str, list[dict[str, Any]]], int]:
    by: dict[str, list[dict[str, Any]]] = {}
    row_count = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                full = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"STOP TRAIN DATA: malformed frontier JSONL line {line_no}: {exc}") from exc
            token = str(full.get("scenario_token", ""))
            if not token:
                raise SystemExit(f"STOP TRAIN DATA: empty scenario_token at frontier line {line_no}")
            by.setdefault(token, []).append({k: full.get(k) for k in _KEEP_KEYS})
            row_count += 1
    if not by:
        raise SystemExit(f"STOP TRAIN DATA: no frontier rows in {path}")
    return by, row_count


def _finite(r: dict[str, Any], key: str, default: float = np.nan) -> float:
    try:
        v = float(r.get(key, default))
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _build(by: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    agg: list[list[float]] = []
    sem: list[list[float]] = []
    delta: list[float] = []
    tok: list[str] = []
    sup: list[float] = []
    scalar: list[float] = []
    action: list[int] = []
    eligible_before_feature = 0
    missing_aggregate = 0
    missing_semantic = 0
    replacement_scenes: set[str] = set()

    for token, rows in by.items():
        if not rows:
            continue
        anchor = int(rows[0].get("anchor_action", -1))
        incumbent = int(rows[0].get("raw_top_action", -1))
        if incumbent < 0 or incumbent == anchor:
            continue
        inc_row = next((r for r in rows if int(r.get("challenger_action", -999)) == incumbent), None)
        if inc_row is None or _finite(inc_row, "icer_admissible", _finite(inc_row, "dacer_admissible", 0.0)) < 0.5:
            continue
        inc_tm = _finite(inc_row, "teacher_margin")
        if not np.isfinite(inc_tm):
            continue

        for r in rows:
            challenger = int(r.get("challenger_action", -1))
            if challenger < 0 or challenger in {anchor, incumbent}:
                continue
            if _finite(r, "icer_admissible", _finite(r, "dacer_admissible", 0.0)) < 0.5:
                continue
            tm = _finite(r, "teacher_margin")
            support = _finite(r, "icer_support_logit")
            dominance = _finite(r, "icer_scalar_dominance_logit")
            if not all(np.isfinite(x) for x in (tm, support, dominance)) or not (support > 0.0 and dominance > 0.0):
                continue
            eligible_before_feature += 1
            a = [_finite(r, f"icer_feature_{name}") for name in _BASE_NAMES]
            s = [_finite(r, f"icer_semantic_family_{name}") for name in _SEMANTIC_NAMES]
            if not all(np.isfinite(x) for x in a):
                missing_aggregate += 1
                continue
            if not all(np.isfinite(x) for x in s):
                missing_semantic += 1
                continue
            agg.append(a)
            sem.append(s)
            delta.append(tm - inc_tm)
            tok.append(token)
            sup.append(support)
            scalar.append(dominance)
            action.append(challenger)
            replacement_scenes.add(token)

    if eligible_before_feature < MIN_EDGES:
        raise SystemExit(f"STOP TRAIN SUPPORT: insufficient frozen replacement population: {eligible_before_feature}")
    if missing_aggregate:
        raise SystemExit(
            f"STOP TRAIN INSTRUMENTATION: missing aggregate evidence features on {missing_aggregate}/{eligible_before_feature} eligible replacement edges"
        )
    if missing_semantic:
        raise SystemExit(
            f"STOP TRAIN INSTRUMENTATION: missing semantic-family features on {missing_semantic}/{eligible_before_feature} eligible replacement edges"
        )
    if len(replacement_scenes) < MIN_SCENES:
        raise SystemExit(f"STOP TRAIN SUPPORT: insufficient replacement scenes: {len(replacement_scenes)} < {MIN_SCENES}")
    A = np.asarray(agg, dtype=np.float64)
    S = np.asarray(sem, dtype=np.float64)
    return {
        "X_aggregate": A,
        "X_semantic": np.concatenate([A, S], axis=1),
        "delta": np.asarray(delta, dtype=np.float64),
        "tok": np.asarray(tok, dtype=object),
        "support": np.asarray(sup, dtype=np.float64),
        "scalar": np.asarray(scalar, dtype=np.float64),
        "action": np.asarray(action, dtype=np.int64),
        "eligible_before_feature": int(eligible_before_feature),
        "replacement_scene_count": int(len(replacement_scenes)),
    }


def _memory(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = np.maximum(X.std(axis=0), 1.0e-6)
    # One equal weight per fixed coordinate.  No family-group weight is added or
    # tuned after seeing V25 fresh results.
    metric_weight = np.full(X.shape[1], 1.0 / X.shape[1], dtype=np.float64)
    Z = ((X - mean[None, :]) / std[None, :]) * np.sqrt(metric_weight[None, :])
    return Z, mean, std, metric_weight


def _score(trainX: np.ndarray, trainY: np.ndarray, qX: np.ndarray) -> np.ndarray:
    Z, mean, std, metric_weight = _memory(trainX)
    Q = ((qX - mean[None, :]) / std[None, :]) * np.sqrt(metric_weight[None, :])
    d2 = np.maximum(
        np.sum(Q * Q, axis=1)[:, None] + np.sum(Z * Z, axis=1)[None, :] - 2.0 * (Q @ Z.T),
        0.0,
    )
    rows = np.arange(len(Q))[:, None]
    bounds: list[np.ndarray] = []
    for k in KS:
        kk = min(k, len(Z))
        nbr = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        dist = np.sqrt(d2[rows, nbr])
        w = 1.0 / np.maximum(dist, 1.0e-6)
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1.0e-12)
        y = trainY[nbr]
        mu = np.sum(w * y, axis=1)
        downside = np.minimum(y, 0.0)
        bounds.append(mu - np.sqrt(np.sum(w * downside * downside, axis=1)))
    return np.min(np.stack(bounds, axis=1), axis=1)


def _selection(data: dict[str, Any], score: np.ndarray, hold: set[str]) -> dict[str, Any]:
    toks, support, scalar, delta, action = (
        data["tok"], data["support"], data["scalar"], data["delta"], data["action"]
    )
    selected: list[float] = []
    opportunities = captured = scene_count = 0
    for token in sorted(hold):
        idx = np.flatnonzero(toks == token)
        if not len(idx):
            continue
        scene_count += 1
        opportunities += int(np.any(delta[idx] > 0.0))
        accepted = idx[(support[idx] > 0.0) & (scalar[idx] > 0.0) & (score[idx] > 0.0)]
        if len(accepted):
            j = sorted(accepted.tolist(), key=lambda q: (-float(scalar[q]), -float(score[q]), int(action[q])))[0]
            selected.append(float(delta[j]))
            captured += int(delta[j] > 0.0)
    arr = np.asarray(selected, dtype=np.float64)
    negative = np.minimum(arr, 0.0)
    return {
        "scene_count": int(scene_count),
        "count": int(len(arr)),
        "precision": float(np.mean(arr > 0.0)) if len(arr) else float("nan"),
        "sum": float(arr.sum()) if len(arr) else 0.0,
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "worst": float(arr.min()) if len(arr) else float("nan"),
        "negative_rms": float(np.sqrt(np.mean(negative * negative))) if len(arr) else float("nan"),
        "negative_sq_sum": float(np.sum(negative * negative)) if len(arr) else 0.0,
        "opportunities": int(opportunities),
        "capture": float(captured / opportunities) if opportunities else float("nan"),
    }


def _crossfit(data: dict[str, Any], feature_mode: str) -> dict[str, Any]:
    X = data["X_aggregate"] if feature_mode == "aggregate" else data["X_semantic"]
    y, toks = data["delta"], data["tok"]
    unique = sorted(set(map(str, toks)))
    if len(X) < MIN_EDGES or len(unique) < MIN_SCENES:
        raise SystemExit(f"STOP TRAIN SUPPORT {feature_mode}: edges={len(X)} scenes={len(unique)}")
    folds: list[dict[str, Any]] = []
    for f in range(FOLDS):
        hold = {t for t in unique if _fold(t) == f}
        if len(hold) < MIN_FOLD_SCENES:
            raise SystemExit(f"STOP TRAIN SPLIT: fold too small {f}: {len(hold)}")
        hold_mask = np.asarray([str(t) in hold for t in toks], dtype=bool)
        fit_mask = ~hold_mask
        score = np.full(len(X), np.nan, dtype=np.float64)
        score[hold_mask] = _score(X[fit_mask], y[fit_mask], X[hold_mask])
        metrics = _selection(data, score, hold)
        metrics["fold"] = int(f)
        metrics["hold_scenes"] = int(len(hold))
        metrics["path_safe"] = bool(metrics["count"] >= MIN_SELECTED and metrics["sum"] >= -EPS)
        folds.append(metrics)
    n = int(sum(x["count"] for x in folds))
    neg_sq = float(sum(x["negative_sq_sum"] for x in folds))
    return {
        "mode": "aggregate_evidence_only" if feature_mode == "aggregate" else "aggregate_plus_identity_preserving_semantic_family",
        "certificate": "downside_rms",
        "folds": folds,
        "all_folds_path_safe": bool(all(x["path_safe"] for x in folds)),
        "fold_pass_count": int(sum(x["path_safe"] for x in folds)),
        "selected_count": n,
        "teacher_improvement_sum": float(sum(x["sum"] for x in folds)),
        "selected_negative_rms": float(np.sqrt(neg_sq / n)) if n else float("nan"),
        "selected_worst": float(min(x["worst"] for x in folds if np.isfinite(x["worst"]))) if n else float("nan"),
        "mean_precision": float(np.nanmean([x["precision"] for x in folds])),
        "mean_capture": float(np.nanmean([x["capture"] for x in folds])),
    }


def _feature_names(feature_mode: str) -> list[str]:
    names = [f"evidence::{name}" for name in _BASE_NAMES]
    if feature_mode == "semantic_family_aligned":
        names += [f"semantic_family::{name}" for name in _SEMANTIC_NAMES]
    return names


def _save_memory(path: Path, data: dict[str, Any], feature_mode: str) -> dict[str, Any]:
    X = data["X_aggregate"] if feature_mode == "evidence_only" else data["X_semantic"]
    Z, mean, std, metric_weight = _memory(X)
    names = _feature_names(feature_mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        memory_metric_z=Z.astype("f4"),
        teacher_improvement=data["delta"].astype("f4"),
        feature_mean=mean.astype("f4"),
        feature_std=std.astype("f4"),
        feature_names=np.asarray(names, dtype="U160"),
        feature_metric_weight=metric_weight.astype("f4"),
        neighbor_k_values=np.asarray(KS, dtype="i4"),
        se_multiplier=np.asarray([1.0], dtype="f4"),
        certificate_kind=np.asarray(["mean_minus_downside_rms"], dtype="U64"),
        downside_multiplier=np.asarray([1.0], dtype="f4"),
    )
    return {
        "path": str(path), "sha256": _sha256_file(path), "row_count": int(len(data["delta"])),
        "feature_count": int(X.shape[1]), "mode": feature_mode, "certificate": "downside_rms",
    }


def _cfg(base: dict[str, Any], memory: dict[str, Any], feature_mode: str, tag: str) -> dict[str, Any]:
    cfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
    ic = _icer(cfg)
    semantic = feature_mode == "semantic_family_aligned"
    ic.update({
        "model_type": (
            "frozen_support_scalar_dominance_plus_semantic_family_aligned_downside_regret_certificate"
            if semantic else "frozen_support_scalar_dominance_plus_evidence_local_downside_regret_certificate"
        ),
        "dominance_policy": "scalar_only",
        "incumbent_retention_policy": "preserve_admissible_incumbent",
        "regret_risk_enabled": True,
        "retention_regret_risk_enabled": False,
        "replacement_regret_risk_enabled": True,
        "regret_risk_model_type": "local_multiscale_downside_regret_certificate",
        "regret_risk_feature_mode": feature_mode,
        "replacement_local_regret_memory_path": memory["path"],
        "replacement_local_regret_memory_sha256": memory["sha256"],
        "replacement_local_regret_neighbor_k_values": list(KS),
        "replacement_local_regret_certificate": "mean_minus_downside_rms",
        "replacement_regret_training_population": "TRAIN_only_final_guard_admissible_support_positive_scalar_dominance_positive_alternatives",
        "replacement_operator": (
            "preserve admissible incumbent by default; replace only if support>0 AND scalar_dominance>0 AND local_downside_regret_certificate>0; "
            "rank by frozen scalar dominance; semantic-family coordinates affect only the certificate neighborhood; no signed-profile, transition, sorted-spectrum, density, or tuned threshold gate"
        ),
        "all_flagged_policy": "preserve_legacy_for_structural_guard",
    })
    version = "V64.3.26-EAF-ICER-SARC-DARM-DBR"
    cfg.setdefault("metadata", {})["algorithm_version"] = version
    cfg.setdefault("provenance", {})["algorithm_version"] = version
    exp = cfg.setdefault("experiment", {})
    exp["name"] = f"v64_3_26_{tag}"
    exp["algorithm"] = (
        "V64.3.26 EAF-ICER-SARC: Evidence-Attributed Incumbent-Contrastive Extremal Recovery "
        "with Semantic-Aligned Downside Regret Certification"
    )
    exp["mechanism_chain"] = (
        "fixed B<=16 -> frozen EAF complete DARM-anchor frontier -> exact selected-evidence attribution -> "
        "complete deployment-admissible frontier -> frozen support/scalar dominance -> identity-preserving semantic-family aligned downside regret certificate -> "
        "incumbent-default extremal replacement -> unchanged one-sided/evidence certificate -> unchanged structural-risk guard -> final decision preservation"
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
        raise SystemExit(f"STOP TRAIN DATA: missing semantic-family TRAIN frontier provenance {edge_path}")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    by, frontier_row_count = _load_minimal_scenes(edge_path)
    if len(by) != EXPECTED_TRAIN_SCENES:
        raise SystemExit(f"STOP TRAIN DATA: expected exactly {EXPECTED_TRAIN_SCENES} frozen TRAIN scenes, got {len(by)}")
    data = _build(by)

    control = _crossfit(data, "aggregate")
    main_cf = _crossfit(data, "semantic")
    tail_nonworse = bool(
        main_cf["selected_negative_rms"] <= control["selected_negative_rms"] + EPS
        and main_cf["selected_worst"] >= control["selected_worst"] - EPS
    )
    tail_strict = bool(
        main_cf["selected_negative_rms"] < control["selected_negative_rms"] - EPS
        or main_cf["selected_worst"] > control["selected_worst"] + EPS
    )
    semantic_incremental = bool(tail_nonworse and tail_strict)
    gate = bool(
        main_cf["all_folds_path_safe"]
        and main_cf["selected_count"] >= MAIN_MIN_SELECTED
        and main_cf["teacher_improvement_sum"] >= -EPS
        and semantic_incremental
    )

    tokens = sorted(by)
    token_path = Path(args.output_train_token_file); token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    y = data["delta"]
    report: dict[str, Any] = {
        "audit": "v64_3_26_eaf_icer_sarc_train_fit",
        "algorithm": "V64.3.26 EAF-ICER-SARC",
        "train_scene_count": int(len(by)), "frontier_row_count": int(frontier_row_count),
        "replacement_edges": int(len(y)), "replacement_scenes": int(data["replacement_scene_count"]),
        "aggregate_feature_count": 18, "semantic_family_feature_count": 10,
        "semantic_family_feature_names": _SEMANTIC_NAMES,
        "population_teacher_positive_fraction": float(np.mean(y > 0.0)),
        "population_teacher_improvement_sum": float(y.sum()),
        "population_teacher_improvement_mean": float(y.mean()),
        "population_teacher_improvement_worst": float(y.min()),
        "fold_seed": FOLD_SEED, "neighbor_k_values": list(KS), "downside_multiplier": 1.0, "decision_boundary": 0.0,
        "crossfit": {"v25_aggregate_downside_control": control, "semantic_family_downside_main": main_cf},
        "semantic_tail_incremental_on_train": semantic_incremental,
        "train_gate_pass": gate,
        "gate_contract": {
            "main": "semantic_family_downside",
            "all_5_scene_folds_selected_path_nonharmful": True,
            "selected_count_min": MAIN_MIN_SELECTED,
            "teacher_improvement_sum_min": 0.0,
            "selected_negative_rms_nonworse_than_v25_aggregate": True,
            "selected_worst_nonworse_than_v25_aggregate": True,
            "at_least_one_tail_metric_strictly_better": True,
            "fresh_validation_must_not_be_used_on_train_gate_fail": True,
        },
        "input_frontier": {"path": str(edge_path), "bytes": int(edge_path.stat().st_size), "sha256": _sha256_file(edge_path)},
        "train_token_manifest": {"path": str(token_path), "count": int(len(tokens)), "sha256": _sha256_file(token_path)},
        "fresh_validation_used": False, "memories": {}, "configs": {},
        "diagnostic_note": (
            "V26 changes only the replacement-certificate representation: fixed semantic-family coordinates preserve selected-evidence identity and candidate/incumbent correspondence. "
            "The V25 aggregate 18-D downside certificate is the sole causal control. No K, downside multiplier, zero boundary, family/group weight, density threshold, selector, EAF, or dominance parameter is tuned."
        ),
    }
    _write_report(Path(args.output_report), report)
    if not gate:
        raise SystemExit(
            "STOP TRAIN SARC: identity-preserving semantic-family representation did not provide 5/5 selected-path safety plus non-worse/strictly-incremental tail over V25 aggregate DRC; "
            "do not spend fresh validation GPU and do not tune family weights/K/downside multiplier/zero/density thresholds"
        )

    base = yaml.safe_load(Path(args.base_v20_dual_config).read_text(encoding="utf-8"))
    variants = {"aggregate_downside": "evidence_only", "semantic_family_downside": "semantic_family_aligned"}
    memories: dict[str, Any] = {}; configs: dict[str, str] = {}
    for tag, mode in variants.items():
        memory_path = out_dir / f"v64_3_26_{tag}_memory.npz"
        memory = _save_memory(memory_path, data, mode)
        config_path = out_dir / f"v64_3_26_{tag}.yaml"
        config_path.write_text(yaml.safe_dump(_cfg(base, memory, mode, tag), sort_keys=False), encoding="utf-8")
        memories[tag] = memory; configs[tag] = str(config_path)
    report["memories"] = memories; report["configs"] = configs
    _write_report(Path(args.output_report), report)
    print(json.dumps({
        "pass": True, "train_gate_pass": gate,
        "main_fold_pass_count": main_cf["fold_pass_count"], "main_count": main_cf["selected_count"],
        "main_sum": main_cf["teacher_improvement_sum"], "semantic_tail_incremental_on_train": semantic_incremental,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
