from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import bdse.tools.fit_v64_3_22_eaf_icer_tcr as v22

FOLDS = 5
# Deliberately retained from V23/V24.  Changing the fold assignment after seeing
# V24 would be an implicit TRAIN split search.
FOLD_SEED = "v64.3.23-eaf-icer-rcr-scene-crossfit-v1"
KS = (32, 64)
MIN_EDGES = 1024
MIN_SCENES = 256
MIN_FOLD_SCENES = 40
MIN_SELECTED = 8
MAIN_MIN_SELECTED = 64
EXPECTED_TRAIN_SCENES = 3000

_BASE_NAMES = list(v22._ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
_KEEP_KEYS = {
    "scenario_token", "anchor_action", "raw_top_action", "challenger_action",
    "icer_admissible", "dacer_admissible", "teacher_margin",
    "icer_support_logit", "icer_scalar_dominance_logit",
} | {f"icer_feature_{name}" for name in _BASE_NAMES}


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
    """Stream the 747MB-style frontier provenance without retaining unused fields.

    V24 used the generic full-row loader.  The DRC fitter needs only the frozen
    replacement population and the 18 aggregate evidence features, so retaining
    the full attribution/diagnostic payload wastes memory without changing any
    numerical semantics.
    """
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
            row = {k: full.get(k) for k in _KEEP_KEYS}
            by.setdefault(token, []).append(row)
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
    X: list[list[float]] = []
    delta: list[float] = []
    tok: list[str] = []
    sup: list[float] = []
    scalar: list[float] = []
    action: list[int] = []
    eligible_before_feature = 0
    missing_feature = 0
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
            feat = [_finite(r, f"icer_feature_{name}") for name in _BASE_NAMES]
            if not all(np.isfinite(x) for x in feat):
                missing_feature += 1
                continue
            X.append(feat)
            delta.append(tm - inc_tm)
            tok.append(token)
            sup.append(support)
            scalar.append(dominance)
            action.append(challenger)
            replacement_scenes.add(token)

    if eligible_before_feature < MIN_EDGES:
        raise SystemExit(f"STOP TRAIN SUPPORT: insufficient frozen replacement population: {eligible_before_feature}")
    # Aggregate evidence features are part of the frozen runtime contract.  A
    # missing feature is an instrumentation failure, not a sample to silently drop.
    if missing_feature:
        raise SystemExit(
            f"STOP TRAIN INSTRUMENTATION: missing aggregate evidence features on "
            f"{missing_feature}/{eligible_before_feature} eligible replacement edges"
        )
    if len(replacement_scenes) < MIN_SCENES:
        raise SystemExit(
            f"STOP TRAIN SUPPORT: insufficient replacement scenes: {len(replacement_scenes)} < {MIN_SCENES}"
        )
    return {
        "X": np.asarray(X, dtype=np.float64),
        "delta": np.asarray(delta, dtype=np.float64),
        "tok": np.asarray(tok, dtype=object),
        "support": np.asarray(sup, dtype=np.float64),
        "scalar": np.asarray(scalar, dtype=np.float64),
        "action": np.asarray(action, dtype=np.int64),
        "eligible_before_feature": int(eligible_before_feature),
        "missing_feature": int(missing_feature),
        "replacement_scene_count": int(len(replacement_scenes)),
    }


def _memory(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = np.maximum(X.std(axis=0), 1.0e-6)
    metric_weight = np.full(X.shape[1], 1.0 / X.shape[1], dtype=np.float64)
    Z = ((X - mean[None, :]) / std[None, :]) * np.sqrt(metric_weight[None, :])
    return Z, mean, std, metric_weight


def _score(trainX: np.ndarray, trainY: np.ndarray, qX: np.ndarray, cert: str) -> np.ndarray:
    Z, mean, std, metric_weight = _memory(trainX)
    Q = ((qX - mean[None, :]) / std[None, :]) * np.sqrt(metric_weight[None, :])
    d2 = np.maximum(
        np.sum(Q * Q, axis=1)[:, None]
        + np.sum(Z * Z, axis=1)[None, :]
        - 2.0 * (Q @ Z.T),
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
        if cert == "mean_se":
            var = np.sum(w * (y - mu[:, None]) ** 2, axis=1)
            neff = 1.0 / np.maximum(np.sum(w * w, axis=1), 1.0e-12)
            bound = mu - np.sqrt(var / np.maximum(neff, 1.0))
        elif cert == "downside_rms":
            downside = np.minimum(y, 0.0)
            bound = mu - np.sqrt(np.sum(w * downside * downside, axis=1))
        else:
            raise ValueError(cert)
        bounds.append(bound)
    return np.min(np.stack(bounds, axis=1), axis=1)


def _selection(data: dict[str, Any], score: np.ndarray, hold: set[str]) -> dict[str, Any]:
    toks = data["tok"]
    support = data["support"]
    scalar = data["scalar"]
    delta = data["delta"]
    action = data["action"]
    selected: list[float] = []
    opportunities = captured = scene_count = 0
    selected_tokens: list[str] = []

    for token in sorted(hold):
        idx = np.flatnonzero(toks == token)
        if not len(idx):
            continue
        scene_count += 1
        opportunities += int(np.any(delta[idx] > 0.0))
        accepted = idx[(support[idx] > 0.0) & (scalar[idx] > 0.0) & (score[idx] > 0.0)]
        if len(accepted):
            j = sorted(
                accepted.tolist(),
                key=lambda q: (-float(scalar[q]), -float(score[q]), int(action[q])),
            )[0]
            selected.append(float(delta[j]))
            selected_tokens.append(token)
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
        "opportunities": int(opportunities),
        "capture": float(captured / opportunities) if opportunities else float("nan"),
        "selected_token_count": int(len(selected_tokens)),
    }


def _crossfit(data: dict[str, Any], cert: str) -> dict[str, Any]:
    X = data["X"]
    y = data["delta"]
    toks = data["tok"]
    unique = sorted(set(map(str, toks)))
    if len(X) < MIN_EDGES or len(unique) < MIN_SCENES:
        raise SystemExit(f"STOP TRAIN SUPPORT {cert}: edges={len(X)} scenes={len(unique)}")
    folds: list[dict[str, Any]] = []
    for f in range(FOLDS):
        hold = {t for t in unique if _fold(t) == f}
        if len(hold) < MIN_FOLD_SCENES:
            raise SystemExit(f"STOP TRAIN SPLIT: fold too small {f}: {len(hold)}")
        hold_mask = np.asarray([str(t) in hold for t in toks], dtype=bool)
        fit_mask = ~hold_mask
        score = np.full(len(X), np.nan, dtype=np.float64)
        score[hold_mask] = _score(X[fit_mask], y[fit_mask], X[hold_mask], cert)
        metrics = _selection(data, score, hold)
        metrics["fold"] = int(f)
        metrics["hold_scenes"] = int(len(hold))
        metrics["path_safe"] = bool(metrics["count"] >= MIN_SELECTED and metrics["sum"] >= -1.0e-9)
        folds.append(metrics)
    return {
        "mode": "aggregate_evidence_only",
        "certificate": cert,
        "folds": folds,
        "all_folds_path_safe": bool(all(x["path_safe"] for x in folds)),
        "fold_pass_count": int(sum(x["path_safe"] for x in folds)),
        "selected_count": int(sum(x["count"] for x in folds)),
        "teacher_improvement_sum": float(sum(x["sum"] for x in folds)),
        "mean_precision": float(np.nanmean([x["precision"] for x in folds])),
        "mean_capture": float(np.nanmean([x["capture"] for x in folds])),
    }


def _save_memory(path: Path, data: dict[str, Any], cert: str) -> dict[str, Any]:
    Z, mean, std, metric_weight = _memory(data["X"])
    names = [f"evidence::{name}" for name in _BASE_NAMES]
    kind = "mean_minus_downside_rms" if cert == "downside_rms" else "mean_minus_standard_error"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        memory_metric_z=Z.astype("f4"),
        teacher_improvement=data["delta"].astype("f4"),
        feature_mean=mean.astype("f4"),
        feature_std=std.astype("f4"),
        feature_names=np.asarray(names, dtype="U128"),
        feature_metric_weight=metric_weight.astype("f4"),
        neighbor_k_values=np.asarray(KS, dtype="i4"),
        se_multiplier=np.asarray([1.0], dtype="f4"),
        certificate_kind=np.asarray([kind], dtype="U64"),
        downside_multiplier=np.asarray([1.0], dtype="f4"),
    )
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "row_count": int(len(data["delta"])),
        "feature_count": int(data["X"].shape[1]),
        "mode": "evidence_only",
        "certificate": cert,
    }


def _cfg(base: dict[str, Any], memory: dict[str, Any], cert: str, tag: str) -> dict[str, Any]:
    cfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
    ic = _icer(cfg)
    downside = cert == "downside_rms"
    ic.update(
        {
            "model_type": "frozen_support_scalar_dominance_plus_evidence_local_downside_regret_certificate"
            if downside
            else "frozen_support_scalar_dominance_plus_evidence_local_mean_se_regret_certificate",
            "dominance_policy": "scalar_only",
            "incumbent_retention_policy": "preserve_admissible_incumbent",
            "regret_risk_enabled": True,
            "retention_regret_risk_enabled": False,
            "replacement_regret_risk_enabled": True,
            "regret_risk_model_type": "local_multiscale_downside_regret_certificate"
            if downside
            else "local_multiscale_regret_lower_bound",
            "regret_risk_feature_mode": "evidence_only",
            "replacement_local_regret_memory_path": memory["path"],
            "replacement_local_regret_memory_sha256": memory["sha256"],
            "replacement_local_regret_neighbor_k_values": list(KS),
            "replacement_local_regret_certificate": "mean_minus_downside_rms"
            if downside
            else "mean_minus_standard_error",
            "replacement_regret_training_population": "TRAIN_only_final_guard_admissible_support_positive_scalar_dominance_positive_alternatives",
            "replacement_operator": "preserve admissible incumbent by default; replace only if support>0 AND scalar_dominance>0 AND local_regret_certificate>0; rank by frozen scalar dominance; no signed-profile, transition, or full-attribution-spectrum gate",
            "all_flagged_policy": "preserve_legacy_for_structural_guard",
        }
    )
    version = "V64.3.25-EAF-ICER-DRC-DARM-DBR"
    cfg.setdefault("metadata", {})["algorithm_version"] = version
    cfg.setdefault("provenance", {})["algorithm_version"] = version
    exp = cfg.setdefault("experiment", {})
    exp["name"] = f"v64_3_25_{tag}"
    exp["algorithm"] = (
        "V64.3.25 EAF-ICER-DRC: Evidence-Attributed Incumbent-Contrastive Extremal Recovery "
        "with Evidence-Local Downside Regret Certification"
    )
    exp["mechanism_chain"] = (
        "fixed B<=16 -> frozen EAF complete DARM-anchor frontier -> auditable selected-evidence attribution upstream -> "
        "complete deployment-admissible frontier -> frozen support/scalar dominance -> aggregate evidence-local downside regret certificate -> "
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
        raise SystemExit(f"STOP TRAIN DATA: missing frontier provenance {edge_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by, frontier_row_count = _load_minimal_scenes(edge_path)
    if len(by) != EXPECTED_TRAIN_SCENES:
        raise SystemExit(f"STOP TRAIN DATA: expected exactly {EXPECTED_TRAIN_SCENES} frozen TRAIN scenes, got {len(by)}")
    data = _build(by)
    crossfit = {
        "aggregate_meanse": _crossfit(data, "mean_se"),
        "aggregate_downside": _crossfit(data, "downside_rms"),
    }
    main_cf = crossfit["aggregate_downside"]
    gate = bool(
        main_cf["all_folds_path_safe"]
        and main_cf["selected_count"] >= MAIN_MIN_SELECTED
        and main_cf["teacher_improvement_sum"] >= -1.0e-9
    )

    tokens = sorted(by)
    token_path = Path(args.output_train_token_file)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")

    y = data["delta"]
    report: dict[str, Any] = {
        "audit": "v64_3_25_eaf_icer_drc_train_fit",
        "algorithm": "V64.3.25 EAF-ICER-DRC",
        "train_scene_count": int(len(by)),
        "frontier_row_count": int(frontier_row_count),
        "replacement_edges": int(len(y)),
        "replacement_scenes": int(data["replacement_scene_count"]),
        "aggregate_feature_coverage": 1.0,
        "population_teacher_positive_fraction": float(np.mean(y > 0.0)),
        "population_teacher_improvement_sum": float(y.sum()),
        "population_teacher_improvement_mean": float(y.mean()),
        "population_teacher_improvement_worst": float(y.min()),
        "fold_seed": FOLD_SEED,
        "neighbor_k_values": list(KS),
        "downside_multiplier": 1.0,
        "decision_boundary": 0.0,
        "crossfit": crossfit,
        "train_gate_pass": gate,
        "gate_contract": {
            "main": "aggregate_downside",
            "all_5_scene_folds_selected_path_nonharmful": True,
            "selected_count_min": MAIN_MIN_SELECTED,
            "teacher_improvement_sum_min": 0.0,
            "fresh_validation_must_not_be_used_on_train_gate_fail": True,
        },
        "input_frontier": {
            "path": str(edge_path),
            "bytes": int(edge_path.stat().st_size),
            "sha256": _sha256_file(edge_path),
        },
        "train_token_manifest": {
            "path": str(token_path),
            "count": int(len(tokens)),
            "sha256": _sha256_file(token_path),
        },
        "fresh_validation_used": False,
        "memories": {},
        "configs": {},
        "diagnostic_note": (
            "V24 attribution-resolved variants are intentionally not carried forward: the pre-registered V24 TRAIN branch selected aggregate-downside. "
            "This fitter writes the full TRAIN report before any fail-closed exit so a legitimate STOP remains auditable."
        ),
    }
    _write_report(Path(args.output_report), report)

    if not gate:
        raise SystemExit(
            "STOP TRAIN DRC: aggregate evidence-local downside certificate is not selected-path safe in all fixed scene folds; "
            "do not spend fresh validation GPU and do not tune K/downside multiplier/zero boundary"
        )

    base = yaml.safe_load(Path(args.base_v20_dual_config).read_text(encoding="utf-8"))
    variants = {
        "aggregate_meanse": "mean_se",
        "aggregate_downside": "downside_rms",
    }
    memories: dict[str, Any] = {}
    configs: dict[str, str] = {}
    for tag, cert in variants.items():
        memory_path = out_dir / f"v64_3_25_{tag}_memory.npz"
        memory = _save_memory(memory_path, data, cert)
        config_path = out_dir / f"v64_3_25_{tag}.yaml"
        config_path.write_text(yaml.safe_dump(_cfg(base, memory, cert, tag), sort_keys=False), encoding="utf-8")
        memories[tag] = memory
        configs[tag] = str(config_path)

    report["memories"] = memories
    report["configs"] = configs
    _write_report(Path(args.output_report), report)
    print(
        json.dumps(
            {
                "pass": True,
                "train_gate_pass": gate,
                "main_sum": main_cf["teacher_improvement_sum"],
                "main_fold_pass_count": main_cf["fold_pass_count"],
                "main_count": main_cf["selected_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
