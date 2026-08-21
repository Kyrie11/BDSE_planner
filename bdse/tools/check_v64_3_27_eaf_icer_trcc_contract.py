from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

import bdse.tools.fit_v64_3_22_eaf_icer_tcr as v22
from bdse.planner.tournament import _ICER_SEMANTIC_TYPE_FEATURE_NAMES


def _ic(cfg):
    return cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_memory(path: Path, sha: str, expected_names: list[str], errors: list[str], label: str) -> dict[str, object]:
    info: dict[str, object] = {}
    if not path.is_file():
        errors.append(f"missing {label} memory {path}")
        return info
    got_sha = _sha(path)
    if got_sha != sha:
        errors.append(f"{label} memory SHA mismatch")
    try:
        with np.load(path, allow_pickle=False) as z:
            names = [str(x) for x in z["feature_names"].reshape(-1)]
            weights = np.asarray(z["feature_metric_weight"], dtype=float).reshape(-1)
            ks = [int(x) for x in z["neighbor_k_values"].reshape(-1)]
            kind = str(z["certificate_kind"].reshape(-1)[0])
            dm = float(z["downside_multiplier"].reshape(-1)[0])
            mem = np.asarray(z["memory_metric_z"])
            y = np.asarray(z["teacher_improvement"]).reshape(-1)
        if names != expected_names:
            errors.append(f"{label} memory feature schema mismatch")
        d = len(expected_names)
        if len(weights) != d or not np.allclose(weights, np.full(d, 1.0 / d), atol=1e-7):
            errors.append(f"{label} metric weights must be fixed equal per-coordinate")
        if ks != [32, 64]: errors.append(f"{label} memory K must remain 32/64")
        if kind != "mean_minus_downside_rms": errors.append(f"{label} certificate kind mismatch")
        if abs(dm - 1.0) > 1e-8: errors.append(f"{label} downside multiplier must remain 1.0")
        if mem.shape != (len(y), d): errors.append(f"{label} memory shape mismatch")
        info = {"rows": int(len(y)), "features": d, "kind": kind, "sha256": got_sha}
    except Exception as exc:
        errors.append(f"{label} memory read failed: {exc}")
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--expect", choices=["aggregate-downside", "type-confirmed"], required=True)
    ap.add_argument("--frozen-v20-dual-config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(args.frozen_v20_dual_config).read_text(encoding="utf-8"))
    ic, bic = _ic(cfg), _ic(base)
    main = args.expect == "type-confirmed"
    errors: list[str] = []
    if not ic.get("enabled"): errors.append("ICER disabled")
    if ic.get("dominance_policy") != "scalar_only": errors.append("dominance must remain frozen scalar-only")
    if ic.get("incumbent_retention_policy") != "preserve_admissible_incumbent": errors.append("incumbent default-preservation missing")
    if not ic.get("regret_risk_enabled") or ic.get("retention_regret_risk_enabled") or not ic.get("replacement_regret_risk_enabled"):
        errors.append("replacement-only risk contract broken")
    if ic.get("regret_risk_feature_mode") != "evidence_only": errors.append("aggregate proposal feature mode must remain evidence_only")
    expected_model = "local_multiscale_downside_regret_with_type_confirmation" if main else "local_multiscale_downside_regret_certificate"
    if ic.get("regret_risk_model_type") != expected_model: errors.append("risk model mismatch")
    if ic.get("replacement_local_regret_certificate") != "mean_minus_downside_rms": errors.append("aggregate certificate metadata mismatch")
    if list(ic.get("replacement_local_regret_neighbor_k_values", [])) != [32, 64]: errors.append("aggregate K must remain fixed 32/64")
    if ic.get("all_flagged_policy") != "preserve_legacy_for_structural_guard": errors.append("structural delegation changed")

    agg_names = [f"evidence::{n}" for n in v22._ICER_REGRET_RISK_EVIDENCE_BASE_NAMES]
    memories = {
        "aggregate": _check_memory(
            Path(str(ic.get("replacement_local_regret_memory_path", ""))),
            str(ic.get("replacement_local_regret_memory_sha256", "")),
            agg_names, errors, "aggregate",
        )
    }
    if main:
        if ic.get("replacement_confirmation_regret_risk_feature_mode") != "semantic_type_only": errors.append("type confirmation feature mode mismatch")
        if list(ic.get("replacement_confirmation_local_regret_neighbor_k_values", [])) != [32, 64]: errors.append("type confirmation K must remain fixed 32/64")
        if ic.get("replacement_confirmation_local_regret_certificate") != "mean_minus_downside_rms": errors.append("type confirmation certificate metadata mismatch")
        if ic.get("replacement_selection_monotonicity") != "selected_replacements_are_subset_of_V25_aggregate_DRC_selected_replacements_no_fallback": errors.append("no-fallback subset invariant missing")
        type_names = [f"semantic_type::{n}" for n in _ICER_SEMANTIC_TYPE_FEATURE_NAMES]
        memories["type_confirmation"] = _check_memory(
            Path(str(ic.get("replacement_confirmation_local_regret_memory_path", ""))),
            str(ic.get("replacement_confirmation_local_regret_memory_sha256", "")),
            type_names, errors, "type_confirmation",
        )

    frozen = [
        "support_feature_names", "support_feature_mean", "support_feature_std", "support_weights", "support_bias",
        "scalar_dominance_feature_names", "scalar_dominance_base_feature_names", "scalar_dominance_feature_mean",
        "scalar_dominance_feature_std", "scalar_dominance_weights", "scalar_dominance_bias",
        "profile_dominance_feature_names", "profile_dominance_base_feature_names", "profile_dominance_feature_mean",
        "profile_dominance_feature_std", "profile_dominance_weights", "profile_dominance_bias",
    ]
    for key in frozen:
        if ic.get(key) != bic.get(key): errors.append("frozen head changed: " + key)

    report = {
        "pass": not errors, "errors": errors, "expect": args.expect, "memories": memories,
        "frozen_head_identity": not any(x.startswith("frozen head") for x in errors),
        "selection_subset_contract": bool(main),
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if errors: raise SystemExit("STOP CONTRACT: " + "; ".join(errors))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
