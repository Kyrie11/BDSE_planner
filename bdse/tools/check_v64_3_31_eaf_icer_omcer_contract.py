from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

import bdse.tools.fit_v64_3_22_eaf_icer_tcr as v22

EXPECTED_CAT_THRESHOLD = -0.5
EXPECTED_K = [32, 64]
OPERATOR_NAME = "operator::min_support_scalar_dominance_logit"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _icer(cfg: dict) -> dict:
    return cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Hard contract audit for V64.3.31 OMCER.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--frozen-v20-config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(args.frozen_v20_config).read_text(encoding="utf-8"))
    errors: list[str] = []

    if cfg.get("evidence", {}).get("budget") != 16:
        errors.append("literal retained evidence budget changed")
    sel = cfg.get("selector", {}) or {}
    if sel.get("proposal_top_m") != 24:
        errors.append("proposal Top-M changed")
    if "proposal_conditioned_witness_rebinding" in sel:
        errors.append("failed V30 PCWER remains in V31 selector")
    if "frontier_contrast_rebinding" in sel:
        errors.append("failed V29 FCR remains in V31 selector")
    if sel.get("selector_cap_mode") != "anytime_adverse_certificate":
        errors.append("frozen AOCC selector cap mode changed")
    if sel.get("evidence_certificate_mode") != "exact_downstream_winner_preservation":
        errors.append("frozen AOCC evidence certificate changed")
    if not bool(sel.get("force_fill_budget", False)) or int(sel.get("min_selected_atoms", -1)) != 16:
        errors.append("frozen B=16 fill/cardinality policy changed")

    ic = _icer(cfg)
    bic = _icer(base)
    if not ic.get("enabled"):
        errors.append("ICER disabled")
    if ic.get("dominance_policy") != "scalar_only":
        errors.append("scalar-only dominance policy changed")
    if ic.get("incumbent_retention_policy") != "preserve_admissible_incumbent":
        errors.append("asymmetric incumbent preservation changed")
    if not ic.get("regret_risk_enabled") or ic.get("retention_regret_risk_enabled") or not ic.get("replacement_regret_risk_enabled"):
        errors.append("replacement-only OMCER contract broken")
    if ic.get("regret_risk_model_type") != "local_multiscale_catastrophic_excess_regret_certificate":
        errors.append("OMCER risk model type changed")
    if ic.get("regret_risk_feature_mode") != "operator_margin_evidence":
        errors.append("OMCER feature mode changed")
    if ic.get("replacement_local_regret_certificate") != "mean_minus_catastrophic_excess_rms":
        errors.append("OMCER certificate changed")
    if list(ic.get("replacement_local_regret_neighbor_k_values", [])) != EXPECTED_K:
        errors.append("OMCER K changed")
    if abs(float(ic.get("replacement_local_regret_catastrophic_delta_threshold", 999.0)) - EXPECTED_CAT_THRESHOLD) > 1e-9:
        errors.append("catastrophic threshold changed")
    if ic.get("operator_conditioning_statistic") != "min(support_logit,scalar_dominance_logit)":
        errors.append("operator margin definition changed")
    if ic.get("all_flagged_policy") != "preserve_legacy_for_structural_guard":
        errors.append("structural guard delegation changed")
    if "before extremization" not in str(ic.get("replacement_operator", "")):
        errors.append("risk admissibility is not explicitly pre-extremal")

    # Support/scalar readouts stay frozen. V31 changes only the risk statistic.
    frozen_heads = [
        "support_feature_names", "support_feature_mean", "support_feature_std", "support_weights", "support_bias",
        "scalar_dominance_feature_names", "scalar_dominance_base_feature_names", "scalar_dominance_feature_mean",
        "scalar_dominance_feature_std", "scalar_dominance_weights", "scalar_dominance_bias",
        "profile_dominance_feature_names", "profile_dominance_base_feature_names", "profile_dominance_feature_mean",
        "profile_dominance_feature_std", "profile_dominance_weights", "profile_dominance_bias",
    ]
    for key in frozen_heads:
        if ic.get(key) != bic.get(key):
            errors.append("frozen ICER head changed: " + key)

    for key in ic:
        low = str(key).lower()
        if key.startswith("replacement_confirmation") or "tail_mode" in low:
            errors.append(f"failed V28 PTMC confirmation leaked into V31: {key}")

    mem_path = Path(str(ic.get("replacement_local_regret_memory_path", "")))
    mem_sha = str(ic.get("replacement_local_regret_memory_sha256", ""))
    memory_info: dict = {}
    if not mem_path.is_file():
        errors.append(f"missing OMCER memory {mem_path}")
    else:
        got = _sha(mem_path)
        if got != mem_sha:
            errors.append("OMCER memory SHA mismatch")
        try:
            with np.load(mem_path, allow_pickle=False) as z:
                names = [str(x) for x in z["feature_names"].reshape(-1)]
                weights = np.asarray(z["feature_metric_weight"], dtype=float).reshape(-1)
                ks = [int(x) for x in z["neighbor_k_values"].reshape(-1)]
                kind = str(z["certificate_kind"].reshape(-1)[0])
                dm = float(z["downside_multiplier"].reshape(-1)[0])
                cat = float(z["catastrophic_delta_threshold"].reshape(-1)[0])
                mem = np.asarray(z["memory_metric_z"])
                y = np.asarray(z["teacher_improvement"]).reshape(-1)
            expected = [f"evidence::{n}" for n in v22._ICER_REGRET_RISK_EVIDENCE_BASE_NAMES] + [OPERATOR_NAME]
            d = len(expected)
            if names != expected:
                errors.append("OMCER memory schema changed")
            if len(weights) != d or not np.allclose(weights, np.full(d, 1.0 / d), atol=1e-7):
                errors.append("OMCER metric weights changed")
            if ks != EXPECTED_K:
                errors.append("OMCER memory K changed")
            if kind != "mean_minus_catastrophic_excess_rms":
                errors.append("OMCER memory certificate changed")
            if abs(dm - 1.0) > 1e-8:
                errors.append("OMCER downside multiplier changed")
            if abs(cat - EXPECTED_CAT_THRESHOLD) > 1e-8:
                errors.append("OMCER catastrophic threshold changed")
            if mem.shape != (len(y), d):
                errors.append("OMCER memory shape mismatch")
            if len(y) != 1455:
                errors.append(f"OMCER training population changed: {len(y)} != 1455")
            memory_info = {
                "rows": int(len(y)), "features": int(d), "sha256": got,
                "kind": kind, "catastrophic_delta_threshold": cat,
            }
        except Exception as exc:
            errors.append(f"OMCER memory read failed: {exc}")

    report = {
        "pass": not errors,
        "errors": errors,
        "algorithm_version": cfg.get("metadata", {}).get("algorithm_version"),
        "memory": memory_info,
        "fixed_constants": {
            "B": 16, "M": 24, "K": EXPECTED_K, "downside_multiplier": 1.0,
            "decision_boundary": 0.0, "catastrophic_delta_threshold": EXPECTED_CAT_THRESHOLD,
        },
        "failed_rebinding_removed": "proposal_conditioned_witness_rebinding" not in sel and "frontier_contrast_rebinding" not in sel,
        "pre_extremal_operator_conditioned_risk": True,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if errors:
        raise SystemExit("STOP CONTRACT: " + "; ".join(errors))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
