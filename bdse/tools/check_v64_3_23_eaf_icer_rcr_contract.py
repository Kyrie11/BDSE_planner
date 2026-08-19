from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml

import bdse.tools.fit_v64_3_22_eaf_icer_tcr as v22


def _icer(c):
    return c["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Check frozen V64.3.23 EAF-ICER-RCR contract.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--expect", choices=["evidence-local-scalar", "evidence-local-rcr", "transition-local-rcr"], required=True)
    ap.add_argument("--frozen-v20-dual-config", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    c = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    b = yaml.safe_load(Path(a.frozen_v20_dual_config).read_text(encoding="utf-8"))
    ic, bic = _icer(c), _icer(b)
    errs: list[str] = []
    expected_mode = "evidence_only" if a.expect.startswith("evidence-local") else "transition_conditioned"
    expected_dom = "scalar_only" if a.expect == "evidence-local-scalar" else "scalar_positive_dual_mean_positive"
    if not bool(ic.get("enabled", False)):
        errs.append("ICER disabled")
    if str(ic.get("model_type", "")) != "frozen_support_dominance_plus_local_multiscale_regret_coherence":
        errs.append("model type mismatch")
    if str(ic.get("dominance_policy", "")) != expected_dom:
        errs.append("dominance policy mismatch")
    if str(ic.get("incumbent_retention_policy", "")) != "preserve_admissible_incumbent":
        errs.append("incumbent default-preservation contract missing")
    if not bool(ic.get("regret_risk_enabled", False)) or bool(ic.get("retention_regret_risk_enabled", True)) or not bool(ic.get("replacement_regret_risk_enabled", False)):
        errs.append("asymmetric replacement-only risk contract broken")
    if str(ic.get("regret_risk_model_type", "")) != "local_multiscale_regret_lower_bound":
        errs.append("local risk model mismatch")
    if str(ic.get("regret_risk_feature_mode", "")) != expected_mode:
        errs.append("risk feature mode mismatch")
    if str(ic.get("regret_risk_threshold_policy", "")) != "fixed_zero_multiscale_local_lower_bound_no_validation_sweep":
        errs.append("fixed zero lower-bound contract missing")
    if str(ic.get("all_flagged_policy", "")) != "preserve_legacy_for_structural_guard":
        errs.append("structural delegation changed")
    if expected_mode == "evidence_only" and not bool(ic.get("train_crossfit_replacement_path_safe", False)):
        errs.append("TRAIN evidence-local path gate missing")
    if expected_mode == "transition_conditioned" and str(ic.get("transition_conditioning_role", "")) != "controlled_ablation_not_required_for_main_promotion":
        errs.append("transition conditioning must remain a controlled ablation")

    memory_path = Path(str(ic.get("replacement_local_regret_memory_path", "")))
    expected_sha = str(ic.get("replacement_local_regret_memory_sha256", ""))
    memory_ok = memory_path.is_file()
    memory_info = {}
    if not memory_ok:
        errs.append(f"missing local memory: {memory_path}")
    else:
        got_sha = hashlib.sha256(memory_path.read_bytes()).hexdigest()
        if got_sha != expected_sha:
            errs.append("local memory SHA mismatch")
        try:
            with np.load(memory_path, allow_pickle=False) as z:
                names = [str(x) for x in np.asarray(z["feature_names"]).reshape(-1).tolist()]
                expected_names = v22._runtime_feature_names(expected_mode)
                weights = np.asarray(z["feature_metric_weight"], dtype=float).reshape(-1)
                ks = [int(x) for x in np.asarray(z["neighbor_k_values"]).reshape(-1).tolist()]
                se = float(np.asarray(z["se_multiplier"]).reshape(-1)[0])
                mem = np.asarray(z["memory_metric_z"])
                y = np.asarray(z["teacher_improvement"]).reshape(-1)
            if names != expected_names:
                errs.append("local memory feature schema mismatch")
            if ks != [32, 64]:
                errs.append("local neighbor scales must be fixed K32/K64")
            if abs(se - 1.0) > 1e-8:
                errs.append("local lower bound must use fixed one-standard-error multiplier")
            if mem.ndim != 2 or mem.shape[0] != len(y) or mem.shape[1] != len(names) or len(weights) != len(names):
                errs.append("local memory array shape mismatch")
            if expected_mode == "evidence_only":
                if abs(float(weights.sum()) - 1.0) > 1e-5:
                    errs.append("evidence metric weight sum mismatch")
            else:
                if len(weights) != 59 or any(abs(float(weights[s:e].sum()) - 1.0) > 1e-5 for s, e in [(0,18),(18,39),(39,59)]):
                    errs.append("transition metric must balance evidence/semantic/geometry groups equally")
            memory_info = {"rows": int(mem.shape[0]), "features": int(mem.shape[1]), "neighbor_k_values": ks, "se_multiplier": se, "sha256": got_sha}
        except Exception as e:
            errs.append(f"local memory read failed: {e}")

    frozen_keys = [
        "support_feature_names", "support_feature_mean", "support_feature_std", "support_weights", "support_bias",
        "scalar_dominance_feature_names", "scalar_dominance_base_feature_names", "scalar_dominance_feature_mean", "scalar_dominance_feature_std", "scalar_dominance_weights", "scalar_dominance_bias",
        "profile_dominance_feature_names", "profile_dominance_base_feature_names", "profile_dominance_feature_mean", "profile_dominance_feature_std", "profile_dominance_weights", "profile_dominance_bias",
    ]
    for k in frozen_keys:
        if ic.get(k) != bic.get(k):
            errs.append(f"frozen head changed: {k}")
    report = {
        "pass": not errs,
        "errors": errs,
        "expect": a.expect,
        "frozen_head_identity": not any(x.startswith("frozen head") for x in errs),
        "memory": memory_info,
        "no_validation_threshold_or_neighbor_tuning": str(ic.get("regret_risk_threshold_policy", "")).startswith("fixed_zero"),
    }
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if errs:
        raise SystemExit("STOP CONTRACT: " + "; ".join(errs))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
