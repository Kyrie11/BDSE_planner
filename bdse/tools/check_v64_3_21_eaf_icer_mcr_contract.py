from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Any
import yaml

from bdse.planner.tournament import _ICER_DOMINANCE_PROFILE_BASE_NAMES


def _icer(cfg: dict[str,Any])->dict[str,Any]:
    return ((((cfg.get("runtime",{}) or {}).get("decisive_frontier_value",{}) or {}).get("incumbent_contrastive_extremal_recovery",{}) or {}))


def _head(ic: dict[str,Any], prefix: str)->dict[str,Any]:
    return {k:v for k,v in ic.items() if k.startswith(prefix)}


def _finite(xs)->bool:
    try: return all(math.isfinite(float(x)) for x in xs)
    except Exception: return False


def main()->None:
    ap=argparse.ArgumentParser(description="Audit V64.3.21 EAF-ICER-MCR frozen-head + retention contract.")
    ap.add_argument("--config",required=True); ap.add_argument("--expect",choices=["mcr-scalar-retention","mcr-mean","mcr-consensus"],required=True)
    ap.add_argument("--frozen-v20-dual-config",required=True); ap.add_argument("--output")
    a=ap.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text(encoding="utf-8")); old=yaml.safe_load(Path(a.frozen_v20_dual_config).read_text(encoding="utf-8"))
    ic, oi=_icer(cfg),_icer(old)
    expected_policy="dual_positive_consensus_mean" if a.expect=="mcr-consensus" else "dual_equal_mean"
    expected_retention="selected_incumbent_scalar_margin_mse" if a.expect=="mcr-scalar-retention" else "selected_incumbent_profile_margin_mse"
    expected_features=list(_ICER_DOMINANCE_PROFILE_BASE_NAMES[:18] if a.expect=="mcr-scalar-retention" else _ICER_DOMINANCE_PROFILE_BASE_NAMES)
    meta=cfg.get("metadata",{}) or {}; prov=cfg.get("provenance",{}) or {}
    rnames=list(ic.get("retention_feature_names",[]) or [])
    checks={
        "version":meta.get("algorithm_version")=="V64.3.21-EAF-ICER-MCR-DARM-DBR" and prov.get("algorithm_version")==meta.get("algorithm_version"),
        "enabled":bool(ic.get("enabled",False)),
        "dominance_policy":str(ic.get("dominance_policy",""))==expected_policy,
        "all_flagged_deployment_complete":str(ic.get("all_flagged_policy",""))=="preserve_legacy_for_structural_guard",
        "safe_domain_learning_only":bool(ic.get("require_safe_available_for_learned_intervention",False)),
        "support_head_frozen":_head(ic,"support_")==_head(oi,"support_"),
        "scalar_dominance_head_frozen":_head(ic,"scalar_dominance_")==_head(oi,"scalar_dominance_"),
        "profile_dominance_head_frozen":_head(ic,"profile_dominance_")==_head(oi,"profile_dominance_"),
        "zero_reference_logits":abs(float(ic.get("anchor_logit",99)))<1e-12 and abs(float(ic.get("incumbent_logit",99)))<1e-12,
        "retention_policy":str(ic.get("incumbent_retention_policy",""))==expected_retention,
        "retention_schema":rnames==expected_features,
        "retention_vectors":len(ic.get("retention_feature_mean",[]))==len(rnames)==len(ic.get("retention_feature_std",[]))==len(ic.get("retention_weights",[])),
        "retention_numeric_finite":_finite(ic.get("retention_feature_mean",[])) and _finite(ic.get("retention_feature_std",[])) and _finite(ic.get("retention_weights",[])) and math.isfinite(float(ic.get("retention_bias",float("nan")))) ,
        "retention_objective_fixed":str(ic.get("retention_training_objective",""))=="fixed_linear_mse_plus_l2_1e-3_zero_is_semantic_teacher_tie",
        "retention_zero_boundary":str(ic.get("retention_threshold_policy",""))=="fixed_zero_predicted_teacher_margin_no_validation_threshold_sweep",
        "target_is_selected_incumbent_margin":str(ic.get("retention_training_target",""))=="normalized_teacher_margin_JT_anchor_minus_JT_selected_incumbent",
        "no_validation_threshold_sweep":"no_validation_threshold_sweep" in str(ic.get("threshold_policy","")),
        "training_reuse":str(ic.get("training_reuse",""))=="exact_v64_3_20_support_and_v64_3_19_dominance_heads_frozen_retention_fit_train_only",
        "evaluation_only":bool((cfg.get("training",{}) or {}).get("evaluation_only",False)),
    }
    report={"audit":"v64_3_21_eaf_icer_mcr_contract","expect":a.expect,"passed":all(checks.values()),"checks":checks}
    if a.output:
        p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))
    if not report["passed"]: raise SystemExit(2)

if __name__=="__main__": main()
