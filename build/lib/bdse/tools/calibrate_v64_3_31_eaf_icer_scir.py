from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ALPHA = 0.05
MIN_PROPOSALS = 64


def _f(r: dict[str, Any], k: str, d: float = float("nan")) -> float:
    try: v=float(r.get(k,d))
    except (TypeError,ValueError): return d
    return v if math.isfinite(v) else d


def main() -> None:
    ap=argparse.ArgumentParser(description="Calibrate V64.3.31 SCIR one-sided selection-conformal overprediction quantile on an independent calibration block.")
    ap.add_argument("--calibration-edges",required=True)
    ap.add_argument("--rank-config",required=True)
    ap.add_argument("--output-main-config",required=True)
    ap.add_argument("--output-report",required=True)
    ap.add_argument("--alpha",type=float,default=ALPHA)
    args=ap.parse_args()
    if abs(float(args.alpha)-ALPHA)>1e-12:
        raise SystemExit("V64.3.31 fixes conformal alpha=0.05 before calibration; do not sweep alpha")

    groups: dict[str,list[dict[str,Any]]]={}
    with Path(args.calibration_edges).open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r=json.loads(line); groups.setdefault(str(r.get("scenario_token","")),[]).append(r)
    residuals=[]; outcomes=[]; mus=[]; tokens=[]
    for tok,g in groups.items():
        if not g: continue
        if _f(g[0],"icer_scir_proposal_exists",0.0)<0.5: continue
        inc=int(g[0].get("raw_top_action",-1)); prop=int(g[0].get("icer_scir_proposal_action",inc))
        if prop==inc: continue
        ir=[r for r in g if int(r.get("challenger_action",-2))==inc]
        pr=[r for r in g if int(r.get("challenger_action",-2))==prop]
        if not ir or not pr: continue
        y=_f(pr[0],"teacher_margin")-_f(ir[0],"teacher_margin")
        mu=_f(pr[0],"icer_scir_predicted_improvement")
        if not (math.isfinite(y) and math.isfinite(mu)): continue
        residuals.append(float(mu-y)); outcomes.append(float(y)); mus.append(float(mu)); tokens.append(tok)
    n=len(residuals)
    if n<MIN_PROPOSALS:
        raise SystemExit(f"V64.3.31 independent calibration has too few selected proposals: {n} < {MIN_PROPOSALS}")
    r=np.sort(np.asarray(residuals,dtype=np.float64))
    # Split-conformal one-sided finite-sample order statistic.
    k=int(math.ceil((n+1)*(1.0-ALPHA)))
    k=max(1,min(k,n))
    q=max(0.0,float(r[k-1]))
    y=np.asarray(outcomes,dtype=np.float64); mu=np.asarray(mus,dtype=np.float64)
    lcb=mu-q
    empirical_coverage=float(np.mean(y>=lcb-1e-12))
    accepted=lcb>0.0
    harmful=(y<=0.0)
    report={
        "audit":"v64_3_31_eaf_icer_scir_independent_calibration",
        "alpha":ALPHA,
        "calibration_scene_count":len(groups),
        "selected_proposal_count":n,
        "finite_sample_order_index_1based":k,
        "conformal_overprediction_quantile":q,
        "empirical_lower_bound_coverage":empirical_coverage,
        "rank_proposal_positive_fraction":float((y>0.0).mean()),
        "rank_proposal_teacher_improvement_sum":float(y.sum()),
        "rank_proposal_worst":float(y.min()),
        "calibrated_accept_count":int(accepted.sum()),
        "calibrated_accept_fraction_of_proposals":float(accepted.mean()),
        "calibrated_accepted_harmful_count":int(np.sum(accepted & harmful)),
        "calibrated_accepted_harmful_fraction_of_all_proposals":float(np.mean(accepted & harmful)),
        "contract":{
            "calibration_population_selected_label_free_before_teacher_access":True,
            "model_and_proposal_operator_frozen_before_calibration":True,
            "alpha_frozen_no_sweep":True,
            "selected_proposal_count_min":MIN_PROPOSALS,
            "main_is_veto_only_subset_of_rank_proposals":True,
        },
        "theorem_scope":"Under exchangeability of the independent calibration and future proposal population with the predictor/proposal rule frozen before calibration, the one-sided split-conformal lower bound has marginal coverage >=1-alpha (up to the standard finite-sample order statistic); harmful accepted interventions are a subset of lower-bound miscoverage events.",
    }
    rp=Path(args.output_report); rp.parent.mkdir(parents=True,exist_ok=True); rp.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")

    cfg=yaml.safe_load(Path(args.rank_config).read_text(encoding="utf-8"))
    ic=cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    scir=ic.get("selection_conditioned_intervention_recovery",{}) or {}
    if not bool(scir.get("enabled",False)) or str(scir.get("mode",""))!="rank_only":
        raise SystemExit("rank config is not a V64.3.31 SCIR rank-only artifact")
    if abs(float(scir.get("conformal_alpha",ALPHA))-ALPHA)>1e-12:
        raise SystemExit("rank config conformal alpha mismatch")
    scir["mode"]="conformal_veto"
    scir["conformal_overprediction_quantile"]=q
    scir["conformal_calibration_status"]="independent_cal500_frozen_before_double_fresh"
    scir["conformal_calibration_selected_proposals"]=n
    scir["conformal_order_index_1based"]=k
    ic["selection_conditioned_intervention_recovery"]=scir
    cfg.setdefault("metadata",{})["algorithm_version"]="V64.3.31-EAF-ICER-SCIR"
    cfg.setdefault("provenance",{})["algorithm_version"]="V64.3.31-EAF-ICER-SCIR"
    exp=cfg.setdefault("experiment",{})
    exp["name"]="v64_3_31_eaf_icer_scir"
    exp["algorithm"]="V64.3.31 EAF-ICER-SCIR: Selection-Conditioned Intervention Recovery with independent selected-path conformal certificate"
    exp["mechanism_chain"]="bounded B16 interface -> exact EAF attribution -> deployment-admissible direct intervention -> same-scene incumbent-contrastive continuous improvement -> one extremal proposal -> independent selected-path conformal lower bound -> accept same proposal or incumbent (no fallback) -> unchanged final/structural guards"
    out=Path(args.output_main_config); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
    print(json.dumps({"pass":True,"selected_proposals":n,"q":q,"empirical_coverage":empirical_coverage,"output_main_config":str(out)},sort_keys=True))


if __name__=="__main__": main()
