from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Any
import numpy as np

from bdse.tools.check_v64_3_19_eaf_icer_screen import _icer_edge_diag, _metric_pack, _f, _auc


def _load_rows(path: str)->dict[str,dict[str,Any]]:
    rows=[json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    out={str(r.get("scenario_token","")):r for r in rows}
    if len(out)!=len(rows): raise SystemExit(f"duplicate scenario tokens in {path}")
    return out


def _edge_groups(path: str, allowed: set[str]|None=None)->dict[str,list[dict[str,Any]]]:
    g:dict[str,list[dict[str,Any]]]={}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        r=json.loads(line); t=str(r.get("scenario_token",""))
        if allowed is not None and t not in allowed: continue
        g.setdefault(t,[]).append(r)
    return g


def _retention_diag(edge_path: str, safe: set[str])->dict[str,float]:
    ys=[]; ps=[]; fallback_sum=0.0; fallback_n=0; n=0
    for rs in _edge_groups(edge_path,safe).values():
        if not rs: continue
        a=int(rs[0].get("anchor_action",-1)); lg=int(rs[0].get("raw_top_action",-1))
        if lg==a: continue
        lr=next((r for r in rs if int(r.get("challenger_action",-999))==lg),None)
        if lr is None or _f(lr,"icer_admissible",0)<.5: continue
        tm=_f(lr,"teacher_margin"); pred=_f(lr,"icer_incumbent_retention_margin")
        if not (math.isfinite(tm) and math.isfinite(pred)): continue
        n+=1; ys.append(int(tm>0)); ps.append(pred)
        if pred<0: fallback_n+=1; fallback_sum+=tm
    sign=float(np.mean([(p>=0)==(y>0) for p,y in zip(ps,ys)])) if ys else float("nan")
    return {"admissible_selected_incumbent_count":float(n),"retention_auc":_auc(ys,ps) if ys else float("nan"),"retention_sign_accuracy":sign,
            "predicted_anchor_fallback_rate":float(fallback_n/n) if n else float("nan"),"predicted_anchor_fallback_teacher_margin_sum":float(fallback_sum)}


def _path_diag(raw_rows:dict[str,dict[str,Any]], alg_rows:dict[str,dict[str,Any]], edge_path:str, safe:set[str], allflag:set[str])->dict[str,Any]:
    groups=_edge_groups(edge_path)
    buckets={k:{"count":0,"regret_delta_sum":0.0,"regret_delta_mean":float("nan")} for k in ["direct_incumbent_to_alternative","admissible_incumbent_to_anchor","inadmissible_incumbent_anchor_relative","keep_legacy","all_flagged_delegated","other"]}
    for t,r in alg_rows.items():
        delta=_f(r,"teacher_regret",0)-_f(raw_rows[t],"teacher_regret",0)
        if t in allflag: k="all_flagged_delegated"
        else:
            rs=groups.get(t,[])
            if not rs: k="other"
            else:
                a=int(rs[0].get("anchor_action",-1)); lg=int(rs[0].get("raw_top_action",-1)); sel=int(rs[0].get("icer_selected_action",lg))
                lr=next((x for x in rs if int(x.get("challenger_action",-999))==lg),None)
                legacy_adm=bool(lr is not None and _f(lr,"icer_admissible",0)>=.5)
                if legacy_adm and sel not in {a,lg}: k="direct_incumbent_to_alternative"
                elif legacy_adm and sel==a: k="admissible_incumbent_to_anchor"
                elif not legacy_adm and sel!=lg: k="inadmissible_incumbent_anchor_relative"
                elif sel==lg: k="keep_legacy"
                else: k="other"
        buckets[k]["count"]+=1; buckets[k]["regret_delta_sum"]+=delta
    for d in buckets.values():
        if d["count"]: d["regret_delta_mean"]=d["regret_delta_sum"]/d["count"]
    return buckets


def _guard_block_rate(rows:dict[str,dict[str,Any]], tokens:set[str])->float:
    if not tokens: return float("nan")
    return float(np.mean([_f(rows[t],"pair_action_anchor_guard_blocked_flip",0)>=.5 for t in tokens]))


def _identity_rate(a,b,tokens,key="bdse_action"):
    if not tokens:return float("nan")
    return float(np.mean([int(a[t].get(key,-99999))==int(b[t].get(key,-99998)) for t in tokens]))


def main()->None:
    ap=argparse.ArgumentParser(description="One 500-scene V64.3.21 MCR replication block checker.")
    ap.add_argument("--split-name",required=True)
    for name in ["raw","v20","scalar-retention","profile-mean","consensus"]:
        ap.add_argument(f"--{name}-metrics",required=True); ap.add_argument(f"--{name}-rows",required=True)
        if name!="raw": ap.add_argument(f"--{name}-edges",required=True)
    ap.add_argument("--output",required=True); a=ap.parse_args()
    tags=["raw","v20","scalar_retention","profile_mean","consensus"]
    metrics={t:json.load(open(getattr(a,t+"_metrics"))) for t in tags}
    rows={t:_load_rows(getattr(a,t+"_rows")) for t in tags}
    tokens=set(rows["raw"])
    if len(tokens)<480 or any(set(rows[t])!=tokens for t in tags[1:]): raise SystemExit("paired fresh token identity mismatch")
    allflag={t for t in tokens if _f(rows["raw"][t],"all_actions_safety_flagged_rate",0)>=.5}; safe=tokens-allflag
    edge_paths={t:getattr(a,t+"_edges") for t in tags[1:]}
    ed={t:_icer_edge_diag(Path(edge_paths[t]),safe) for t in tags[1:]}
    retention={t:_retention_diag(edge_paths[t],safe) for t in ["scalar_retention","profile_mean","consensus"]}
    path={t:_path_diag(rows["raw"],rows[t],edge_paths[t],safe,allflag) for t in tags[1:]}
    M={t:_metric_pack(metrics[t]) for t in tags}
    anchor={"match":_f(metrics["raw"],"selected_local_anchor_action_match"),"regret":_f(metrics["raw"],"selected_local_anchor_teacher_regret")}
    structural={
        "all_flagged_scene_count":float(len(allflag)),
        "consensus_all_flagged_final_identity_vs_raw":_identity_rate(rows["consensus"],rows["raw"],allflag),
        "profile_mean_all_flagged_final_identity_vs_raw":_identity_rate(rows["profile_mean"],rows["raw"],allflag),
        "consensus_all_flagged_delegation_rate":float(np.mean([_f(rows["consensus"][t],"decisive_frontier_icer_structural_domain_delegated",0)>=.5 for t in allflag])) if allflag else float("nan"),
        "consensus_safe_guard_block_rate":_guard_block_rate(rows["consensus"],safe),
        "v20_safe_guard_block_rate":_guard_block_rate(rows["v20"],safe),
    }
    frozen_keys=["selected_local_anchor_action_match","pair_full_interface_action_match","local_pair_full_interface_action_match","evidence_certificate_fraction","decision_budget_atom_count","proposal_candidate_atom_count","proposal_decisive_atom_recall","selected_decisive_atom_recall","effective_selected_decisive_atom_recall"]
    frozen={k:bool(math.isfinite(_f(metrics["raw"],k)) and math.isfinite(_f(metrics["consensus"],k)) and abs(_f(metrics["raw"],k)-_f(metrics["consensus"],k))<=1e-6) for k in frozen_keys}
    c=ed["consensus"]; p=ed["profile_mean"]; s=ed["scalar_retention"]
    instrumentation=c["scene_count"]>=450 and c["admissible_edge_count"]>=1800 and c["direct_counterfactual_dominance_edge_count"]>=400 and _f(metrics["consensus"],"decisive_frontier_value_complete_star_coverage")>=.99 and all(frozen.values())
    structural_ok=(len(allflag)>=3 and structural["consensus_all_flagged_final_identity_vs_raw"]==1.0 and structural["profile_mean_all_flagged_final_identity_vs_raw"]==1.0 and structural["consensus_all_flagged_delegation_rate"]==1.0 and structural["consensus_safe_guard_block_rate"]<=.001)
    candidate=c["multi_admissible_proposal_rate"]>=.25 and c["admissible_candidates_per_proposal_mean"]>=3.0
    signal=c["support_auc"]>=.65 and c["direct_counterfactual_dominance_auc"]>=.70
    magnitude=(retention["profile_mean"]["retention_auc"]>=.65 and retention["profile_mean"]["retention_sign_accuracy"]>=.65 and path["profile_mean"]["admissible_incumbent_to_anchor"]["regret_delta_sum"]<=0.0)
    recovery=(c["alternative_recovery_rate"]>=.03 and c["alternative_recovery_precision"]>=.80 and c["direct_incumbent_replacement_rate"]>=.02 and c["direct_incumbent_replacement_precision"]>=.60 and c["direct_incumbent_opportunity_capture_rate"]>=.08 and c["selected_nonanchor_teacher_better_rate"]>=.80 and c["alternative_teacher_margin_mean"]>0)
    consensus_gain=(c["direct_incumbent_replacement_precision"]>=p["direct_incumbent_replacement_precision"]+.01 and c["direct_incumbent_opportunity_capture_rate"]>=p["direct_incumbent_opportunity_capture_rate"]-.06 and M["consensus"]["regret"]<=M["profile_mean"]["regret"]*1.02 and M["consensus"]["match"]>=M["profile_mean"]["match"]-.005)
    profile_retention_incremental=(retention["profile_mean"]["retention_auc"]>=retention["scalar_retention"]["retention_auc"]+.005 or M["profile_mean"]["regret"]<=M["scalar_retention"]["regret"]*.99 or path["profile_mean"]["admissible_incumbent_to_anchor"]["regret_delta_sum"]<path["scalar_retention"]["admissible_incumbent_to_anchor"]["regret_delta_sum"])
    ret=M["consensus"]["beneficial"]/max(M["raw"]["beneficial"],1e-12) if M["raw"]["beneficial"]>0 else float("nan")
    preservation=M["raw"]["harmful"]-M["consensus"]["harmful"]>=.05 and ret>=.35 and M["consensus"]["beneficial"]>M["consensus"]["harmful"] and M["consensus"]["flip"]>=.03 and M["consensus"]["flip"]<M["raw"]["flip"]
    endpoint=M["consensus"]["match"]>=anchor["match"]+.005 and M["consensus"]["regret"]<=M["raw"]["regret"]*1.02
    full=bool(instrumentation and structural_ok and candidate and signal and magnitude and recovery and consensus_gain and preservation and endpoint)
    if full: next_action="split_pass_freeze_do_not_tune_wait_for_second_independent_fresh_block"
    elif not instrumentation: next_action="engineering_or_frozen_interface_failure"
    elif not structural_ok: next_action="deployment_domain_accounting_or_structural_delegation_failure"
    elif not candidate or not signal: next_action="frontier_or_reliability_signal_regression_do_not_tune_thresholds"
    elif not magnitude: next_action="selected_incumbent_magnitude_retention_failed_do_not_change_dominance_or_budget_audit_incumbent_conditioning"
    elif not recovery: next_action="incumbent_replacement_failed_despite_retention_do_not_tune_thresholds_audit_contrastive_false_positive_extremes"
    elif not consensus_gain: next_action="corroboration_not_helpful_keep_profile_mean_as_candidate_do_not_tune_view_weights"
    elif not preservation: next_action="recovery_harms_preservation_keep_guards_frozen_audit_selected_extremes"
    else: next_action="mechanism_passes_but_endpoint_still_fails_enter_regret_magnitude_ordering_only_after_replication_audit"
    report={"audit":"v64_3_21_eaf_icer_mcr_split","split_name":a.split_name,"full_split_pass":full,"instrumentation_valid":instrumentation,"deployment_complete_domain_alignment":structural_ok,"candidate_support_valid":candidate,"fresh_support_and_dominance_signal":signal,"selected_incumbent_magnitude_retention":magnitude,"counterfactual_recovery_mechanism":recovery,"corroboration_gain":consensus_gain,"profile_retention_incremental_diagnostic":profile_retention_incremental,"preservation_gain":preservation,"endpoint_gain":endpoint,"next_action":next_action,"structural":structural,"retention":retention,"edge_diagnostics":ed,"path_diagnostics":path,"metrics":{"anchor":anchor,**M,"consensus_beneficial_retention_vs_raw":ret},"frozen_interface":frozen,
    "thresholds":{"support_auc_min":.65,"direct_dominance_auc_min":.70,"retention_auc_min":.65,"retention_sign_accuracy_min":.65,"incumbent_to_anchor_regret_delta_sum_max":0.0,"direct_replacement_precision_min":.60,"direct_capture_min":.08,"safe_selected_nonanchor_teacher_better_min":.80,"consensus_precision_gain_min":.01,"consensus_capture_max_drop":.06,"harmful_abs_reduction_min":.05,"beneficial_retention_min":.35,"match_over_anchor_min":.005,"regret_vs_raw_tolerance":.02}}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))

if __name__=="__main__":main()
