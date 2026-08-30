from __future__ import annotations
import argparse,json,math
from pathlib import Path

from bdse.tools.fit_v64_3_38_eaf_icer_davr import CAPTURE_TOL, CAT, NOOP_REDUCTION_MIN
from bdse.tools.check_v64_3_39_eaf_icer_cfsr_split import (
    EPS,_containment,_f,_icer_edge_diag,_load_rows,_mechanism_gate,_metric_pack,_query_diag,_replacement_tail_diag,_selected_policy_diag,_structural,
)


def main():
    ap=argparse.ArgumentParser(description="Audit one V64.3.40 SDFR fresh block")
    tags=["raw","v20","preserve","rsmr","dense","hurdle","sign_shift","sdfr_raw","sdfr"]
    ap.add_argument("--split-name",required=True)
    for tag in tags:
        ap.add_argument(f"--{tag.replace('_','-')}-metrics",dest=f"{tag}_metrics",required=True); ap.add_argument(f"--{tag.replace('_','-')}-rows",dest=f"{tag}_rows",required=True)
        if tag!="raw": ap.add_argument(f"--{tag.replace('_','-')}-edges",dest=f"{tag}_edges",required=True)
    ap.add_argument("--output",required=True); a=ap.parse_args()
    metrics={t:json.load(open(getattr(a,t+"_metrics"))) for t in tags}; rows={t:_load_rows(getattr(a,t+"_rows")) for t in tags}; toks=set(rows["raw"])
    if len(toks)!=500 or any(set(rows[t])!=toks for t in tags[1:]): raise SystemExit("STOP DATA: V40 nine arms must contain exact paired 500 scenes")
    flagged={t for t in toks if _f(rows["raw"][t],"all_actions_safety_flagged_rate",0.0)>=0.5}; safe=toks-flagged
    valtags=["preserve","rsmr","dense","hurdle","sign_shift","sdfr_raw","sdfr"]
    q={t:_query_diag(rows["v20"],rows[t],toks) for t in valtags}; query_ok=all(v["all_query_counts_exact_scene_parity"] for v in q.values())
    structural={t:_structural(rows[t],rows["raw"],flagged) for t in ["v20"]+valtags}; struct_ok=all((not flagged) or (structural[t]["final_identity_vs_raw"]==1.0 and structural[t]["icer_structural_delegation_rate"]==1.0) for t in valtags)
    cont={t:_containment(rows["rsmr"],rows[t],safe) for t in ["dense","hurdle","sign_shift","sdfr_raw","sdfr"]}; contain_ok=all(x["monotone_selected_policy_containment_valid"] for x in cont.values())
    edge={t:_icer_edge_diag(Path(getattr(a,t+"_edges")),safe) for t in tags if t!="raw"}; tail={t:_replacement_tail_diag(rows["raw"],rows[t],getattr(a,t+"_edges"),safe) for t in tags if t!="raw"}
    policy={t:_selected_policy_diag(getattr(a,t+"_edges"),safe) for t in ["rsmr","dense","hurdle","sign_shift","sdfr_raw","sdfr"]}; M={t:_metric_pack(metrics[t]) for t in tags}
    gates={t:_mechanism_gate(policy[t],policy["rsmr"]) for t in ["dense","hurdle","sign_shift","sdfr_raw","sdfr"]}
    preserve_cap=float(edge["preserve"]["direct_incumbent_opportunity_capture_rate"]); main_cap=float(edge["sdfr"]["direct_incumbent_opportunity_capture_rate"])
    coverage=bool(math.isfinite(main_cap) and math.isfinite(preserve_cap) and main_cap>=preserve_cap+0.03-EPS)
    endp=bool(M["sdfr"]["match"]>=M["preserve"]["match"]-0.002 and M["sdfr"]["regret"]<=M["preserve"]["regret"]*1.005 and M["sdfr"]["match"]>=M["v20"]["match"]-0.002 and M["sdfr"]["regret"]<=M["v20"]["regret"]*1.005)
    eng=bool(query_ok and struct_ok and contain_ok); full=bool(eng and gates["sdfr"]["pass"] and coverage and endp)
    if not eng: nxt="STOP_fix_V40_engineering_or_frozen_winner_containment_before_scientific_interpretation"
    elif gates["hurdle"]["pass"] and not gates["sdfr"]["pass"]: nxt="population_distribution_factorization_suffices_selected_component_adaptation_unnecessary_or_harmful"
    elif gates["sign_shift"]["pass"] and not gates["sdfr"]["pass"]: nxt="selected_sign_frequency_shift_is_sufficient_simplify_discard_magnitude_adaptation"
    elif not gates["sdfr"]["existence_and_capture"]: nxt="distribution_factorization_does_not_close_capture_stop_current_19D_selected_value_distribution_route"
    elif not gates["sdfr"]["hard_tail"]: nxt="distribution_factorization_preserves_recovery_but_hard_tail_remains_next_requires_value_specific_representation_not_more_target_factorization"
    elif not coverage: nxt="SDFR_direct_mechanism_passes_but_useful_recovery_gain_over_preserve_insufficient"
    elif not endp: nxt="SDFR_direct_mechanism_passes_but_endpoint_does_not_convert_audit_runtime_path_composition"
    else: nxt="if_second_fresh_block_also_passes_freeze_SDFR_and_run_one_independent_full_validation_reproduction"
    rep={"audit":"v64_3_40_eaf_icer_sdfr_split","split_name":a.split_name,"full_split_pass":full,"engineering_valid":eng,"mechanism_gates":gates,"sdfr_capture_gain_over_preserve":main_cap-preserve_cap if math.isfinite(main_cap) and math.isfinite(preserve_cap) else float("nan"),"sdfr_meaningful_coverage":coverage,"endpoint_noninferior":endp,"next_action":nxt,"query_parity":q,"structural":structural,"containment":cont,"edge_diagnostics":edge,"selected_policy_diagnostics":policy,"direct_selected_path_tail":tail,"metrics":M,"frozen_contract":{"catastrophic_threshold":CAT,"noop_reduction_fraction_min":NOOP_REDUCTION_MIN,"capture_tolerance":CAPTURE_TOL,"promotion_capture_gain_over_preserve_min":0.03,"no_AB_pooling":True,"RSMR_winner_frozen_before_any_distribution_readout":True,"distribution_heads_never_rerank":True,"selected_distribution_adaptation_scalar_only":True,"CAL500_translation_unit_slope":True,"no_second_best_fallback":True,"no_runtime_threshold_lambda_alpha_feature_candidate_count_or_temperature_sweep":True}}
    Path(a.output).write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n"); print(json.dumps(rep,indent=2,sort_keys=True))


if __name__=="__main__": main()
