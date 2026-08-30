from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.tools.fit_v64_3_33_eaf_icer_spcr import FEATURE_NAMES, _scene_samples, _select
from bdse.tools.fit_v64_3_34_eaf_icer_rsmr import _structured_scores
from bdse.tools.fit_v64_3_38_eaf_icer_davr import MIN_VALUE_CAL_PROPOSALS
from bdse.tools.fit_v64_3_39_eaf_icer_cfsr import _fit_translation

EXPECTED_SCENES = 500
PROB_EPS = 1.0e-4


def _tokens(path: Path) -> list[str]:
    return [str(json.loads(x).get("scenario_token", "")) for x in path.read_text().splitlines() if x.strip()]


def _linear(x: np.ndarray, sc: dict[str, Any], prefix: str) -> float:
    mean=np.asarray(sc.get(f"post_selection_hurdle_{prefix}_feature_mean",[]),dtype=np.float64)
    std=np.asarray(sc.get(f"post_selection_hurdle_{prefix}_feature_std",[]),dtype=np.float64)
    w=np.asarray(sc.get(f"post_selection_hurdle_{prefix}_weights",[]),dtype=np.float64)
    b=float(sc.get(f"post_selection_hurdle_{prefix}_bias",float("nan")))
    if any(v.size!=len(FEATURE_NAMES) for v in [mean,std,w]) or not math.isfinite(b): raise SystemExit(f"V40 CAL malformed hurdle {prefix}")
    z=(np.asarray(x,dtype=np.float64)-mean)/np.maximum(std,1.0e-6)
    return float(z@w+b)


def _sigmoid(x: float) -> float:
    if x>=0.0:
        e=math.exp(-min(x,60.0)); return 1.0/(1.0+e)
    e=math.exp(max(x,-60.0)); return e/(1.0+e)


def _sdfr_value(x: np.ndarray, sc: dict[str, Any]) -> float:
    p=float(np.clip(_linear(x,sc,"sign"),PROB_EPS,1.0-PROB_EPS))
    mp=float(np.clip(_linear(x,sc,"positive_magnitude"),0.0,40.0)); mn=float(np.clip(_linear(x,sc,"negative_magnitude"),0.0,40.0))
    shift=float(sc.get("post_selection_hurdle_selected_logit_shift",float("nan")))
    sp=float(sc.get("post_selection_hurdle_selected_positive_magnitude_scale",float("nan")))
    sn=float(sc.get("post_selection_hurdle_selected_negative_magnitude_scale",float("nan")))
    if not all(math.isfinite(v) for v in [shift,sp,sn]) or sp<0.0 or sn<0.0: raise SystemExit("V40 CAL malformed selected hurdle adaptation")
    l=math.log(p/max(1.0-p,1.0e-12))+shift; ps=_sigmoid(l)
    return float(np.clip(ps*(sp*mp)-(1.0-ps)*(sn*mn),-40.0,40.0))


def _parts(cfg: dict[str, Any]):
    sc=cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    if not bool(sc.get("enabled")) or str(sc.get("post_selection_value_mode",""))!="dense_edge_hurdle_selected": raise SystemExit("V40 CAL requires raw SDFR config")
    if list(sc.get("feature_names",[]))!=FEATURE_NAMES: raise SystemExit("V40 CAL RSMR feature schema mismatch")
    rmean=np.asarray(sc.get("feature_mean",[]),dtype=np.float64); rstd=np.asarray(sc.get("feature_std",[]),dtype=np.float64); rw=np.asarray(sc.get("weights",[]),dtype=np.float64)
    if any(v.size!=len(FEATURE_NAMES) for v in [rmean,rstd,rw]) or np.max(np.abs(rmean))>1e-12 or abs(float(sc.get("bias",0.0)))>1e-12: raise SystemExit("V40 CAL frozen RSMR malformed")
    return sc,(rw,rstd)


def _decorate_main(cfg: dict[str, Any], bias: float) -> dict[str, Any]:
    out=yaml.safe_load(yaml.safe_dump(cfg,sort_keys=False)); sc=out["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    sc.update({"post_selection_value_mode":"dense_edge_hurdle_selected_shift","post_selection_selected_bias":float(bias),"post_selection_value_training":"all_edge_hurdle_distribution_plus_cross_fitted_selected_component_adaptation_plus_independent_CAL500_unit_slope_translation","post_selection_operator":"freeze_RSMR_winner_then_selected_distribution_expected_value_plus_translation_accept_same_winner_iff_positive_else_incumbent"})
    out.setdefault("metadata",{})["algorithm_version"]="V64.3.40-EAF-ICER-SDFR"; out.setdefault("provenance",{})["algorithm_version"]="V64.3.40-EAF-ICER-SDFR"; out.setdefault("experiment",{})["name"]="v64_3_40_eaf_icer_sdfr"; out["experiment"]["algorithm"]="V64.3.40 selection-distribution factorized recovery"
    return out


def main()->None:
    ap=argparse.ArgumentParser(description="V64.3.40 independent CAL500 unit-slope translation")
    ap.add_argument("--calibration-rows",required=True); ap.add_argument("--calibration-edges",required=True); ap.add_argument("--sdfr-config",required=True); ap.add_argument("--output-sdfr-main-config",required=True); ap.add_argument("--output-report",required=True); a=ap.parse_args()
    rt=_tokens(Path(a.calibration_rows))
    if len(rt)!=EXPECTED_SCENES or len(set(rt))!=EXPECTED_SCENES: raise SystemExit("V40 CAL rows must be exactly 500 unique scenes")
    groups={t:[] for t in rt}; allowed=set(rt)
    for line in Path(a.calibration_edges).read_text().splitlines():
        if not line.strip(): continue
        r=json.loads(line); t=str(r.get("scenario_token",""))
        if t not in allowed: raise SystemExit(f"V40 CAL edge token outside rows: {t}")
        groups[t].append(r)
    cfg=yaml.safe_load(Path(a.sdfr_config).read_text()); sc,rsm=_parts(cfg); rw,rstd=rsm
    ys=[]; vals=[]; used=[]; replay_max=0.0
    for t in rt:
        ss=_scene_samples(groups[t])
        if not ss: continue
        score=_structured_scores(ss,(rw,rstd,{})); idx=_select(ss,score)
        if idx is None: continue
        x=np.asarray(ss[idx]["x"],dtype=np.float64); v=_sdfr_value(x,sc)
        rr=next((r for r in groups[t] if int(r.get("challenger_action",-2))==int(ss[idx]["action"])),None)
        if rr is None: raise SystemExit("V40 CAL selected action missing from edge replay")
        try: logged=float(rr.get("icer_scir_raw_predicted_improvement",rr.get("icer_scir_predicted_improvement",float("nan"))))
        except (TypeError,ValueError): logged=float("nan")
        if math.isfinite(logged): replay_max=max(replay_max,abs(logged-float(score[idx])))
        ys.append(float(ss[idx]["y"])); vals.append(v); used.append(t)
    if len(used)<MIN_VALUE_CAL_PROPOSALS: raise SystemExit(f"V40 CAL selected proposals {len(used)} < {MIN_VALUE_CAL_PROPOSALS}")
    if replay_max>1.0e-5: raise SystemExit(f"V40 CAL frozen RSMR replay mismatch {replay_max}")
    y=np.asarray(ys); v=np.asarray(vals); fit=_fit_translation(v,y,"selected_distribution_factorized_value")
    pred=v+float(fit["selected_policy_bias"])
    Path(a.output_sdfr_main_config).write_text(yaml.safe_dump(_decorate_main(cfg,fit["selected_policy_bias"]),sort_keys=False))
    rep={"audit":"v64_3_40_eaf_icer_sdfr_independent_CAL500_translation_fit","calibration_total_scene_count":EXPECTED_SCENES,"selected_policy_proposal_count":len(used),"selected_policy_proposal_count_min":MIN_VALUE_CAL_PROPOSALS,"calibration_tokens_sha256":hashlib.sha256(("\n".join(rt)+"\n").encode()).hexdigest(),"selected_policy_tokens_sha256":hashlib.sha256(("\n".join(used)+"\n").encode()).hexdigest(),"frozen_rsmr_score_replay_max_abs":float(replay_max),"proposal_teacher_improvement_sum":float(y.sum()),"proposal_precision":float(np.mean(y>0.0)),"proposal_worst":float(y.min()),"sdfr_translation_fit":fit,"sdfr_translation_sign_accuracy":float(np.mean((pred>0.0)==(y>0.0)))}
    Path(a.output_report).write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n"); print(json.dumps({"pass":True,"selected_policy_proposals":len(used),"sdfr_main_config":a.output_sdfr_main_config},sort_keys=True))


if __name__=="__main__": main()
