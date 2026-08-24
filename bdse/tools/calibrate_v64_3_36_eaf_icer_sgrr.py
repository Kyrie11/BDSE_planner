from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.tools.fit_v64_3_33_eaf_icer_spcr import BASE_FEATURE_NAMES, RIDGE_LAMBDA
from bdse.tools.fit_v64_3_36_eaf_icer_sgrr import (
    BASEPOINT_NAMES, GEOMETRY_NAMES, MIN_RESERVATION_CAL_PROPOSALS,
    _fit_reservation, _geometry,
)

EXPECTED_SCENES = 500


def _f(r: dict[str, Any], k: str, d: float = float("nan")) -> float:
    try: v = float(r.get(k, d))
    except (TypeError, ValueError): return d
    return v if math.isfinite(v) else d


def _tokens(path: Path) -> list[str]:
    return [str(json.loads(x).get("scenario_token", "")) for x in path.read_text().splitlines() if x.strip()]


def _context(rs: list[dict[str, Any]], inc: int) -> np.ndarray:
    ir = next((r for r in rs if int(r.get("challenger_action", -2)) == inc), None)
    if ir is None:
        raise ValueError("V36 CAL scene lacks incumbent edge")
    vals = [_f(ir, f"icer_feature_{n}") for n in BASE_FEATURE_NAMES] + [_f(ir, "icer_support_logit")]
    a = np.asarray(vals, dtype=np.float64)
    if not np.all(np.isfinite(a)): raise ValueError("V36 CAL incumbent context is nonfinite")
    return a


def _decorate(cfg: dict[str, Any], model, mode: str, names: list[str], version: str, expname: str) -> dict[str, Any]:
    w, scale, info = model
    out = yaml.safe_load(yaml.safe_dump(cfg, sort_keys=False))
    sc = out["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    if not bool(sc.get("enabled")) or bool(sc.get("scene_reservation_enabled", False)):
        raise SystemExit("V36 rank config must be frozen RSMR without a pre-existing reservation")
    sc.update({
        "scene_reservation_enabled": True,
        "scene_reservation_feature_mode": mode,
        "scene_reservation_feature_names": names,
        "scene_reservation_feature_mean": [0.0] * len(names),
        "scene_reservation_feature_std": [float(x) for x in scale],
        "scene_reservation_weights": [float(x) for x in w],
        "scene_reservation_bias": 0.0,
        "scene_reservation_max": 40.0,
        "scene_reservation_target": "selected_policy_overprediction_positive_part",
        "scene_reservation_training": "independent_CAL500_frozen_RSMR_policy_outputs_only",
        "scene_reservation_operator": "nonnegative_common_subtraction_monotone_subset_no_rerank_no_fallback",
    })
    out.setdefault("metadata", {})["algorithm_version"] = version
    out.setdefault("provenance", {})["algorithm_version"] = version
    out.setdefault("experiment", {})["name"] = expname
    out["experiment"]["algorithm"] = f"V64.3.36 frozen-RSMR {mode} scene-reservation recovery"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit V64.3.36 clean basepoint and selection-geometry reservations on independent CAL500")
    ap.add_argument("--calibration-rows", required=True); ap.add_argument("--calibration-edges", required=True); ap.add_argument("--rsmr-config", required=True)
    ap.add_argument("--output-basepoint-config", required=True); ap.add_argument("--output-geometry-config", required=True); ap.add_argument("--output-report", required=True)
    a = ap.parse_args()
    rt = _tokens(Path(a.calibration_rows))
    if len(rt) != EXPECTED_SCENES or len(set(rt)) != EXPECTED_SCENES:
        raise SystemExit("V36 CAL rows must contain exactly 500 unique scenes")
    allowed = set(rt); groups = {t: [] for t in rt}
    for line in Path(a.calibration_edges).read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line); t = str(r.get("scenario_token", ""))
        if t not in allowed: raise SystemExit(f"V36 CAL edge token outside rows: {t}")
        groups[t].append(r)

    GX=[]; BX=[]; target=[]; ys=[]; used=[]
    for t in rt:
        rs=groups[t]
        if not rs: continue
        inc=int(rs[0].get("raw_top_action",-1)); ir=next((r for r in rs if int(r.get("challenger_action",-2))==inc),None)
        if ir is None or _f(ir,"icer_admissible",0.0)<0.5: continue
        itm=_f(ir,"teacher_margin")
        if not math.isfinite(itm): continue
        alts=[]
        for r in rs:
            act=int(r.get("challenger_action",-2))
            if act==inc or _f(r,"icer_admissible",0.0)<0.5 or _f(r,"icer_support_logit",-math.inf)<=0.0: continue
            s=_f(r,"icer_scir_predicted_improvement"); y=_f(r,"teacher_margin")-itm
            if math.isfinite(s) and math.isfinite(y):
                alts.append((act,s,y,_f(r,"icer_support_logit",-math.inf),_f(r,"raw_margin",-math.inf),int(_f(r,"dacer_utility_prior",0.0)>=0.5)))
        if not alts: continue
        scores=np.asarray([x[1] for x in alts],dtype=np.float64)
        positive=[j for j,x in enumerate(alts) if x[1]>0.0]
        if not positive: continue
        j=sorted(positive,key=lambda j:(-alts[j][1],-alts[j][3],-alts[j][4],-alts[j][5],alts[j][0]))[0]
        s=float(alts[j][1]); y=float(alts[j][2])
        GX.append(_geometry(scores)); BX.append(_context(rs,inc)); target.append(max(0.0,s-y)); ys.append(y); used.append(t)
    if len(used)<MIN_RESERVATION_CAL_PROPOSALS:
        raise SystemExit(f"V36 CAL500 frozen RSMR produced too few selected-policy proposals: {len(used)} < {MIN_RESERVATION_CAL_PROPOSALS}")
    gm=_fit_reservation(np.stack(GX),np.asarray(target),mode="selection_geometry")
    bm=_fit_reservation(np.stack(BX),np.asarray(target),mode="incumbent_basepoint")
    cfg=yaml.safe_load(Path(a.rsmr_config).read_text())
    bcfg=_decorate(cfg,bm,"incumbent_basepoint",BASEPOINT_NAMES,"V64.3.36-EAF-ICER-BPR","v64_3_36_basepoint_reservation")
    gcfg=_decorate(cfg,gm,"selection_geometry",GEOMETRY_NAMES,"V64.3.36-EAF-ICER-SGRR","v64_3_36_eaf_icer_sgrr")
    Path(a.output_basepoint_config).write_text(yaml.safe_dump(bcfg,sort_keys=False)); Path(a.output_geometry_config).write_text(yaml.safe_dump(gcfg,sort_keys=False))
    arr=np.asarray(ys,dtype=np.float64)
    report={
        "audit":"v64_3_36_eaf_icer_sgrr_independent_CAL500_reservation_fit",
        "calibration_total_scene_count":EXPECTED_SCENES,"selected_policy_proposal_count":len(used),"selected_policy_proposal_count_min":MIN_RESERVATION_CAL_PROPOSALS,
        "calibration_tokens_sha256":hashlib.sha256(("\n".join(rt)+"\n").encode()).hexdigest(),
        "selected_policy_tokens_sha256":hashlib.sha256(("\n".join(used)+"\n").encode()).hexdigest(),
        "proposal_teacher_improvement_sum_before_reservation":float(arr.sum()),"proposal_precision_before_reservation":float((arr>0).mean()),"proposal_worst_before_reservation":float(arr.min()),
        "basepoint_reservation_fit":bm[2],"selection_geometry_reservation_fit":gm[2],
        "causal_contract":"RSMR candidate weights and winner ordering are frozen before CAL. Each reservation is nonnegative and common to all challengers, so it can only veto the exact frozen winner to incumbent; it cannot re-rank, create a new proposal, or fall through to second best.",
    }
    Path(a.output_report).write_text(json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps({"pass":True,"selected_policy_proposals":len(used),"basepoint_config":a.output_basepoint_config,"geometry_config":a.output_geometry_config},sort_keys=True))


if __name__ == "__main__": main()
