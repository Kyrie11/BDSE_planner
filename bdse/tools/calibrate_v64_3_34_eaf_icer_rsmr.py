from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ALPHA = 0.05
EXPECTED_SCENES = 500
MIN_SELECTED_POLICY_PROPOSALS = 64


def _f(r: dict[str, Any], k: str, d: float = float("nan")) -> float:
    try:
        v = float(r.get(k, d))
    except (TypeError, ValueError):
        return d
    return v if math.isfinite(v) else d


def _q(scores: list[float]) -> tuple[float, int]:
    a = np.asarray([float(x) for x in scores if math.isfinite(float(x))], dtype=np.float64)
    if a.size == 0:
        raise SystemExit("V64.3.34 selected-policy calibration has no finite proposal residuals")
    a.sort(); k = int(math.ceil((a.size + 1) * (1.0 - ALPHA))); k = min(max(k, 1), int(a.size))
    return max(0.0, float(a[k - 1])), k


def _tokens(path: Path) -> list[str]:
    return [str(json.loads(x).get("scenario_token", "")) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate V64.3.34 RSMR on the proposal emitted by the frozen regret-structured margin selector.")
    ap.add_argument("--calibration-rows", required=True); ap.add_argument("--calibration-edges", required=True); ap.add_argument("--rank-config", required=True)
    ap.add_argument("--output-main-config", required=True); ap.add_argument("--output-report", required=True); ap.add_argument("--alpha", type=float, default=ALPHA)
    a = ap.parse_args()
    if abs(float(a.alpha) - ALPHA) > 1e-12:
        raise SystemExit("V64.3.34 alpha is frozen at 0.05; no sweep permitted")
    rt = _tokens(Path(a.calibration_rows))
    if len(rt) != EXPECTED_SCENES or len(set(rt)) != EXPECTED_SCENES:
        raise SystemExit("V64.3.34 CAL rows must contain exactly 500 unique scenes")
    allowed = set(rt); groups: dict[str, list[dict[str, Any]]] = {t: [] for t in rt}
    for line in Path(a.calibration_edges).read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        r = json.loads(line); t = str(r.get("scenario_token", ""))
        if t not in allowed: raise SystemExit(f"V64.3.34 CAL edge token outside rows: {t}")
        groups[t].append(r)

    residuals: list[float] = []; selected_y: list[float] = []; noopp = 0; proposal_tokens: list[str] = []
    for t in rt:
        rs = groups[t]
        if not rs: continue
        inc = int(rs[0].get("raw_top_action", -1)); by = {int(r.get("challenger_action", -2)): r for r in rs}; ir = by.get(inc)
        if ir is None or _f(ir, "icer_admissible", 0.0) < 0.5: continue
        itm = _f(ir, "teacher_margin")
        if not math.isfinite(itm): continue
        alts: list[tuple[int,float,float,float,float,int]] = []
        for r in rs:
            act = int(r.get("challenger_action", -2))
            if act == inc or _f(r,"icer_admissible",0.0)<0.5 or _f(r,"icer_support_logit",-math.inf)<=0.0: continue
            y = _f(r,"teacher_margin") - itm; s = _f(r,"icer_scir_predicted_improvement")
            if math.isfinite(y) and math.isfinite(s) and s > 0.0:
                alts.append((act,y,s,_f(r,"icer_support_logit",-math.inf),_f(r,"raw_margin",-math.inf),int(_f(r,"dacer_utility_prior",0.0)>=0.5)))
        if not alts: continue
        best = sorted(alts, key=lambda x:(-x[2],-x[3],-x[4],-x[5],x[0]))[0]
        proposal_tokens.append(t); residuals.append(float(best[2]-best[1])); selected_y.append(float(best[1]))
        noopp += int(not any((_f(r,"teacher_margin")-itm)>0 for r in rs if int(r.get("challenger_action",-2))!=inc and _f(r,"icer_admissible",0.0)>=0.5 and _f(r,"icer_support_logit",-math.inf)>0.0))

    if len(residuals) < MIN_SELECTED_POLICY_PROPOSALS:
        raise SystemExit(f"V64.3.34 CAL500 structured selector produced too few proposals: {len(residuals)} < {MIN_SELECTED_POLICY_PROPOSALS}")
    q, k = _q(residuals)
    cfg = yaml.safe_load(Path(a.rank_config).read_text(encoding="utf-8"))
    ic = cfg.get("runtime",{}).get("decisive_frontier_value",{}).get("incumbent_contrastive_extremal_recovery",{}) or {}
    sc = ic.get("selection_conditioned_intervention_recovery",{}) or {}
    if not bool(sc.get("enabled")) or sc.get("mode") not in {"rank_only","mean_rank"}:
        raise SystemExit("V64.3.34 rank config is not an RSMR structured rank control")
    sc["mode"] = "conformal_veto"
    sc["conformal_overprediction_quantile"] = float(q)
    sc["calibration_status"] = "independent_CAL500_selected_policy_proposals_only_frozen_before_A_B"
    sc["proposal_operator"] = "frozen_regret_structured_margin_argmax_then_selected_policy_conformal_veto"
    ic["selection_conditioned_intervention_recovery"] = sc
    cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"] = ic
    cfg.setdefault("metadata",{})["algorithm_version"] = "V64.3.34-EAF-ICER-RSMR"
    cfg.setdefault("provenance",{})["algorithm_version"] = "V64.3.34-EAF-ICER-RSMR"
    exp = cfg.setdefault("experiment",{}); exp["name"]="v64_3_34_eaf_icer_rsmr"; exp["algorithm"]="V64.3.34 RSMR: incumbent-augmented regret-structured margin ordering with selected-policy conformal certificate"
    exp["mechanism_chain"]="bounded B16 interface -> exact EAF -> admissible direct scene -> incumbent pseudo-item + scene-max teacher-regret structured margin score -> frozen policy proposal -> independent selected-policy conformal lower bound -> incumbent default/no fallback"
    Path(a.output_main_config).write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")

    arr=np.asarray(selected_y,dtype=np.float64); neg=np.minimum(arr,0.0)
    report={
        "audit":"v64_3_34_eaf_icer_rsmr_independent_calibration","alpha":ALPHA,"calibration_total_scene_count":EXPECTED_SCENES,
        "selected_policy_proposal_count":len(residuals),"selected_policy_proposal_count_min":MIN_SELECTED_POLICY_PROPOSALS,
        "selected_policy_conformal_quantile":float(q),"conformal_order_index_1based":int(k),
        "proposal_teacher_improvement_sum_before_certificate":float(arr.sum()),"proposal_precision_before_certificate":float((arr>0).mean()),
        "proposal_worst_before_certificate":float(arr.min()),"proposal_negative_rms_before_certificate":float(np.sqrt(np.mean(neg*neg))),
        "no_positive_opportunity_proposal_count":int(noopp),
        "calibration_uses_promotion_labels":False,"fit_uses_calibration_labels":False,
        "calibration_tokens_sha256":__import__('hashlib').sha256(("\n".join(rt)+"\n").encode()).hexdigest(),
        "selected_policy_tokens_sha256":__import__('hashlib').sha256(("\n".join(proposal_tokens)+"\n").encode()).hexdigest(),
        "theorem_scope":"With the structured selector and all model parameters frozen before CAL, the one selected proposal (when one exists) is an exchangeable policy output under CAL/future scene exchangeability. Split conformal on selected-policy residual score-y gives marginal one-sided coverage for that policy output. The certificate can only accept the same proposal or return the incumbent; it does not re-rank or fall through. This is not per-scene conditional, distribution-shift, or closed-loop absolute safety.",
    }
    Path(a.output_report).write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps({"pass":True,"q":q,"selected_policy_proposals":len(residuals),"output_main_config":a.output_main_config},sort_keys=True))

if __name__ == "__main__": main()
