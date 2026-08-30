from __future__ import annotations

"""V64.3.50 SIOR: fit selected-outcome retention from paired closed-loop interventions.

V49 failed the preregistered offline selected-risk identification gate.  V50
therefore freezes selector, state, model class, regularization, and retention
budget, and changes only the *outcome evidence source*: the label is whether the
actual full-set RSMR winner produces a positive paired closed-loop score effect
without degrading any preregistered hard-safety metric relative to preserving
the incumbent in the identical scene.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.tools.fit_v64_3_33_eaf_icer_spcr import FOLDS, _fold
from bdse.tools.fit_v64_3_38_eaf_icer_davr import _auc
from bdse.tools.fit_v64_3_48_eaf_icer_ocrr import _fit_sign_ranker, _risk, _conformal_threshold

ALPHA_RET = 0.0779185520361991  # frozen V48 capture-derived false-veto budget
V49_FAILURE = "selection_interventional_risk_does_not_outperform_observational_selected_risk_close_current_offline_selected_risk_family"
PAIR_COLLECTION_PROTOCOL_VERSION = "v50-live-selected-event-cohort-v1"


def _read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(newline="", encoding="utf-8")))


def _candidate_features(path: Path) -> dict[str, dict[str, Any]]:
    """Load the frozen V49 full-set winner state used for identity/OBS replay.

    V50 no longer uses these offline Q/P/E values as the causal-state features
    for the paired outcome.  They remain necessary to (a) lock the exact V49
    proposal action/fold identity and (b) reconstruct the historical OBS-SIGN
    model for a fair transport baseline evaluated on the *live* intervention
    Q/P/E state.
    """
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tok = str(r["scenario_token"])
        act = int(r.get("full_selected_action", -1))
        if act < 0:
            continue
        cand = [c for c in r.get("candidates", []) if int(c.get("action", -999)) == act]
        if len(cand) != 1:
            raise RuntimeError(f"V50 ENGINEERING STOP: selected candidate not unique for {tok}/{act}")
        c = cand[0]
        out[tok] = {
            "scenario_token": tok,
            "rsm_selected_action": act,
            "rsm_selected_teacher_improvement": float(c["y"]),
            "quality_value": float(c["quality_value"]),
            "plan_control_value": float(c["plan_control_value"]),
            "ego_ref_value": float(c["ego_ref_value"]),
            "outer_test_fold": int(r["outer_test_fold"]),
            "candidate_count": int(r.get("candidate_count", 1)),
        }
    if len(out) != 502:
        raise RuntimeError(f"V50 ENGINEERING STOP: expected exact 502 full-set RSMR selected candidate states, got {len(out)}")
    return out


def _join(
    v49_scene: Path,
    offline_features: dict[str, dict[str, Any]],
    paired: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Join the frozen offline cohort with V50 live treatment-opportunity status.

    The V49 502-scene population is a discovery cohort, not a guarantee that the
    native closed-loop RSMR policy will emit a proposal in every replay.  Only
    synchronized live proposal events have a defined selected-action treatment
    effect.  Symmetric no-proposal scenes remain in ``cohort_audit`` as a
    label-free transport stratum but are excluded from SIOR fitting/calibration.
    """
    scene = {str(r["scenario_token"]): r for r in _read_csv(v49_scene)}
    if len(scene) != 782:
        raise RuntimeError("V50 ENGINEERING STOP: V49 scene audit is not 782 unique rows")
    pairs = {str(r["scenario_token"]): r for r in _read_csv(paired)}
    if set(pairs) != set(offline_features):
        raise RuntimeError(
            f"V50 ENGINEERING STOP: paired outcome/token mismatch: paired={len(pairs)} feature={len(offline_features)} "
            f"missing={len(set(offline_features)-set(pairs))} extra={len(set(pairs)-set(offline_features))}"
        )
    eligible_rows: list[dict[str, Any]] = []
    cohort_audit: list[dict[str, Any]] = []
    per_fold = {k: {"offline_discovery": 0, "live_eligible": 0, "no_live_proposal": 0} for k in range(FOLDS)}
    for tok in sorted(offline_features):
        f = offline_features[tok]
        p = pairs[tok]
        version = str(p.get("collection_protocol_version", "")).strip()
        if version != PAIR_COLLECTION_PROTOCOL_VERSION:
            raise RuntimeError(
                "V50 ENGINEERING STOP: paired outcome protocol version mismatch for "
                f"{tok}: got={version or 'missing'} expected={PAIR_COLLECTION_PROTOCOL_VERSION}. "
                "Remove only the V50 paired_train directory and recollect; never mix old and live-eligibility rows."
            )
        srow = scene[tok]
        fold = int(f["outer_test_fold"])
        per_fold[fold]["offline_discovery"] += 1
        recorded_offline = int(float(p.get("offline_v49_action_slot", p.get("frozen_v49_proposal_action", f["rsm_selected_action"]))))
        frozen = int(f["rsm_selected_action"])
        if recorded_offline != frozen:
            raise RuntimeError(
                f"V50 ENGINEERING STOP: offline V49 cohort action-slot provenance changed "
                f"{tok}: paired_record={recorded_offline} frozen_audit={frozen}"
            )
        status = str(p.get("pair_status", "")).strip()
        try:
            eligible = int(float(p.get("live_intervention_eligible", 1 if status == "eligible_intervened" else 0)))
        except Exception as exc:
            raise RuntimeError(f"V50 ENGINEERING STOP: invalid live_intervention_eligible for {tok}") from exc
        if eligible not in {0, 1}:
            raise RuntimeError(f"V50 ENGINEERING STOP: invalid live_intervention_eligible={eligible} for {tok}")
        audit_base: dict[str, Any] = {
            "scenario_token": tok,
            "outer_test_fold": fold,
            "offline_v49_action_slot": frozen,
            "pair_status": status,
            "live_intervention_eligible": eligible,
            "offline_quality_value": float(f["quality_value"]),
            "offline_plan_control_value": float(f["plan_control_value"]),
            "offline_ego_ref_value": float(f["ego_ref_value"]),
            "v49_obs_sign_risk_offline_state": float(srow["v49_obs_sign_risk"]),
        }
        if eligible == 0:
            if status != "no_live_proposal":
                raise RuntimeError(f"V50 ENGINEERING STOP: ineligible pair has unexpected status for {tok}: {status!r}")
            if int(float(p.get("proposal_action", -1))) != -1 or int(float(p.get("intervention_iteration", -1))) != -1:
                raise RuntimeError(f"V50 ENGINEERING STOP: no-live-proposal row carries a fake intervention for {tok}")
            if int(float(p.get("preintervention_pair_aligned", 0))) != 1:
                raise RuntimeError(f"V50 ENGINEERING STOP: no-live-proposal pair was not fully aligned for {tok}")
            if int(float(p.get("no_treatment_metric_equivalent", 0))) != 1:
                raise RuntimeError(f"V50 ENGINEERING STOP: no-live-proposal metric equivalence not certified for {tok}")
            per_fold[fold]["no_live_proposal"] += 1
            cohort_audit.append({
                **audit_base,
                "live_rsm_selected_action": -1,
                "live_proposal_fingerprint": "",
                "intervention_iteration": -1,
                "intervention_time_s": "",
                "quality_value": "",
                "plan_control_value": "",
                "ego_ref_value": "",
                "paired_score_delta": "",
                "hard_noninferior": "",
                "safe_benefit": "",
            })
            continue

        if status != "eligible_intervened":
            raise RuntimeError(f"V50 ENGINEERING STOP: live-eligible pair has unexpected status for {tok}: {status!r}")
        proposal = int(float(p["proposal_action"]))
        if proposal < 0:
            raise RuntimeError(f"V50 ENGINEERING STOP: invalid live RSMR proposal slot for {tok}: {proposal}")
        fp = str(p.get("live_proposal_fingerprint", "")).strip()
        if not fp:
            raise RuntimeError(f"V50 ENGINEERING STOP: missing live proposal fingerprint for {tok}")
        it = int(float(p["intervention_iteration"]))
        if it < 0:
            raise RuntimeError(f"V50 ENGINEERING STOP: invalid intervention iteration for {tok}: {it}")
        if int(float(p.get("preintervention_pair_aligned", 0))) != 1:
            raise RuntimeError(f"V50 ENGINEERING STOP: pre-intervention CONTROL/TREATMENT alignment not certified for {tok}")
        safe = bool(int(float(p["safe_benefit"])))
        hard_ok = bool(int(float(p["hard_noninferior"])))
        delta = float(p["paired_score_delta"])
        if safe != bool(hard_ok and delta > 0.0):
            raise RuntimeError(f"V50 ENGINEERING STOP: inconsistent paired safe-benefit label {tok}")
        q = float(p["live_quality_value"]); pv = float(p["live_plan_control_value"]); e = float(p["live_ego_ref_value"])
        if not all(math.isfinite(x) for x in (q, pv, e)):
            raise RuntimeError(f"V50 ENGINEERING STOP: non-finite live Q/P/E for {tok}: {(q,pv,e)}")
        row = {
            "scenario_token": tok,
            "outer_test_fold": fold,
            "rsm_selected_action": frozen,
            "live_rsm_selected_action": proposal,
            "offline_v49_action_slot": frozen,
            "live_vs_offline_action_slot_equal": int(proposal == frozen),
            "live_proposal_fingerprint": fp,
            "quality_value": q,
            "plan_control_value": pv,
            "ego_ref_value": e,
            "candidate_count": int(f["candidate_count"]),
            "intervention_iteration": it,
            "intervention_time_s": float(p.get("intervention_time_s", 0.0)),
            "rsm_selected_teacher_improvement": 1.0 if safe else -1.0,
            "paired_score_delta": delta,
            "hard_noninferior": hard_ok,
            "safe_benefit": safe,
            "hard_regressions": str(p.get("hard_regressions", "")),
            "v49_obs_sign_risk_offline_state": float(srow["v49_obs_sign_risk"]),
            "offline_quality_value": float(f["quality_value"]),
            "offline_plan_control_value": float(f["plan_control_value"]),
            "offline_ego_ref_value": float(f["ego_ref_value"]),
        }
        per_fold[fold]["live_eligible"] += 1
        eligible_rows.append(row)
        cohort_audit.append({**audit_base, **row, "pair_status": status, "live_intervention_eligible": 1})

    folds = sorted(set(int(r["outer_test_fold"]) for r in cohort_audit))
    if folds != list(range(FOLDS)) or any(_fold(str(r["scenario_token"])) != int(r["outer_test_fold"]) for r in cohort_audit):
        raise RuntimeError("V50 ENGINEERING STOP: V49/V50 outer-fold identity mismatch")
    transport = {
        "offline_discovery_cohort": len(offline_features),
        "live_intervention_eligible": len(eligible_rows),
        "no_live_proposal": len(offline_features) - len(eligible_rows),
        "live_eligibility_rate": float(len(eligible_rows) / max(len(offline_features), 1)),
        "per_fold": per_fold,
        "interpretation": "no_live_proposal is a label-free offline-to-live selection-transport stratum, not a negative SIOR outcome",
    }
    return eligible_rows, cohort_audit, transport


def _statistical_admissibility(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Check only estimator-required support; do not invent an eligibility-rate threshold.

    These counts are inherited from the frozen V48/V49 estimator implementation:
    pairwise fitting requires >=32 positive and >=32 nonpositive training events,
    and conformal calibration requires >=16 positives.  AUC additionally requires
    both outcome classes in every held-out test fold.  Failure is a scientific
    population-support STOP, not an engineering exception and not a reason to
    tune the estimator after observing outcomes.
    """
    folds: list[dict[str, Any]] = []
    ok = True
    for k in range(FOLDS):
        cf = (k + 1) % FOLDS
        fit = [r for r in rows if int(r["outer_test_fold"]) not in {k, cf}]
        cal = [r for r in rows if int(r["outer_test_fold"]) == cf]
        test = [r for r in rows if int(r["outer_test_fold"]) == k]
        fp = sum(bool(r["safe_benefit"]) for r in fit); fn = len(fit) - fp
        cp = sum(bool(r["safe_benefit"]) for r in cal)
        tp = sum(bool(r["safe_benefit"]) for r in test); tn = len(test) - tp
        fold_ok = bool(fp >= 32 and fn >= 32 and cp >= 16 and tp >= 1 and tn >= 1)
        ok = ok and fold_ok
        folds.append({
            "fold": k, "calibration_fold": cf,
            "fit_count": len(fit), "fit_safe_benefit": fp, "fit_nonbenefit": fn,
            "cal_count": len(cal), "cal_safe_benefit": cp,
            "test_count": len(test), "test_safe_benefit": tp, "test_nonbenefit": tn,
            "admissible": fold_ok,
        })
    final_fit = [r for r in rows if int(r["outer_test_fold"]) != 0]
    final_cal = [r for r in rows if int(r["outer_test_fold"]) == 0]
    fpos = sum(bool(r["safe_benefit"]) for r in final_fit); fneg = len(final_fit) - fpos
    cpos = sum(bool(r["safe_benefit"]) for r in final_cal)
    final_ok = bool(fpos >= 32 and fneg >= 32 and cpos >= 16)
    ok = ok and final_ok
    return {
        "pass": bool(ok),
        "requirements_source": "frozen V48/V49 pairwise-ranker and conformal-calibration minimum support",
        "folds": folds,
        "final_fit_count": len(final_fit), "final_fit_safe_benefit": fpos, "final_fit_nonbenefit": fneg,
        "final_cal_count": len(final_cal), "final_cal_safe_benefit": cpos,
        "final_fit_admissible": final_ok,
    }


def _write_cohort_audit(path: Path, cohort_rows: list[dict[str, Any]], oof_risk: dict[str, float] | None = None, keep_by: dict[str, bool] | None = None) -> None:
    oof_risk = oof_risk or {}; keep_by = keep_by or {}
    enriched = []
    for r in cohort_rows:
        tok = str(r["scenario_token"])
        enriched.append({
            **r,
            "v50_oof_cl_risk": oof_risk.get(tok, ""),
            "v50_oof_retained": int(keep_by[tok]) if tok in keep_by else "",
        })
    fields = sorted({k for r in enriched for k in r})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(enriched)


def _metric(rows: list[dict[str, Any]], risks: np.ndarray, tau: float) -> dict[str, Any]:
    keep = np.asarray(risks, dtype=np.float64) <= float(tau)
    benefit = np.asarray([bool(r["safe_benefit"]) for r in rows], dtype=bool)
    hard_bad = np.asarray([not bool(r["hard_noninferior"]) for r in rows], dtype=bool)
    delta = np.asarray([float(r["paired_score_delta"]) for r in rows], dtype=np.float64)
    neg = delta[keep & (delta < 0.0)]
    return {
        "population": int(len(rows)), "retained": int(keep.sum()),
        "safe_benefit_population": int(benefit.sum()), "safe_benefit_retained": int((keep & benefit).sum()),
        "safe_benefit_retention": float((keep & benefit).sum() / max(int(benefit.sum()), 1)),
        "nonbenefit_population": int((~benefit).sum()), "nonbenefit_retained": int((keep & ~benefit).sum()),
        "hard_regression_population": int(hard_bad.sum()), "hard_regression_retained": int((keep & hard_bad).sum()),
        "paired_score_delta_sum_all_rsmr": float(delta.sum()),
        "paired_score_delta_sum_retained": float(delta[keep].sum()),
        "paired_negative_rms_retained": float(np.sqrt(np.mean(neg * neg))) if neg.size else 0.0,
        "worst_retained_delta": float(delta[keep].min()) if keep.any() else 0.0,
        "threshold": float(tau),
    }


def _baseline_metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    delta=np.asarray([float(r["paired_score_delta"]) for r in rows]); benefit=np.asarray([bool(r["safe_benefit"]) for r in rows]); hard=np.asarray([not bool(r["hard_noninferior"]) for r in rows]); neg=delta[delta<0]
    return {"population":len(rows),"safe_benefit_population":int(benefit.sum()),"nonbenefit_population":int((~benefit).sum()),"hard_regression_population":int(hard.sum()),"paired_score_delta_sum":float(delta.sum()),"paired_negative_rms":float(np.sqrt(np.mean(neg*neg))) if neg.size else 0.0,"worst_delta":float(delta.min())}


def _check_v49(report: Path) -> dict[str, Any]:
    r=json.loads(report.read_text(encoding="utf-8")); n=r.get("nested_crossfit",{})
    if n.get("train_gate_pass") is not False or n.get("failure_diagnosis") != V49_FAILURE:
        raise RuntimeError("V50 ENGINEERING STOP: V49 preregistered offline-family failure signature changed")
    ident=n.get("risk_identification",{})
    # V49's persisted schema is flat. The V49 artifacts are byte-locked by the
    # launcher, so V50 only needs to verify that the preregistered identification
    # failure remains semantically true; duplicating one exact AUC constant here
    # is redundant and previously used the wrong nested schema.
    try:
        ego_auc=float(ident["aggregate_ego_ref_auc"])
        obs_auc=float(ident["aggregate_obs_sign_auc"])
        siir_auc=float(ident["aggregate_siir_auc"])
        better_ego=int(ident["siir_better_ego_fold_count"])
        better_obs=int(ident["siir_better_obs_fold_count"])
        identified=ident["identified"]
    except (KeyError,TypeError,ValueError) as e:
        raise RuntimeError(f"V50 ENGINEERING STOP: V49 risk-identification schema changed: {e}") from e
    if not all(math.isfinite(x) for x in (ego_auc,obs_auc,siir_auc)):
        raise RuntimeError("V50 ENGINEERING STOP: V49 risk-identification AUC is non-finite")
    recomputed=bool(siir_auc>max(ego_auc,obs_auc)+1.0e-12 and better_ego>=4 and better_obs>=4)
    if identified is not False or recomputed:
        raise RuntimeError(
            "V50 ENGINEERING STOP: V49 preregistered SIIR identification-failure semantics changed: "
            f"identified={identified} aucs={(ego_auc,obs_auc,siir_auc)} folds={(better_ego,better_obs)}"
        )
    return r


def _write_config(src: Path, model: dict[str, Any], tau: float, output: Path) -> None:
    cfg=yaml.safe_load(src.read_text(encoding="utf-8")); sc=cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    rcfg=sc.get("operator_conditioned_risk_retention",{})
    rcfg["use_extremal_multiplicity"]=False
    rcfg["aggregation"]="sign_only"
    rcfg["components"]={"sign_risk": model}
    rcfg["retention_threshold"]=float(tau)
    rcfg["training_evidence_source"]="paired_closed_loop_one_shot_full_set_RSMR_vs_incumbent"
    rcfg["paired_hard_noninferiority_required"] = True
    rcfg["alpha_retention_budget"] = float(ALPHA_RET)
    sc["operator_conditioned_risk_retention"]=rcfg
    sc["post_selection_value_training"]="paired_closed_loop_selected_outcome_pairwise_sign_risk_fixed_lambda_1_same_QPE_no_multiplicity"
    sc["post_selection_operator"]="freeze_full_set_RSMR_winner_then_SIOR_veto_only_same_winner_or_incumbent_no_rerank_no_fallback"
    cfg.pop("selected_outcome_probe",None)
    cfg.setdefault("metadata",{})["algorithm_version"]="V64.3.50-EAF-ICER-SIOR"
    cfg.setdefault("provenance",{})["algorithm_version"]="V64.3.50-EAF-ICER-SIOR"
    cfg.setdefault("experiment",{})["algorithm"]="V64.3.50-EAF-ICER-SIOR"
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding="utf-8")


def main() -> None:
    ap=argparse.ArgumentParser(description="Fit V64.3.50 SIOR from paired closed-loop selected outcomes")
    ap.add_argument("--v49-fit-report",type=Path,required=True)
    ap.add_argument("--v49-scene-audit",type=Path,required=True)
    ap.add_argument("--v49-candidate-audit",type=Path,required=True)
    ap.add_argument("--paired-outcomes",type=Path,required=True)
    ap.add_argument("--v49-siir-config",type=Path,required=True)
    ap.add_argument("--output-config",type=Path,required=True)
    ap.add_argument("--output-report",type=Path,required=True)
    ap.add_argument("--output-scene-audit",type=Path,required=True)
    a=ap.parse_args(); _check_v49(a.v49_fit_report)
    offline_features=_candidate_features(a.v49_candidate_audit)
    rows, cohort_audit, cohort_transport=_join(a.v49_scene_audit,offline_features,a.paired_outcomes)
    support=_statistical_admissibility(rows)
    if not support["pass"]:
        report={
          "algorithm":"V64.3.50-EAF-ICER-SIOR", "train_gate_pass":False,
          "failure_diagnosis":"live_selected_event_population_insufficient_for_frozen_sior_estimator",
          "cohort_transport":cohort_transport, "statistical_admissibility":support,
          "preregistered_next":"Scientific STOP before SIOR fitting/fresh. Do not relabel no-live-proposal scenes as negatives or tune estimator minima; diagnose offline-to-live selector transport and evidence-source coverage."
        }
        a.output_report.parent.mkdir(parents=True,exist_ok=True); a.output_report.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
        _write_cohort_audit(a.output_scene_audit,cohort_audit)
        raise SystemExit("V64.3.50 SIOR live selected-event population is insufficient for the frozen estimator; scientific STOP before fresh")
    base=_baseline_metric(rows)
    fold_reports=[]; oof_risk={}; better_obs=0; better_ego=0
    for k in range(FOLDS):
        cf=(k+1)%FOLDS
        fit=[r for r in rows if int(r["outer_test_fold"]) not in {k,cf}]
        cal=[r for r in rows if int(r["outer_test_fold"])==cf]
        test=[r for r in rows if int(r["outer_test_fold"])==k]
        m=_fit_sign_ranker(fit,use_multiplicity=False)
        tau,caldiag=_conformal_threshold(cal,m,ALPHA_RET)
        # Reconstruct the historical observational selected-sign model on the
        # frozen V49 offline training events, then transport that unchanged model
        # to the same live intervention Q/P/E test states used by SIOR.  This is
        # the fair comparator after event-state alignment.
        obs_fit=[r for r in offline_features.values() if int(r["outer_test_fold"]) not in {k,cf}]
        obs_model=_fit_sign_ranker(obs_fit,use_multiplicity=False)
        rr=np.asarray([_risk(r,m) for r in test]); ybad=np.asarray([not bool(r["safe_benefit"]) for r in test],dtype=np.int64)
        obs_rr=np.asarray([_risk(r,obs_model) for r in test])
        auc=float(_auc(ybad,rr)); obs=float(_auc(ybad,obs_rr)); ego=float(_auc(ybad,np.asarray([-float(r["ego_ref_value"]) for r in test])))
        better_obs += int(auc>obs); better_ego += int(auc>ego)
        for r,v in zip(test,rr): oof_risk[str(r["scenario_token"])]=float(v)
        mm=_metric(test,rr,tau)
        fold_reports.append({"fold":k,"calibration_fold":cf,"fit_count":len(fit),"cal_count":len(cal),"test_count":len(test),"cl_sior_bad_risk_auc":auc,"transported_v49_obs_bad_risk_auc_on_live_QPE_cl_labels":obs,"neg_ego_ref_bad_risk_auc_on_cl_labels":ego,"calibration":caldiag,"deployment":mm})
    allrisk=np.asarray([oof_risk[str(r["scenario_token"])] for r in rows]); ybad=np.asarray([not bool(r["safe_benefit"]) for r in rows],dtype=np.int64)
    # OOF historical-OBS risk transported to live event state, reconstructed with
    # the exact same fold exclusions as the SIOR comparison.
    obs_oof: dict[str,float] = {}
    for k in range(FOLDS):
        cf=(k+1)%FOLDS
        obs_fit=[r for r in offline_features.values() if int(r["outer_test_fold"]) not in {k,cf}]
        obs_model=_fit_sign_ranker(obs_fit,use_multiplicity=False)
        for r in rows:
            if int(r["outer_test_fold"])==k:
                obs_oof[str(r["scenario_token"])]=float(_risk(r,obs_model))
    obsrisk=np.asarray([obs_oof[str(r["scenario_token"])] for r in rows])
    auc=float(_auc(ybad,allrisk)); obs=float(_auc(ybad,obsrisk)); ego=float(_auc(ybad,np.asarray([-float(r["ego_ref_value"]) for r in rows])))
    # OOF thresholds are fold-specific; reconstruct keep decisions per fold.
    keep_by={}
    for fr in fold_reports:
        tau=float(fr["calibration"]["threshold"]); k=int(fr["fold"])
        for r in rows:
            if int(r["outer_test_fold"])==k: keep_by[str(r["scenario_token"])]=oof_risk[str(r["scenario_token"])]<=tau
    benefit=np.asarray([bool(r["safe_benefit"]) for r in rows]); hard=np.asarray([not bool(r["hard_noninferior"]) for r in rows]); delta=np.asarray([float(r["paired_score_delta"]) for r in rows]); keep=np.asarray([keep_by[str(r["scenario_token"])] for r in rows])
    neg=delta[keep&(delta<0)]
    agg={"population":len(rows),"retained":int(keep.sum()),"safe_benefit_population":int(benefit.sum()),"safe_benefit_retained":int((keep&benefit).sum()),"safe_benefit_retention":float((keep&benefit).sum()/max(int(benefit.sum()),1)),"nonbenefit_population":int((~benefit).sum()),"nonbenefit_retained":int((keep&~benefit).sum()),"hard_regression_population":int(hard.sum()),"hard_regression_retained":int((keep&hard).sum()),"paired_score_delta_sum_all_rsmr":float(delta.sum()),"paired_score_delta_sum_retained":float(delta[keep].sum()),"paired_negative_rms_retained":float(np.sqrt(np.mean(neg*neg))) if neg.size else 0.0,"worst_retained_delta":float(delta[keep].min()) if keep.any() else 0.0}
    ident=bool(auc>obs and auc>ego and better_obs>=4 and better_ego>=4)
    capture_ok=agg["safe_benefit_retention"]+1e-12 >= 1.0-ALPHA_RET
    hard_ok=agg["hard_regression_retained"]==0
    nonbenefit_ok=agg["nonbenefit_retained"] <= math.floor(0.8*agg["nonbenefit_population"]+1e-12)
    sum_ok=agg["paired_score_delta_sum_retained"] >= max(0.0,agg["paired_score_delta_sum_all_rsmr"])-1e-12
    base_neg=float(base["paired_negative_rms"]); neg_ok=agg["paired_negative_rms_retained"] <= base_neg+1e-12
    fold_sum_ok=all(float(fr["deployment"]["paired_score_delta_sum_retained"])>=-1e-12 for fr in fold_reports)
    fold_hard_ok=all(int(fr["deployment"]["hard_regression_retained"])==0 for fr in fold_reports)
    deployment=bool(capture_ok and hard_ok and nonbenefit_ok and sum_ok and neg_ok and fold_sum_ok and fold_hard_ok)
    passed=bool(ident and deployment)
    # Preserve the V48/V49 final-fit calibration independence: fold 0 is a
    # dedicated calibration block and never participates in the final ranker fit.
    final_fit_rows=[r for r in rows if int(r["outer_test_fold"]) != 0]
    final_cal_rows=[r for r in rows if int(r["outer_test_fold"]) == 0]
    full_model=_fit_sign_ranker(final_fit_rows,use_multiplicity=False)
    full_tau,full_cal=_conformal_threshold(final_cal_rows,full_model,ALPHA_RET)
    full_cal["final_fit_folds"]=[1,2,3,4]; full_cal["final_calibration_fold"]=0
    _write_config(a.v49_siir_config,full_model,full_tau,a.output_config)
    report={
      "algorithm":"V64.3.50-EAF-ICER-SIOR","train_gate_pass":passed,
      "mechanism":"same frozen Q/P/E coordinate definitions, ranker class and calibration; V49 provides a frozen offline discovery cohort, while SIOR labels are generated only for synchronized live RSMR proposal events by paired one-shot closed-loop intervention",
      "event_state_alignment":{"absolute_iteration_zero_required":False,"control_treatment_preintervention_trace_required_equal":True,"cross_state_offline_action_slot_equality_required":False,"same_live_proposal_fingerprint_required":True,"live_qpe_must_match_between_arms":True,"offline_v49_qpe_used_for_sior_features":False},
      "cohort_transport":cohort_transport,
      "statistical_admissibility":support,
      "paired_outcome_definition":{"positive":"treatment aggregate score > control AND no hard metric regresses","hard_metrics":["no_ego_at_fault_collisions","time_to_collision_within_bound","drivable_area_compliance","driving_direction_compliance"]},
      "baseline_full_rsmr_one_shot":base,
      "risk_identification":{"cl_sior_bad_auc":auc,"transported_v49_obs_bad_auc_on_same_live_QPE_cl_labels":obs,"neg_ego_ref_bad_auc_on_same_cl_labels":ego,"folds_better_than_transported_v49_obs":better_obs,"folds_better_than_ego":better_ego,"identified":ident},
      "nested_oof_deployment":agg,"folds":fold_reports,
      "gates":{"live_selected_population_statistically_admissible":support["pass"],"identification":ident,"safe_benefit_retention":capture_ok,"zero_hard_regression_retained":hard_ok,"nonbenefit_reduction_20pct":nonbenefit_ok,"aggregate_noninferiority_and_nonnegative":sum_ok,"negative_rms_noninferiority":neg_ok,"five_of_five_fold_sum_nonnegative":fold_sum_ok,"five_of_five_fold_zero_hard_regression":fold_hard_ok,"deployment":deployment},
      "full_fit":{"model":full_model,"calibration":full_cal},
      "preregistered_next":"If TRAIN fails, do not tune QPE/loss/lambda/threshold or return to offline selection intervention; diagnose whether paired selected outcomes are unidentifiable from the frozen state and move to richer causal state/evidence only if mechanism evidence supports it. If TRAIN passes, freeze V50 and run untouched paired closed-loop A500 and B500 independently; no pooling/tuning.",
    }
    a.output_report.parent.mkdir(parents=True,exist_ok=True); a.output_report.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    _write_cohort_audit(a.output_scene_audit,cohort_audit,oof_risk,keep_by)
    print(json.dumps({"train_gate_pass":passed,"risk_identification":report["risk_identification"],"deployment":agg,"gates":report["gates"]},indent=2,sort_keys=True))
    if not passed:
        raise SystemExit("V64.3.50 SIOR nested paired-closed-loop TRAIN gate failed; scientific STOP before untouched fresh closed-loop selection")


if __name__=="__main__": main()
