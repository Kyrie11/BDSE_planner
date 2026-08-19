from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES, _ICER_TYPED_EVIDENCE_FEATURE_NAMES


def _icer(c):
    return c["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]


def _expected(expect: str) -> tuple[str, str, str, bool]:
    if expect == "evidence-lcb":
        return "evidence_only", "local_multiscale_regret_lower_bound", "dominance_first", False
    if expect == "evidence-tail":
        return "evidence_only", "local_multiscale_tail_coherence", "dominance_first", True
    if expect == "typed-lcb":
        return "typed_interaction", "local_multiscale_regret_lower_bound", "dominance_first", False
    if expect == "typed-tail-dominance":
        return "typed_interaction", "local_multiscale_tail_coherence", "dominance_first", True
    if expect == "typed-tail-risk-first":
        return "typed_interaction", "local_multiscale_tail_coherence", "regret_risk_first", True
    raise ValueError(expect)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check frozen V64.3.24 typed-tail ICER contract.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--expect", choices=["evidence-lcb", "evidence-tail", "typed-lcb", "typed-tail-dominance", "typed-tail-risk-first"], required=True)
    ap.add_argument("--frozen-v20-dual-config", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    c = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    b = yaml.safe_load(Path(a.frozen_v20_dual_config).read_text(encoding="utf-8"))
    ic, bic = _icer(c), _icer(b)
    mode, model, rank, tail = _expected(a.expect)
    errs: list[str] = []

    if not bool(ic.get("enabled", False)): errs.append("ICER disabled")
    if str(ic.get("dominance_policy", "")) != "scalar_only": errs.append("V24 must keep frozen scalar dominance only")
    if str(ic.get("incumbent_retention_policy", "")) != "preserve_admissible_incumbent": errs.append("incumbent preservation changed")
    if not bool(ic.get("regret_risk_enabled", False)) or bool(ic.get("retention_regret_risk_enabled", True)) or not bool(ic.get("replacement_regret_risk_enabled", False)):
        errs.append("replacement-only regret contract broken")
    if str(ic.get("regret_risk_feature_mode", "")) != mode: errs.append("feature mode mismatch")
    if str(ic.get("regret_risk_model_type", "")) != model: errs.append("risk model mismatch")
    if str(ic.get("replacement_rank_policy", "")) != rank: errs.append("replacement rank policy mismatch")
    if str(ic.get("regret_risk_threshold_policy", "")) != "fixed_zero_no_validation_sweep": errs.append("zero-boundary/no-sweep contract missing")
    if str(ic.get("all_flagged_policy", "")) != "preserve_legacy_for_structural_guard": errs.append("structural delegation changed")
    if mode == "typed_interaction" and str(ic.get("typed_evidence_contract", "")) != "selected_B_atom_type_and_predicted_g_candidate_minus_incumbent_only_no_new_query":
        errs.append("typed evidence no-new-query contract missing")

    mem_path = Path(str(ic.get("replacement_local_regret_memory_path", "")))
    mem_info: dict[str, object] = {}
    if not mem_path.is_file():
        errs.append(f"missing memory: {mem_path}")
    else:
        got = hashlib.sha256(mem_path.read_bytes()).hexdigest()
        if got != str(ic.get("replacement_local_regret_memory_sha256", "")): errs.append("memory SHA mismatch")
        try:
            with np.load(mem_path, allow_pickle=False) as z:
                names = [str(x) for x in np.asarray(z["feature_names"]).reshape(-1).tolist()]
                weights = np.asarray(z["feature_metric_weight"], dtype=float).reshape(-1)
                ks = [int(x) for x in np.asarray(z["neighbor_k_values"]).reshape(-1).tolist()]
                se = float(np.asarray(z["se_multiplier"]).reshape(-1)[0])
                rows = int(np.asarray(z["teacher_improvement"]).reshape(-1).shape[0])
                expected_names = [f"evidence::{n}" for n in _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES]
                if mode == "typed_interaction": expected_names += [f"typed::{n}" for n in _ICER_TYPED_EVIDENCE_FEATURE_NAMES]
                if names != expected_names: errs.append("memory feature schema mismatch")
                if ks != [32, 64] or abs(se - 1.0) > 1e-8: errs.append("K32/K64 one-SE contract changed")
                e = len(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
                if mode == "evidence_only":
                    if len(weights) != e or abs(float(weights.sum()) - 1.0) > 1e-5: errs.append("evidence metric weights invalid")
                else:
                    t = len(_ICER_TYPED_EVIDENCE_FEATURE_NAMES)
                    if len(weights) != e+t or abs(float(weights[:e].sum())-1.0)>1e-5 or abs(float(weights[e:].sum())-1.0)>1e-5:
                        errs.append("typed metric must equal-balance evidence and typed views")
                if tail:
                    if "material_delta_threshold" not in z.files or "tail_se_multiplier" not in z.files:
                        errs.append("tail memory metadata missing")
                    else:
                        tau = float(np.asarray(z["material_delta_threshold"]).reshape(-1)[0])
                        tse = float(np.asarray(z["tail_se_multiplier"]).reshape(-1)[0])
                        base_tau = float(b.get("fallback", {}).get("tau_delta_normalized", np.nan))
                        if not np.isfinite(base_tau) or abs(tau-base_tau) > 1e-8: errs.append("material threshold not frozen V20 tau")
                        if abs(tse-1.0) > 1e-8: errs.append("tail SE multiplier must remain 1")
                mem_info = {"path": str(mem_path), "sha256": got, "rows": rows, "features": len(names), "neighbor_k_values": ks}
        except Exception as e:
            errs.append(f"memory read failed: {e}")

    frozen_keys = [
        "support_feature_names", "support_feature_mean", "support_feature_std", "support_weights", "support_bias",
        "scalar_dominance_feature_names", "scalar_dominance_base_feature_names", "scalar_dominance_feature_mean", "scalar_dominance_feature_std", "scalar_dominance_weights", "scalar_dominance_bias",
        "profile_dominance_feature_names", "profile_dominance_base_feature_names", "profile_dominance_feature_mean", "profile_dominance_feature_std", "profile_dominance_weights", "profile_dominance_bias",
    ]
    for k in frozen_keys:
        if ic.get(k) != bic.get(k): errs.append(f"frozen head changed: {k}")

    report = {
        "pass": not errs, "errors": errs, "expect": a.expect, "feature_mode": mode, "risk_model": model,
        "replacement_rank_policy": rank, "tail_objective": tail, "memory": mem_info,
        "frozen_head_identity": not any(e.startswith("frozen head") for e in errs),
        "no_validation_threshold_or_K_tuning": True,
    }
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if errs: raise SystemExit("STOP CONTRACT: " + "; ".join(errs))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__": main()
