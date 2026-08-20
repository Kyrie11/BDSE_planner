from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

import bdse.tools.fit_v64_3_22_eaf_icer_tcr as v22
from bdse.planner.tournament import _ICER_SEMANTIC_FAMILY_FEATURE_NAMES


def _ic(cfg):
    return cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--expect", choices=["aggregate-downside", "semantic-family-downside"], required=True)
    ap.add_argument("--frozen-v20-dual-config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(args.frozen_v20_dual_config).read_text(encoding="utf-8"))
    ic, bic = _ic(cfg), _ic(base)
    semantic = args.expect == "semantic-family-downside"
    errors: list[str] = []
    expected_mode = "semantic_family_aligned" if semantic else "evidence_only"
    expected_model_type = (
        "frozen_support_scalar_dominance_plus_semantic_family_aligned_downside_regret_certificate"
        if semantic else "frozen_support_scalar_dominance_plus_evidence_local_downside_regret_certificate"
    )
    if not ic.get("enabled"): errors.append("ICER disabled")
    if ic.get("model_type") != expected_model_type: errors.append("model_type metadata/semantics mismatch")
    if ic.get("dominance_policy") != "scalar_only": errors.append("dominance must remain frozen scalar-only")
    if ic.get("incumbent_retention_policy") != "preserve_admissible_incumbent": errors.append("incumbent default-preservation missing")
    if not ic.get("regret_risk_enabled") or ic.get("retention_regret_risk_enabled") or not ic.get("replacement_regret_risk_enabled"):
        errors.append("replacement-only risk contract broken")
    if ic.get("regret_risk_feature_mode") != expected_mode: errors.append("regret-risk feature mode mismatch")
    if ic.get("regret_risk_model_type") != "local_multiscale_downside_regret_certificate": errors.append("risk model mismatch")
    if ic.get("replacement_local_regret_certificate") != "mean_minus_downside_rms": errors.append("certificate metadata mismatch")
    if list(ic.get("replacement_local_regret_neighbor_k_values", [])) != [32, 64]: errors.append("runtime K must remain fixed 32/64")
    if ic.get("all_flagged_policy") != "preserve_legacy_for_structural_guard": errors.append("structural delegation changed")

    path = Path(str(ic.get("replacement_local_regret_memory_path", "")))
    expected_sha = str(ic.get("replacement_local_regret_memory_sha256", ""))
    info: dict[str, object] = {}
    if not path.is_file():
        errors.append(f"missing memory {path}")
    else:
        got_sha = _sha(path)
        if got_sha != expected_sha: errors.append("memory SHA mismatch")
        try:
            with np.load(path, allow_pickle=False) as z:
                names = [str(x) for x in z["feature_names"].reshape(-1)]
                weights = np.asarray(z["feature_metric_weight"], dtype=float).reshape(-1)
                ks = [int(x) for x in z["neighbor_k_values"].reshape(-1)]
                kind = str(z["certificate_kind"].reshape(-1)[0])
                dm = float(z["downside_multiplier"].reshape(-1)[0])
                mem = np.asarray(z["memory_metric_z"]); y = np.asarray(z["teacher_improvement"]).reshape(-1)
            expected_names = [f"evidence::{n}" for n in v22._ICER_REGRET_RISK_EVIDENCE_BASE_NAMES]
            if semantic:
                expected_names += [f"semantic_family::{n}" for n in _ICER_SEMANTIC_FAMILY_FEATURE_NAMES]
            if names != expected_names: errors.append("memory feature schema mismatch")
            expected_dim = 28 if semantic else 18
            if len(names) != expected_dim or len(weights) != expected_dim or not np.allclose(weights, np.full(expected_dim, 1.0/expected_dim), atol=1e-7):
                errors.append("metric must use fixed equal per-dimension weights with no semantic group weight")
            if ks != [32, 64]: errors.append("memory K must remain fixed 32/64")
            if kind != "mean_minus_downside_rms": errors.append("certificate kind mismatch")
            if abs(dm - 1.0) > 1e-8: errors.append("downside multiplier must remain fixed 1.0")
            if mem.shape != (len(y), len(names)): errors.append("memory shape mismatch")
            info = {"rows": int(len(y)), "features": len(names), "kind": kind, "sha256": got_sha}
        except Exception as exc:
            errors.append(f"memory read failed: {exc}")

    frozen = [
        "support_feature_names", "support_feature_mean", "support_feature_std", "support_weights", "support_bias",
        "scalar_dominance_feature_names", "scalar_dominance_base_feature_names", "scalar_dominance_feature_mean",
        "scalar_dominance_feature_std", "scalar_dominance_weights", "scalar_dominance_bias",
        "profile_dominance_feature_names", "profile_dominance_base_feature_names", "profile_dominance_feature_mean",
        "profile_dominance_feature_std", "profile_dominance_weights", "profile_dominance_bias",
    ]
    for key in frozen:
        if ic.get(key) != bic.get(key): errors.append("frozen head changed: " + key)

    report = {"pass": not errors, "errors": errors, "expect": args.expect, "memory": info,
              "frozen_head_identity": not any(x.startswith("frozen head") for x in errors)}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if errors: raise SystemExit("STOP CONTRACT: " + "; ".join(errors))
    print(json.dumps(report, sort_keys=True))

if __name__ == "__main__": main()
