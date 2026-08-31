from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _write(base: dict, arm: str, out: Path) -> None:
    cfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
    # The paired intervention must be the only way the two arms differ. Disable
    # fallback/reranking so both arms use the same incumbent path before/after
    # the unique full-set RSMR proposal event.
    cfg.setdefault("fallback", {})["enabled"] = False
    cfg["fallback"]["rule_rerank_top_k"] = 0
    cfg["selected_outcome_probe"] = {
        "enabled": True,
        "arm": arm,
        "one_shot": True,
        "proposal_source": "already_frozen_full_set_RSMR_proposal",
        "control_action": "same_ICER_incumbent_baseline",
        "after_intervention": "incumbent_only",
        "teacher_or_future_label_input": False,
        "runtime_deployment_feature": False,
    }
    v = f"V64.3.50-EAF-ICER-PIOR-PROBE-{arm.upper()}"
    cfg.setdefault("metadata", {})["algorithm_version"] = v
    cfg.setdefault("provenance", {})["algorithm_version"] = v
    cfg.setdefault("experiment", {})["name"] = f"v64_3_50_eaf_icer_pior_probe_{arm}"
    cfg["experiment"]["algorithm"] = v
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Create paired V64.3.50 one-shot selected-outcome intervention configs.")
    ap.add_argument("--v49-siir-config", type=Path, required=True)
    ap.add_argument("--output-treatment", type=Path, required=True)
    ap.add_argument("--output-control", type=Path, required=True)
    a = ap.parse_args()
    base = yaml.safe_load(a.v49_siir_config.read_text(encoding="utf-8"))
    # Fail closed: V50 instrumentation assumes the V49 full-set RSMR proposal is
    # still exposed by the unchanged ICER post-selection path.
    sc = base["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    if not bool(sc.get("post_selection_value_enabled", False)):
        raise RuntimeError("V64.3.50 PIOR requires V49 post-selection path to expose the frozen RSMR proposal")
    _write(base, "treatment", a.output_treatment)
    _write(base, "control", a.output_control)
    print(f"PASS V64.3.50 PIOR probe configs: {a.output_treatment} {a.output_control}")


if __name__ == "__main__":
    main()
