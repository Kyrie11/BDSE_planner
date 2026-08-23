from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import bdse.tools.fit_v64_3_25_eaf_icer_drc as v25


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _v30_cfg(base: dict[str, Any], memory: dict[str, Any], cert: str, tag: str) -> dict[str, Any]:
    cfg = v25._cfg(base, memory, cert, tag)
    version = "V64.3.30-EAF-ICER-FBIC-DRC"
    cfg.setdefault("metadata", {})["algorithm_version"] = version
    cfg["metadata"]["fixed_planner_interface_evidence_budget"] = 16
    cfg["metadata"]["baseline_evidence_budget"] = 16
    cfg["metadata"]["capacity_ceiling_retained_interface_budget"] = 24
    cfg["metadata"]["fixed_proposal_top_m"] = 24
    cfg["metadata"]["fbic_full_bank_capacity_probe"] = True
    cfg.setdefault("provenance", {})["algorithm_version"] = version
    cfg["provenance"]["screening_only"] = True
    exp = cfg.setdefault("experiment", {})
    exp["name"] = f"v64_3_30_fbic_{tag}"
    exp["algorithm"] = (
        "V64.3.30 EAF-ICER-FBIC: full already-queried M=24 retained-interface capacity ceiling "
        "+ unchanged V25 aggregate DRC recipe"
    )
    exp["evaluation_role"] = "causal_capacity_ceiling_main" if cert == "downside_rms" else "capacity_estimator_control"
    exp["interface_budget_accounting"] = (
        "AOCC baseline selection is still constructed at historical B=16. FBIC then exposes every already-queried "
        "decision atom in the frozen M=24 bank to the downstream interface only in the non-structural domain. "
        "No additional evidence/model query is issued; the retained interface ceiling alone is B<=24."
    )
    exp["mechanism_chain"] = (
        "fixed queried M=24 bank -> historical B16 AOCC construction -> safe-domain FBIC B24 retained-interface ceiling -> "
        "frozen EAF/ICER support+scalar semantics -> unchanged aggregate evidence-local DRC recipe re-estimated on the same "
        "frozen 3000 TRAIN scenes -> incumbent-default extremal replacement -> frozen structural guard"
    )
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit the unchanged V25 DRC recipe on the V64.3.30 FBIC TRAIN representation.")
    ap.add_argument("--train-frontier-edges", required=True)
    ap.add_argument("--base-fbic-v20-config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--output-train-token-file", required=True)
    ap.add_argument("--output-report", required=True)
    a = ap.parse_args()

    edge_path = Path(a.train_frontier_edges)
    if not edge_path.is_file() or edge_path.stat().st_size <= 0:
        raise SystemExit(f"STOP TRAIN DATA: missing frontier provenance {edge_path}")
    out_dir = Path(a.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    by, frontier_row_count = v25._load_minimal_scenes(edge_path)
    if len(by) != v25.EXPECTED_TRAIN_SCENES:
        raise SystemExit(f"STOP TRAIN DATA: expected {v25.EXPECTED_TRAIN_SCENES} frozen TRAIN scenes, got {len(by)}")
    data = v25._build(by)
    crossfit = {
        "aggregate_meanse": v25._crossfit(data, "mean_se"),
        "aggregate_downside": v25._crossfit(data, "downside_rms"),
    }
    main_cf = crossfit["aggregate_downside"]
    gate = bool(
        main_cf["all_folds_path_safe"]
        and main_cf["selected_count"] >= v25.MAIN_MIN_SELECTED
        and main_cf["teacher_improvement_sum"] >= -1.0e-9
    )

    tokens = sorted(by)
    token_path = Path(a.output_train_token_file); token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    y = data["delta"]
    report: dict[str, Any] = {
        "audit": "v64_3_30_eaf_icer_fbic_train_fit",
        "algorithm": "V64.3.30 EAF-ICER-FBIC with unchanged V25 DRC recipe",
        "train_scene_count": int(len(by)),
        "frontier_row_count": int(frontier_row_count),
        "replacement_edges": int(len(y)),
        "replacement_scenes": int(data["replacement_scene_count"]),
        "population_teacher_positive_fraction": float(np.mean(y > 0.0)),
        "population_teacher_improvement_sum": float(y.sum()),
        "population_teacher_improvement_worst": float(y.min()),
        "fold_seed": v25.FOLD_SEED,
        "neighbor_k_values": list(v25.KS),
        "decision_boundary": 0.0,
        "crossfit": crossfit,
        "train_gate_pass": gate,
        "fresh_validation_used": False,
        "causal_contract": {
            "DRC_estimator_family_changed": False,
            "K_changed": False,
            "threshold_changed": False,
            "teacher_feature_added": False,
            "only_selected_evidence_distribution_changed_by_FBIC": True,
        },
        "memories": {},
        "configs": {},
    }
    _write(Path(a.output_report), report)
    if not gate:
        raise SystemExit(
            "STOP TRAIN FBIC-DRC: unchanged aggregate downside recipe is not fold-safe on the capacity-complete TRAIN representation; "
            "do not tune K/threshold/downside multiplier and do not spend new fresh scenes"
        )

    base = yaml.safe_load(Path(a.base_fbic_v20_config).read_text(encoding="utf-8"))
    variants = {"aggregate_meanse": "mean_se", "aggregate_downside": "downside_rms"}
    for tag, cert in variants.items():
        memory_path = out_dir / f"v64_3_30_fbic_{tag}_memory.npz"
        memory = v25._save_memory(memory_path, data, cert)
        cfg_path = out_dir / f"v64_3_30_fbic_{tag}.yaml"
        cfg_path.write_text(yaml.safe_dump(_v30_cfg(base, memory, cert, tag), sort_keys=False, width=120), encoding="utf-8")
        report["memories"][tag] = memory
        report["configs"][tag] = str(cfg_path)
    _write(Path(a.output_report), report)
    print(json.dumps({
        "pass": True,
        "train_gate_pass": gate,
        "main_fold_pass_count": main_cf["fold_pass_count"],
        "main_selected_count": main_cf["selected_count"],
        "main_teacher_improvement_sum": main_cf["teacher_improvement_sum"],
    }, indent=2))


if __name__ == "__main__":
    main()
