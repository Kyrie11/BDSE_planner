from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


CACHE_SOURCES = {
    "cache",
    "cached",
    "cache_only",
    "cache_prefix_runtime_extension",
    "verified_prefix_runtime_extension",
    "cache_supported_prefix",
}
RUNTIME_SOURCES = {
    "runtime",
    "runtime_recompute",
    "recompute",
    "canonical_runtime",
    "cache_verified",
    "verified_cache",
}


def _load(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return obj


def _check(path: Path, role: str) -> dict[str, Any]:
    cfg = _load(path)
    exp = cfg.get("experiment", {}) or {}
    model = cfg.get("model", {}) or {}
    evidence = cfg.get("evidence", {}) or {}
    runtime = cfg.get("runtime", {}) or {}
    adapter = model.get("query_extension_adapter", {}) or {}
    critical_adapter = model.get("critical_proposal_adapter", {}) or {}
    training_cfg = cfg.get("training", {}) or {}
    crit = training_cfg.get("exact_winner_flip_criticality", {}) or {}
    source = str(runtime.get("dense_query_feature_source", "cache_or_recompute")).strip().lower()

    checks: dict[str, bool] = {
        "experiment_is_v64_family": str(exp.get("name", "")).startswith(("v64_", "v64.")),
        "fixed_evidence_budget_16": int(evidence.get("budget", -1)) == 16,
        "query_feature_dim_18": int(model.get("query_feature_dim", -1)) == 18,
        "legacy_support_dim_12": int(model.get("query_legacy_support_dim", -1)) == 12,
        "query_extension_adapter_enabled": bool(adapter.get("enabled", False)),
        "query_extension_adapter_zero_init": bool(adapter.get("zero_init", False)),
        "query_source_supported": source in CACHE_SOURCES | RUNTIME_SOURCES,
    }
    if role == "train":
        checks.update(
            {
                "literal_winner_flip_criticality_enabled": bool(crit.get("enabled", False)),
                "teacher_interface_target": str(crit.get("target_source", "")).lower()
                in {"teacher", "teacher_interface", "teacher_exact"},
            }
        )
    if str(exp.get("name", "")).startswith("v64_3_"):
        trainable = [str(x) for x in training_cfg.get("trainable_modules", [])]
        checks.update(
            {
                "v64_3_query_extension_nominally_noop": abs(float(adapter.get("scale", 1.0))) <= 1.0e-12,
                "v64_3_critical_proposal_adapter_enabled": bool(critical_adapter.get("enabled", False)),
                "v64_3_critical_proposal_adapter_zero_init": bool(critical_adapter.get("zero_init", False)),
            }
        )
        if role == "train":
            checks.update(
                {
                    "v64_3_critical_adapter_trainable": "critical_proposal_adapter" in trainable,
                    "v64_3_legacy_proposal_frozen": not any(
                        x in trainable
                        for x in ("proposal_head", "family_head", "family_embed", "family_activity_proj", "proposal_feature_proj")
                    ),
                    "v64_3_query_extension_frozen": "query_extension_proj" not in trainable,
                }
            )
    failures = [name for name, ok in checks.items() if not ok]
    return {
        "path": str(path),
        "role": role,
        "experiment_name": exp.get("name"),
        "dense_query_feature_source": source,
        "checks": checks,
        "pass": not failures,
        "failures": failures,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Fail fast on stale/non-V64 pipeline configs.")
    ap.add_argument("--train-config", type=Path, required=True)
    ap.add_argument("--eval-config", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    report = {
        "audit": "v64_pipeline_config_contract",
        "train": _check(args.train_config, "train"),
        "eval": _check(args.eval_config, "eval"),
    }
    report["pass"] = bool(report["train"]["pass"] and report["eval"]["pass"])
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
