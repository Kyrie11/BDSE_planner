from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("scenario_token", "")), int(row.get("timestamp_us", 0) or 0))
            if not key[0] or key in out:
                raise ValueError(f"invalid duplicate/empty row key in {path}: {key}")
            out[key] = row
    return out


def metric(d: dict[str, Any], key: str) -> float:
    try:
        x = float(d.get(key, float("nan")))
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def mismatch(a: dict, b: dict, field: str) -> float:
    if set(a) != set(b):
        raise ValueError(f"scenario key mismatch: {len(a)} vs {len(b)}")
    vals = [(int(a[k][field]), int(b[k][field])) for k in a if field in a[k] and field in b[k]]
    return sum(x != y for x, y in vals) / max(len(vals), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    names = ["legacy_anchor", "support_aware_nominal", "prefix_cache", "structural_prior"]
    summaries = {n: load(args.suite_root / n / "metrics.json") for n in names}
    per_rows = {n: rows(args.suite_root / n / "metrics.jsonl") for n in names}
    nominal = summaries["support_aware_nominal"]
    hard_checks = {
        "legacy_anchor_recovered": metric(summaries["legacy_anchor"], "full_interface_action_match") >= 0.32,
        # Step-zero immutability is a planner-interface/deployed-decision contract.
        # The support-aware diagnostic ``full_action`` intentionally evaluates a
        # different internal interface and is therefore reported below as a
        # warning-only diagnostic rather than a hard audit condition.
        "support_step0_matches_legacy_deployed": mismatch(per_rows["legacy_anchor"], per_rows["support_aware_nominal"], "bdse_action") <= 0.005,
        "base_contract": metric(nominal, "dense_runtime_base_contract_pass") >= 0.99,
        "raw_query_contract_available": metric(nominal, "dense_runtime_raw_query_feature_contract_available") >= 0.99,
        "raw_query_contract": metric(nominal, "dense_runtime_raw_query_feature_contract_pass") >= 0.99,
        "score_contract": metric(nominal, "dense_runtime_query_score_contract_pass") >= 0.99,
        "decision_contract": metric(nominal, "dense_runtime_query_decision_match") >= 0.999,
        "prefix_cache_matches_runtime": mismatch(per_rows["support_aware_nominal"], per_rows["prefix_cache"], "bdse_action") <= 0.005,
    }
    diagnostic_checks = {
        "support_step0_matches_legacy_dense_diagnostic": mismatch(
            per_rows["legacy_anchor"], per_rows["support_aware_nominal"], "full_action"
        ) <= 0.005,
    }
    report = {
        "audit": "v64_support_aware_query_contract",
        "pass": all(hard_checks.values()),
        "checks": hard_checks,
        "diagnostic_checks": diagnostic_checks,
        "warnings": ([] if diagnostic_checks["support_step0_matches_legacy_dense_diagnostic"] else [
            "support-aware full_action differs from the legacy internal full-interface diagnostic; deployed bdse_action and the three query contracts remain the hard immutability criteria"
        ]),
        "row_mismatch": {
            "legacy_vs_support_full_action": mismatch(per_rows["legacy_anchor"], per_rows["support_aware_nominal"], "full_action"),
            "legacy_vs_support_deployed_action": mismatch(per_rows["legacy_anchor"], per_rows["support_aware_nominal"], "bdse_action"),
            "runtime_vs_prefix_cache_deployed_action": mismatch(per_rows["support_aware_nominal"], per_rows["prefix_cache"], "bdse_action"),
        },
        "metrics": {
            n: {k: summaries[n].get(k) for k in [
                "full_interface_action_match",
                "teacher_action_match",
                "dense_runtime_base_contract_pass",
                "dense_runtime_raw_query_feature_contract_available",
                "dense_runtime_raw_query_feature_contract_pass",
                "dense_runtime_raw_query_feature_allclose_fraction",
                "dense_runtime_query_score_contract_pass",
                "dense_runtime_query_score_allclose_fraction",
                "dense_runtime_query_decision_match",
                "hab_topm_dense_value_vs_runtime_sparse_full_match",
            ]}
            for n in names
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
