from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml


def _drop_metadata_sections(cfg: dict[str, Any]) -> dict[str, Any]:
    x = copy.deepcopy(cfg)
    for key in ("experiment", "metadata", "provenance"):
        x.pop(key, None)
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit the isolated V64.3.30 FBIC B16->B24 capacity intervention.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--frozen-v20-config", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(a.frozen_v20_config).read_text(encoding="utf-8"))
    probe = ((cfg.get("selector", {}) or {}).get("full_bank_capacity_probe", {}) or {})

    contract = {
        "upstream_evidence_budget_frozen_16": float((cfg.get("evidence", {}) or {}).get("budget", -1)) == 16.0,
        "top_m_frozen_24": int((cfg.get("selector", {}) or {}).get("proposal_top_m", -1)) == 24,
        "historical_aocc_baseline_budget_16": float(probe.get("baseline_selector_budget", -1)) == 16.0,
        "probe_enabled": bool(probe.get("enabled", False)),
        "probe_interface_budget_24": float(probe.get("interface_budget", -1)) == 24.0,
        "zero_additional_queries": int(probe.get("additional_evidence_queries", -1)) == 0,
        "teacher_free": probe.get("teacher_labels", None) is False,
        "fcr_disabled": not bool((((cfg.get("selector", {}) or {}).get("frontier_contrast_rebinding", {}) or {}).get("enabled", False))),
        "fallback_budget_stage_frozen_16": list((cfg.get("fallback", {}) or {}).get("budget_stages", [])) == [16],
    }

    # Strong causal-isolation audit: after removing paper/provenance text and the
    # explicitly allowed capacity fields, V30 must be byte-semantically identical
    # to the frozen V20 configuration.
    c = _drop_metadata_sections(cfg)
    b = _drop_metadata_sections(base)
    c.setdefault("selector", {}).pop("full_bank_capacity_probe", None)
    contract["all_noncapacity_algorithm_fields_frozen"] = c == b
    contract["pass"] = bool(all(contract.values()))

    out = {
        "algorithm_version": "V64.3.30-EAF-ICER-FBIC",
        "contract": contract,
        "interpretation": "one-point retained-interface capacity ceiling over the already queried M=24 bank; not a new acquisition objective and not a broad budget sweep",
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    if not contract["pass"]:
        raise SystemExit("STOP CONTRACT: V64.3.30 changed fields outside the pre-registered capacity intervention")


if __name__ == "__main__":
    main()
