from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({k for r in rows for k in r})
    preferred = [x for x in ("budget", "system", "implementation_label", "scenario_count", "successful", "failed", "metric_engine_serialized", "budget_semantics_warning") if x in fields]
    fields = preferred + [x for x in fields if x not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def _score_columns(rows: list[dict[str, Any]]) -> list[str]:
    cols = sorted({k for r in rows for k,v in r.items() if isinstance(v,(int,float)) and ("score" in k.lower() or "collision" in k.lower() or "ttc" in k.lower() or "drivable" in k.lower())})
    return cols[:10]


def _md(path: Path, title: str, rows: list[dict[str, Any]], note: str) -> None:
    scores = _score_columns(rows)
    cols = ["budget", "system", "scenario_count"] + scores
    lines = [f"# {title}", "", note, "", "| " + " | ".join(cols) + " |", "| " + " | ".join(["---"]*len(cols)) + " |"]
    for r in rows:
        vals=[]
        for c in cols:
            v=r.get(c,"")
            if isinstance(v,float): vals.append(f"{v:.6g}")
            else: vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    lines += ["", "All displayed rows were required to use the metric-safe nuPlan wrapper and the same ordered scenario-token manifest."]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-json", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    a=ap.parse_args()
    rows=json.loads(a.input_json.read_text(encoding="utf-8"))
    if not isinstance(rows,list) or not rows: raise SystemExit("empty benchmark rows")
    token_shas={str(r.get("scenario_token_sha256","")) for r in rows}
    if len(token_shas)!=1 or "" in token_shas: raise SystemExit("scenario-token pairing invariant failed")
    unsafe=[r for r in rows if r.get("metric_engine_serialized") is not True]
    if unsafe: raise SystemExit(f"metric-safety provenance missing for {len(unsafe)} rows")
    a.output_root.mkdir(parents=True, exist_ok=True)
    rows=sorted(rows,key=lambda r:(int(r["budget"]),str(r["system"])))
    b16=[r for r in rows if int(r["budget"])==16]
    external=[r for r in rows if str(r["system"])!="bdse"]
    own=[r for r in rows if str(r["system"])=="bdse"]
    if not b16: raise SystemExit("missing B16 primary rows")
    _write_csv(a.output_root/"PRIMARY_B16_MATCHED_INTERFACE.csv",b16)
    _write_csv(a.output_root/"ALL_BUDGETS_B8_B16_B24.csv",rows)
    _write_csv(a.output_root/"EXTERNAL_BUDGET_SPECIFIC_SWEEP.csv",external)
    _write_csv(a.output_root/"BDSE_FROZEN_CROSS_BUDGET_ABLATION.csv",own)
    _md(a.output_root/"PRIMARY_B16_MATCHED_INTERFACE.md","Primary matched-interface comparison (B=16)",b16,"Use this table for the primary fixed-interface comparison. The frozen BDSE method was developed/fitted at B=16; external trainable adapters use their B16-specific checkpoints.")
    _md(a.output_root/"BUDGET_SWEEP_B8_B16_B24.md","Budget sweep (B=8/16/24)",rows,"Interpret BDSE B=8/B=24 as frozen-policy cross-budget robustness ablations, not budget-specific retraining. External trainable adapters use a distinct checkpoint for each budget.")
    audit={
        "scenario_token_sha256": next(iter(token_shas)),
        "row_count": len(rows),
        "budgets": sorted({int(r["budget"]) for r in rows}),
        "systems": sorted({str(r["system"]) for r in rows}),
        "metric_engine_serialized_all": True,
        "primary_budget": 16,
        "bdse_b8_b24_role": "frozen-policy cross-budget robustness ablation",
        "external_trainable_budget_specific": True,
        "pdm_closed_style_official": False,
    }
    (a.output_root/"BENCHMARK_FAIRNESS_AUDIT.json").write_text(json.dumps(audit,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(audit,sort_keys=True))

if __name__=="__main__": main()
