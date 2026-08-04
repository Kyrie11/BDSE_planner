from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _finite(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = float(row.get(key, default))
    except Exception:
        return default
    return value if math.isfinite(value) else default



def _json_number(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def main() -> int:
    p = argparse.ArgumentParser(description="Fail-fast V61 proposal/training health audit")
    p.add_argument("train_log", type=Path)
    p.add_argument("--report-json", type=Path)
    p.add_argument("--max-proposal-loss", type=float, default=100.0)
    p.add_argument("--max-proposal-growth", type=float, default=8.0)
    p.add_argument("--max-logit-rms", type=float, default=20.0)
    p.add_argument("--min-fast-exact-jaccard", type=float, default=0.70)
    p.add_argument("--min-proposal-recall", type=float, default=0.72)
    p.add_argument("--baseline-sparse-full", type=float, default=0.141)
    p.add_argument("--baseline-budget-vs-full", type=float, default=0.172)
    args = p.parse_args()

    rows = _rows(args.train_log)
    if not rows:
        raise SystemExit("empty training log")
    first, last = rows[0], rows[-1]
    prop_first = _finite(first, "L_prop")
    prop_last = _finite(last, "L_prop")
    prop_growth = prop_last / max(prop_first, 1.0e-6) if math.isfinite(prop_first) and math.isfinite(prop_last) else float("nan")
    rms = max((_finite(r, "proposal_logit_rms_mean") for r in rows), default=float("nan"))
    exact = max((_finite(r, "proposal_exact_hab_fraction") for r in rows), default=float("nan"))
    jaccard_values = []
    for row in rows:
        value = _finite(row, "proposal_fast_exact_mask_jaccard")
        frac = _finite(row, "proposal_exact_hab_fraction", 0.0)
        if math.isfinite(value) and frac > 0.0:
            jaccard_values.append(value)
    min_jaccard = min(jaccard_values) if jaccard_values else float("nan")

    validation_rows = [r for r in rows if math.isfinite(_finite(r, "val_proposal_decisive_atom_recall"))]
    latest_val = validation_rows[-1] if validation_rows else None
    proposal_recall = _finite(latest_val or {}, "val_proposal_decisive_atom_recall")
    sparse_full = _finite(latest_val or {}, "val_sparse_full_interface_action_match")
    budget_vs_full = _finite(latest_val or {}, "val_budget_vs_full_match")
    min_feasible = _finite(latest_val or {}, "val_minimum_gate_feasible")

    failures: list[str] = []
    warnings: list[str] = []
    if not math.isfinite(exact) or exact <= 0.0:
        failures.append("exact runtime HAB supervision was not observed")
    if math.isfinite(rms) and rms > args.max_logit_rms:
        failures.append(f"proposal logit RMS={rms:.3f} > {args.max_logit_rms}")
    if math.isfinite(prop_last) and math.isfinite(prop_growth):
        if prop_last > args.max_proposal_loss and prop_growth > args.max_proposal_growth:
            failures.append(
                f"proposal loss runaway: first={prop_first:.3f}, last={prop_last:.3f}, growth={prop_growth:.2f}x"
            )
    if jaccard_values and min_jaccard < args.min_fast_exact_jaccard:
        failures.append(f"fast/exact HAB mask Jaccard={min_jaccard:.3f} < {args.min_fast_exact_jaccard}")
    if latest_val is not None:
        if proposal_recall < args.min_proposal_recall:
            failures.append(f"latest proposal recall={proposal_recall:.6f} < {args.min_proposal_recall}")
        if sparse_full <= args.baseline_sparse_full and budget_vs_full <= args.baseline_budget_vs_full:
            warnings.append(
                "winner bridge has not improved over V60 yet: "
                f"sparse_full={sparse_full:.6f}, budget_vs_full={budget_vs_full:.6f}"
            )
        if min_feasible < 0.5:
            failures.append("latest validation checkpoint is outside the formal Minimum gate")
    else:
        warnings.append("no validation row yet; rerun after epoch 1/3")

    report = {
        "status": "FAIL" if failures else "PASS_WITH_WARNINGS" if warnings else "PASS",
        "rows": len(rows),
        "latest_epoch": int(last.get("epoch", -1)),
        "proposal_loss_first": _json_number(prop_first),
        "proposal_loss_last": _json_number(prop_last),
        "proposal_loss_growth": _json_number(prop_growth),
        "max_proposal_logit_rms": _json_number(rms),
        "max_exact_hab_fraction": _json_number(exact),
        "min_fast_exact_mask_jaccard": _json_number(min_jaccard),
        "latest_validation": {
            "proposal_decisive_recall": _json_number(proposal_recall),
            "sparse_full_interface_action_match": _json_number(sparse_full),
            "budget_vs_full_match": _json_number(budget_vs_full),
            "minimum_gate_feasible": _json_number(min_feasible),
        },
        "failures": failures,
        "warnings": warnings,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(text + "\n", encoding="utf-8")
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
