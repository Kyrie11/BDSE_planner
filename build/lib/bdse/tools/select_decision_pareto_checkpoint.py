from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _finite(row: dict[str, Any], key: str) -> float:
    try:
        value = float(row.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Select an epoch checkpoint on validation only: maximize full-teacher "
            "action preservation subject to a teacher-regret no-harm constraint."
        )
    )
    ap.add_argument("--train-log", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--stem", default="bdse_v64_saqa_bcc")
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-epoch", type=int, default=0)
    ap.add_argument("--teacher-nonworse-tol", type=float, default=0.004)
    ap.add_argument("--regret-nonworse-relative-tol", type=float, default=0.02)
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.train_log).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    anchors = [row for row in rows if int(row.get("epoch", 10**9)) < 0]
    trained = [row for row in rows if int(row.get("epoch", -10**9)) >= int(args.min_epoch)]
    if not anchors or not trained:
        raise SystemExit("requires a val-before-training anchor and trained epoch rows")
    anchor = anchors[-1]
    a_teacher = _finite(anchor, "val_teacher_action_match")
    a_regret = _finite(anchor, "val_teacher_regret")
    if not math.isfinite(a_teacher) or not math.isfinite(a_regret):
        raise SystemExit("anchor lacks finite val_teacher_action_match/val_teacher_regret")

    audited: list[dict[str, Any]] = []
    ckpt_root = Path(args.checkpoint_dir)
    for row in trained:
        epoch = int(row["epoch"])
        teacher = _finite(row, "val_teacher_action_match")
        regret = _finite(row, "val_teacher_regret")
        pairfull = _finite(row, "val_pair_full_interface_action_match")
        bdmu_capture = _finite(row, "val_teacher_bdmu_topm_utility_capture")
        teacher_ok = math.isfinite(teacher) and teacher >= a_teacher - float(args.teacher_nonworse_tol)
        regret_limit = a_regret * (1.0 + float(args.regret_nonworse_relative_tol))
        regret_ok = math.isfinite(regret) and regret <= regret_limit
        ckpt = ckpt_root / f"{args.stem}.epoch_{epoch + 1:04d}.pt"
        audited.append(
            {
                "epoch": epoch,
                "teacher": teacher,
                "teacher_delta": teacher - a_teacher if math.isfinite(teacher) else float("nan"),
                "teacher_regret": regret,
                "teacher_regret_delta": regret - a_regret if math.isfinite(regret) else float("nan"),
                "pairfull": pairfull,
                "bdmu_topm_capture": bdmu_capture,
                "teacher_nonworse": teacher_ok,
                "teacher_regret_nonworse": regret_ok,
                "checkpoint": str(ckpt),
                "checkpoint_exists": ckpt.is_file(),
            }
        )

    feasible = [r for r in audited if r["teacher_nonworse"] and r["teacher_regret_nonworse"] and r["checkpoint_exists"]]
    if not feasible:
        report = {
            "selection": "decision_pareto_validation",
            "selected_checkpoint": None,
            "anchor": {"teacher": a_teacher, "teacher_regret": a_regret},
            "epochs": audited,
            "reason": "no saved epoch satisfies teacher-action and regret no-harm constraints",
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
        raise SystemExit("no Pareto-feasible saved checkpoint")

    def rank(r: dict[str, Any]) -> tuple[float, float, float, float, int]:
        teacher = r["teacher"] if math.isfinite(r["teacher"]) else -1e9
        regret = r["teacher_regret"] if math.isfinite(r["teacher_regret"]) else 1e18
        pairfull = r["pairfull"] if math.isfinite(r["pairfull"]) else -1e9
        capture = r["bdmu_topm_capture"] if math.isfinite(r["bdmu_topm_capture"]) else -1e9
        return (teacher, -regret, pairfull, capture, -int(r["epoch"]))

    selected = max(feasible, key=rank)
    report = {
        "selection": "decision_pareto_validation",
        "policy": "maximize teacher action match; tie-break by lower teacher regret, pair-full match, then BDMU capture",
        "anchor": {"teacher": a_teacher, "teacher_regret": a_regret},
        "constraints": {
            "teacher_nonworse_tolerance": float(args.teacher_nonworse_tol),
            "teacher_regret_nonworse_relative_tolerance": float(args.regret_nonworse_relative_tol),
            "min_epoch": int(args.min_epoch),
        },
        "selected_epoch": selected["epoch"],
        "selected_checkpoint": selected["checkpoint"],
        "selected": selected,
        "epochs": audited,
        "test_split_used_for_selection": False,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    print(selected["checkpoint"])


if __name__ == "__main__":
    main()
