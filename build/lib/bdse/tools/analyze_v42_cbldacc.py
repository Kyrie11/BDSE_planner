from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Slice CBL-DACC fixed-budget recovery outcomes")
    parser.add_argument("cbldacc_jsonl", type=Path)
    parser.add_argument("--v41-jsonl", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = _load_jsonl(args.cbldacc_jsonl)
    if not rows:
        raise SystemExit("empty CBL-DACC JSONL")
    n = len(rows)
    preserved = [r for r in rows if float(r.get("selector_deployment_coreset_target_action_preserved", 0.0)) >= 0.5]
    attempted = [r for r in rows if float(r.get("selector_deployment_coreset_budget_layer_attempted", 0.0)) >= 0.5]
    success = [r for r in rows if float(r.get("selector_deployment_coreset_budget_layer_success", 0.0)) >= 0.5]
    failed = [r for r in rows if float(r.get("selector_deployment_coreset_target_action_preserved", 0.0)) < 0.5]

    lines = [
        f"scenarios: {n}",
        f"target_action_preserved: {len(preserved)}/{n} = {len(preserved) / n:.6f}",
        f"budget_layer_attempted: {len(attempted)}",
        f"budget_layer_success: {len(success)}",
        f"remaining_failures: {len(failed)}",
        f"mean_budget_layer_evaluations_all: {sum(float(r.get('selector_deployment_coreset_budget_layer_evaluations', 0.0)) for r in rows) / n:.3f}",
        f"mean_budget_layer_evaluations_attempted: {sum(float(r.get('selector_deployment_coreset_budget_layer_evaluations', 0.0)) for r in attempted) / max(1, len(attempted)):.3f}",
        f"mean_budget_layer_unique_states_attempted: {sum(float(r.get('selector_deployment_coreset_budget_layer_unique_states', 0.0)) for r in attempted) / max(1, len(attempted)):.3f}",
        f"mean_best_target_rank_failed: {sum(float(r.get('selector_deployment_coreset_budget_layer_best_target_rank', 0.0)) for r in failed) / max(1, len(failed)):.3f}",
        f"mean_best_action_deficit_failed: {sum(float(r.get('selector_deployment_coreset_budget_layer_best_action_deficit', 0.0)) for r in failed) / max(1, len(failed)):.6f}",
    ]

    if args.v41_jsonl is not None and args.v41_jsonl.exists():
        old = {str(r.get("scenario_token", "")): r for r in _load_jsonl(args.v41_jsonl)}
        recovered: list[str] = []
        lost: list[str] = []
        action_changed: list[str] = []
        teacher_gain = 0
        full_gain = 0
        for r in rows:
            token = str(r.get("scenario_token", ""))
            if token not in old:
                continue
            old_row = old[token]
            old_ok = float(old_row.get("selector_deployment_coreset_target_action_preserved", 0.0)) >= 0.5
            new_ok = float(r.get("selector_deployment_coreset_target_action_preserved", 0.0)) >= 0.5
            if not old_ok and new_ok:
                recovered.append(token)
            elif old_ok and not new_ok:
                lost.append(token)
            if int(old_row.get("bdse_action", -1)) != int(r.get("bdse_action", -1)):
                action_changed.append(token)
                teacher_gain += int(r.get("bdse_action", -1) == r.get("teacher_action", -2)) - int(
                    old_row.get("bdse_action", -1) == old_row.get("teacher_action", -2)
                )
                full_gain += int(r.get("bdse_action", -1) == r.get("full_action", -2)) - int(
                    old_row.get("bdse_action", -1) == old_row.get("full_action", -2)
                )
        lines.extend([
            f"v41_failures_recovered: {len(recovered)}",
            f"v41_preserved_lost: {len(lost)}",
            f"v41_actions_changed: {len(action_changed)}",
            f"teacher_match_net_on_changed: {teacher_gain}",
            f"full_match_net_on_changed: {full_gain}",
            "recovered_tokens: " + (", ".join(recovered) if recovered else "none"),
            "lost_tokens: " + (", ".join(lost) if lost else "none"),
        ])

    lines.append(
        "remaining_failure_tokens: "
        + (", ".join(str(r.get("scenario_token", "")) for r in failed) if failed else "none")
    )
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
