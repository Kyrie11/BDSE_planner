from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Slice PR-DACC preservation and beam outcomes")
    parser.add_argument("prdacc_jsonl", type=Path)
    parser.add_argument("--v40-jsonl", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = _load_jsonl(args.prdacc_jsonl)
    if not rows:
        raise SystemExit("empty PR-DACC JSONL")
    n = len(rows)
    preserved = [r for r in rows if float(r.get("selector_deployment_coreset_target_action_preserved", 0.0)) >= 0.5]
    attempted = [r for r in rows if float(r.get("selector_deployment_coreset_beam_attempted", 0.0)) >= 0.5]
    success = [r for r in rows if float(r.get("selector_deployment_coreset_beam_success", 0.0)) >= 0.5]
    failed = [r for r in rows if float(r.get("selector_deployment_coreset_target_action_preserved", 0.0)) < 0.5]

    lines = [
        f"scenarios: {n}",
        f"target_action_preserved: {len(preserved)}/{n} = {len(preserved)/n:.6f}",
        f"beam_attempted: {len(attempted)}",
        f"beam_success: {len(success)}",
        f"remaining_failures: {len(failed)}",
        f"mean_beam_evaluations_all: {sum(float(r.get('selector_deployment_coreset_beam_evaluations', 0.0)) for r in rows)/n:.3f}",
        f"mean_beam_evaluations_attempted: {sum(float(r.get('selector_deployment_coreset_beam_evaluations', 0.0)) for r in attempted)/max(1,len(attempted)):.3f}",
    ]

    if args.v40_jsonl is not None and args.v40_jsonl.exists():
        old = {str(r.get("scenario_token", "")): r for r in _load_jsonl(args.v40_jsonl)}
        recovered = []
        lost = []
        for r in rows:
            token = str(r.get("scenario_token", ""))
            if token not in old:
                continue
            old_ok = float(old[token].get("selector_deployment_coreset_target_action_preserved", 0.0)) >= 0.5
            new_ok = float(r.get("selector_deployment_coreset_target_action_preserved", 0.0)) >= 0.5
            if not old_ok and new_ok:
                recovered.append(token)
            elif old_ok and not new_ok:
                lost.append(token)
        lines.extend([
            f"v40_failures_recovered: {len(recovered)}",
            f"v40_preserved_lost: {len(lost)}",
            "recovered_tokens: " + (", ".join(recovered) if recovered else "none"),
            "lost_tokens: " + (", ".join(lost) if lost else "none"),
        ])

    lines.append("remaining_failure_tokens: " + (", ".join(str(r.get("scenario_token", "")) for r in failed) if failed else "none"))
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
