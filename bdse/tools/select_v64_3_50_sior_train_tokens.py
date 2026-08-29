from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze V50 paired-intervention TRAIN tokens from exact V49 full-set RSMR winners")
    ap.add_argument("--v49-scene-audit", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    rows = list(csv.DictReader(a.v49_scene_audit.open(newline="", encoding="utf-8")))
    if len(rows) != 782:
        raise SystemExit(f"V50 ENGINEERING STOP: expected 782 V49 direct scenes, got {len(rows)}")
    toks = []
    for r in rows:
        try:
            action = int(float(r["rsm_selected_action"]))
        except Exception as exc:
            raise SystemExit(f"V50 ENGINEERING STOP: invalid rsm_selected_action: {r}") from exc
        if action >= 0:
            toks.append(str(r["scenario_token"]))
    if len(toks) != 502 or len(set(toks)) != 502:
        raise SystemExit(f"V50 ENGINEERING STOP: exact frozen RSMR selected population must be 502 unique scenes, got {len(toks)}/{len(set(toks))}")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("\n".join(toks) + "\n", encoding="utf-8")
    print(f"PASS V50 frozen paired-intervention TRAIN population: {len(toks)} unique full-set RSMR winners")


if __name__ == "__main__":
    main()
