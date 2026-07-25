from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the calibrated one-sided adverse epsilon to a BDSE v46 YAML config"
    )
    parser.add_argument("--config-in", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--config-out", required=True)
    parser.add_argument(
        "--do-not-update-tournament",
        action="store_true",
        help="Only update selector.adverse_certificate_epsilon; leave tournament/calibration epsilon unchanged.",
    )
    args = parser.parse_args()

    config_path = Path(args.config_in)
    calibration_path = Path(args.calibration_json)
    output_path = Path(args.config_out)

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    epsilon = float(calibration["recommended_adverse_certificate_epsilon"])
    if epsilon < 0.0:
        raise ValueError("Calibrated epsilon must be non-negative")

    out = deepcopy(cfg)
    out.setdefault("selector", {})["adverse_certificate_epsilon"] = epsilon
    if not args.do_not_update_tournament:
        out.setdefault("tournament", {})["epsilon_cal"] = epsilon
        out.setdefault("calibration", {})["epsilon_cal"] = epsilon

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(out, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "config_in": str(config_path),
                "calibration_json": str(calibration_path),
                "config_out": str(output_path),
                "applied_epsilon": epsilon,
                "updated_tournament": not args.do_not_update_tournament,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
