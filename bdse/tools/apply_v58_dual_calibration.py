from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import yaml


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Apply V58's shared evidence certificate calibration and the separate "
            "proposal-conditional residual-flip calibration."
        )
    )
    p.add_argument("--config", required=True)
    p.add_argument("--calibration-json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--control",
        action="store_true",
        help="Apply the shared evidence certificate but disable the learned residual path.",
    )
    args = p.parse_args()

    config_path = Path(args.config)
    calibration_path = Path(args.calibration_json)
    output_path = Path(args.output)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    cal = json.loads(calibration_path.read_text(encoding="utf-8"))
    evidence_eps = float(cal["recommended_adverse_certificate_epsilon"])
    residual_eps = float(cal.get("recommended_residual_flip_epsilon", 0.0))
    independent = bool(cal.get("independent_calibration", False))
    if evidence_eps < 0.0 or residual_eps < 0.0:
        raise ValueError("Calibrated epsilons must be non-negative")
    if not independent:
        raise ValueError(
            "V58 paper protocol requires a group-disjoint calibration-only split; "
            "the supplied calibration JSON is not independently calibrated"
        )

    out = deepcopy(cfg)
    # Evidence calibration belongs to the AOCC/adverse certificate.  It is not a
    # tournament score offset.  V57's old calibration helper correctly kept these
    # quantities separate; assigning the evidence epsilon to tournament.epsilon_cal
    # would silently change the action rule rather than only calibrating evidence.
    selector = out.setdefault("selector", {})
    selector["adverse_certificate_epsilon"] = evidence_eps
    selector["adverse_certificate_calibrated"] = True
    calibration_cfg = out.setdefault("calibration", {})
    calibration_cfg["independent"] = True
    calibration_cfg["provenance_json"] = str(calibration_path.resolve())

    runtime = out.setdefault("runtime", {})
    dual = runtime.setdefault("dual_certificate", {})
    dual["residual_epsilon_cal"] = 0.0 if args.control else residual_eps
    if args.control:
        runtime["disable_pair_residual_intervention"] = True
    else:
        runtime["disable_pair_residual_intervention"] = False

    provenance = out.setdefault("provenance", {})
    provenance["v58_dual_calibration_json"] = str(calibration_path.resolve())
    provenance["v58_shared_evidence_epsilon"] = evidence_eps
    provenance["v58_residual_flip_epsilon"] = 0.0 if args.control else residual_eps
    provenance["v58_residual_control"] = bool(args.control)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(out, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "config_in": str(config_path),
                "calibration_json": str(calibration_path),
                "config_out": str(output_path),
                "evidence_epsilon": evidence_eps,
                "residual_epsilon": 0.0 if args.control else residual_eps,
                "independent_calibration": independent,
                "control": bool(args.control),
                "tournament_epsilon_unchanged": float(
                    (out.get("tournament", {}) or {}).get("epsilon_cal", 0.0)
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
