from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(report: dict[str, Any], section: str, key: str) -> float:
    try:
        value = float(report[section][key])
    except Exception:
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _check_ratio(
    failures: list[str], label: str, candidate: float, reference: float, minimum: float
) -> None:
    ratio = candidate / max(abs(reference), 1e-9)
    if not math.isfinite(candidate) or not math.isfinite(reference) or ratio < minimum:
        failures.append(
            f"{label}: test/reference ratio={ratio:.4f} < {minimum:.4f} "
            f"(test={candidate}, reference={reference})"
        )


def _check_drop(
    failures: list[str], label: str, candidate: float, reference: float, maximum_drop: float
) -> None:
    drop = reference - candidate
    if not math.isfinite(candidate) or not math.isfinite(reference) or drop > maximum_drop:
        failures.append(
            f"{label}: drop={drop:.4f} > {maximum_drop:.4f} "
            f"(test={candidate}, reference={reference})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check that a test-cache diagnostic report is distributionally and "
            "preprocessing-wise compatible with the validation reference."
        )
    )
    parser.add_argument("reference", type=Path, help="Complete validation diagnostics JSON")
    parser.add_argument("candidate", type=Path, help="Test or other target diagnostics JSON")
    parser.add_argument("--min-map-polygon-ratio", type=float, default=0.80)
    parser.add_argument("--max-safe-candidate-drop", type=float, default=0.10)
    parser.add_argument("--max-b16-oracle-drop", type=float, default=0.10)
    parser.add_argument("--max-full-interface-drop", type=float, default=0.06)
    parser.add_argument("--max-route-tail-m", type=float, default=8.0)
    parser.add_argument("--min-scenarios", type=int, default=0)
    args = parser.parse_args()

    ref = _load(args.reference)
    cand = _load(args.candidate)
    failures: list[str] = []

    n = int(cand.get("num_samples", cand.get("num_loaded", 0)) or 0)
    if n < int(args.min_scenarios):
        failures.append(f"candidate scenario count={n} < {args.min_scenarios}")

    ref_e1 = "E1_teacher_sanity_and_candidate_coverage"
    ref_e4 = "E4_budget_sweep_oracle_teacher_interface"
    ref_e0 = "E0_paper_scale_gate"

    ref_polys = _metric(ref, ref_e1, "drivable_polygon_count")
    cand_polys = _metric(cand, ref_e1, "drivable_polygon_count")
    _check_ratio(failures, "drivable_polygon_count", cand_polys, ref_polys, args.min_map_polygon_ratio)

    ref_safe = float(ref[ref_e0]["checks"]["safe_candidate_exists"]["value"])
    cand_safe = float(cand[ref_e0]["checks"]["safe_candidate_exists"]["value"])
    _check_drop(failures, "safe_candidate_exists", cand_safe, ref_safe, args.max_safe_candidate_drop)

    ref_b16 = float(ref[ref_e0]["checks"]["B16_oracle_decision_sufficiency"]["value"])
    cand_b16 = float(cand[ref_e0]["checks"]["B16_oracle_decision_sufficiency"]["value"])
    _check_drop(failures, "B16_oracle_decision_sufficiency", cand_b16, ref_b16, args.max_b16_oracle_drop)

    ref_full = float(ref[ref_e0]["checks"]["full_interface_action_match"]["value"])
    cand_full = float(cand[ref_e0]["checks"]["full_interface_action_match"]["value"])
    _check_drop(failures, "full_interface_action_match", cand_full, ref_full, args.max_full_interface_drop)

    route_tail = float(cand[ref_e0]["checks"]["logged_ego_route_dist_p95_p90"]["value"])
    if not math.isfinite(route_tail) or route_tail > args.max_route_tail_m:
        failures.append(
            f"logged_ego_route_dist_p95_p90={route_tail} > {args.max_route_tail_m:.3f} m"
        )

    status = "PASS" if not failures else "FAIL"
    print(f"\nDataset diagnostics parity gate [{status}]")
    print(f"  reference={args.reference}")
    print(f"  candidate={args.candidate}; scenarios={n}")
    print(f"  drivable polygons: {cand_polys:.4f} vs {ref_polys:.4f}")
    print(f"  safe candidate: {cand_safe:.4f} vs {ref_safe:.4f}")
    print(f"  B16 oracle: {cand_b16:.4f} vs {ref_b16:.4f}")
    print(f"  full interface: {cand_full:.4f} vs {ref_full:.4f}")
    print(f"  route tail p95-p90: {route_tail:.4f} m")
    for failure in failures:
        print(f"  - {failure}")
    return 0 if not failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
