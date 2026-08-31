from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CITY_TO_RAW = {
    "train_boston": "train_boston",
    "train_pittsburgh": "train_pittsburgh",
    "train_singapore": "train_singapore",
    "train_vegas_2": "train_vegas",
}


def _scalar(z: Any, key: str) -> str:
    if key not in z:
        return ""
    try:
        a = np.asarray(z[key])
        v = a.item() if a.shape == () else a.reshape(-1)[0]
        return str(v)
    except Exception:
        return ""


def _eligible_tokens(candidate_audit: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    out: list[str] = []
    meta: dict[str, dict[str, Any]] = {}
    with candidate_audit.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            tok = str(r["scenario_token"])
            if int(r.get("full_selected_action", -1)) < 0:
                continue
            if tok in meta:
                raise RuntimeError(f"duplicate V49 candidate-audit token: {tok}")
            out.append(tok)
            meta[tok] = {
                "outer_test_fold": int(r.get("outer_test_fold", -1)),
                "candidate_count": int(r.get("candidate_count", 0)),
                "full_selected_action": int(r.get("full_selected_action", -1)),
            }
    if len(out) != 502 or len(set(out)) != 502:
        raise RuntimeError(f"V64.3.50 PIOR requires exact frozen V49 RSMR proposal population 502/502, got {len(out)}/{len(set(out))}")
    return sorted(out), meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Build exact V49 full-set RSMR TRAIN proposal manifest for V64.3.50 paired closed-loop outcome collection.")
    ap.add_argument("--v49-candidate-audit", type=Path, required=True)
    ap.add_argument("--train-cache", type=Path, required=True)
    ap.add_argument("--raw-split-root", type=Path, required=True)
    ap.add_argument("--output-manifest", type=Path, required=True)
    ap.add_argument("--output-token-file", type=Path, required=True)
    args = ap.parse_args()

    tokens, meta = _eligible_tokens(args.v49_candidate_audit)
    wanted = set(tokens)
    found: dict[str, dict[str, Any]] = {}
    if not args.train_cache.is_dir():
        raise FileNotFoundError(args.train_cache)
    for city in CITY_TO_RAW:
        root = args.train_cache / city
        if not root.is_dir():
            raise FileNotFoundError(f"missing TRAIN city cache: {root}")
        for p in sorted(root.rglob("*.npz")):
            try:
                with np.load(p, allow_pickle=False) as z:
                    tok = _scalar(z, "scenario_token")
                    if tok not in wanted or tok in found:
                        continue
                    found[tok] = {
                        **meta[tok],
                        "scenario_token": tok,
                        "cache_city": city,
                        "raw_db_split": CITY_TO_RAW[city],
                        "npz_path": str(p.resolve()),
                        "log_name": _scalar(z, "log_name") or p.parent.name,
                        "scenario_name": _scalar(z, "scenario_name"),
                        "scenario_type": _scalar(z, "scenario_type"),
                    }
            except Exception as exc:
                raise RuntimeError(f"failed to read TRAIN npz {p}: {exc}") from exc
    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(f"V64.3.50 PIOR STOP: {len(missing)} frozen RSMR TRAIN tokens missing from stated bdse_train_v2 layout; first={missing[:10]}")

    raw_dirs = []
    for raw in sorted(set(CITY_TO_RAW.values())):
        d = args.raw_split_root / raw
        if not d.is_dir():
            raise FileNotFoundError(f"missing raw nuPlan DB split directory: {d}")
        dbs = sorted(d.glob("*.db"))
        if not dbs:
            raise RuntimeError(f"raw split directory contains no direct .db files as required: {d}")
        raw_dirs.append(str(d.resolve()))

    rows = [found[t] for t in tokens]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["cache_city"]] = counts.get(r["cache_city"], 0) + 1
    report = {
        "audit": "v64_3_50_pior_train_manifest",
        "algorithm_version": "V64.3.50-EAF-ICER-PIOR",
        "scientific_population": "exact_V49_full_set_RSMR_TRAIN_proposals_only",
        "scenario_count": len(rows),
        "unique_scenario_count": len({r["scenario_token"] for r in rows}),
        "city_counts": counts,
        "raw_db_directories": raw_dirs,
        "dataset_contract": {
            "train_cache_layout": "bdse_train_v2/train_{boston,pittsburgh,singapore,vegas_2}/<log>/*.npz",
            "raw_db_layout": "splits/train_{boston,pittsburgh,singapore,vegas}/*.db",
            "train_vegas_2_maps_to_raw_train_vegas": True,
        },
        "rows": rows,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.output_token_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_token_file.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    print(json.dumps({"pass": True, "scenario_count": len(rows), "city_counts": counts, "output": str(args.output_manifest)}, sort_keys=True))


if __name__ == "__main__":
    main()
