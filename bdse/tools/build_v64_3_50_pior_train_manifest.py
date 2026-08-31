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


def _db_index(raw_split_root: Path) -> dict[str, dict[str, Path]]:
    """Index the user's flat TRAIN DB folders by filename stem.

    V50 originally handed nuPlan all four TRAIN directories. That is scientifically
    correct but can make scenario construction discover/open many irrelevant logs.
    The PIOR manifest already knows the exact log_name for every frozen proposal,
    so resolve the all-or-nothing direct DB subset here and fail closed if any log
    cannot be mapped. This changes only I/O scope, never the scenario population.
    """
    out: dict[str, dict[str, Path]] = {}
    for raw in sorted(set(CITY_TO_RAW.values())):
        d = raw_split_root / raw
        if not d.is_dir():
            raise FileNotFoundError(f"missing raw nuPlan DB split directory: {d}")
        dbs = sorted(d.glob("*.db"), key=lambda p: str(p))
        if not dbs:
            raise RuntimeError(f"raw split directory contains no direct .db files as required: {d}")
        by_stem: dict[str, Path] = {}
        for p in dbs:
            if p.stem in by_stem:
                raise RuntimeError(f"duplicate raw DB stem in {d}: {p.stem}")
            by_stem[p.stem] = p.resolve()
        out[raw] = by_stem
    return out


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

    scanned_npz = 0
    for city in CITY_TO_RAW:
        root = args.train_cache / city
        if not root.is_dir():
            raise FileNotFoundError(f"missing TRAIN city cache: {root}")
        for p in sorted(root.rglob("*.npz")):
            scanned_npz += 1
            try:
                with np.load(p, allow_pickle=False) as z:
                    tok = _scalar(z, "scenario_token")
                    if tok not in wanted or tok in found:
                        continue
                    log_name = _scalar(z, "log_name") or p.parent.name
                    found[tok] = {
                        **meta[tok],
                        "scenario_token": tok,
                        "cache_city": city,
                        "raw_db_split": CITY_TO_RAW[city],
                        "npz_path": str(p.resolve()),
                        "log_name": log_name,
                        "scenario_name": _scalar(z, "scenario_name"),
                        "scenario_type": _scalar(z, "scenario_type"),
                    }
            except Exception as exc:
                raise RuntimeError(f"failed to read TRAIN npz {p}: {exc}") from exc
            if len(found) == len(wanted):
                break
        if len(found) == len(wanted):
            break
    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(f"V64.3.50 PIOR STOP: {len(missing)} frozen RSMR TRAIN tokens missing from stated bdse_train_v2 layout; first={missing[:10]}")

    db_index = _db_index(args.raw_split_root)
    raw_dirs = [str((args.raw_split_root / raw).resolve()) for raw in sorted(db_index)]
    raw_files: list[str] = []
    raw_seen: set[str] = set()
    for tok in tokens:
        row = found[tok]
        raw = str(row["raw_db_split"])
        # nuPlan log_name is normally the DB filename stem. Be permissive about
        # an accidental '.db' suffix while still requiring an exact stem match.
        hint = Path(str(row.get("log_name", ""))).stem
        db = db_index[raw].get(hint)
        if db is None:
            # The user's cache contract also makes the NPZ parent the acquisition
            # log folder. Use it only as a second exact-stem hint, never substring
            # matching or row-order inference.
            parent_hint = Path(str(row["npz_path"])).parent.name
            db = db_index[raw].get(parent_hint)
        if db is None:
            raise RuntimeError(
                f"V64.3.50 PIOR STOP: cannot map token={tok} log_name={row.get('log_name','')} "
                f"to direct raw DB under {args.raw_split_root / raw}; exact DB restriction cannot be proven"
            )
        row["raw_db_file"] = str(db)
        if str(db) not in raw_seen:
            raw_seen.add(str(db))
            raw_files.append(str(db))

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
        "npz_files_scanned_until_complete": int(scanned_npz),
        "raw_db_directories": raw_dirs,
        "raw_db_files": raw_files,
        "raw_db_file_count": len(raw_files),
        "raw_db_restriction": "exact all-or-nothing log_name/NPZ-parent stem mapping for the 502 frozen tokens",
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
    print(json.dumps({
        "pass": True,
        "scenario_count": len(rows),
        "city_counts": counts,
        "raw_db_file_count": len(raw_files),
        "npz_files_scanned_until_complete": int(scanned_npz),
        "output": str(args.output_manifest),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
