from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
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


def _scalar_int(z: Any, key: str) -> int:
    if key not in z:
        return 0
    try:
        a = np.asarray(z[key])
        v = a.item() if a.shape == () else a.reshape(-1)[0]
        return int(v)
    except Exception:
        return 0


def _cache_iteration(path: Path) -> int:
    m = re.search(r"_it(\d+)\.npz$", path.name)
    return int(m.group(1)) if m else -1


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


def _stable_log_name(text: Any) -> str:
    """Recover the nuPlan log identity from a DB/crop name.

    nuPlan split DB files may append a temporal crop suffix such as
    ``_00718_00912`` while cached samples keep the stable log identity
    ``2021...._veh-28``.  This is the same normalization already used by the
    repository's feature/dataset code.
    """
    name = Path(str(text or "")).name
    if name.endswith(".db"):
        name = name[:-3]
    return re.sub(r"_\d{5,6}_\d{5,6}$", "", name)


def _db_index(raw_split_root: Path) -> dict[str, dict[str, Any]]:
    """Index direct TRAIN DB files by exact and stable nuPlan log identity."""
    out: dict[str, dict[str, Any]] = {}
    for raw in sorted(set(CITY_TO_RAW.values())):
        d = raw_split_root / raw
        if not d.is_dir():
            raise FileNotFoundError(f"missing raw nuPlan DB split directory: {d}")
        dbs = sorted((p.resolve() for p in d.glob("*.db")), key=lambda p: str(p))
        if not dbs:
            raise RuntimeError(f"raw split directory contains no direct .db files as required: {d}")
        by_stem: dict[str, Path] = {}
        by_stable: dict[str, list[Path]] = {}
        for p in dbs:
            if p.stem in by_stem:
                raise RuntimeError(f"duplicate raw DB stem in {d}: {p.stem}")
            by_stem[p.stem] = p
            by_stable.setdefault(_stable_log_name(p.stem), []).append(p)
        out[raw] = {"all": dbs, "by_stem": by_stem, "by_stable": by_stable}
    return out


def _db_contains_token(db: Path, token: str) -> bool | None:
    """Best-effort read-only exact token membership check.

    nuPlan versions store the scenario identity as a lidar-pc token, usually a
    BLOB primary key.  Return True/False when a known schema can be queried and
    None when the local DB schema cannot be proven.  None is deliberately not a
    failure: callers keep the whole stable-log DB family and let nuPlan's exact
    ``scenario_filter.scenario_tokens`` perform the final selection.
    """
    token = str(token).strip()
    values: list[Any] = [token]
    try:
        if len(token) % 2 == 0 and re.fullmatch(r"[0-9A-Fa-f]+", token):
            values.insert(0, bytes.fromhex(token))
    except Exception:
        pass
    try:
        conn = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True, timeout=5.0)
    except Exception:
        return None
    try:
        known = False
        for table, column in (("lidar_pc", "token"), ("scenario_tag", "lidar_pc_token")):
            try:
                cols = {str(r[1]) for r in conn.execute(f'PRAGMA table_info("{table}")')}
            except Exception:
                continue
            if column not in cols:
                continue
            known = True
            for value in values:
                try:
                    row = conn.execute(
                        f'SELECT 1 FROM "{table}" WHERE "{column}" = ? LIMIT 1',
                        (value,),
                    ).fetchone()
                except Exception:
                    continue
                if row is not None:
                    return True
        return False if known else None
    finally:
        conn.close()


def _resolve_raw_db_files(*, row: dict[str, Any], token: str, db_index: dict[str, dict[str, Any]]) -> tuple[list[Path], str]:
    """Resolve a safe raw DB set without ever changing the scientific token set.

    Resolution order:
      1) exact DB stem;
      2) stable log family after stripping nuPlan crop suffix;
      3) when a family has multiple DB chunks, exact SQLite token membership;
      4) if naming/schema cannot prove a narrower set, all direct DBs in the
         token's city split.

    Steps 2/4 can return multiple DB files.  This is scientifically safe because
    runtime still filters by the exact frozen ``scenario_token`` list; the DB set
    only controls I/O discovery scope.
    """
    raw = str(row["raw_db_split"])
    idx = db_index[raw]
    hints = []
    for value in (row.get("log_name", ""), Path(str(row.get("npz_path", ""))).parent.name):
        # nuPlan log names contain many dots (e.g. 2021.08.23...._veh-28).
        # pathlib.Path.stem would incorrectly truncate such an identity at the
        # final dot. Strip only a literal DB suffix and preserve every other dot.
        h = str(value or "").strip()
        if h.endswith(".db"):
            h = h[:-3]
        if h and h not in hints:
            hints.append(h)

    # Exact filename stem remains the cheapest path when available.
    for hint in hints:
        db = idx["by_stem"].get(hint)
        if db is not None:
            return [db], "exact_stem"

    # nuPlan commonly appends _<start>_<end> crop ranges to DB filenames.
    family: list[Path] = []
    seen: set[Path] = set()
    for hint in hints:
        stable = _stable_log_name(hint)
        for db in idx["by_stable"].get(stable, []):
            if db not in seen:
                seen.add(db); family.append(db)
    if family:
        if len(family) == 1:
            return family, "stable_log_single"
        exact_hits: list[Path] = []
        schema_unknown = False
        for db in family:
            contains = _db_contains_token(db, token)
            if contains is True:
                exact_hits.append(db)
            elif contains is None:
                schema_unknown = True
        if len(exact_hits) == 1:
            return exact_hits, "stable_log_sqlite_token_exact"
        if len(exact_hits) > 1:
            raise RuntimeError(
                f"V64.3.50 PIOR STOP: scenario token={token} appears in multiple raw DB chunks "
                f"for stable log={_stable_log_name(hints[0] if hints else '')}: {[str(x) for x in exact_hits]}"
            )
        # No exact hit can mean a devkit/schema variation rather than missing data.
        # Keep all DB chunks belonging to the proven stable log family and let the
        # exact scenario-token filter validate the population at simulation time.
        return family, "stable_log_family_schema_fallback" if schema_unknown else "stable_log_family_token_not_indexed"

    # Last-resort correctness fallback.  Passing all DBs in the *known city* does
    # not alter the requested 502 tokens because nuPlan receives the exact token
    # filter.  It only gives up part of the startup optimization for this row.
    return list(idx["all"]), "city_split_fallback"


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
                    timestamp_us = _scalar_int(z, "timestamp_us")
                    cache_iteration = _cache_iteration(p)
                    if timestamp_us <= 0:
                        raise RuntimeError(
                            f"V64.3.50 PIOR STOP: frozen token={tok} has no valid cache timestamp_us in {p}; "
                            "paired frozen-anchor intervention identity cannot be proven"
                        )
                    if cache_iteration != 0:
                        raise RuntimeError(
                            f"V64.3.50 PIOR STOP: frozen token={tok} comes from cache iteration={cache_iteration}, expected it000000; "
                            "V50.2 repair is defined only for the preregistered frozen proposal anchor event"
                        )
                    action = int(meta[tok]["full_selected_action"])
                    trajectories = np.asarray(z["candidate_trajectories"], dtype=np.float32)
                    candidate_valid = np.asarray(z["candidate_valid"], dtype=bool).reshape(-1)
                    maneuver_ids = np.asarray(z["candidate_maneuver_ids"], dtype=np.int64).reshape(-1)
                    if trajectories.ndim != 3 or trajectories.shape[0] != candidate_valid.shape[0]:
                        raise RuntimeError(
                            f"V64.3.50.3 PIOR STOP: malformed frozen candidate bank token={tok} "
                            f"trajectories={trajectories.shape} valid={candidate_valid.shape}"
                        )
                    # V49 candidate_count is the size of the post-admissibility RSMR
                    # population, not the raw CandidateBank.K.  Do not compare those
                    # quantities.  Persist the raw cached bank size separately as an
                    # engineering identity invariant for closed-loop replay.
                    if not (0 <= action < trajectories.shape[0]) or not bool(candidate_valid[action]):
                        raise RuntimeError(
                            f"V64.3.50.3 PIOR STOP: frozen V49 proposal is not a valid cache action "
                            f"token={tok} action={action} K={trajectories.shape[0]}"
                        )
                    frozen_traj = np.ascontiguousarray(trajectories[action], dtype=np.float32)
                    found[tok] = {
                        **meta[tok],
                        "scenario_token": tok,
                        "cache_city": city,
                        "raw_db_split": CITY_TO_RAW[city],
                        "npz_path": str(p.resolve()),
                        "cache_iteration": int(cache_iteration),
                        "timestamp_us": int(timestamp_us),
                        "log_name": log_name,
                        "scenario_name": _scalar(z, "scenario_name"),
                        "scenario_type": _scalar(z, "scenario_type"),
                        # V50.3 engineering identity repair: an integer slot is not
                        # a stable physical action under a changed closed-loop state.
                        # Persist the selected local trajectory and maneuver so the
                        # runtime can prove that the intervention is the exact cached
                        # V49 proposal rather than merely the same slot number.
                        "frozen_candidate_bank_size": int(trajectories.shape[0]),
                        "frozen_proposal_trajectory": frozen_traj.tolist(),
                        "frozen_proposal_trajectory_sha256": hashlib.sha256(frozen_traj.tobytes(order="C")).hexdigest(),
                        "frozen_proposal_maneuver_id": int(maneuver_ids[action]) if action < maneuver_ids.shape[0] else -1,
                        "frozen_proposal_valid": True,
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

    # Every preregistered V49 proposal comes from *_it000000.npz. Store the
    # exact frozen proposal *anchor-event* timestamp. V50.3 changes only nuPlan's
    # scenario extraction offset so each paired simulation begins at this anchor;
    # no planner-controlled pre-roll is permitted before the intervention.
    # Timestamp collisions across the full 502
    # population are allowed: the runner deterministically separates colliding
    # timestamps into different subprocess batches, where scenario_filter tokens
    # provide the remaining identity constraint.
    ts_counts: dict[int, int] = {}
    for tok in tokens:
        ts = int(found[tok]["timestamp_us"])
        ts_counts[ts] = ts_counts.get(ts, 0) + 1

    db_index = _db_index(args.raw_split_root)
    raw_dirs = [str((args.raw_split_root / raw).resolve()) for raw in sorted(db_index)]
    raw_files: list[str] = []
    raw_seen: set[str] = set()
    resolution_counts: dict[str, int] = {}
    for tok in tokens:
        row = found[tok]
        dbs, mode = _resolve_raw_db_files(row=row, token=tok, db_index=db_index)
        if not dbs:
            raise RuntimeError(f"V64.3.50 PIOR STOP: no raw DB candidates for token={tok}")
        row["raw_db_files"] = [str(x) for x in dbs]
        # Backward-compatible convenience field only when the mapping is unique.
        row["raw_db_file"] = str(dbs[0]) if len(dbs) == 1 else ""
        row["raw_db_resolution_mode"] = mode
        resolution_counts[mode] = resolution_counts.get(mode, 0) + 1
        for db in dbs:
            if str(db) not in raw_seen:
                raw_seen.add(str(db))
                raw_files.append(str(db))

    rows = [found[t] for t in tokens]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["cache_city"]] = counts.get(r["cache_city"], 0) + 1
    report = {
        "audit": "v64_3_50_pior_train_manifest",
        "algorithm_version": "V64.3.50.3-EAF-ICER-PIOR-ANCHOR-IDENTITY-REPAIR",
        "scientific_population": "exact_V49_full_set_RSMR_TRAIN_proposals_only",
        "scenario_count": len(rows),
        "unique_scenario_count": len({r["scenario_token"] for r in rows}),
        "anchor_timestamp_unique_count": len(ts_counts),
        "anchor_timestamp_collision_count": int(sum(max(v - 1, 0) for v in ts_counts.values())),
        "city_counts": counts,
        "npz_files_scanned_until_complete": int(scanned_npz),
        "raw_db_directories": raw_dirs,
        "raw_db_files": raw_files,
        "raw_db_file_count": len(raw_files),
        "raw_db_resolution_counts": resolution_counts,
        "raw_db_restriction": (
            "safe per-token DB-set restriction: exact stem -> stable nuPlan log family -> optional read-only SQLite token disambiguation; "
            "unresolved naming/schema falls back to the token city split while exact scenario_filter.scenario_tokens preserves the frozen population"
        ),
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
        "raw_db_resolution_counts": resolution_counts,
        "npz_files_scanned_until_complete": int(scanned_npz),
        "output": str(args.output_manifest),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
