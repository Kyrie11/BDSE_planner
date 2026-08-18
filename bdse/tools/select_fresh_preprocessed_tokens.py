from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bdse.data.nuplan_dataset import PreprocessedBDSEDataset


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Select a fresh preprocessed-cache scenario-token set using identity + fixed hash only."
    )
    ap.add_argument("--preprocessed-dir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--exclude-tokens", required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--hash-seed", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--audit-output", required=True)
    args = ap.parse_args()

    exclude = {x.strip() for x in Path(args.exclude_tokens).read_text(encoding="utf-8").splitlines() if x.strip()}
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    ds = PreprocessedBDSEDataset(args.preprocessed_dir, split=args.split)
    token_paths: dict[str, list[str]] = {}
    unresolved = 0
    for path in ds.build_index():
        token = ds.scenario_token_from_cache_path(path)
        if token is None:
            unresolved += 1
            continue
        token_paths.setdefault(token, []).append(str(path))
    tokens = set(token_paths)
    eligible = tokens - exclude
    rank = lambda t: hashlib.sha256((args.hash_seed + "::" + t).encode()).hexdigest()
    selected = sorted(eligible, key=lambda t: (rank(t), t))[: args.count]
    if len(selected) < args.count:
        raise SystemExit(
            f"STOP DATA SPLIT: only {len(selected)} clean canonical cache tokens remain; "
            "do not select from labels/metrics to fill the screen"
        )
    duplicate_token_paths = sum(len(v) > 1 for v in token_paths.values())
    audit = {
        "audit": "preprocessed_identity_hash_fresh_selection",
        "split": str(args.split),
        "cache_path_count": int(sum(len(v) for v in token_paths.values()) + unresolved),
        "canonical_token_count": int(len(tokens)),
        "noncanonical_path_count": int(unresolved),
        "duplicate_token_path_count": int(duplicate_token_paths),
        "design_exclusion_count": int(len(exclude)),
        "excluded_present_in_cache": int(len(tokens & exclude)),
        "eligible_unique": int(len(eligible)),
        "fresh_count": int(len(selected)),
        "fresh_overlap_design": int(len(set(selected) & exclude)),
        "hash_seed": str(args.hash_seed),
        "selection_uses_labels": False,
        "selection_loads_npz_samples": False,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(selected) + "\n", encoding="utf-8")
    Path(args.audit_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_output).write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
