from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from bdse.data.nuplan_dataset import PreprocessedBDSEDataset


def _tokens(root: Path, splits: list[str]) -> list[str]:
    paths = PreprocessedBDSEDataset(root, split=splits, max_scenarios=None).build_index()
    out: list[str] = []
    seen: set[str] = set()
    for p in paths:
        try:
            with np.load(p, allow_pickle=False) as z:
                raw = z["scenario_token"]
                token = str(raw.item() if raw.shape == () else raw.reshape(-1)[0])
        except Exception as exc:
            raise RuntimeError(f"failed to read scenario_token from {p}: {exc}") from exc
        if token and token not in seen:
            seen.add(token); out.append(token)
    if not out:
        raise RuntimeError(f"no scenario tokens under root={root} splits={splits}")
    return out


def _sha(tokens: list[str]) -> str:
    return hashlib.sha256(("\n".join(tokens) + "\n").encode()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description="Export and audit scenario-token manifests for the user's custom BDSE train/val/test caches.")
    p.add_argument("--train-root", type=Path, required=True)
    p.add_argument("--val-root", type=Path, required=True)
    p.add_argument("--test-root", type=Path, required=True)
    p.add_argument("--train-splits", nargs="+", default=["train_boston", "train_pittsburgh", "train_singapore", "train_vegas_2"])
    p.add_argument("--val-split", default="val")
    p.add_argument("--test-split", default="public_set_test")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = _tokens(args.train_root, list(args.train_splits))
    val = _tokens(args.val_root, [args.val_split])
    test = _tokens(args.test_root, [args.test_split])
    sets = {"train": set(train), "val": set(val), "test": set(test)}
    overlap = {
        "train_val": sorted(sets["train"] & sets["val"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "val_test": sorted(sets["val"] & sets["test"]),
    }
    for name, tokens in (("train", train), ("val", val), ("test", test)):
        (args.output_dir / f"{name}_scenario_tokens.txt").write_text("\n".join(tokens) + "\n", encoding="utf-8")
    report = {
        "roots": {"train": str(args.train_root.resolve()), "val": str(args.val_root.resolve()), "test": str(args.test_root.resolve())},
        "splits": {"train": list(args.train_splits), "val": [args.val_split], "test": [args.test_split]},
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "sha256": {"train": _sha(train), "val": _sha(val), "test": _sha(test)},
        "overlap_counts": {k: len(v) for k, v in overlap.items()},
        "disjoint": all(len(v) == 0 for v in overlap.values()),
        "overlap_examples": {k: v[:50] for k, v in overlap.items()},
    }
    (args.output_dir / "dataset_split_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["disjoint"]:
        raise SystemExit("train/val/test scenario-token overlap detected; do not report final test results until the split is repaired")


if __name__ == "__main__":
    main()
