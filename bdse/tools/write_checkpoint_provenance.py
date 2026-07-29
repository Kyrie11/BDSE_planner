from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from bdse.config import load_config


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _manifest_records(root: Path | None) -> list[dict[str, Any]]:
    if root is None or not root.exists():
        return []
    manifests = sorted(root.rglob("manifest.jsonl"))
    records: list[dict[str, Any]] = []
    for path in manifests:
        rec = _file_record(path)
        rec["relative_path"] = str(path.relative_to(root))
        records.append(rec)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-cache", default=None)
    parser.add_argument("--val-cache", default=None)
    parser.add_argument("--source", default="rebuilt_current_code")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    config = Path(args.config)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not config.is_file():
        raise FileNotFoundError(config)

    cfg = load_config(str(config))
    train_root = Path(args.train_cache) if args.train_cache else None
    val_root = Path(args.val_cache) if args.val_cache else None
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source),
        "checkpoint": _file_record(checkpoint),
        "config": _file_record(config),
        "seed": int(cfg.get("seed", 17)),
        "training": {
            "epochs": int(cfg.get("training", {}).get("epochs", 0)),
            "batch_size_per_rank": int(cfg.get("training", {}).get("batch_size", 0)),
            "foundation_stage": cfg.get("training", {}).get("foundation_stage"),
            "allow_oracle_only_selector_training": bool(
                cfg.get("training", {}).get("allow_oracle_only_selector_training", False)
            ),
        },
        "train_cache": {
            "root": str(train_root.resolve()) if train_root and train_root.exists() else None,
            "manifests": _manifest_records(train_root),
        },
        "validation_cache": {
            "root": str(val_root.resolve()) if val_root and val_root.exists() else None,
            "manifests": _manifest_records(val_root),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "comparison_rule": (
            "Use this checkpoint for both the v50 warm start and the frozen matched control. "
            "Do not compare absolute metrics directly with runs initialized from the deleted historical v30 checkpoint."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
