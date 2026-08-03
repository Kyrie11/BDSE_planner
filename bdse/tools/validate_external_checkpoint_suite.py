from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from bdse.external_baselines.models import external_reference, external_variant
from bdse.utils import torch_load_any

EXPECTED = ("gameformer", "dtpp", "plantf", "pluto")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description="Validate matched external-baseline checkpoints and dataset manifests.")
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--expected-train-count", type=int, default=50000)
    p.add_argument("--expected-val-count", type=int, default=500)
    p.add_argument("--expected-splits", nargs="+", default=["train_boston", "train_pittsburgh", "train_singapore", "train_vegas_2"])
    p.add_argument("--allow-missing-manifest", action="store_true")
    args = p.parse_args()

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    train_digests: set[str] = set()
    val_digests: set[str] = set()
    protocol_signatures: set[str] = set()
    for name in EXPECTED:
        path = args.checkpoint_root / f"{name}_budgeted.best.pt"
        if not path.is_file():
            errors.append(f"missing checkpoint: {path}")
            continue
        ckpt = torch_load_any(path, map_location="cpu")
        if not isinstance(ckpt, dict):
            errors.append(f"checkpoint is not a dict: {path}")
            continue
        cfg = ckpt.get("cfg", {}) or {}
        variant = external_variant(cfg)
        if variant != name:
            errors.append(f"variant mismatch: {path.name} contains {variant!r}, expected {name!r}")
        manifest = ckpt.get("training_manifest")
        if not isinstance(manifest, dict):
            sidecar = path.with_name(path.name.replace(".best.pt", ".data_manifest.json"))
            if sidecar.is_file():
                manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            if not args.allow_missing_manifest:
                errors.append(f"missing training manifest: {path}")
            manifest = {}
        train = manifest.get("train") or {}
        val = manifest.get("validation") or {}
        protocol = manifest.get("protocol") or {}
        train_digest = str(train.get("ordered_path_size_sha256", ""))
        val_digest = str(val.get("ordered_path_size_sha256", ""))
        if train_digest:
            train_digests.add(train_digest)
        if val_digest:
            val_digests.add(val_digest)
        if train and int(train.get("count", -1)) != args.expected_train_count:
            errors.append(f"{name}: train count={train.get('count')} expected={args.expected_train_count}")
        if val and int(val.get("count", -1)) != args.expected_val_count:
            errors.append(f"{name}: val count={val.get('count')} expected={args.expected_val_count}")
        if train and list(train.get("split", [])) != list(args.expected_splits):
            errors.append(f"{name}: train splits={train.get('split')} expected={args.expected_splits}")
        signature_payload = {
            "seed": manifest.get("seed"),
            "train_digest": train_digest,
            "val_digest": val_digest,
            "protocol": protocol,
        }
        signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()
        protocol_signatures.add(signature)
        rows.append({
            "variant": name,
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": sha256(path),
            "epoch": ckpt.get("epoch"),
            "selection_metric": ckpt.get("selection_metric"),
            "best_metric": ckpt.get("best_metric"),
            "train_count": train.get("count"),
            "train_sha256": train_digest,
            "val_count": val.get("count"),
            "val_sha256": val_digest,
            "seed": manifest.get("seed"),
            "implementation": external_reference(name),
        })
    if len(train_digests) > 1:
        errors.append(f"train dataset mismatch across checkpoints: {sorted(train_digests)}")
    if len(val_digests) > 1:
        errors.append(f"validation dataset mismatch across checkpoints: {sorted(val_digests)}")
    if len(protocol_signatures) > 1:
        errors.append("training protocol mismatch across checkpoints (seed/dataset/protocol differ)")
    report = {
        "pass": not errors and len(rows) == len(EXPECTED),
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "expected_names": [f"{x}_budgeted.best.pt" for x in EXPECTED],
        "rows": rows,
        "errors": errors,
    }
    output = args.output or (args.checkpoint_root / "external_checkpoint_suite_validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
