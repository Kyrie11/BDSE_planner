from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from bdse.data.nuplan_dataset import PreprocessedBDSEDataset, normalize_split_name


def _stable_hash(text: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()


def _manifest_metadata(root: Path) -> dict[Path, dict[str, Any]]:
    out: dict[Path, dict[str, Any]] = {}
    for manifest in sorted(root.rglob("manifest.jsonl")):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw = str(rec.get("path", ""))
            if not raw:
                continue
            path = Path(raw)
            if not path.is_absolute():
                path = root / path
            try:
                path = path.resolve()
            except OSError:
                continue
            out[path] = rec
    return out


def _bucket_for(path: Path, requested_split: str, rec: dict[str, Any] | None) -> str:
    if rec and rec.get("split"):
        return str(rec["split"])
    norm = normalize_split_name(requested_split)
    for part in path.parts:
        if normalize_split_name(part) == norm and part != norm:
            return part
    return norm


def _group_for(path: Path, rec: dict[str, Any] | None) -> str:
    if rec:
        for key in ("log_name", "database_log_name", "db_name"):
            value = str(rec.get(key, "")).strip()
            if value:
                return value
    # Preprocessed caches are normally .../<concrete_split>/<log_name>/<sample>.npz.
    return path.parent.name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create log-group-disjoint val_tune and val_calib manifests from an official nuPlan validation cache."
    )
    parser.add_argument("--preprocessed-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--max-scenarios", type=int, default=None)
    args = parser.parse_args()
    if not 0.05 <= args.calibration_fraction <= 0.50:
        raise ValueError("--calibration-fraction must be in [0.05, 0.50]")

    root = Path(args.preprocessed_dir).resolve()
    output_root = Path(args.output_root).resolve()
    dataset = PreprocessedBDSEDataset(root, split=args.split, max_scenarios=args.max_scenarios)
    paths = [p.resolve() for p in dataset.build_index()]
    metadata = _manifest_metadata(root)

    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in paths:
        rec = metadata.get(path)
        grouped[(_bucket_for(path, args.split, rec), _group_for(path, rec))].append(path)

    by_bucket: dict[str, list[str]] = defaultdict(list)
    for bucket, group in grouped:
        by_bucket[bucket].append(group)

    calibration_groups: set[tuple[str, str]] = set()
    for bucket, groups in sorted(by_bucket.items()):
        unique = sorted(set(groups), key=lambda g: _stable_hash(f"{bucket}:{g}", args.seed))
        n_cal = max(1, int(round(len(unique) * args.calibration_fraction))) if len(unique) > 1 else 0
        # Never consume the only group in a bucket; a one-group bucket cannot be
        # split without leakage and is kept in tuning with an explicit warning.
        n_cal = min(n_cal, max(0, len(unique) - 1))
        calibration_groups.update((bucket, group) for group in unique[:n_cal])

    tune_records: list[dict[str, Any]] = []
    calib_records: list[dict[str, Any]] = []
    for (bucket, group), group_paths in sorted(grouped.items()):
        target = calib_records if (bucket, group) in calibration_groups else tune_records
        split_name = "val_calib" if target is calib_records else "val_tune"
        for path in sorted(group_paths):
            rec = metadata.get(path, {})
            target.append({
                "path": str(path),
                "split": split_name,
                "source_split": str(bucket),
                "log_name": str(group),
                "scenario_token": str(rec.get("scenario_token", path.stem)),
            })

    tune_groups = {(r["source_split"], r["log_name"]) for r in tune_records}
    calib_groups = {(r["source_split"], r["log_name"]) for r in calib_records}
    overlap = tune_groups & calib_groups
    if overlap:
        raise RuntimeError(f"group leakage detected: {sorted(overlap)[:5]}")
    if not calib_records:
        raise RuntimeError("No calibration records were created; the cache needs at least two log groups in one validation bucket")

    for split_name, records in (("val_tune", tune_records), ("val_calib", calib_records)):
        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "manifest.jsonl").write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
            encoding="utf-8",
        )

    provenance = {
        "method": "deterministic log-group-disjoint validation partition",
        "source_root": str(root),
        "source_split": str(args.split),
        "seed": int(args.seed),
        "calibration_fraction_requested": float(args.calibration_fraction),
        "tune_scene_count": len(tune_records),
        "calibration_scene_count": len(calib_records),
        "tune_group_count": len(tune_groups),
        "calibration_group_count": len(calib_groups),
        "group_disjoint": True,
        "no_group_overlap": not bool(overlap),
        "calibration_role": "calibration_only",
        "checkpoint_selection_role": "val_tune_only",
        "tune_manifest": str(output_root / "val_tune" / "manifest.jsonl"),
        "calibration_manifest": str(output_root / "val_calib" / "manifest.jsonl"),
        "tune_manifest_sha256": hashlib.sha256((output_root / "val_tune" / "manifest.jsonl").read_bytes()).hexdigest(),
        "calibration_manifest_sha256": hashlib.sha256((output_root / "val_calib" / "manifest.jsonl").read_bytes()).hexdigest(),
        "warning": "The checkpoint, hyperparameters, and stopping rule must never use val_calib. Official test remains untouched until final evaluation.",
    }
    provenance_path = output_root / "calibration_split_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
