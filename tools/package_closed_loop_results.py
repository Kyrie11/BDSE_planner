#!/usr/bin/env python3
"""Package nuPlan closed-loop artifacts with a reproducible manifest.

Example:
  python tools/package_closed_loop_results.py \
    --root ~/code/BDSE_planner/outputs_v32_runtime_ckpt/closed_loop \
    --output ~/code/BDSE_planner/outputs_v32_runtime_ckpt_closed_loop_results.tar.gz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    size_bytes: int
    sha256: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _is_requested_artifact(path: Path, challenge_dir: Path) -> bool:
    try:
        rel = path.relative_to(challenge_dir)
    except ValueError:
        return False
    if not rel.parts:
        return False
    first = rel.parts[0]
    return (
        first in {"aggregator_metric", "metrics"}
        or rel.as_posix() == "runner_report.parquet"
        or first.startswith("nuboard")
    )


def discover_artifacts(root: Path) -> tuple[list[Path], list[str]]:
    files: set[Path] = set()
    warnings: list[str] = []
    challenge_dirs = sorted(
        p for p in root.rglob("closed_loop_nonreactive_agents") if p.is_dir()
    )
    if not challenge_dirs:
        warnings.append(f"No closed_loop_nonreactive_agents directory found under {root}")
        return [], warnings

    for challenge in challenge_dirs:
        found_for_run = 0
        for path in challenge.rglob("*"):
            if path.is_file() and _is_requested_artifact(path, challenge):
                files.add(path.resolve())
                found_for_run += 1
        for required in ("aggregator_metric", "metrics", "runner_report.parquet"):
            target = challenge / required
            if not target.exists():
                warnings.append(f"Missing {target.relative_to(root)}")
        if found_for_run == 0:
            warnings.append(f"No requested artifacts found in {challenge.relative_to(root)}")
    return sorted(files), warnings


def archive_name(path: Path, root: Path) -> str:
    return (Path(root.name) / path.relative_to(root)).as_posix()


def write_archive(output: Path, root: Path, files: Iterable[Path], manifest_bytes: bytes) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffixes = [x.lower() for x in output.suffixes]
    if suffixes[-2:] == [".tar", ".gz"] or output.suffix.lower() in {".tgz", ".tar"}:
        mode = "w:gz" if output.suffix.lower() != ".tar" else "w"
        with tarfile.open(output, mode) as archive:
            for path in files:
                archive.add(path, arcname=archive_name(path, root), recursive=False)
            info = tarfile.TarInfo(name=f"{root.name}/manifest.json")
            info.size = len(manifest_bytes)
            info.mtime = 0
            import io
            archive.addfile(info, io.BytesIO(manifest_bytes))
        return
    if output.suffix.lower() == ".zip":
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                archive.write(path, archive_name(path, root))
            archive.writestr(f"{root.name}/manifest.json", manifest_bytes)
        return
    raise ValueError("--output must end with .tar.gz, .tgz, .tar, or .zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="closed_loop output root")
    parser.add_argument("--output", required=True, type=Path, help="archive path")
    parser.add_argument("--strict", action="store_true", help="fail when required artifacts are missing")
    parser.add_argument("--dry-run", action="store_true", help="print files without creating an archive")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Root does not exist or is not a directory: {root}")
    files, warnings = discover_artifacts(root)
    if not files:
        for warning in warnings:
            print(f"WARNING: {warning}")
        raise SystemExit("No matching closed-loop artifacts found")
    if args.strict and warnings:
        for warning in warnings:
            print(f"ERROR: {warning}")
        raise SystemExit(2)

    entries = [
        ManifestEntry(
            path=archive_name(path, root),
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in files
    ]
    manifest = {
        "schema_version": 1,
        "source_root": str(root),
        "file_count": len(entries),
        "total_size_bytes": sum(x.size_bytes for x in entries),
        "warnings": warnings,
        "files": [asdict(x) for x in entries],
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    for entry in entries:
        print(entry.path)
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Files: {len(entries)}, bytes: {manifest['total_size_bytes']}")
    if args.dry_run:
        return 0
    write_archive(output, root, files, manifest_bytes)
    print(f"Created: {output}")
    print(f"SHA256: {sha256_file(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
