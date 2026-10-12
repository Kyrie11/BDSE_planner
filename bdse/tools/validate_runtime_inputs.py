"""Validate the inputs for real nuPlan runtime/closed-loop experiments.

The old shell wrappers collapsed all input failures into one message.  This
small dependency-free checker reports every resolved path and fails before a
GPU or Hydra process is started.  It deliberately does not inspect NPZ cache
files: a real closed-loop run needs raw nuPlan DB files and an ordered token
manifest whose tokens belong to those DBs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _resolve(value: str, root: Path) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(value)))
    return p if p.is_absolute() else (root / p).resolve()


def _tokens(path: Path) -> tuple[list[str], str]:
    raw = path.read_text(encoding="utf-8").splitlines()
    tokens = [line.strip() for line in raw if line.strip() and not line.lstrip().startswith("#")]
    canonical = "\n".join(tokens) + ("\n" if tokens else "")
    return tokens, hashlib.sha256(canonical.encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--raw-db-root", required=True)
    ap.add_argument("--token-manifest", required=True)
    ap.add_argument("--data-root")
    ap.add_argument("--map-root")
    ap.add_argument("--json-output", type=Path)
    args = ap.parse_args()
    root = args.repo_root.resolve()
    paths = {
        "config": _resolve(args.config, root),
        "checkpoint": _resolve(args.checkpoint, root),
        "raw_db_root": _resolve(args.raw_db_root, root),
        "token_manifest": _resolve(args.token_manifest, root),
    }
    if args.data_root:
        paths["data_root"] = _resolve(args.data_root, root)
    if args.map_root:
        paths["map_root"] = _resolve(args.map_root, root)
    errors: list[str] = []
    for name in ("config", "checkpoint", "token_manifest"):
        if not paths[name].is_file():
            errors.append(f"{name}: file does not exist: {paths[name]}")
        elif paths[name].stat().st_size == 0:
            errors.append(f"{name}: file is empty: {paths[name]}")
    db_files: list[Path] = []
    db_root = paths["raw_db_root"]
    if not db_root.is_dir():
        errors.append(f"raw_db_root: directory does not exist: {db_root}")
    else:
        db_files = sorted(db_root.rglob("*.db"))
        if not db_files:
            errors.append(f"raw_db_root: no *.db files found recursively under: {db_root}")
    tokens: list[str] = []
    token_sha = ""
    if paths["token_manifest"].is_file() and paths["token_manifest"].stat().st_size:
        tokens, token_sha = _tokens(paths["token_manifest"])
        if not tokens:
            errors.append(f"token_manifest: no non-comment tokens found: {paths['token_manifest']}")
        if len(tokens) != len(set(tokens)):
            errors.append(f"token_manifest: duplicate tokens found: {paths['token_manifest']}")
    result = {
        "cwd": str(Path.cwd().resolve()),
        "repo_root": str(root),
        "paths": {k: str(v) for k, v in paths.items()},
        "raw_db_file_count": len(db_files),
        "raw_db_examples": [str(p) for p in db_files[:10]],
        "token_count": len(tokens),
        "token_sha256": token_sha,
        "ok": not errors,
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_output:
        out = args.json_output if args.json_output.is_absolute() else (root / args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
