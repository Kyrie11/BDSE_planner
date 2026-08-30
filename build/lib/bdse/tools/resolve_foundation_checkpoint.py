from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from bdse.config import load_config
from bdse.model.bdse_model import BDSEModel


_ALGORITHM_MARKERS = (
    "d3ce",
    "dbce",
    "dbap",
    "aocc",
    "pb_rads",
    "rads",
    "sapdacc",
    "cbldacc",
    "prdacc",
    "lexdacc",
)
_V30_MARKERS = (
    "bdse_v30_pmvrbsr",
    "v30_rebuilt",
    "foundation_v30",
    "v30-compatible",
    "v30_compatible",
)


@dataclass
class CandidateReport:
    path: str
    readable: bool
    safe_foundation: bool
    score: float
    matched_key_fraction: float
    matched_numel_fraction: float
    matched_keys: int
    target_keys: int
    missing_keys: int
    unexpected_keys: int
    source_config: str
    source_output: str
    source_warm_start: str
    source_epoch: int | None
    source_metric: float | None
    v30_identity_evidence: list[str]
    algorithm_markers: list[str]
    rejection_reasons: list[str]
    error: str = ""


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _normalize_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in state.items():
        name = str(key)
        if name.startswith("module."):
            name = name[len("module.") :]
        normalized[name] = value
    return normalized


def _extract_state(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, Mapping):
        raw = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    else:
        raw = checkpoint
    if not isinstance(raw, Mapping):
        raise TypeError("checkpoint does not contain a model/state_dict mapping")
    return _normalize_state_dict(raw)


def _nested_get(mapping: Any, *keys: str, default: Any = "") -> Any:
    cur = mapping
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _string(value: Any) -> str:
    return "" if value is None else str(value)


def _identity_evidence(path: Path, checkpoint: Any) -> tuple[list[str], list[str], str, str, str]:
    args = checkpoint.get("args", {}) if isinstance(checkpoint, Mapping) else {}
    cfg = checkpoint.get("cfg", {}) if isinstance(checkpoint, Mapping) else {}
    source_config = _string(args.get("config") if isinstance(args, Mapping) else "")
    source_output = _string(args.get("output") if isinstance(args, Mapping) else "")
    source_warm = _string(args.get("warm_start_from") if isinstance(args, Mapping) else "")

    evidence: list[str] = []
    searchable = {
        "checkpoint_filename": path.name.lower(),
        "checkpoint_path": str(path).lower(),
        "args.config": source_config.lower(),
        "args.output": source_output.lower(),
    }
    for source_name, text in searchable.items():
        for marker in _V30_MARKERS:
            if marker in text:
                evidence.append(f"{source_name}:{marker}")
    # The old training scripts stored the complete config.  A non-residual pair
    # head is consistent with the v30 foundation, but is deliberately only weak
    # evidence because several later methods inherited that setting.
    residual_over_local = _nested_get(cfg, "model", "pair_head_residual_over_local", default=None)
    if residual_over_local is False:
        evidence.append("cfg:model.pair_head_residual_over_local=false")

    algorithm_markers: list[str] = []
    # Parent directory names such as ``runtime_v30ckpt`` describe an evaluation
    # output root, not necessarily the identity of the checkpoint file inside it.
    # Reject algorithm adaptation only when the checkpoint filename or its own
    # stored training config/output metadata carries the marker.
    combined = " ".join(
        searchable[key]
        for key in ("checkpoint_filename", "args.config", "args.output")
    )
    for marker in _ALGORITHM_MARKERS:
        if marker in combined:
            algorithm_markers.append(marker)
    return sorted(set(evidence)), sorted(set(algorithm_markers)), source_config, source_output, source_warm


def analyze_checkpoint(
    path: Path,
    target_state: Mapping[str, Any],
    *,
    allow_algorithm_checkpoints: bool = False,
) -> CandidateReport:
    try:
        checkpoint = _torch_load(path)
        state = _extract_state(checkpoint)
        target = _normalize_state_dict(target_state)
        matched = {
            key
            for key, value in state.items()
            if key in target
            and hasattr(value, "shape")
            and hasattr(target[key], "shape")
            and tuple(value.shape) == tuple(target[key].shape)
        }
        target_numel = sum(int(value.numel()) for value in target.values() if hasattr(value, "numel"))
        matched_numel = sum(int(target[key].numel()) for key in matched if hasattr(target[key], "numel"))
        key_fraction = len(matched) / max(1, len(target))
        numel_fraction = matched_numel / max(1, target_numel)
        evidence, algorithm_markers, source_config, source_output, source_warm = _identity_evidence(path, checkpoint)

        strong_identity = any(
            item.startswith("checkpoint_filename:")
            or item.startswith("args.config:")
            or item.startswith("args.output:")
            for item in evidence
        )
        rejection: list[str] = []
        if numel_fraction < 0.80:
            rejection.append(f"matched_numel_fraction<{0.80:.2f}")
        if key_fraction < 0.70:
            rejection.append(f"matched_key_fraction<{0.70:.2f}")
        if not strong_identity and not allow_algorithm_checkpoints:
            rejection.append("no_strong_v30_identity_evidence")
        if algorithm_markers and not allow_algorithm_checkpoints:
            rejection.append("algorithm_specific_checkpoint")

        score = 100.0 * numel_fraction + 40.0 * key_fraction
        if strong_identity:
            score += 200.0
        if any(item.startswith("checkpoint_filename:bdse_v30_pmvrbsr") for item in evidence):
            score += 300.0
        if path.name.endswith(".best.pt"):
            score += 20.0
        if algorithm_markers:
            score -= 150.0

        metric = None
        if isinstance(checkpoint, Mapping):
            raw_metric = checkpoint.get("best_metric")
            if isinstance(raw_metric, (int, float)):
                metric = float(raw_metric)
        epoch = None
        if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("epoch"), (int, float)):
            epoch = int(checkpoint["epoch"])

        return CandidateReport(
            path=str(path.resolve()),
            readable=True,
            safe_foundation=not rejection,
            score=float(score),
            matched_key_fraction=float(key_fraction),
            matched_numel_fraction=float(numel_fraction),
            matched_keys=len(matched),
            target_keys=len(target),
            missing_keys=max(0, len(target) - len(matched)),
            unexpected_keys=max(0, len(state) - len(matched)),
            source_config=source_config,
            source_output=source_output,
            source_warm_start=source_warm,
            source_epoch=epoch,
            source_metric=metric,
            v30_identity_evidence=evidence,
            algorithm_markers=algorithm_markers,
            rejection_reasons=rejection,
        )
    except Exception as exc:  # inventory must continue past corrupt/foreign files
        return CandidateReport(
            path=str(path.resolve()),
            readable=False,
            safe_foundation=False,
            score=float("-inf"),
            matched_key_fraction=0.0,
            matched_numel_fraction=0.0,
            matched_keys=0,
            target_keys=len(target_state),
            missing_keys=len(target_state),
            unexpected_keys=0,
            source_config="",
            source_output="",
            source_warm_start="",
            source_epoch=None,
            source_metric=None,
            v30_identity_evidence=[],
            algorithm_markers=[],
            rejection_reasons=["unreadable"],
            error=f"{type(exc).__name__}: {exc}",
        )


def _iter_checkpoints(search_root: Path, patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in search_root.glob(pattern):
            if path.is_file() and path.suffix == ".pt":
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    paths.append(path)
    return sorted(paths, key=lambda p: str(p))


def build_inventory(
    *,
    config_path: Path,
    search_root: Path,
    patterns: Iterable[str],
    allow_algorithm_checkpoints: bool,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    target_model = BDSEModel(cfg)
    target_state = target_model.state_dict()
    checkpoints = _iter_checkpoints(search_root, patterns)
    reports = [
        analyze_checkpoint(path, target_state, allow_algorithm_checkpoints=allow_algorithm_checkpoints)
        for path in checkpoints
    ]
    reports.sort(key=lambda item: (item.safe_foundation, item.score), reverse=True)
    selected = next((item.path for item in reports if item.safe_foundation), "")
    return {
        "schema_version": 1,
        "config": str(config_path.resolve()),
        "search_root": str(search_root.resolve()),
        "patterns": list(patterns),
        "allow_algorithm_checkpoints": bool(allow_algorithm_checkpoints),
        "selected_safe_foundation": selected,
        "candidate_count": len(reports),
        "safe_candidate_count": sum(1 for item in reports if item.safe_foundation),
        "candidates": [asdict(item) for item in reports],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory retained BDSE checkpoints and conservatively recover only an explicitly "
            "identified v30-compatible foundation copy. Later algorithm checkpoints are reported "
            "but rejected by default because they confound v50 attribution."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--search-root", type=Path, default=Path("."))
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        default=None,
        help="Glob relative to --search-root. Repeatable. Defaults scan outputs_v40* through outputs_v50*.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--allow-algorithm-checkpoints", action="store_true")
    parser.add_argument("--print-selected", action="store_true", help="Print only the selected safe path to stdout.")
    args = parser.parse_args()

    patterns = args.patterns or [
        "outputs_v4[0-9]*/**/*v30*.pt",
        "outputs_v50*/**/*v30*.pt",
        "outputs_v4[0-9]*/**/*.best.pt",
        "outputs_v50*/**/*.best.pt",
    ]
    inventory = build_inventory(
        config_path=args.config,
        search_root=args.search_root,
        patterns=patterns,
        allow_algorithm_checkpoints=bool(args.allow_algorithm_checkpoints),
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8")

    selected = str(inventory["selected_safe_foundation"])
    if args.print_selected:
        if selected:
            print(selected)
        return

    print(json.dumps({
        "selected_safe_foundation": selected,
        "candidate_count": inventory["candidate_count"],
        "safe_candidate_count": inventory["safe_candidate_count"],
        "output_json": str(args.output_json) if args.output_json else "",
    }, indent=2, sort_keys=True))
    if not selected:
        print(
            "No conservatively recoverable v30 checkpoint was found. Rebuild the matched foundation; "
            "do not silently substitute a v47-v49 algorithm checkpoint for the paper main run.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
