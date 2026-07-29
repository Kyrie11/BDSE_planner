from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _manifest_ids(root: str | None) -> tuple[set[str], int, int, list[str]]:
    if not root:
        return set(), 0, 0, []
    base = Path(root)
    manifests = sorted(base.rglob("manifest.jsonl"))
    ids: list[str] = []
    for manifest in manifests:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            token = str(row.get("scenario_token", row.get("token", "")))
            ts = int(row.get("timestamp_us", row.get("timestamp", 0)) or 0)
            ids.append(f"{token}@{ts}")
    failed = sum(
        1
        for fp in base.rglob("failed_preprocess.jsonl")
        for line in fp.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return set(ids), len(ids), failed, [str(p) for p in manifests]


def _config_mismatches(a: dict[str, Any] | None, b: dict[str, Any] | None, prefix: str = "") -> list[str]:
    if a is None or b is None:
        return [f"{prefix or 'config'}:missing"]
    out: list[str] = []
    keys = sorted(set(a) | set(b))
    for key in keys:
        name = f"{prefix}.{key}" if prefix else str(key)
        va, vb = a.get(key), b.get(key)
        if isinstance(va, dict) and isinstance(vb, dict):
            out.extend(_config_mismatches(va, vb, name))
        elif va != vb:
            out.append(f"{name}:{va!r}!={vb!r}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Separate test-cache build integrity from legitimate split distribution shift.")
    ap.add_argument("--test-diagnostics", required=True)
    ap.add_argument("--val-diagnostics")
    ap.add_argument("--test-cache")
    ap.add_argument("--train-cache")
    ap.add_argument("--val-cache")
    ap.add_argument("--min-preliminary-samples", type=int, default=10000)
    ap.add_argument("--max-failed-fraction", type=float, default=0.01)
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--output")
    args = ap.parse_args()

    test = _json(args.test_diagnostics) or {}
    val = _json(args.val_diagnostics)
    n = int(test.get("num_samples", test.get("num_loaded", 0)) or 0)
    skipped = int(test.get("num_skipped_missing_labels", 0) or 0)
    ident = test.get("split_identity", {}) or {}
    duplicate_count = int(ident.get("identity_duplicate_count", -1))
    unique_count = int(ident.get("identity_unique_count", 0))
    config_diff = _config_mismatches(test.get("config_summary"), None if val is None else val.get("config_summary")) if val else []

    test_ids, manifest_rows, failed, manifests = _manifest_ids(args.test_cache)
    train_ids, _, _, _ = _manifest_ids(args.train_cache)
    val_ids, _, _, _ = _manifest_ids(args.val_cache)
    overlap_train = len(test_ids & train_ids) if test_ids and train_ids else None
    overlap_val = len(test_ids & val_ids) if test_ids and val_ids else None
    failed_fraction = failed / max(manifest_rows + failed, 1) if manifests else None

    hard_failures: list[str] = []
    warnings: list[str] = []
    if n < args.min_preliminary_samples:
        hard_failures.append(f"sample_count={n} < {args.min_preliminary_samples}")
    if skipped != 0:
        hard_failures.append(f"missing_label_skips={skipped}")
    if duplicate_count > 0 or (unique_count and unique_count != n):
        hard_failures.append(f"identity_integrity duplicate={duplicate_count} unique={unique_count} n={n}")
    if val and config_diff:
        hard_failures.append("preprocess/config mismatch with val: " + "; ".join(config_diff[:20]))
    if overlap_train not in (None, 0):
        hard_failures.append(f"test/train identity overlap={overlap_train}")
    if overlap_val not in (None, 0):
        hard_failures.append(f"test/val identity overlap={overlap_val}")
    if failed_fraction is not None and failed_fraction > args.max_failed_fraction:
        hard_failures.append(f"failed_preprocess_fraction={failed_fraction:.6f} > {args.max_failed_fraction:.6f}")
    if not manifests:
        warnings.append("No test cache manifest was supplied; cross-split leakage, failed-preprocess bias, log/city coverage, and completion cannot be proven from aggregate diagnostics alone.")
    if not args.allow_incomplete:
        warnings.append("Completion cannot be inferred unless an expected-scenario index/count is supplied; use this result only after preprocessing finishes.")
    else:
        warnings.append("Incomplete cache explicitly allowed: status is preliminary, not a final paper test result.")

    status = "FAIL" if hard_failures else ("PRELIMINARY_PASS" if args.allow_incomplete else "INTEGRITY_PASS_COMPLETION_UNVERIFIED")
    report = {
        "status": status,
        "num_samples": n,
        "identity_unique_count": unique_count,
        "identity_duplicate_count": duplicate_count,
        "num_skipped_missing_labels": skipped,
        "config_matches_val": bool(val is not None and not config_diff),
        "config_mismatches": config_diff,
        "manifest_rows": manifest_rows,
        "failed_preprocess_count": failed if manifests else None,
        "failed_preprocess_fraction": failed_fraction,
        "test_train_overlap": overlap_train,
        "test_val_overlap": overlap_val,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "interpretation": "Distribution differences are reported separately and are not build failures. Do not tune thresholds or checkpoints on this test split.",
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(1 if hard_failures else 0)


if __name__ == "__main__":
    main()
