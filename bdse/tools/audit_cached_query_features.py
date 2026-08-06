from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from bdse.config import load_config
from bdse.data.cache_schema import load_sample_npz
from bdse.data.nuplan_dataset import PreprocessedBDSEDataset
from bdse.planner.evidence_atoms import ATOM_QUERY_DIM, compute_query_features


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _query_code_sha256() -> str:
    return _sha256_bytes((inspect.getsource(compute_query_features) + f"\nATOM_QUERY_DIM={ATOM_QUERY_DIM}\n").encode("utf-8"))


def _query_config_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    # Query features currently depend on candidate temporal sampling and the
    # teacher evaluation stride.  Include broader relevant sections so future
    # feature additions invalidate old audits conservatively.
    payload = {
        "candidate": cfg.get("candidate", {}),
        "teacher": cfg.get("teacher", {}),
        "evidence": cfg.get("evidence", {}),
        "runtime_safety": cfg.get("runtime_safety", {}),
        "model_query_feature_dim": (cfg.get("model", {}) or {}).get("query_feature_dim", ATOM_QUERY_DIM),
    }
    return payload


def _query_config_sha256(cfg: dict[str, Any]) -> str:
    return _sha256_bytes(json.dumps(_query_config_payload(cfg), sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _verify_report(path: Path, cfg: dict[str, Any], tolerance: float) -> int:
    report = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if not bool(report.get("pass", False)):
        failures.append("audit report is not PASS")
    if str(report.get("query_code_sha256", "")) != _query_code_sha256():
        failures.append("compute_query_features code fingerprint changed")
    if str(report.get("query_config_sha256", "")) != _query_config_sha256(cfg):
        failures.append("query-relevant config fingerprint changed")
    max_abs = float(report.get("max_abs_error", float("inf")))
    if not math.isfinite(max_abs) or max_abs > tolerance:
        failures.append(f"max_abs_error={max_abs} > tolerance={tolerance}")
    if int(report.get("shape_failure_count", 1)) != 0:
        failures.append(f"shape_failure_count={report.get('shape_failure_count')}")
    if failures:
        print(json.dumps({"verified": False, "failures": failures}, indent=2))
        return 3
    print(json.dumps({"verified": True, "report": str(path)}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Audit shape-valid cached evidence query features against canonical runtime recomputation")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--preprocessed-dir", type=Path)
    p.add_argument("--split", nargs="+", default=["train"])
    p.add_argument("--max-scenarios", type=int, default=512)
    p.add_argument("--seed", type=int, default=63)
    p.add_argument("--tolerance", type=float, default=1.0e-5)
    p.add_argument("--output", type=Path)
    p.add_argument("--verify-report", type=Path)
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.verify_report is not None:
        return _verify_report(args.verify_report, cfg, float(args.tolerance))
    if args.preprocessed_dir is None or args.output is None:
        p.error("--preprocessed-dir and --output are required unless --verify-report is used")

    dataset = PreprocessedBDSEDataset(args.preprocessed_dir, split=args.split)
    paths = list(dataset.build_index())
    if not paths:
        raise SystemExit("No samples found")
    rng = np.random.default_rng(int(args.seed))
    count = min(max(int(args.max_scenarios), 1), len(paths))
    indices = np.sort(rng.choice(len(paths), size=count, replace=False))
    errors: list[float] = []
    exact_rows = 0
    shape_failures: list[dict[str, Any]] = []
    worst: list[tuple[float, str]] = []
    qdim = int((cfg.get("model", {}) or {}).get("query_feature_dim", ATOM_QUERY_DIM))
    for idx in indices.tolist():
        sample = load_sample_npz(
            paths[idx],
            include_label_future=False,
            include_candidate_metadata=False,
            include_runtime_metadata=False,
            include_route_ids=False,
            include_evidence_aux_metadata=False,
            allow_pickle=False,
        )
        E = sample.evidence_bank.E
        K = sample.candidates.K
        cached = np.asarray(sample.evidence_bank.query_features, dtype=np.float32)
        if cached.ndim != 3 or cached.shape[0] < E or cached.shape[1] < K or cached.shape[2] < qdim:
            shape_failures.append({
                "path": str(paths[idx]),
                "cached_shape": list(cached.shape),
                "required_shape": [E, K, qdim],
            })
            continue
        runtime = compute_query_features(
            sample.evidence_bank.atoms,
            sample.candidates,
            sample.runtime,
            cfg,
        )
        d = min(qdim, runtime.shape[2])
        err = np.abs(cached[:E, :K, :d] - runtime[:E, :K, :d])
        max_abs = float(err.max()) if err.size else 0.0
        errors.append(max_abs)
        exact_rows += int(max_abs == 0.0)
        worst.append((max_abs, str(paths[idx])))
    arr = np.asarray(errors, dtype=np.float64)
    tol = float(args.tolerance)
    report = {
        "audit": "cached_query_features_vs_canonical_runtime",
        "pass": bool(len(shape_failures) == 0 and arr.size == count and np.isfinite(arr).all() and float(arr.max(initial=0.0)) <= tol),
        "preprocessed_dir": str(args.preprocessed_dir.resolve()),
        "splits": list(args.split),
        "dataset_size": len(paths),
        "sampled_scenarios": count,
        "compared_scenarios": int(arr.size),
        "shape_failure_count": len(shape_failures),
        "shape_failures": shape_failures[:20],
        "tolerance": tol,
        "max_abs_error": float(arr.max(initial=float("nan"))) if arr.size else float("nan"),
        "mean_scene_max_abs_error": float(arr.mean()) if arr.size else float("nan"),
        "p50_scene_max_abs_error": float(np.quantile(arr, 0.50)) if arr.size else float("nan"),
        "p95_scene_max_abs_error": float(np.quantile(arr, 0.95)) if arr.size else float("nan"),
        "p99_scene_max_abs_error": float(np.quantile(arr, 0.99)) if arr.size else float("nan"),
        "exact_scene_fraction": float(exact_rows / max(int(arr.size), 1)),
        "worst_samples": [
            {"max_abs_error": float(err), "path": path}
            for err, path in sorted(worst, reverse=True)[:20]
        ],
        "query_code_sha256": _query_code_sha256(),
        "query_config_sha256": _query_config_sha256(cfg),
        "query_config_payload": _query_config_payload(cfg),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
