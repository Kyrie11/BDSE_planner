from __future__ import annotations

"""TRAIN-only behavior supervision for V64.3.44 PCOR.

This tool is deliberately outside the deployed planner.  It may read logged agent
future from frozen TRAIN cache to learn which *runtime-only* response mode best
matches observed behavior.  The resulting model is later fitted with nested
outer-fold isolation.  Deployment never consumes this sidecar or logged future.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.data.cache_schema import CandidateBank, Sample, load_sample_npz
from bdse.data.nuplan_dataset import PreprocessedBDSEDataset
from bdse.planner.response_modes import build_response_modes
from bdse.planner.response_value_observables import (
    PLAN_RESPONSE_CONDITIONING_NAMES,
    RUNTIME_RESPONSE_MODE_NAMES,
    runtime_plan_response_mode_features,
)


def _load_tokens(path: str | Path) -> list[str]:
    vals = [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]
    if len(vals) != len(set(vals)):
        raise ValueError("V44 behavior token file contains duplicates")
    return vals


def _logged_ego_bank(sample: Sample) -> CandidateBank | None:
    if sample.label_future is None:
        return None
    ego = np.asarray(sample.label_future.logged_ego, dtype=np.float32)
    if ego.ndim != 2 or ego.shape[0] <= 0 or ego.shape[1] < 4 or not np.all(np.isfinite(ego[:, :4])):
        return None
    return CandidateBank(
        trajectories=ego[None, :, :],
        valid_mask=np.asarray([True], dtype=bool),
        maneuver_ids=np.asarray([0], dtype=np.int64),
        theta=[{}],
        dynamic_flags=[{}],
        metadata=[{"source": "logged_ego_behavior_supervision_only"}],
    )


def behavior_supervision_example(sample: Sample, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Build one scene-level mode label and logged-plan conditioning feature.

    The target is the runtime-only mode whose agent trajectories have the lowest
    mean displacement error to logged agent future on the already-frozen runtime
    soft-check horizon.  No teacher improvement/action label is used.
    """
    if sample.label_future is None:
        return None
    bank = _logged_ego_bank(sample)
    if bank is None:
        return None
    logged = np.asarray(sample.label_future.logged_agents, dtype=np.float64)
    lvalid = np.asarray(sample.label_future.agent_valid, dtype=bool).reshape(-1)
    rvalid = np.asarray(sample.runtime.agent_valid, dtype=bool).reshape(-1)
    if logged.ndim < 3 or logged.shape[0] <= 0 or logged.shape[1] <= 0:
        return None
    n = min(logged.shape[0], lvalid.size, rvalid.size)
    valid = lvalid[:n] & rvalid[:n]
    if not np.any(valid):
        return None

    modes = {m.name: m for m in build_response_modes(sample.runtime, None, cfg) if m.name in RUNTIME_RESPONSE_MODE_NAMES}
    # build_response_modes may disable a configured mode.  V44's behavior basis is
    # fixed by design, so synthesize missing runtime-only modes through the raw
    # feature helper's fixed bank by requiring all names below.  In historical
    # configs all five are enabled; fail closed otherwise instead of changing basis.
    if set(modes) != set(RUNTIME_RESPONSE_MODE_NAMES):
        raise ValueError(f"V44 requires fixed response basis {RUNTIME_RESPONSE_MODE_NAMES}, got {sorted(modes)}")

    dt = float(cfg.get("candidate", {}).get("step_s", 0.1))
    soft_h = float((cfg.get("runtime_safety", {}) or {}).get("soft_check_horizon_s", float("inf")))
    Te = min(logged.shape[1], min(np.asarray(m.agent_futures).shape[1] for m in modes.values()))
    if np.isfinite(soft_h):
        Te = min(Te, max(1, int(round(soft_h / max(dt, 1.0e-6)))))
    if Te <= 0:
        return None
    errors: list[float] = []
    for name in RUNTIME_RESPONSE_MODE_NAMES:
        fut = np.asarray(modes[name].agent_futures, dtype=np.float64)[:n, :Te, :2]
        gt = logged[:n, :Te, :2]
        d = np.linalg.norm(fut - gt, axis=2)
        errors.append(float(np.mean(d[valid])))
    target = int(np.argmin(np.asarray(errors, dtype=np.float64)))
    raw, names = runtime_plan_response_mode_features(sample.runtime, bank, cfg)
    if names != PLAN_RESPONSE_CONDITIONING_NAMES or raw.shape != (1, len(names)):
        raise ValueError("V44 logged-plan conditioning feature schema mismatch")
    return {
        "scenario_token": str(sample.scenario_token),
        "target_mode_index": target,
        "target_mode": RUNTIME_RESPONSE_MODE_NAMES[target],
        "mode_ade_m": errors,
        "conditioning_feature_names": list(names),
        "conditioning_features": [float(x) for x in raw[0]],
        "valid_agent_count": int(np.sum(valid)),
        "supervision_source": "TRAIN_logged_agent_future_nearest_runtime_mode_only",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed-dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--scenario-token-file", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.base_config).read_text())
    tokens = _load_tokens(args.scenario_token_file)
    want = set(tokens)
    ds = PreprocessedBDSEDataset(args.preprocessed_dir, split=args.split, scenario_tokens=want)
    paths = ds.build_index()
    by_token: dict[str, Path] = {}
    for p in paths:
        try:
            with np.load(p, allow_pickle=True) as z:
                tok = str(z["scenario_token"].item() if z["scenario_token"].shape == () else z["scenario_token"].reshape(-1)[0])
        except Exception:
            continue
        if tok in want:
            by_token[tok] = Path(p)
    missing = want - set(by_token)
    if missing:
        raise SystemExit(f"STOP V44 DATA: behavior supervision cache missing {len(missing)} frozen TRAIN tokens")
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = 0
    mode_counts = np.zeros((len(RUNTIME_RESPONSE_MODE_NAMES),), dtype=np.int64)
    valid_agents = 0
    for tok in tokens:
        sample = load_sample_npz(by_token[tok], include_label_future=True, include_candidate_metadata=False, include_evidence_aux_metadata=False)
        ex = behavior_supervision_example(sample, cfg)
        if ex is None:
            skipped += 1
            continue
        rows.append(ex); mode_counts[int(ex["target_mode_index"])] += 1; valid_agents += int(ex["valid_agent_count"])
    if len(rows) < 512:
        raise SystemExit(f"STOP V44 DATA: only {len(rows)} behavior-supervised TRAIN scenes (<512)")
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    audit = {
        "audit": "v64_3_44_behavior_supervision",
        "requested_tokens": len(tokens),
        "supervised_scenes": len(rows),
        "skipped_scenes_without_usable_logged_agent_future": skipped,
        "total_valid_agent_instances": valid_agents,
        "mode_names": list(RUNTIME_RESPONSE_MODE_NAMES),
        "mode_target_counts": {m: int(mode_counts[i]) for i, m in enumerate(RUNTIME_RESPONSE_MODE_NAMES)},
        "feature_names": list(PLAN_RESPONSE_CONDITIONING_NAMES),
        "deployment_uses_logged_future": False,
    }
    Path(str(out) + ".audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
