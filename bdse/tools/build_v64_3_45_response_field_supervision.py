from __future__ import annotations

"""TRAIN-only agent-local continuous response supervision for V64.3.45 PIRF.

Logged agent future is used only here to fit a physically bounded longitudinal
response target.  No teacher value, teacher improvement, selected action, or
future datum is exposed at deployment.  Each supervised row belongs to one
agent, while later fitting uses scene-total-one weights so scenes with many
tracked agents do not dominate.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.data.cache_schema import CandidateBank, Sample, load_sample_npz
from bdse.data.nuplan_dataset import PreprocessedBDSEDataset
from bdse.planner.interaction_response_field import (
    RESPONSE_FIELD_LOCAL_FEATURE_NAMES,
    RESPONSE_FIELD_PLAN_FEATURE_NAMES,
    logged_longitudinal_response_target,
    response_field_local_agent_features,
    response_field_plan_agent_features,
)


def _load_tokens(path: str | Path) -> list[str]:
    vals = [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]
    if len(vals) != len(set(vals)):
        raise ValueError("V45 response-field token file contains duplicates")
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
        theta=[{}], dynamic_flags=[{}], metadata=[{"source": "TRAIN_logged_ego_response_field_supervision_only"}],
    )


def response_field_supervision_rows(sample: Sample, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if sample.label_future is None:
        return []
    bank = _logged_ego_bank(sample)
    if bank is None:
        return []
    logged = np.asarray(sample.label_future.logged_agents, dtype=np.float64)
    lvalid = np.asarray(sample.label_future.agent_valid, dtype=bool).reshape(-1)
    rvalid = np.asarray(sample.runtime.agent_valid, dtype=bool).reshape(-1)
    cur = np.asarray(sample.runtime.current_agents, dtype=np.float64)
    if logged.ndim != 3 or logged.shape[0] <= 0 or logged.shape[1] <= 0:
        return []
    n = min(logged.shape[0], lvalid.size, rvalid.size, len(cur))
    if n <= 0:
        return []
    local = response_field_local_agent_features(sample.runtime, cfg)
    plan, exposure = response_field_plan_agent_features(sample.runtime, bank, cfg)
    if local.shape[0] < n or plan.shape[:2] != (1, local.shape[0]):
        raise ValueError("V45 response-field feature shape mismatch")
    out: list[dict[str, Any]] = []
    for j in range(n):
        if not (lvalid[j] and rvalid[j]):
            continue
        gt = logged[j, :, :2]
        if gt.size == 0 or not np.any(np.isfinite(gt)):
            continue
        target = logged_longitudinal_response_target(sample.runtime, logged, j, cfg)
        lf = np.asarray(local[j], dtype=np.float64)
        pf = np.asarray(plan[0, j], dtype=np.float64)
        if not (np.all(np.isfinite(lf)) and np.all(np.isfinite(pf)) and np.isfinite(target)):
            continue
        out.append({
            "scenario_token": str(sample.scenario_token),
            "agent_index": int(j),
            "target_longitudinal_accel_mps2": float(target),
            "local_feature_names": list(RESPONSE_FIELD_LOCAL_FEATURE_NAMES),
            "local_features": [float(x) for x in lf],
            "plan_feature_names": list(RESPONSE_FIELD_PLAN_FEATURE_NAMES),
            "plan_features_logged_ego": [float(x) for x in pf],
            "logged_ego_interaction_exposure": float(exposure[0, j]),
            "supervision_source": "TRAIN_logged_agent_future_continuous_longitudinal_response_only",
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed-dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--scenario-token-file", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.base_config).read_text())
    tokens = _load_tokens(args.scenario_token_file); want = set(tokens)
    ds = PreprocessedBDSEDataset(args.preprocessed_dir, split=args.split, scenario_tokens=want)
    paths = ds.build_index(); by_token: dict[str, Path] = {}
    for p in paths:
        try:
            with np.load(p, allow_pickle=True) as z:
                tok = str(z["scenario_token"].item() if z["scenario_token"].shape == () else z["scenario_token"].reshape(-1)[0])
        except Exception:
            continue
        if tok in want: by_token[tok] = Path(p)
    missing = want - set(by_token)
    if missing:
        raise SystemExit(f"STOP V45 DATA: response supervision cache missing {len(missing)} frozen TRAIN tokens")

    rows: list[dict[str, Any]] = []; scene_counts: dict[str, int] = {}; skipped = 0
    targets: list[float] = []; exposures: list[float] = []
    for tok in tokens:
        sample = load_sample_npz(by_token[tok], include_label_future=True, include_candidate_metadata=False, include_evidence_aux_metadata=False)
        rr = response_field_supervision_rows(sample, cfg)
        if not rr:
            skipped += 1; continue
        scene_counts[tok] = len(rr); rows.extend(rr)
        targets.extend(float(r["target_longitudinal_accel_mps2"]) for r in rr)
        exposures.extend(float(r["logged_ego_interaction_exposure"]) for r in rr)
    if len(scene_counts) < 512 or len(rows) < 4096:
        raise SystemExit(f"STOP V45 DATA: only {len(scene_counts)} scenes / {len(rows)} agent rows")
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, sort_keys=True) + "\n")
    ta = np.asarray(targets, dtype=np.float64); ex = np.asarray(exposures, dtype=np.float64)
    audit = {
        "audit": "v64_3_45_agent_local_continuous_response_supervision",
        "requested_tokens": len(tokens), "supervised_scenes": len(scene_counts), "agent_rows": len(rows),
        "skipped_scenes": skipped,
        "local_feature_names": list(RESPONSE_FIELD_LOCAL_FEATURE_NAMES),
        "plan_feature_names": list(RESPONSE_FIELD_PLAN_FEATURE_NAMES),
        "target_mean_mps2": float(np.mean(ta)), "target_rms_mps2": float(np.sqrt(np.mean(ta * ta))),
        "target_clip_min_fraction": float(np.mean(ta <= -2.0 + 1e-9)),
        "target_clip_max_fraction": float(np.mean(ta >= 0.5 - 1e-9)),
        "interaction_exposure_mean": float(np.mean(ex)), "interaction_exposure_nonzero_fraction": float(np.mean(ex > 1e-12)),
        "deployment_uses_logged_future": False, "uses_teacher_value_or_improvement": False,
    }
    Path(str(out)+".audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
