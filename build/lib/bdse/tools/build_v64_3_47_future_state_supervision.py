from __future__ import annotations

"""TRAIN-only nuisance supervision for V64.3.47 FSFR.

Produces two disjoint supervision tables:
- agent-local lateral response drift from logged agent future;
- candidate demo/reference component from logged ego future.

Neither table contains teacher total cost, teacher improvement, selected action,
or outer-test deployment outputs.  Logged future is never consumed by runtime.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.data.cache_schema import CandidateBank, Sample, load_sample_npz
from bdse.data.nuplan_dataset import PreprocessedBDSEDataset
from bdse.planner.future_state_factorization import (
    EGO_REFERENCE_FEATURE_NAMES,
    ego_reference_candidate_features,
    logged_demo_component_targets,
    logged_lateral_drift_target,
)
from bdse.planner.interaction_response_field import (
    RESPONSE_FIELD_LOCAL_FEATURE_NAMES,
    RESPONSE_FIELD_PLAN_FEATURE_NAMES,
    response_field_local_agent_features,
    response_field_plan_agent_features,
)


def _load_tokens(path: str | Path) -> list[str]:
    vals = [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]
    if len(vals) != len(set(vals)):
        raise ValueError("V47 token file contains duplicates")
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
        theta=[{}], dynamic_flags=[{}], metadata=[{"source": "V47_TRAIN_logged_ego_plan_features_only"}],
    )


def _agent_rows(sample: Sample, cfg: dict[str, Any]) -> list[dict[str, Any]]:
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
    local = response_field_local_agent_features(sample.runtime, cfg)
    plan, exposure = response_field_plan_agent_features(sample.runtime, bank, cfg)
    out: list[dict[str, Any]] = []
    for j in range(n):
        if not (lvalid[j] and rvalid[j]):
            continue
        gt = logged[j, :, :2]
        if gt.size == 0 or not np.any(np.isfinite(gt)):
            continue
        target = logged_lateral_drift_target(sample.runtime, logged, j, cfg)
        lf = np.asarray(local[j], dtype=np.float64)
        pf = np.asarray(plan[0, j], dtype=np.float64)
        if not (np.all(np.isfinite(lf)) and np.all(np.isfinite(pf)) and np.isfinite(target)):
            continue
        out.append({
            "scenario_token": str(sample.scenario_token),
            "agent_index": int(j),
            "target_lateral_drift_mps": float(target),
            "local_feature_names": list(RESPONSE_FIELD_LOCAL_FEATURE_NAMES),
            "local_features": [float(x) for x in lf],
            "plan_feature_names": list(RESPONSE_FIELD_PLAN_FEATURE_NAMES),
            "plan_features_logged_ego": [float(x) for x in pf],
            "logged_ego_interaction_exposure": float(exposure[0, j]),
            "supervision_source": "TRAIN_logged_agent_future_lateral_drift_only",
        })
    return out


def _ego_rows(sample: Sample, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if sample.label_future is None:
        return []
    features = ego_reference_candidate_features(sample.runtime, sample.candidates, cfg)
    target = logged_demo_component_targets(sample.runtime, sample.label_future.logged_ego, sample.candidates, cfg)
    tcfg = cfg.get("teacher", {}) if isinstance(cfg, dict) else {}
    w = float(tcfg.get("demo_weight", 1.0)) / max(float(tcfg.get("demo_scale", 120.0)), 1.0e-6)
    out: list[dict[str, Any]] = []
    valid = np.asarray(sample.candidates.valid_mask, dtype=bool).reshape(-1)
    for a in range(int(sample.candidates.K)):
        if a >= len(valid) or not valid[a]:
            continue
        x = np.asarray(features[a], dtype=np.float64)
        y = float(target[a])
        if not (np.all(np.isfinite(x)) and np.isfinite(y)):
            continue
        out.append({
            "scenario_token": str(sample.scenario_token),
            "action": int(a),
            "feature_names": list(EGO_REFERENCE_FEATURE_NAMES),
            "features": [float(z) for z in x],
            "target_demo_component": y,
            "cv_demo_proxy": float(max(x[0] * w, 0.0)),
            "supervision_source": "TRAIN_logged_ego_future_demo_component_only",
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed-dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--scenario-token-file", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--output-agent", required=True)
    ap.add_argument("--output-ego", required=True)
    ap.add_argument("--output-audit", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.base_config).read_text())
    tokens = _load_tokens(a.scenario_token_file); want = set(tokens)
    if len(tokens) != 3000:
        raise SystemExit(f"STOP V47 DATA: expected frozen 3000 TRAIN tokens, got {len(tokens)}")
    ds = PreprocessedBDSEDataset(a.preprocessed_dir, split=a.split, scenario_tokens=want)
    paths = ds.build_index(); bytok: dict[str, Path] = {}
    for p in paths:
        try:
            with np.load(p, allow_pickle=True) as z:
                tok = str(z["scenario_token"].item() if z["scenario_token"].shape == () else z["scenario_token"].reshape(-1)[0])
        except Exception:
            continue
        if tok in want:
            bytok[tok] = Path(p)
    missing = want - set(bytok)
    if missing:
        raise SystemExit(f"STOP V47 DATA: cache missing {len(missing)} frozen TRAIN tokens")

    ar: list[dict[str, Any]] = []; er: list[dict[str, Any]] = []
    agent_scenes = ego_scenes = 0
    for tok in tokens:
        s = load_sample_npz(bytok[tok], include_label_future=True, include_candidate_metadata=False, include_evidence_aux_metadata=False)
        aa = _agent_rows(s, cfg); ee = _ego_rows(s, cfg)
        if aa: agent_scenes += 1; ar.extend(aa)
        if ee: ego_scenes += 1; er.extend(ee)
    if agent_scenes < 512 or len(ar) < 4096:
        raise SystemExit(f"STOP V47 DATA: lateral supervision only {agent_scenes} scenes/{len(ar)} rows")
    if ego_scenes < 512 or len(er) < 4096:
        raise SystemExit(f"STOP V47 DATA: ego-reference supervision only {ego_scenes} scenes/{len(er)} rows")
    pa, pe = Path(a.output_agent), Path(a.output_ego)
    pa.parent.mkdir(parents=True, exist_ok=True); pe.parent.mkdir(parents=True, exist_ok=True)
    with pa.open("w") as f:
        for r in ar: f.write(json.dumps(r, sort_keys=True) + "\n")
    with pe.open("w") as f:
        for r in er: f.write(json.dumps(r, sort_keys=True) + "\n")
    lat = np.asarray([r["target_lateral_drift_mps"] for r in ar], dtype=np.float64)
    demo = np.asarray([r["target_demo_component"] for r in er], dtype=np.float64)
    proxy = np.asarray([r["cv_demo_proxy"] for r in er], dtype=np.float64)
    audit = {
        "audit": "v64_3_47_future_state_supervision",
        "requested_tokens": len(tokens),
        "agent_supervised_scenes": agent_scenes,
        "agent_rows": len(ar),
        "ego_supervised_scenes": ego_scenes,
        "ego_candidate_rows": len(er),
        "lateral_target_mean_mps": float(np.mean(lat)),
        "lateral_target_rms_mps": float(np.sqrt(np.mean(lat * lat))),
        "demo_component_mean": float(np.mean(demo)),
        "cv_demo_proxy_mean": float(np.mean(proxy)),
        "deployment_uses_logged_future": False,
        "uses_teacher_total_or_improvement": False,
        "agent_target_uses_logged_agent_future_only": True,
        "ego_target_uses_logged_ego_future_demo_component_only": True,
    }
    Path(a.output_audit).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
