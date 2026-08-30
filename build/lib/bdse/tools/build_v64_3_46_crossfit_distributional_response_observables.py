from __future__ import annotations

"""Build honest V46 DIRP distributional response/profile observables.

For outer fold k, both first- and second-moment response models exclude k and
(k+1)%5.  The second moment learns E[a^2|x, ego-plan] directly rather than
squaring an in-sample residual, so the outer-test distributional diagnostic is
honest.  Runtime sidecars are recomputed from current state/candidates only.
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.data.cache_schema import load_sample_npz
from bdse.data.nuplan_dataset import PreprocessedBDSEDataset
from bdse.planner.distributional_interaction_response import DIRP_OBSERVABLE_NAMES, runtime_distributional_interaction_response_observable_costs
from bdse.planner.interaction_response_field import RESPONSE_FIELD_LOCAL_FEATURE_NAMES, RESPONSE_FIELD_PLAN_FEATURE_NAMES
from bdse.tools.build_v64_3_45_crossfit_response_observables import (
    _fit as _fit_mean,
    _predict as _predict_mean,
    _read as _read_supervision,
    _scene_weights,
    _weighted_scale,
)
from bdse.tools.fit_v64_3_33_eaf_icer_spcr import FOLDS, RIDGE_LAMBDA, _fold

EPS = 1.0e-12


def _fit_ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, bias: bool) -> tuple[np.ndarray, np.ndarray, float]:
    scale = _weighted_scale(X, w)
    Z = X / scale[None, :]
    if bias:
        A = np.concatenate([Z, np.ones((len(Z), 1), dtype=np.float64)], axis=1)
    else:
        A = Z
    sw = np.sqrt(np.maximum(w, 0.0))
    Aw = A * sw[:, None]
    yw = y * sw
    reg = np.eye(A.shape[1], dtype=np.float64) * RIDGE_LAMBDA
    if bias:
        reg[-1, -1] = 0.0
    coef = np.linalg.solve(Aw.T @ Aw + reg, Aw.T @ yw)
    if bias:
        return scale, coef[:-1], float(coef[-1])
    return scale, coef, 0.0


def _fit_second_moment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    Xl = np.asarray([r["local_features"] for r in rows], dtype=np.float64)
    Xp = np.asarray([r["plan_features_logged_ego"] for r in rows], dtype=np.float64)
    y2 = np.asarray([float(r["target_longitudinal_accel_mps2"]) ** 2 for r in rows], dtype=np.float64)
    sw = _scene_weights(rows)
    ls, lw, lb = _fit_ridge(Xl, y2, sw, True)
    local = np.maximum(0.0, Xl / np.maximum(ls[None, :], 1.0e-6) @ lw + lb)
    resid = y2 - local
    exposure = np.maximum(np.asarray([r["logged_ego_interaction_exposure"] for r in rows], dtype=np.float64), 0.0)
    toks = [str(r["scenario_token"]) for r in rows]
    sums: dict[str, float] = {}
    for t, e in zip(toks, exposure):
        sums[t] = sums.get(t, 0.0) + float(e)
    pw_scene = np.asarray([
        sw[i] * (exposure[i] / max(sums[toks[i]], EPS) if sums[toks[i]] > EPS else 0.0)
        for i in range(len(rows))
    ], dtype=np.float64)
    ps, pw, _ = _fit_ridge(Xp, resid, pw_scene, False)
    const = float(np.sum(sw * y2) / max(float(sw.sum()), EPS))
    return {
        "enabled": True,
        "model": "agent_local_plan_conditioned_acceleration_second_moment",
        "lambda": RIDGE_LAMBDA,
        "local_feature_names": list(RESPONSE_FIELD_LOCAL_FEATURE_NAMES),
        "local_feature_scale": ls,
        "local_weights": lw,
        "local_bias": lb,
        "plan_enabled": True,
        "plan_feature_names": list(RESPONSE_FIELD_PLAN_FEATURE_NAMES),
        "plan_feature_scale": ps,
        "plan_weights": pw,
        "plan_bias": 0.0,
        "constant_second_moment_baseline": const,
    }


def _predict_second_moment_rows(rows: list[dict[str, Any]], m: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xl = np.asarray([r["local_features"] for r in rows], dtype=np.float64)
    Xp = np.asarray([r["plan_features_logged_ego"] for r in rows], dtype=np.float64)
    y2 = np.asarray([float(r["target_longitudinal_accel_mps2"]) ** 2 for r in rows], dtype=np.float64)
    local = np.maximum(0.0, Xl / np.maximum(m["local_feature_scale"][None, :], 1.0e-6) @ m["local_weights"] + m["local_bias"])
    plan = np.maximum(0.0, local + Xp / np.maximum(m["plan_feature_scale"][None, :], 1.0e-6) @ m["plan_weights"])
    const = np.full_like(y2, float(m["constant_second_moment_baseline"]))
    return y2, const, local, plan


def _scene_equal_mse(rows: list[dict[str, Any]], y: np.ndarray, p: np.ndarray) -> float:
    by: dict[str, list[float]] = {}
    for r, z in zip(rows, (y - p) ** 2):
        by.setdefault(str(r["scenario_token"]), []).append(float(z))
    return float(np.mean([np.mean(v) for v in by.values()])) if by else float("nan")


def _serial(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _serial(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_serial(v) for v in x]
    return x


def _cfg_with_models(base: dict[str, Any], mean: dict[str, Any], m2: dict[str, Any]) -> dict[str, Any]:
    c = copy.deepcopy(base)
    ic = c.setdefault("runtime", {}).setdefault("decisive_frontier_value", {}).setdefault("incumbent_contrastive_extremal_recovery", {})
    ic["instrument_value_observables"] = True
    ic["instrument_interaction_response_field_observables"] = True
    ic["instrument_distributional_interaction_response_observables"] = True
    sc = ic.setdefault("selection_conditioned_intervention_recovery", {})
    sc["interaction_response_field"] = _serial(mean)
    sc["distributional_interaction_response_field"] = _serial(m2)
    return c


def _read_v45_sidecar(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        c = np.asarray(r["costs"], dtype=np.float64)
        if c.ndim != 2 or c.shape[1] != 3:
            raise ValueError("V46 prerequisite V45 sidecar shape mismatch")
        out[str(r["scenario_token"])] = c[:, 2]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--supervision", required=True)
    ap.add_argument("--v45-sidecar", required=True)
    ap.add_argument("--v45-response-report", required=True)
    ap.add_argument("--preprocessed-dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--scenario-token-file", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--output-sidecar", required=True)
    ap.add_argument("--output-model", required=True)
    ap.add_argument("--output-report", required=True)
    a = ap.parse_args()

    rows = _read_supervision(Path(a.supervision))
    v45_side = _read_v45_sidecar(Path(a.v45_sidecar))
    v45_report = json.loads(Path(a.v45_response_report).read_text())
    exp = v45_report.get("aggregate", {})
    expected = (0.30137842796229286, 0.12486025654085724, 0.12385468573917016)
    got = (float(exp.get("cv_mse", -1)), float(exp.get("local_mse", -1)), float(exp.get("plan_mse", -1)))
    if any(abs(g - e) > 1.0e-12 for g, e in zip(got, expected)) or not bool(v45_report.get("plan_response_identified", False)):
        raise SystemExit(f"STOP V46 prerequisite: V45 response signature changed {got}")

    base = yaml.safe_load(Path(a.base_config).read_text())
    tokens = [x.strip() for x in Path(a.scenario_token_file).read_text().splitlines() if x.strip()]
    want = set(tokens)
    if len(tokens) != 3000 or len(want) != 3000:
        raise SystemExit("STOP V46 DATA: frozen TRAIN token set changed")

    byfold = {k: [] for k in range(FOLDS)}
    for r in rows:
        t = str(r["scenario_token"])
        if t in want:
            byfold[_fold(t)].append(r)

    mean_models: dict[int, dict[str, Any]] = {}
    m2_models: dict[int, dict[str, Any]] = {}
    fold_diag: list[dict[str, Any]] = []
    for k in range(FOLDS):
        cf = (k + 1) % FOLDS
        fit = [r for r in rows if str(r["scenario_token"]) in want and _fold(str(r["scenario_token"])) not in {k, cf}]
        test = byfold[k]
        mm = _fit_mean(fit)
        sm = _fit_second_moment(fit)
        mean_models[k] = mm
        m2_models[k] = sm
        y2, c2, l2, p2 = _predict_second_moment_rows(test, sm)
        cm = _scene_equal_mse(test, y2, c2)
        lm = _scene_equal_mse(test, y2, l2)
        pm = _scene_equal_mse(test, y2, p2)
        fold_diag.append({
            "fold": k,
            "fit_agent_rows": len(fit),
            "test_agent_rows": len(test),
            "test_scenes": len(set(str(r["scenario_token"]) for r in test)),
            "constant_second_moment_mse": cm,
            "local_second_moment_mse": lm,
            "plan_second_moment_mse": pm,
            "local_second_moment_better_than_constant": bool(lm < cm),
            "plan_second_moment_better_than_local": bool(pm < lm),
        })

    full_mean = _fit_mean([r for r in rows if str(r["scenario_token"]) in want])
    full_m2 = _fit_second_moment([r for r in rows if str(r["scenario_token"]) in want])

    ds = PreprocessedBDSEDataset(a.preprocessed_dir, split=a.split, scenario_tokens=want)
    paths = ds.build_index()
    bytok: dict[str, Path] = {}
    for p in paths:
        try:
            with np.load(p, allow_pickle=True) as z:
                tok = str(z["scenario_token"].item() if z["scenario_token"].shape == () else z["scenario_token"].reshape(-1)[0])
        except Exception:
            continue
        if tok in want:
            bytok[tok] = Path(p)
    miss = want - set(bytok)
    if miss:
        raise SystemExit(f"STOP V46 DATA: current-state cache missing {len(miss)} frozen tokens")

    out = Path(a.output_sidecar)
    out.parent.mkdir(parents=True, exist_ok=True)
    replay_max = 0.0
    with out.open("w") as f:
        for tok in tokens:
            s = load_sample_npz(bytok[tok], include_label_future=False, include_candidate_metadata=False, include_evidence_aux_metadata=False)
            cfg = _cfg_with_models(base, mean_models[_fold(tok)], m2_models[_fold(tok)])
            cost, names = runtime_distributional_interaction_response_observable_costs(s.runtime, s.candidates, cfg)
            if names != DIRP_OBSERVABLE_NAMES or cost.shape != (s.candidates.K, len(DIRP_OBSERVABLE_NAMES)):
                raise SystemExit(f"STOP V46 DATA: DIRP cost schema malformed for {tok}")
            if tok not in v45_side or v45_side[tok].shape[0] != s.candidates.K:
                raise SystemExit(f"STOP V46 prerequisite: V45 sidecar missing/mismatched {tok}")
            diff = float(np.max(np.abs(cost[:, 0] - v45_side[tok]))) if s.candidates.K else 0.0
            replay_max = max(replay_max, diff)
            if diff > 1.0e-10:
                raise SystemExit(f"STOP V46 ENGINEERING: V45 PLAN occupancy replay drift {tok}: {diff}")
            f.write(json.dumps({"scenario_token": tok, "outer_fold": _fold(tok), "observable_names": names, "costs": cost.tolist()}, sort_keys=True) + "\n")

    agg = {
        "constant_second_moment_mse": float(np.mean([x["constant_second_moment_mse"] for x in fold_diag])),
        "local_second_moment_mse": float(np.mean([x["local_second_moment_mse"] for x in fold_diag])),
        "plan_second_moment_mse": float(np.mean([x["plan_second_moment_mse"] for x in fold_diag])),
    }
    local_count = int(sum(x["local_second_moment_better_than_constant"] for x in fold_diag))
    plan_count = int(sum(x["plan_second_moment_better_than_local"] for x in fold_diag))
    report = {
        "audit": "v64_3_46_crossfit_distributional_response_field",
        "folds": fold_diag,
        "aggregate": agg,
        "local_second_moment_better_fold_count": local_count,
        "plan_second_moment_better_fold_count": plan_count,
        "local_second_moment_identified": bool(agg["local_second_moment_mse"] < agg["constant_second_moment_mse"] and local_count >= 4),
        "plan_second_moment_identified": bool(agg["plan_second_moment_mse"] < agg["local_second_moment_mse"] and plan_count >= 4),
        "v45_plan_occupancy_exact_replay_max_abs": replay_max,
        "deployment_uses_logged_future": False,
        "quadrature": {"offsets": [-float(np.sqrt(3.0)), 0.0, float(np.sqrt(3.0))], "weights": [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]},
        "full_train_mean_model": _serial(full_mean),
        "full_train_second_moment_model": _serial(full_m2),
    }
    Path(a.output_model).write_text(json.dumps({"mean_response_model": _serial(full_mean), "second_moment_model": _serial(full_m2)}, indent=2, sort_keys=True))
    Path(a.output_report).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
