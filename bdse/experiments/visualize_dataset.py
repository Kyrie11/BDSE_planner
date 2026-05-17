from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bdse.config import load_config
from bdse.data.nuplan_dataset import PreprocessedBDSEDataset
from bdse.planner.evidence_atoms import hard_event_matrix


def _sample_metrics(s, cfg):
    valid = s.candidates.valid_mask.astype(bool)
    log_costs = np.linalg.norm(
        s.candidates.trajectories[:, :, :2] - s.label_future.logged_ego[None, :, :2], axis=-1
    ).mean(axis=1)
    log_costs = np.where(valid, log_costs, np.inf)
    log_nearest = int(np.argmin(log_costs)) if np.isfinite(log_costs).any() else -1
    hard_events = hard_event_matrix(s.evidence_bank.atoms, s.candidates, s.runtime, s.label_future, cfg)
    return {
        "token": s.scenario_token,
        "timestamp_us": int(s.timestamp_us),
        "valid_candidate_count": int(valid.sum()),
        "candidate_log_ade_min": float(log_costs[log_nearest]) if log_nearest >= 0 else float("nan"),
        "log_nearest": log_nearest,
        "teacher": int(s.teacher.a_star),
        "teacher_hard_violation": bool(s.teacher.hard_violation_mask[int(s.teacher.a_star)]),
        "safe_candidate_exists": bool(((~s.teacher.hard_violation_mask) & valid).any()),
        "pair_count": 0 if s.pairs is None else int(len(s.pairs.pairs)),
        "atom_count": int(len(s.evidence_bank.atoms)),
        "hard_event_atoms": int(hard_events[:, valid].any(axis=1).sum()) if valid.any() else 0,
    }


def _plot_sample(s, cfg, out_path: Path, title: str = ""):
    valid = s.candidates.valid_mask.astype(bool)
    a_star = int(s.teacher.a_star)
    log_costs = np.linalg.norm(
        s.candidates.trajectories[:, :, :2] - s.label_future.logged_ego[None, :, :2], axis=-1
    ).mean(axis=1)
    log_costs = np.where(valid, log_costs, np.inf)
    log_nearest = int(np.argmin(log_costs)) if np.isfinite(log_costs).any() else -1

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title or f"{s.scenario_token} @ {s.timestamp_us}")

    for poly in s.runtime.map_features.get("drivable_polygons", []):
        xy = np.asarray(poly.get("xy", []), dtype=np.float32).reshape(-1, 2)
        if len(xy) >= 3:
            closed = np.vstack([xy, xy[0]])
            ax.plot(closed[:, 0], closed[:, 1], linewidth=0.8, alpha=0.25)

    route = np.asarray(s.runtime.map_features.get("route_centerline", []), dtype=np.float32).reshape(-1, 2)
    if len(route) >= 2:
        ax.plot(route[:, 0], route[:, 1], linewidth=2.0, label="route")

    for i in np.flatnonzero(valid):
        tr = s.candidates.trajectories[i]
        lw = 1.0
        alpha = 0.35
        label = None
        if i == a_star:
            lw, alpha, label = 3.0, 0.95, "teacher"
        elif i == log_nearest:
            lw, alpha, label = 2.5, 0.85, "log-nearest"
        ax.plot(tr[:, 0], tr[:, 1], linewidth=lw, alpha=alpha, label=label)

    gt = s.label_future.logged_ego
    ax.plot(gt[:, 0], gt[:, 1], linestyle="--", linewidth=2.0, label="logged ego")

    for j, ok in enumerate(s.label_future.agent_valid.astype(bool)):
        if not ok:
            continue
        cur = s.runtime.current_agents[j]
        if np.linalg.norm(cur[:2]) > 90.0:
            continue
        ax.scatter([cur[0]], [cur[1]], s=10)
        fut = s.label_future.logged_agents[j]
        ax.plot(fut[:, 0], fut[:, 1], linewidth=0.8, alpha=0.4)

    for sl in s.runtime.map_features.get("stop_lines", []):
        xy = np.asarray(sl.get("xy", []), dtype=np.float32).reshape(-1, 2)
        if len(xy) >= 2:
            ax.plot(xy[:, 0], xy[:, 1], linewidth=2.0, alpha=0.7, label="stop line" if sl.get("red", False) else None)

    ax.scatter([0], [0], marker="x", s=80, label="ego now")
    ax.set_xlim(-30, 100)
    ax.set_ylim(-50, 50)
    ax.grid(True, linewidth=0.3)
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_summary(metrics: list[dict], out_path: Path):
    keys = ["valid_candidate_count", "candidate_log_ade_min", "pair_count", "atom_count", "hard_event_atoms"]
    for key in keys:
        vals = np.asarray([m[key] for m in metrics if np.isfinite(m[key])], dtype=np.float32)
        if not len(vals):
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(vals, bins=30)
        ax.set_title(key)
        ax.set_xlabel(key)
        ax.set_ylabel("samples")
        fig.tight_layout()
        fig.savefig(out_path / f"hist_{key}.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="bdse/configs/full_preprocess.yaml")
    parser.add_argument("--preprocessed-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--num-plots", type=int, default=12)
    parser.add_argument("--output-dir", type=str, default="outputs/dataset_viz")
    parser.add_argument("--rank-by", type=str, default="candidate_log_ade_min", choices=["candidate_log_ade_min", "pair_count", "valid_candidate_count", "hard_event_atoms"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = PreprocessedBDSEDataset(args.preprocessed_dir, split=args.split, max_scenarios=args.max_samples)
    metrics = []
    for i in range(len(ds)):
        s = ds[i]
        if s.teacher is None or s.label_future is None:
            continue
        m = _sample_metrics(s, cfg)
        m["index"] = i
        metrics.append(m)
    (out_dir / "sample_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _plot_summary(metrics, out_dir)

    reverse = args.rank_by != "valid_candidate_count"
    chosen = sorted(metrics, key=lambda m: (m[args.rank_by], m["index"]), reverse=reverse)[: args.num_plots]
    for rank, m in enumerate(chosen):
        s = ds[m["index"]]
        title = f"rank={rank} idx={m['index']} ADE={m['candidate_log_ade_min']:.2f} valid={m['valid_candidate_count']} pairs={m['pair_count']}"
        _plot_sample(s, cfg, out_dir / f"sample_{rank:02d}_idx{m['index']:05d}.png", title)
    print(json.dumps({"num_metrics": len(metrics), "output_dir": str(out_dir), "rank_by": args.rank_by}, indent=2))


if __name__ == "__main__":
    main()
