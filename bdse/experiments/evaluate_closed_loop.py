from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_CANONICAL_NUPLAN_CHALLENGES: dict[str, dict[str, str]] = {
    "closed_loop_nonreactive_agents": {
        "simulation": "closed_loop_nonreactive_agents",
        "simulation_metric": "simulation_closed_loop_nonreactive_agents",
        "metric_aggregator": "closed_loop_nonreactive_agents_weighted_average",
    },
    "closed_loop_reactive_agents": {
        "simulation": "closed_loop_reactive_agents",
        "simulation_metric": "simulation_closed_loop_reactive_agents",
        "metric_aggregator": "closed_loop_reactive_agents_weighted_average",
    },
    "open_loop_boxes": {
        "simulation": "open_loop_boxes",
        "simulation_metric": "simulation_open_loop_boxes",
        "metric_aggregator": "open_loop_boxes_weighted_average",
    },
}

_NUPLAN_OVERRIDE_ALIASES: dict[str, dict[str, str]] = {
    "simulation": {
        "closed_loop_nonreactive_agent": "closed_loop_nonreactive_agents",
        "closed_loop_nonreactive_agents": "closed_loop_nonreactive_agents",
        "simulation_closed_loop_nonreactive_agent": "closed_loop_nonreactive_agents",
        "simulation_closed_loop_nonreactive_agents": "closed_loop_nonreactive_agents",
        "closed_loop_reactive_agent": "closed_loop_reactive_agents",
        "closed_loop_reactive_agents": "closed_loop_reactive_agents",
        "simulation_closed_loop_reactive_agent": "closed_loop_reactive_agents",
        "simulation_closed_loop_reactive_agents": "closed_loop_reactive_agents",
        "open_loop_box": "open_loop_boxes",
        "open_loop_boxes": "open_loop_boxes",
        "simulation_open_loop_box": "open_loop_boxes",
        "simulation_open_loop_boxes": "open_loop_boxes",
    },
    "simulation_metric": {
        "closed_loop_nonreactive_agent": "simulation_closed_loop_nonreactive_agents",
        "closed_loop_nonreactive_agents": "simulation_closed_loop_nonreactive_agents",
        "simulation_closed_loop_nonreactive_agent": "simulation_closed_loop_nonreactive_agents",
        "simulation_closed_loop_nonreactive_agents": "simulation_closed_loop_nonreactive_agents",
        "closed_loop_reactive_agent": "simulation_closed_loop_reactive_agents",
        "closed_loop_reactive_agents": "simulation_closed_loop_reactive_agents",
        "simulation_closed_loop_reactive_agent": "simulation_closed_loop_reactive_agents",
        "simulation_closed_loop_reactive_agents": "simulation_closed_loop_reactive_agents",
        "open_loop_box": "simulation_open_loop_boxes",
        "open_loop_boxes": "simulation_open_loop_boxes",
        "simulation_open_loop_box": "simulation_open_loop_boxes",
        "simulation_open_loop_boxes": "simulation_open_loop_boxes",
    },
    "metric_aggregator": {
        "closed_loop_nonreactive_agent_weighted_average": "closed_loop_nonreactive_agents_weighted_average",
        "closed_loop_nonreactive_agents_weighted_average": "closed_loop_nonreactive_agents_weighted_average",
        "closed_loop_nonreactive_agent": "closed_loop_nonreactive_agents_weighted_average",
        "closed_loop_nonreactive_agents": "closed_loop_nonreactive_agents_weighted_average",
        "closed_loop_reactive_agent_weighted_average": "closed_loop_reactive_agents_weighted_average",
        "closed_loop_reactive_agents_weighted_average": "closed_loop_reactive_agents_weighted_average",
        "closed_loop_reactive_agent": "closed_loop_reactive_agents_weighted_average",
        "closed_loop_reactive_agents": "closed_loop_reactive_agents_weighted_average",
        "open_loop_box_weighted_average": "open_loop_boxes_weighted_average",
        "open_loop_boxes_weighted_average": "open_loop_boxes_weighted_average",
        "open_loop_box": "open_loop_boxes_weighted_average",
        "open_loop_boxes": "open_loop_boxes_weighted_average",
    },
}


def _strip_hydra_wrapping(value: str) -> str:
    value = str(value).strip()
    while len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
        value = value[1:-1].strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if "," not in inner:
            value = inner.strip()
    return value


def _canonical_override_value(group: str, value: str) -> str:
    raw = _strip_hydra_wrapping(value)
    return _NUPLAN_OVERRIDE_ALIASES.get(group, {}).get(raw, raw)


def _canonical_challenge_name(challenge: str) -> str:
    canonical = _canonical_override_value("simulation", challenge)
    if canonical not in _CANONICAL_NUPLAN_CHALLENGES:
        valid = ", ".join(sorted(_CANONICAL_NUPLAN_CHALLENGES))
        raise ValueError(f"Unknown nuPlan challenge '{challenge}'. Expected one of: {valid}")
    return canonical


def _challenge_scoped_output_dir(output_dir: str, challenge: str) -> str:
    """Ensure nuPlan metric aggregation can find challenge-specific metric files.

    nuPlan's metric aggregator filters parquet paths by ``metric_aggregator.challenge``.
    If users override ``output_dir`` to a generic path, the simulation can succeed
    but aggregation can report ``No metric files found`` because the challenge name
    is absent from every metric-file path.  Appending the challenge here preserves
    user control over the root directory while keeping official aggregation semantics.
    """
    out = Path(output_dir).expanduser()
    if challenge in str(out):
        return str(out)
    return str(out / challenge)


def _replace_simple_override(item: str, key: str, value: str) -> str:
    for prefix in ("++", "+", ""):
        marker = f"{prefix}{key}="
        if item.startswith(marker):
            return f"{prefix}{key}={value}"
    marker = f"/{key}="
    if item.startswith(marker):
        return f"/{key}={value}"
    return item


def _normalize_known_nuplan_overrides(overrides: list[str]) -> list[str]:
    """Normalize fragile nuPlan Hydra group names before launching Hydra.

    Several nuPlan groups use plural names (for example
    ``simulation_closed_loop_nonreactive_agents``).  A one-character typo such as
    ``..._agent`` fails during Hydra composition before BDSE is instantiated.
    This helper keeps user-supplied choices but canonicalizes known aliases and
    singular/plural variants for the official simulation, simulation_metric, and
    metric_aggregator groups.
    """
    normalized: list[str] = []
    for item in overrides:
        new_item = item
        matched = False
        for key in ("simulation", "simulation_metric", "metric_aggregator"):
            for prefix in ("++", "+", "", "/"):
                marker = f"{prefix}{key}="
                if item.startswith(marker):
                    value = item[len(marker):]
                    canonical = _canonical_override_value(key, value)
                    new_item = _replace_simple_override(item, key, canonical)
                    matched = True
                    break
            if matched:
                break
        normalized.append(new_item)
    return normalized


def _split_overrides(raw: list[str]) -> list[str]:
    if raw and raw[0] == "--":
        return raw[1:]
    return raw


def _append_if_missing(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _has_override(overrides: list[str], key: str) -> bool:
    """Return True when a Hydra override already sets or deletes ``key``.

    Examples matched for key="scenario_builder.data_root":
    - scenario_builder.data_root=/path
    - +scenario_builder.data_root=/path
    - ++scenario_builder.data_root=/path
    - ~scenario_builder.data_root
    """
    prefixes = (f"{key}=", f"+{key}=", f"++{key}=", f"~{key}")
    return any(item.startswith(prefixes) for item in overrides)


def _override_value(overrides: list[str], key: str) -> str | None:
    """Return the value assigned to a simple Hydra override, if present."""
    prefixes = (f"{key}=", f"+{key}=", f"++{key}=")
    for item in overrides:
        for prefix in prefixes:
            if item.startswith(prefix):
                return item[len(prefix):]
    return None


def _remove_override(overrides: list[str], key: str) -> list[str]:
    prefixes = (f"{key}=", f"+{key}=", f"++{key}=", f"~{key}")
    return [item for item in overrides if not item.startswith(prefixes)]


def _hydra_list(values: list[str]) -> str:
    """Format a simple Hydra list override for absolute DB file/dir paths."""
    return "[" + ",".join(str(Path(v).expanduser()) for v in values) + "]"


def _db_load_paths_from_root(root: str | Path) -> list[str]:
    """Return nuPlan-compatible DB load paths for a local root.

    nuPlan's ``discover_log_dbs`` can load a single ``.db`` file, a directory
    whose *immediate* children are ``*.db`` files, or a list of such files /
    directories. It does not recursively scan arbitrary split roots. BDSE
    validation caches are commonly laid out as::

        val_root/2021.06.07.11.59.52_veh-35/*.db
        val_root/2021.06.08.19.16.23_veh-26/*.db

    In that layout, passing ``val_root`` as ``scenario_builder.data_root``
    yields ``No log files found``. This helper rewrites the top-level root to a
    Hydra ``scenario_builder.db_files=[child_dir_1,child_dir_2,...]`` override,
    where each returned directory directly contains at least one ``.db`` file.
    """
    path = Path(root).expanduser()
    if path.suffix == ".db":
        return [str(path)] if path.is_file() else []
    if not path.is_dir():
        return []

    # Fast path: this directory itself is a valid nuPlan DB load directory.
    if any(path.glob("*.db")):
        return [str(path)]

    # Common BDSE split layout: val/<log_name>/*.db. Prefer direct children so
    # the generated command remains readable and deterministic.
    direct_child_db_dirs = sorted(
        [child for child in path.iterdir() if child.is_dir() and any(child.glob("*.db"))],
        key=lambda p: str(p),
    )
    if direct_child_db_dirs:
        return [str(p) for p in direct_child_db_dirs]

    # Fallback for one more unexpected nesting level. Return unique DB parent
    # directories; each is directly consumable by nuPlan.
    db_parent_dirs = sorted({p.parent for p in path.rglob("*.db")}, key=lambda p: str(p))
    return [str(p) for p in db_parent_dirs]


def _expand_db_load_paths(paths: list[str] | tuple[str, ...]) -> list[str]:
    """Expand user-provided DB files/roots to nuPlan-compatible load paths."""
    expanded: list[str] = []
    for item in paths:
        item_path = Path(item).expanduser()
        if item_path.exists():
            discovered = _db_load_paths_from_root(item_path)
            expanded.extend(discovered if discovered else [str(item_path)])
        else:
            expanded.append(str(item_path))

    # Preserve order while deduplicating.
    seen: set[str] = set()
    unique: list[str] = []
    for item in expanded:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _rewrite_local_db_root_overrides(overrides: list[str]) -> list[str]:
    """Rewrite local scenario_builder.data_root overrides to robust db_files lists.

    This is intentionally conservative: only existing local paths are rewritten.
    Remote paths, unresolved environment interpolations, or non-existing paths are
    left untouched so nuPlan/Hydra can handle them exactly as requested.
    """
    if _has_override(overrides, "scenario_builder.db_files"):
        return overrides
    data_root = _override_value(overrides, "scenario_builder.data_root")
    if not data_root:
        return overrides
    if data_root.startswith("s3://") or "${" in data_root:
        return overrides
    db_load_paths = _db_load_paths_from_root(data_root)
    if not db_load_paths:
        return overrides
    rewritten = _remove_override(overrides, "scenario_builder.data_root")
    rewritten.insert(0, f"scenario_builder.db_files={_hydra_list(db_load_paths)}")
    return rewritten


def _append_custom_db_safety_overrides(overrides: list[str]) -> list[str]:
    """Avoid hidden default log-name filters when evaluating custom DB subsets."""
    has_custom_db = _has_override(overrides, "scenario_builder.db_files") or _has_override(overrides, "scenario_builder.data_root")
    if has_custom_db and not _has_override(overrides, "scenario_filter.log_names"):
        overrides.append("scenario_filter.log_names=null")
    return overrides


def build_nuplan_command(args: argparse.Namespace, overrides: list[str]) -> tuple[list[str], list[str]]:
    env_overrides = [
        f"BDSE_CHECKPOINT={str(Path(args.checkpoint).resolve())}",
        f"BDSE_CONFIG={str(Path(args.config).resolve()) if args.config else ''}",
        f"BDSE_DEVICE={args.device}",
    ]
    if args.nuplan_data_root:
        env_overrides.append(f"NUPLAN_DATA_ROOT={str(Path(args.nuplan_data_root).expanduser())}")
    if args.nuplan_map_root:
        env_overrides.append(f"NUPLAN_MAPS_ROOT={str(Path(args.nuplan_map_root).expanduser())}")
    if args.nuplan_exp_root:
        env_overrides.append(f"NUPLAN_EXP_ROOT={str(Path(args.nuplan_exp_root).expanduser())}")
    if args.hydra_full_error:
        env_overrides.append("HYDRA_FULL_ERROR=1")

    challenge = _canonical_challenge_name(args.challenge)
    output_dir = _challenge_scoped_output_dir(args.output_dir, challenge)
    base = [
        sys.executable,
        "-m",
        args.nuplan_module,
        # Keep nuPlan's official shared config search paths and append BDSE's
        # planner config package. Passing only bdse.nuplan_config here overrides
        # the default Hydra searchpath and makes nuPlan unable to resolve groups
        # such as simulation_metric/default_metrics and metric_aggregator/*.
        "hydra.searchpath=[pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments,pkg://bdse.nuplan_config]",
        "planner=bdse_planner",
        # In some nuPlan-devkit/Hydra combinations, default_simulation.yaml is
        # already the root config and does not list a `simulation` defaults
        # group. Using `simulation=...` then fails with:
        #   Could not override 'simulation'. No match in the defaults list.
        # Appending the challenge-specific config is the compatible form.
        f"+simulation={challenge}",
        f"output_dir={output_dir}",
        f"experiment_uid={args.experiment_uid}",
    ]
    if args.scenario_builder:
        base.append(f"scenario_builder={args.scenario_builder}")
    if args.scenario_filter:
        base.append(f"scenario_filter={args.scenario_filter}")
    if args.worker:
        base.append(f"worker={args.worker}")
    if args.metric_aggregator:
        base.append(f"metric_aggregator={_canonical_override_value('metric_aggregator', args.metric_aggregator)}")

    final_overrides = _normalize_known_nuplan_overrides(list(overrides))

    # nuPlan's official scenario_builder=nuplan config derives its default DB
    # path from NUPLAN_DATA_ROOT as ``${NUPLAN_DATA_ROOT}/nuplan-v1.1/trainval``.
    # BDSE preprocessing commonly writes nuPlan-compatible DB caches elsewhere
    # (for example data/cache/bdse_val_v2/val).  Forward an explicit DB root or
    # DB file/dir list to ScenarioBuilder so closed-loop evaluation does not fall
    # back to a missing raw trainval directory.  User-supplied raw Hydra overrides
    # after ``--`` still take precedence.
    if getattr(args, "nuplan_db_files", None):
        if not _has_override(final_overrides, "scenario_builder.db_files"):
            db_load_paths = _expand_db_load_paths(args.nuplan_db_files)
            final_overrides.insert(0, f"scenario_builder.db_files={_hydra_list(db_load_paths)}")
    elif getattr(args, "nuplan_db_root", None):
        if not _has_override(final_overrides, "scenario_builder.data_root") and not _has_override(final_overrides, "scenario_builder.db_files"):
            root_path = Path(args.nuplan_db_root).expanduser()
            db_load_paths = _db_load_paths_from_root(root_path)
            if db_load_paths:
                final_overrides.insert(0, f"scenario_builder.db_files={_hydra_list(db_load_paths)}")
            elif root_path.exists():
                raise FileNotFoundError(
                    f"No .db files found under --nuplan-db-root={root_path}. "
                    "Closed-loop simulation needs nuPlan log DB files, not only tensor/cache artifacts."
                )
            else:
                final_overrides.insert(0, f"scenario_builder.data_root={str(root_path)}")

    # If a raw override such as scenario_builder.data_root=... is provided, make
    # it robust for nested split/city cache layouts before handing it to nuPlan.
    final_overrides = _rewrite_local_db_root_overrides(final_overrides)
    final_overrides = _append_custom_db_safety_overrides(final_overrides)

    # Keep nuPlan's splitter default by default. The official config resolves it
    # to a no-op YAML; if a local devkit checkout is incomplete, BDSE also ships
    # a no-op splitter/nuplan.yaml as a compatibility fallback.
    if args.disable_splitter and not any(x.startswith("splitter=") or x == "~splitter" for x in final_overrides):
        final_overrides.insert(0, "~splitter")
    return env_overrides, base + final_overrides


def _run_nuplan_command(cmd: list[str], env: dict[str, str], *, retry_without_splitter: bool) -> None:
    # Do not retry by assigning a YAML null to a defaults-list group: Hydra
    # treats group overrides as string/list selections. With nuPlan's common
    # config searchpath restored, splitter/nuplan resolves to the official no-op
    # YAML, so no fallback is needed.
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a nuPlan closed-loop simulation with BDSEPlanner through Hydra. "
            "Arguments after '--' are forwarded as raw nuPlan Hydra overrides."
        )
    )
    parser.add_argument("--config", type=str, default="bdse/configs/full_preprocess.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", help="Planner device passed to BDSEnuPlanPlanner. auto uses CUDA when available.")
    parser.add_argument("--challenge", type=str, default="closed_loop_nonreactive_agents")
    parser.add_argument("--output-dir", type=str, default="outputs/closed_loop_bdse")
    parser.add_argument("--experiment-uid", type=str, default="bdse_closed_loop")
    parser.add_argument("--nuplan-module", type=str, default="nuplan.planning.script.run_simulation")
    parser.add_argument("--scenario-builder", type=str, default="nuplan")
    parser.add_argument("--scenario-filter", type=str, default=None)
    parser.add_argument("--worker", type=str, default=None)
    parser.add_argument("--metric-aggregator", type=str, default=None)
    parser.add_argument("--nuplan-data-root", type=str, default=None, help="Optional NUPLAN_DATA_ROOT passed to nuPlan.")
    parser.add_argument("--nuplan-map-root", type=str, default=None, help="Optional NUPLAN_MAPS_ROOT passed to nuPlan.")
    parser.add_argument("--nuplan-exp-root", type=str, default=None, help="Optional NUPLAN_EXP_ROOT passed to nuPlan.")
    parser.add_argument(
        "--nuplan-db-root",
        type=str,
        default=None,
        help=(
            "Directory or .db file passed to nuPlan ScenarioBuilder. "
            "For nested split roots, BDSE expands it to scenario_builder.db_files automatically."
        ),
    )
    parser.add_argument(
        "--nuplan-db-files",
        nargs="+",
        default=None,
        help=(
            "One or more nuPlan .db files or directories passed as scenario_builder.db_files. "
            "Use this when the split is spread over multiple folders such as train_boston/train_pittsburgh."
        ),
    )
    parser.add_argument("--hydra-full-error", action="store_true", help="Set HYDRA_FULL_ERROR=1 for full Hydra stack traces.")
    parser.add_argument("--disable-splitter", dest="disable_splitter", action="store_true", default=False, help="Remove nuPlan's splitter default if your local Hydra setup explicitly requires it.")
    parser.add_argument("--keep-splitter", dest="disable_splitter", action="store_false", help="Keep nuPlan's splitter default. This is now the default.")
    parser.add_argument("--dry-run", action="store_true", help="Print the nuPlan command without executing it.")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.config and not Path(args.config).exists():
        raise FileNotFoundError(f"BDSE config not found: {args.config}")
    Path(_challenge_scoped_output_dir(args.output_dir, _canonical_challenge_name(args.challenge))).mkdir(parents=True, exist_ok=True)
    overrides = _split_overrides(args.overrides)
    env_assignments, cmd = build_nuplan_command(args, overrides)
    print(" ".join(env_assignments + cmd))
    if args.dry_run:
        return
    env = os.environ.copy()
    for item in env_assignments:
        k, v = item.split("=", 1)
        env[k] = v
    _run_nuplan_command(cmd, env, retry_without_splitter=bool(args.disable_splitter))


if __name__ == "__main__":
    main()
