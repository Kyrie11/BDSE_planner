from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class RuntimeFeatures:
    ego_history: np.ndarray
    agent_history: np.ndarray
    agent_valid: np.ndarray
    current_agents: np.ndarray
    traffic_lights: list[dict[str, Any]]
    map_features: dict[str, Any]
    route_roadblock_ids: list[str]
    mission_goal: np.ndarray | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def runtime_only_dict(self) -> dict[str, Any]:
        return {
            "ego_history": self.ego_history,
            "agent_history": self.agent_history,
            "agent_valid": self.agent_valid,
            "current_agents": self.current_agents,
            "traffic_lights": self.traffic_lights,
            "map_features": self.map_features,
            "route_roadblock_ids": self.route_roadblock_ids,
            "mission_goal": self.mission_goal,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class LabelOnlyFuture:
    logged_ego: np.ndarray
    logged_agents: np.ndarray
    agent_valid: np.ndarray
    future_traffic_lights: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CandidateBank:
    trajectories: np.ndarray
    valid_mask: np.ndarray
    maneuver_ids: np.ndarray
    theta: list[dict[str, Any]]
    dynamic_flags: list[dict[str, bool]]
    metadata: list[dict[str, Any]]

    @property
    def K(self) -> int:
        return int(self.trajectories.shape[0])

    @property
    def T(self) -> int:
        return int(self.trajectories.shape[1])

    def valid_indices(self) -> np.ndarray:
        return np.flatnonzero(self.valid_mask.astype(bool))


@dataclass(slots=True)
class EvidenceAtom:
    atom_id: int
    type: str
    anchor: dict[str, Any]
    budget_cost: float
    is_hard: bool
    family: str
    active_mask: bool


@dataclass(slots=True)
class EvidenceBank:
    atoms: list[EvidenceAtom]
    query_features: np.ndarray
    active_mask: np.ndarray

    @property
    def E(self) -> int:
        return len(self.atoms)

    def budget_costs(self) -> np.ndarray:
        return np.asarray([a.budget_cost for a in self.atoms], dtype=np.float32)

    def hard_mask(self) -> np.ndarray:
        return np.asarray([a.is_hard for a in self.atoms], dtype=bool)

    def family_ids(self) -> list[str]:
        return [a.family for a in self.atoms]


@dataclass(slots=True)
class TeacherLabels:
    J_base: np.ndarray
    g_evid: np.ndarray
    J_evid: np.ndarray
    J_T: np.ndarray
    a_star: int
    hard_violation_mask: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def validate_partition(self, valid_mask: np.ndarray | None = None, atol: float = 1e-5) -> None:
        if self.g_evid.ndim != 2:
            raise AssertionError("g_evid must have shape [E,K]")
        evid_sum = self.g_evid.sum(axis=0)
        mask = np.ones_like(self.J_T, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
        if not np.allclose(self.J_evid[mask], evid_sum[mask], atol=atol):
            raise AssertionError("J_evid must equal atom-level sum_i g_i_T")
        if not np.allclose(self.J_T[mask], self.J_base[mask] + self.J_evid[mask], atol=atol):
            raise AssertionError("J_T must equal J_base_T + J_evid_T")


@dataclass(slots=True)
class PairLabels:
    pairs: np.ndarray
    margins: np.ndarray
    weights: np.ndarray
    residuals: np.ndarray
    valid_mask: np.ndarray

    def validate_positive_direction(self) -> None:
        if len(self.pairs) and np.any(self.margins[self.valid_mask] <= 0):
            raise AssertionError("Every valid pair must store the better action first with positive M_T(a,b)")


@dataclass(slots=True)
class Sample:
    scenario_token: str
    timestamp_us: int
    runtime: RuntimeFeatures
    label_future: LabelOnlyFuture | None
    candidates: CandidateBank
    evidence_bank: EvidenceBank
    teacher: TeacherLabels | None
    pairs: PairLabels | None


def pad_array(arr: np.ndarray, shape: tuple[int, ...], value: float = 0.0, dtype=np.float32) -> np.ndarray:
    out = np.full(shape, value, dtype=dtype)
    arr = np.asarray(arr, dtype=dtype)
    slices = tuple(slice(0, min(a, b)) for a, b in zip(arr.shape, shape))
    out[slices] = arr[slices]
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"__ndarray__": True, "dtype": str(value.dtype), "shape": list(value.shape), "data": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _from_jsonable(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__ndarray__"):
        arr = np.asarray(value.get("data", []), dtype=np.dtype(value.get("dtype", "float32")))
        shape = tuple(int(x) for x in value.get("shape", arr.shape))
        try:
            return arr.reshape(shape)
        except ValueError:
            return arr
    if isinstance(value, dict):
        return {k: _from_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_jsonable(v) for v in value]
    return value


def _json_dumps(value: Any) -> np.ndarray:
    return np.asarray(json.dumps(_jsonable(value), separators=(",", ":")))


def _json_loads_npz(data: Any, key: str, default: Any) -> Any:
    if key not in data.files:
        return default
    raw = data[key]
    try:
        text = str(raw.item()) if raw.shape == () else str(raw.tolist())
        return _from_jsonable(json.loads(text))
    except Exception:
        return default


def _string_list(arr: np.ndarray) -> list[str]:
    if arr.size == 0:
        return []
    return [str(x) for x in arr.reshape(-1).tolist()]


def save_sample_npz(sample: Sample, path: str | Path, compressed: bool = False) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    save_fn = np.savez_compressed if compressed else np.savez
    save_fn(
        p,
        scenario_token=np.asarray(sample.scenario_token),
        timestamp_us=np.asarray(sample.timestamp_us, dtype=np.int64),
        runtime_ego_history=sample.runtime.ego_history,
        runtime_agent_history=sample.runtime.agent_history,
        runtime_agent_valid=sample.runtime.agent_valid,
        runtime_current_agents=sample.runtime.current_agents,
        runtime_traffic_lights_json=_json_dumps(sample.runtime.traffic_lights),
        runtime_map_features_json=_json_dumps(sample.runtime.map_features),
        runtime_metadata_json=_json_dumps(sample.runtime.metadata),
        route_roadblock_ids=np.asarray(sample.runtime.route_roadblock_ids, dtype=str),
        mission_goal=np.asarray([] if sample.runtime.mission_goal is None else sample.runtime.mission_goal),
        label_logged_ego=np.asarray([] if sample.label_future is None else sample.label_future.logged_ego),
        label_logged_agents=np.asarray([] if sample.label_future is None else sample.label_future.logged_agents),
        label_agent_valid=np.asarray([] if sample.label_future is None else sample.label_future.agent_valid),
        label_future_traffic_lights_json=_json_dumps([] if sample.label_future is None else sample.label_future.future_traffic_lights),
        label_future_metadata_json=_json_dumps({} if sample.label_future is None else sample.label_future.metadata),
        candidate_trajectories=sample.candidates.trajectories,
        candidate_valid=sample.candidates.valid_mask,
        candidate_maneuver_ids=sample.candidates.maneuver_ids,
        candidate_theta_json=_json_dumps(sample.candidates.theta),
        candidate_dynamic_flags_json=_json_dumps(sample.candidates.dynamic_flags),
        candidate_metadata_json=_json_dumps(sample.candidates.metadata),
        evidence_query_features=sample.evidence_bank.query_features,
        evidence_active=sample.evidence_bank.active_mask,
        evidence_types=np.asarray([a.type for a in sample.evidence_bank.atoms], dtype=str),
        evidence_families=np.asarray([a.family for a in sample.evidence_bank.atoms], dtype=str),
        evidence_is_hard=np.asarray([a.is_hard for a in sample.evidence_bank.atoms], dtype=bool),
        evidence_budget_costs=sample.evidence_bank.budget_costs(),
        evidence_anchors_json=_json_dumps([a.anchor for a in sample.evidence_bank.atoms]),
        teacher_J_base=np.asarray([] if sample.teacher is None else sample.teacher.J_base),
        teacher_g_evid=np.asarray([] if sample.teacher is None else sample.teacher.g_evid),
        teacher_J_evid=np.asarray([] if sample.teacher is None else sample.teacher.J_evid),
        teacher_J_T=np.asarray([] if sample.teacher is None else sample.teacher.J_T),
        teacher_a_star=np.asarray(-1 if sample.teacher is None else sample.teacher.a_star, dtype=np.int64),
        teacher_hard_violation=np.asarray([] if sample.teacher is None else sample.teacher.hard_violation_mask),
        teacher_diagnostics_json=_json_dumps({} if sample.teacher is None else sample.teacher.diagnostics),
        pair_indices=np.asarray([] if sample.pairs is None else sample.pairs.pairs),
        pair_margins=np.asarray([] if sample.pairs is None else sample.pairs.margins),
        pair_weights=np.asarray([] if sample.pairs is None else sample.pairs.weights),
        pair_residuals=np.asarray([] if sample.pairs is None else sample.pairs.residuals),
        pair_valid=np.asarray([] if sample.pairs is None else sample.pairs.valid_mask),
    )


def load_sample_npz(path: str | Path) -> Sample:
    p = Path(path)
    with np.load(p, allow_pickle=True) as z:
        scenario_token = str(z["scenario_token"].item() if z["scenario_token"].shape == () else z["scenario_token"].reshape(-1)[0])
        timestamp_us = int(z["timestamp_us"].item())
        mission_goal = np.asarray(z["mission_goal"], dtype=np.float32)
        if mission_goal.size == 0:
            mission_goal_val = None
        else:
            mission_goal_val = mission_goal.astype(np.float32)
        map_features = _json_loads_npz(z, "runtime_map_features_json", {})
        if not map_features:
            map_features = {"route_centerline": np.array([[0.0, 0.0], [160.0, 0.0]], dtype=np.float32), "route_corridor_width": 4.0, "map_valid": False}
        runtime = RuntimeFeatures(
            ego_history=np.asarray(z["runtime_ego_history"], dtype=np.float32),
            agent_history=np.asarray(z["runtime_agent_history"], dtype=np.float32),
            agent_valid=np.asarray(z["runtime_agent_valid"], dtype=bool),
            current_agents=np.asarray(z["runtime_current_agents"], dtype=np.float32),
            traffic_lights=_json_loads_npz(z, "runtime_traffic_lights_json", []),
            map_features=map_features,
            route_roadblock_ids=_string_list(np.asarray(z["route_roadblock_ids"])),
            mission_goal=mission_goal_val,
            metadata=_json_loads_npz(z, "runtime_metadata_json", {}),
        )
        if "label_logged_ego" in z.files and np.asarray(z["label_logged_ego"]).size:
            label_future = LabelOnlyFuture(
                logged_ego=np.asarray(z["label_logged_ego"], dtype=np.float32),
                logged_agents=np.asarray(z["label_logged_agents"], dtype=np.float32),
                agent_valid=np.asarray(z["label_agent_valid"], dtype=bool),
                future_traffic_lights=_json_loads_npz(z, "label_future_traffic_lights_json", []),
                metadata=_json_loads_npz(z, "label_future_metadata_json", {}),
            )
        else:
            label_future = None
        trajectories = np.asarray(z["candidate_trajectories"], dtype=np.float32)
        K = int(trajectories.shape[0])
        candidates = CandidateBank(
            trajectories=trajectories,
            valid_mask=np.asarray(z["candidate_valid"], dtype=bool),
            maneuver_ids=np.asarray(z["candidate_maneuver_ids"], dtype=np.int64),
            theta=_json_loads_npz(z, "candidate_theta_json", [{} for _ in range(K)]),
            dynamic_flags=_json_loads_npz(z, "candidate_dynamic_flags_json", [{} for _ in range(K)]),
            metadata=_json_loads_npz(z, "candidate_metadata_json", [{} for _ in range(K)]),
        )
        types = _string_list(np.asarray(z["evidence_types"]))
        families = _string_list(np.asarray(z["evidence_families"]))
        hard = np.asarray(z["evidence_is_hard"], dtype=bool)
        costs = np.asarray(z["evidence_budget_costs"], dtype=np.float32)
        active = np.asarray(z["evidence_active"], dtype=bool)
        anchors = _json_loads_npz(z, "evidence_anchors_json", [{} for _ in types])
        if not isinstance(anchors, list) or len(anchors) != len(types):
            anchors = [{} for _ in types]
        atoms = [
            EvidenceAtom(int(i), str(types[i]), dict(anchors[i] or {}), float(costs[i]), bool(hard[i]), str(families[i]), bool(active[i]))
            for i in range(len(types))
        ]
        evidence_bank = EvidenceBank(atoms=atoms, query_features=np.asarray(z["evidence_query_features"], dtype=np.float32), active_mask=active)
        if np.asarray(z["teacher_J_T"]).size:
            teacher = TeacherLabels(
                J_base=np.asarray(z["teacher_J_base"], dtype=np.float64),
                g_evid=np.asarray(z["teacher_g_evid"], dtype=np.float32),
                J_evid=np.asarray(z["teacher_J_evid"], dtype=np.float64),
                J_T=np.asarray(z["teacher_J_T"], dtype=np.float64),
                a_star=int(np.asarray(z["teacher_a_star"]).item()),
                hard_violation_mask=np.asarray(z["teacher_hard_violation"], dtype=bool),
                diagnostics=_json_loads_npz(z, "teacher_diagnostics_json", {}),
            )
        else:
            teacher = None
        pair_indices = np.asarray(z["pair_indices"], dtype=np.int64)
        if pair_indices.size:
            pair_indices = pair_indices.reshape(-1, 2)
            pairs = PairLabels(
                pairs=pair_indices,
                margins=np.asarray(z["pair_margins"], dtype=np.float32),
                weights=np.asarray(z["pair_weights"], dtype=np.float32),
                residuals=np.asarray(z["pair_residuals"], dtype=np.float32),
                valid_mask=np.asarray(z["pair_valid"], dtype=bool),
            )
        else:
            pairs = None
    return Sample(scenario_token, timestamp_us, runtime, label_future, candidates, evidence_bank, teacher, pairs)


def batch_samples(samples: list[Sample]) -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError("Cannot batch an empty sample list")
    return {
        "ego_history": np.stack([s.runtime.ego_history for s in samples]),
        "agent_history": np.stack([s.runtime.agent_history for s in samples]),
        "agent_valid": np.stack([s.runtime.agent_valid for s in samples]),
        "current_agents": np.stack([s.runtime.current_agents for s in samples]),
        "candidate_trajectories": np.stack([s.candidates.trajectories for s in samples]),
        "candidate_valid": np.stack([s.candidates.valid_mask for s in samples]),
        "candidate_maneuver_ids": np.stack([s.candidates.maneuver_ids for s in samples]),
        "evidence_query_features": np.stack([s.evidence_bank.query_features for s in samples]),
        "evidence_active": np.stack([s.evidence_bank.active_mask for s in samples]),
        "teacher_J_base": np.stack([s.teacher.J_base for s in samples if s.teacher is not None]),
        "teacher_g_evid": np.stack([s.teacher.g_evid for s in samples if s.teacher is not None]),
        "teacher_J_T": np.stack([s.teacher.J_T for s in samples if s.teacher is not None]),
    }
