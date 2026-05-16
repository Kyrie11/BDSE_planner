from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
        route_roadblock_ids=np.asarray(sample.runtime.route_roadblock_ids, dtype=object),
        mission_goal=np.asarray([] if sample.runtime.mission_goal is None else sample.runtime.mission_goal),
        candidate_trajectories=sample.candidates.trajectories,
        candidate_valid=sample.candidates.valid_mask,
        candidate_maneuver_ids=sample.candidates.maneuver_ids,
        evidence_query_features=sample.evidence_bank.query_features,
        evidence_active=sample.evidence_bank.active_mask,
        evidence_types=np.asarray([a.type for a in sample.evidence_bank.atoms], dtype=object),
        evidence_families=np.asarray([a.family for a in sample.evidence_bank.atoms], dtype=object),
        evidence_is_hard=np.asarray([a.is_hard for a in sample.evidence_bank.atoms], dtype=bool),
        evidence_budget_costs=sample.evidence_bank.budget_costs(),
        teacher_J_base=np.asarray([] if sample.teacher is None else sample.teacher.J_base),
        teacher_g_evid=np.asarray([] if sample.teacher is None else sample.teacher.g_evid),
        teacher_J_evid=np.asarray([] if sample.teacher is None else sample.teacher.J_evid),
        teacher_J_T=np.asarray([] if sample.teacher is None else sample.teacher.J_T),
        teacher_a_star=np.asarray(-1 if sample.teacher is None else sample.teacher.a_star, dtype=np.int64),
        teacher_hard_violation=np.asarray([] if sample.teacher is None else sample.teacher.hard_violation_mask),
        pair_indices=np.asarray([] if sample.pairs is None else sample.pairs.pairs),
        pair_margins=np.asarray([] if sample.pairs is None else sample.pairs.margins),
        pair_weights=np.asarray([] if sample.pairs is None else sample.pairs.weights),
        pair_residuals=np.asarray([] if sample.pairs is None else sample.pairs.residuals),
        pair_valid=np.asarray([] if sample.pairs is None else sample.pairs.valid_mask),
    )


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
