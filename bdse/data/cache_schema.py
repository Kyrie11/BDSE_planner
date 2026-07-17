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
    # New paper-aligned local-certificate metadata.  Defaults keep old caches/tests
    # readable while allowing atoms to store (tau,r,phi,Omega,D,lambda,c).
    validity_domain: dict[str, Any] = field(default_factory=dict)
    response_modes: list[str] = field(default_factory=list)
    aggregator: str = "mean"
    lambda_weight: float = 1.0
    cheap_features: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class EvidenceBank:
    atoms: list[EvidenceAtom]
    query_features: np.ndarray
    active_mask: np.ndarray
    proposal_features: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.proposal_features is None:
            self.proposal_features = np.zeros((len(self.atoms), 16), dtype=np.float32)
        else:
            self.proposal_features = np.asarray(self.proposal_features, dtype=np.float32)

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

    def __post_init__(self) -> None:
        self.pairs = np.asarray(self.pairs, dtype=np.int64).reshape(-1, 2)
        n = int(self.pairs.shape[0])
        self.margins = np.asarray(self.margins, dtype=np.float32).reshape(-1)
        self.weights = np.asarray(self.weights, dtype=np.float32).reshape(-1)
        self.residuals = np.asarray(self.residuals, dtype=np.float32).reshape(-1)
        self.valid_mask = np.asarray(self.valid_mask, dtype=bool).reshape(-1)
        for name in ("margins", "weights", "residuals", "valid_mask"):
            arr = getattr(self, name)
            if arr.shape[0] != n:
                raise ValueError(f"PairLabels.{name} must have length {n}, got {arr.shape[0]}")

    def validate_positive_direction(self, atol: float = 1e-6) -> None:
        if self.pairs.ndim != 2 or self.pairs.shape[1] != 2:
            raise AssertionError("pairs must have shape [P,2]")
        valid = np.asarray(self.valid_mask, dtype=bool)
        if valid.any() and np.any(self.margins[valid] <= -float(atol)):
            bad = int(np.flatnonzero(valid & (self.margins <= -float(atol)))[0])
            raise AssertionError(f"Pair margin must be non-negative for valid pairs; pair {bad} has margin {self.margins[bad]}")
        if valid.any() and np.any(self.weights[valid] < 0.0):
            raise AssertionError("Pair weights must be non-negative")


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
        evidence_proposal_features=np.asarray(sample.evidence_bank.proposal_features if sample.evidence_bank.proposal_features is not None else np.zeros((sample.evidence_bank.E, 16), dtype=np.float32)),
        evidence_active=sample.evidence_bank.active_mask,
        evidence_types=np.asarray([a.type for a in sample.evidence_bank.atoms], dtype=str),
        evidence_families=np.asarray([a.family for a in sample.evidence_bank.atoms], dtype=str),
        evidence_is_hard=np.asarray([a.is_hard for a in sample.evidence_bank.atoms], dtype=bool),
        evidence_budget_costs=sample.evidence_bank.budget_costs(),
        evidence_anchors_json=_json_dumps([a.anchor for a in sample.evidence_bank.atoms]),
        evidence_validity_domains_json=_json_dumps([a.validity_domain for a in sample.evidence_bank.atoms]),
        evidence_response_modes_json=_json_dumps([a.response_modes for a in sample.evidence_bank.atoms]),
        evidence_aggregators=np.asarray([a.aggregator for a in sample.evidence_bank.atoms], dtype=str),
        evidence_lambda_weights=np.asarray([a.lambda_weight for a in sample.evidence_bank.atoms], dtype=np.float32),
        evidence_cheap_features_json=_json_dumps([a.cheap_features for a in sample.evidence_bank.atoms]),
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


def load_sample_npz(
    path: str | Path,
    *,
    include_label_future: bool = True,
    include_candidate_metadata: bool = True,
    include_runtime_metadata: bool = True,
    include_route_ids: bool = True,
    include_evidence_aux_metadata: bool = True,
    allow_pickle: bool = True,
) -> Sample:
    p = Path(path)
    with np.load(p, allow_pickle=allow_pickle) as z:
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
            route_roadblock_ids=_string_list(np.asarray(z["route_roadblock_ids"])) if include_route_ids else [],
            mission_goal=mission_goal_val,
            metadata=_json_loads_npz(z, "runtime_metadata_json", {}) if include_runtime_metadata else {},
        )
        if include_label_future and "label_logged_ego" in z.files and np.asarray(z["label_logged_ego"]).size:
            label_future = LabelOnlyFuture(
                logged_ego=np.asarray(z["label_logged_ego"], dtype=np.float32),
                logged_agents=np.asarray(z["label_logged_agents"], dtype=np.float32),
                agent_valid=np.asarray(z["label_agent_valid"], dtype=bool),
                future_traffic_lights=_json_loads_npz(z, "label_future_traffic_lights_json", []),
                metadata=_json_loads_npz(z, "label_future_metadata_json", {}),
            )
        else:
            # Training and runtime tensorization do not use label-only futures.
            # Skipping these arrays avoids copying large logged-agent future tensors
            # through DataLoader workers while preserving the default full loader.
            label_future = None
        trajectories = np.asarray(z["candidate_trajectories"], dtype=np.float32)
        K = int(trajectories.shape[0])
        if include_candidate_metadata:
            theta = _json_loads_npz(z, "candidate_theta_json", [{} for _ in range(K)])
            dynamic_flags = _json_loads_npz(z, "candidate_dynamic_flags_json", [{} for _ in range(K)])
            candidate_metadata = _json_loads_npz(z, "candidate_metadata_json", [{} for _ in range(K)])
        else:
            # The training tensorizer only needs trajectories, validity, and maneuver ids.
            theta = [{} for _ in range(K)]
            dynamic_flags = [{} for _ in range(K)]
            candidate_metadata = [{} for _ in range(K)]
        candidates = CandidateBank(
            trajectories=trajectories,
            valid_mask=np.asarray(z["candidate_valid"], dtype=bool),
            maneuver_ids=np.asarray(z["candidate_maneuver_ids"], dtype=np.int64),
            theta=theta,
            dynamic_flags=dynamic_flags,
            metadata=candidate_metadata,
        )
        types = _string_list(np.asarray(z["evidence_types"]))
        families = _string_list(np.asarray(z["evidence_families"]))
        hard = np.asarray(z["evidence_is_hard"], dtype=bool)
        costs = np.asarray(z["evidence_budget_costs"], dtype=np.float32)
        active = np.asarray(z["evidence_active"], dtype=bool)
        anchors = _json_loads_npz(z, "evidence_anchors_json", [{} for _ in types])
        if not isinstance(anchors, list) or len(anchors) != len(types):
            anchors = [{} for _ in types]
        if include_evidence_aux_metadata:
            domains = _json_loads_npz(z, "evidence_validity_domains_json", [{} for _ in types])
            modes = _json_loads_npz(z, "evidence_response_modes_json", [[] for _ in types])
            aggs = _string_list(np.asarray(z["evidence_aggregators"])) if "evidence_aggregators" in z.files else ["mean" for _ in types]
        else:
            domains = [{} for _ in types]
            modes = [[] for _ in types]
            aggs = ["mean" for _ in types]
        cheap = _json_loads_npz(z, "evidence_cheap_features_json", [{} for _ in types])
        lambdas = np.asarray(z["evidence_lambda_weights"], dtype=np.float32) if "evidence_lambda_weights" in z.files else np.ones((len(types),), dtype=np.float32)
        def _take(seq, i, default):
            return seq[i] if isinstance(seq, list) and i < len(seq) else default
        atoms = [
            EvidenceAtom(
                int(i), str(types[i]), dict(anchors[i] or {}), float(costs[i]), bool(hard[i]), str(families[i]), bool(active[i]),
                validity_domain=dict(_take(domains, i, {}) or {}),
                response_modes=list(_take(modes, i, []) or []),
                aggregator=str(aggs[i] if i < len(aggs) else "mean"),
                lambda_weight=float(lambdas[i]) if i < len(lambdas) else 1.0,
                cheap_features=dict(_take(cheap, i, {}) or {}),
            )
            for i in range(len(types))
        ]
        proposal = np.asarray(z["evidence_proposal_features"], dtype=np.float32) if "evidence_proposal_features" in z.files else None
        evidence_bank = EvidenceBank(atoms=atoms, query_features=np.asarray(z["evidence_query_features"], dtype=np.float32), active_mask=active, proposal_features=proposal)
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
            pair_margins = np.asarray(z["pair_margins"], dtype=np.float32)
            pair_weights = np.asarray(z["pair_weights"], dtype=np.float32)
            pair_residuals = np.asarray(z["pair_residuals"], dtype=np.float32)
            pair_valid = np.asarray(z["pair_valid"], dtype=bool)
        else:
            pair_indices = np.zeros((0, 2), dtype=np.int64)
            pair_margins = np.zeros((0,), dtype=np.float32)
            pair_weights = np.zeros((0,), dtype=np.float32)
            pair_residuals = np.zeros((0,), dtype=np.float32)
            pair_valid = np.zeros((0,), dtype=bool)
        pairs = PairLabels(
            pairs=pair_indices,
            margins=pair_margins,
            weights=pair_weights,
            residuals=pair_residuals,
            valid_mask=pair_valid,
        )
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
