"""Pure NumPy construction of optional semantic attribution diagnostics."""
from __future__ import annotations

import numpy as np


FAMILY_IDS = (1, 2, 3, 4, 5)
FAMILY_NAMES = {
    1: "feasibility",
    2: "reachability_interaction",
    3: "precedence",
    4: "decision_boundary",
    5: "dynamic_regularity",
}
FAMILY_FEATURE_NAMES = (
    [f"candidate_family_{FAMILY_NAMES[f]}_signed_sum" for f in FAMILY_IDS]
    + [f"delta_family_{FAMILY_NAMES[f]}_signed_sum" for f in FAMILY_IDS]
)

TYPE_NAMES = (
    "occupancy", "ttc", "gap", "drivable_area", "wrong_way", "speed_limit",
    "red_light", "route_connector", "local_comfort_accel", "local_comfort_jerk",
    "local_comfort_curvature", "local_comfort_brake",
)
TYPE_FEATURE_NAMES = (
    [f"candidate_type_{name}_signed_sum" for name in TYPE_NAMES]
    + [f"delta_type_{name}_signed_sum" for name in TYPE_NAMES]
)


def _contributions(values, candidate_count: int) -> np.ndarray:
    if values is None:
        return np.zeros((0, candidate_count), dtype=np.float64)
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < candidate_count:
        return np.zeros((0, candidate_count), dtype=np.float64)
    return raw[:, :candidate_count]


def family_feature_matrix(values, family_ids, valid_mask, anchor_action, incumbent_action):
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    candidate_count = len(valid)
    out = np.zeros((candidate_count, len(FAMILY_FEATURE_NAMES)), dtype=np.float64)
    contrib = _contributions(values, candidate_count)
    # Upstream selection metadata exists in every run, whereas contribution rows
    # only exist when decisive-frontier attribution is active.  Zero rows mean
    # "diagnostic unavailable", not a malformed selected-atom schema.
    if contrib.shape[0] == 0 or family_ids is None:
        return out, list(FAMILY_FEATURE_NAMES)
    families = np.asarray(family_ids, dtype=np.int64).reshape(-1)
    if contrib.shape[0] != families.size:
        raise ValueError(
            "EAF-ICER semantic-family attribution requires one family id per selected atom: "
            f"contrib_rows={contrib.shape[0]} family_ids={families.size}"
        )
    unknown = sorted(set(int(x) for x in families.tolist()) - {0, *FAMILY_IDS})
    if unknown:
        raise ValueError(f"EAF-ICER semantic-family attribution received unknown family ids {unknown}")
    anchor = int(anchor_action)
    incumbent = int(incumbent_action)
    incumbent_vec = (
        contrib[:, incumbent]
        if 0 <= incumbent < candidate_count and incumbent != anchor
        else np.zeros((contrib.shape[0],), dtype=np.float64)
    )
    for candidate in np.flatnonzero(valid).tolist():
        if candidate == anchor:
            continue
        candidate_vec = contrib[:, candidate]
        delta = candidate_vec - incumbent_vec
        for column, family in enumerate(FAMILY_IDS):
            mask = families == family
            if np.any(mask):
                out[candidate, column] = float(np.sum(candidate_vec[mask]))
                out[candidate, len(FAMILY_IDS) + column] = float(np.sum(delta[mask]))
    return out, list(FAMILY_FEATURE_NAMES)


def type_feature_matrix(values, type_names, valid_mask, anchor_action, incumbent_action):
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    candidate_count = len(valid)
    out = np.zeros((candidate_count, len(TYPE_FEATURE_NAMES)), dtype=np.float64)
    contrib = _contributions(values, candidate_count)
    if contrib.shape[0] == 0 or type_names is None:
        return out, list(TYPE_FEATURE_NAMES)
    types = np.asarray([str(x) for x in list(type_names)], dtype=object).reshape(-1)
    if contrib.shape[0] != types.size:
        raise ValueError(
            "EAF-ICER type-resolved attribution requires one atom type per selected attribution row: "
            f"contrib_rows={contrib.shape[0]} type_names={types.size}"
        )
    unknown = sorted(set(types.tolist()) - set(TYPE_NAMES))
    if unknown:
        raise ValueError(f"EAF-ICER type-resolved attribution received unknown selected atom types {unknown}")
    anchor = int(anchor_action)
    incumbent = int(incumbent_action)
    incumbent_vec = (
        contrib[:, incumbent]
        if 0 <= incumbent < candidate_count and incumbent != anchor
        else np.zeros((contrib.shape[0],), dtype=np.float64)
    )
    for candidate in np.flatnonzero(valid).tolist():
        if candidate == anchor:
            continue
        candidate_vec = contrib[:, candidate]
        delta = candidate_vec - incumbent_vec
        for column, name in enumerate(TYPE_NAMES):
            mask = types == name
            if np.any(mask):
                out[candidate, column] = float(np.sum(candidate_vec[mask]))
                out[candidate, len(TYPE_NAMES) + column] = float(np.sum(delta[mask]))
    return out, list(TYPE_FEATURE_NAMES)
