from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bdse.planner.pair_screen import build_rival_sets_from_base
from bdse.planner.selector import budgeted_margin, full_interface_margin
from bdse.utils import softmin_np


@dataclass(slots=True)
class TournamentResult:
    action_index: int
    scores: np.ndarray
    margins: np.ndarray
    rival_sets: list[list[int]]
    diagnostics: dict[str, Any]


def build_rival_sets(
    predicted_full_margin: np.ndarray,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    L_infer: int = 16,
    eta_pred: float = 1.0,
) -> list[list[int]]:
    M = np.asarray(predicted_full_margin, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    K = len(valid)
    full_cost_proxy = -M.mean(axis=1)
    rivals: list[list[int]] = []
    for a in range(K):
        if not valid[a]:
            rivals.append([])
            continue
        cand = [b for b in range(K) if b != a and valid[b]]
        top = sorted(cand, key=lambda b: (float(full_cost_proxy[b]), abs(float(M[a, b])), b))[:L_infer]
        near = [b for b in cand if abs(float(M[a, b])) < eta_pred]
        safety = [b for b in cand if runtime_safety_flags[b]]
        all_rivals = sorted(set(top + near + safety), key=lambda b: (abs(float(M[a, b])), b))
        if len(all_rivals) > max(L_infer, len(safety)):
            safety_set = set(safety)
            kept = [b for b in all_rivals if b in safety_set]
            for b in all_rivals:
                if b not in safety_set and len(kept) < L_infer:
                    kept.append(b)
            all_rivals = kept
        rivals.append(all_rivals)
    return rivals


def tournament_scores(
    margins: np.ndarray,
    valid_mask: np.ndarray,
    rival_sets: list[list[int]],
    use_softmin: bool = True,
    softmin_tau: float = 1.0,
    beta_uncertainty: float = 0.0,
    sigma: np.ndarray | None = None,
) -> np.ndarray:
    K = margins.shape[0]
    scores = np.full((K,), -1e9, dtype=np.float32)
    for a in range(K):
        if not valid_mask[a]:
            continue
        rivals = rival_sets[a]
        if not rivals:
            scores[a] = 0.0
            continue
        vals = np.asarray([margins[a, b] for b in rivals], dtype=np.float32)
        if sigma is not None and beta_uncertainty > 0:
            vals = vals - float(beta_uncertainty) * np.asarray([sigma[a, b] for b in rivals], dtype=np.float32)
        scores[a] = float(softmin_np(vals, softmin_tau if use_softmin else 0.0))
    return scores


def run_tournament(
    predicted_base_cost: np.ndarray,
    predicted_atom_costs: np.ndarray,
    selected_atoms: list[int] | np.ndarray,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    cfg: dict[str, Any],
    sigma: np.ndarray | None = None,
) -> TournamentResult:
    tc = cfg.get("tournament", {})
    sc = cfg.get("selector", {})
    rivals = build_rival_sets_from_base(
        predicted_base_cost,
        valid_mask,
        runtime_safety_flags,
        L_infer=int(tc.get("L_infer", 16)),
        eta0=float(sc.get("eta_pred", 1.0)),
    )
    M_B = budgeted_margin(predicted_base_cost, predicted_atom_costs, selected_atoms)
    epsilon_cal = float(tc.get("epsilon_cal", cfg.get("calibration", {}).get("epsilon_cal", 0.0)))
    M_eval = M_B - epsilon_cal
    scores = tournament_scores(
        M_eval,
        np.asarray(valid_mask, dtype=bool),
        rivals,
        use_softmin=bool(tc.get("use_softmin", True)),
        softmin_tau=float(tc.get("softmin_tau", 1.0)),
        beta_uncertainty=float(tc.get("beta_uncertainty", 0.0)),
        sigma=sigma,
    )
    action = int(np.argmax(scores))
    sorted_scores = np.sort(scores[np.asarray(valid_mask, dtype=bool)])
    delta = float(sorted_scores[-1] - sorted_scores[-2]) if len(sorted_scores) >= 2 else float("inf")
    safety_idx = np.flatnonzero(np.asarray(runtime_safety_flags, dtype=bool) & np.asarray(valid_mask, dtype=bool))
    if safety_idx.size and action not in safety_idx:
        safety_lcb_min = float(np.min(M_eval[action, safety_idx]))
    elif action in safety_idx:
        safety_lcb_min = -float("inf")
    else:
        safety_lcb_min = float("inf")
    return TournamentResult(
        action_index=action,
        scores=scores,
        margins=M_eval,
        rival_sets=rivals,
        diagnostics={
            "delta_hat_B": delta,
            "selected_atoms": list(map(int, selected_atoms)),
            "valid_actions": int(np.asarray(valid_mask).sum()),
            "rival_source": "base_score_cheap_flags",
            "epsilon_cal": epsilon_cal,
            "safety_lcb_min": safety_lcb_min,
            "selected_action_safety_flag": bool(np.asarray(runtime_safety_flags, dtype=bool)[action]) if 0 <= action < len(runtime_safety_flags) else False,
        },
    )

def full_interface_action(
    predicted_base_cost: np.ndarray,
    predicted_atom_costs: np.ndarray,
    valid_mask: np.ndarray,
    cfg: dict[str, Any],
) -> int:
    M = full_interface_margin(predicted_base_cost, predicted_atom_costs)
    K = len(valid_mask)
    rivals = [[b for b in range(K) if b != a and valid_mask[b]] for a in range(K)]
    scores = tournament_scores(M, np.asarray(valid_mask, dtype=bool), rivals, True, float(cfg.get("tournament", {}).get("softmin_tau", 1.0)))
    return int(np.argmax(scores))


def assert_antisymmetric(M: np.ndarray, valid_mask: np.ndarray, atol: float = 1e-5) -> None:
    valid = np.asarray(valid_mask, dtype=bool)
    for a in np.flatnonzero(valid):
        for b in np.flatnonzero(valid):
            if abs(float(M[a, b] + M[b, a])) > atol:
                raise AssertionError("Pairwise margins must be antisymmetric")
