from __future__ import annotations

"""V64.3.50 selected-outcome intervention probe.

This module is deliberately *not* a deployment policy.  It is a TRAIN/fresh
closed-loop evidence collector that creates a paired causal contrast for the
already frozen full-set RSMR proposal:

- control: preserve the incumbent whenever a direct RSMR proposal exists;
- treatment: execute the first direct RSMR proposal exactly once, then preserve
  the incumbent for every later direct proposal in that scenario.

The proposal identity is never recomputed, re-ranked, or replaced by a runner-up.
The probe consumes only tournament diagnostics that already describe the frozen
RSMR proposal and incumbent.  It never receives teacher/open-loop outcome labels.
"""

from dataclasses import dataclass
from typing import Any


ROLES = {"control", "treatment"}


@dataclass
class SelectedOutcomeProbeState:
    intervention_consumed: bool = False
    first_proposal_seen: bool = False
    first_proposal_action: int = -1
    first_baseline_action: int = -1
    proposal_event_count: int = 0
    executed_intervention_count: int = 0

    def reset(self) -> None:
        self.intervention_consumed = False
        self.first_proposal_seen = False
        self.first_proposal_action = -1
        self.first_baseline_action = -1
        self.proposal_event_count = 0
        self.executed_intervention_count = 0


def _as_int(diag: dict[str, Any], key: str, default: int = -1) -> int:
    try:
        return int(round(float(diag.get(key, default))))
    except Exception:
        return int(default)


def _as_bool(diag: dict[str, Any], key: str, default: bool = False) -> bool:
    try:
        return bool(float(diag.get(key, 1.0 if default else 0.0)) > 0.5)
    except Exception:
        return bool(default)


def apply_selected_outcome_probe(
    current_action: int,
    tournament_diagnostics: dict[str, Any],
    cfg: dict[str, Any],
    state: SelectedOutcomeProbeState,
) -> tuple[int, dict[str, Any]]:
    """Apply the V50 paired intervention contract to one planner decision.

    The function assumes the supplied tournament diagnostics were produced by
    the frozen full-set RSMR direct-recovery path.  It fail-closes if the
    treatment proposal is not the action that RSMR itself selected before the
    probe; this prevents a post-selection or structural-guard bypass from being
    mislabeled as an RSMR intervention.
    """
    pcfg = cfg.get("selected_outcome_probe", {}) if isinstance(cfg, dict) else {}
    enabled = bool(pcfg.get("enabled", False))
    if not enabled:
        return int(current_action), {
            "enabled": False,
            "role": "disabled",
            "proposal_exists": False,
            "intervention_executed": False,
        }

    role = str(pcfg.get("role", "")).strip().lower()
    if role not in ROLES:
        raise ValueError(f"V50 selected-outcome probe role must be one of {sorted(ROLES)}, got {role!r}")
    if bool((cfg.get("fallback", {}) or {}).get("enabled", True)):
        raise ValueError("V50 selected-outcome probe requires fallback.enabled=false")

    diag = dict(tournament_diagnostics or {})
    proposal_exists = _as_bool(diag, "decisive_frontier_icer_scir_proposal_exists", False)
    proposal_action = _as_int(diag, "decisive_frontier_icer_scir_proposal_action", -1)
    baseline_action = _as_int(diag, "decisive_frontier_icer_baseline_action", -1)
    rsmr_selected = _as_int(diag, "decisive_frontier_icer_selected_action", int(current_action))

    out_action = int(current_action)
    executed = False
    first_now = False
    if proposal_exists:
        if proposal_action < 0 or baseline_action < 0:
            raise ValueError("V50 probe saw proposal_exists without valid proposal/baseline actions")
        state.proposal_event_count += 1
        if not state.first_proposal_seen:
            state.first_proposal_seen = True
            state.first_proposal_action = int(proposal_action)
            state.first_baseline_action = int(baseline_action)
            first_now = True

        if role == "control":
            out_action = int(baseline_action)
        elif not state.intervention_consumed:
            # The causal treatment must be the actual frozen full-set RSMR action,
            # not a proposal that a later guard/risk head had already displaced.
            if int(current_action) != int(proposal_action) or int(rsmr_selected) != int(proposal_action):
                raise ValueError(
                    "V50 treatment probe refuses to execute a proposal that is not the frozen RSMR selected action "
                    f"(current={current_action}, selected={rsmr_selected}, proposal={proposal_action})"
                )
            out_action = int(proposal_action)
            state.intervention_consumed = True
            state.executed_intervention_count += 1
            executed = True
        else:
            out_action = int(baseline_action)

    pdiag = {
        "enabled": True,
        "role": role,
        "proposal_exists": bool(proposal_exists),
        "proposal_action": int(proposal_action),
        "baseline_action": int(baseline_action),
        "rsmr_selected_action": int(rsmr_selected),
        "pre_probe_action": int(current_action),
        "post_probe_action": int(out_action),
        "first_proposal_now": bool(first_now),
        "first_proposal_seen": bool(state.first_proposal_seen),
        "first_proposal_action": int(state.first_proposal_action),
        "first_baseline_action": int(state.first_baseline_action),
        "intervention_executed": bool(executed),
        "intervention_consumed": bool(state.intervention_consumed),
        "proposal_event_count": int(state.proposal_event_count),
        "executed_intervention_count": int(state.executed_intervention_count),
    }
    return int(out_action), pdiag
