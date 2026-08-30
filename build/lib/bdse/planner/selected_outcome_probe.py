from __future__ import annotations

"""V64.3.50 selected-outcome intervention probe.

This module is deliberately *not* a deployment policy.  It is a TRAIN/fresh
closed-loop evidence collector that creates a paired causal contrast for the
first live full-set winner produced by the already frozen RSMR selector:

- control: preserve the incumbent whenever a direct RSMR proposal exists;
- treatment: execute the first direct RSMR proposal exactly once, then preserve
  the incumbent for every later direct proposal in that scenario.

The selector is never modified, re-ranked, or replaced by a runner-up. Candidate
``action`` integers are local slots in a bank rebuilt at each live state, so an
offline V49 slot number is provenance for the frozen scene cohort, not a global
proposal identifier that can be compared across planner states.
For V50 event-state alignment, the probe may be layered on the frozen V49
post-selection config solely so the already-defined Q/P/E coordinates are
instrumented at the *actual live proposal event*.  The probe ignores the old
post-selection accept/veto decision and acts only on the pre-post-selection
RSMR proposal diagnostics; this prevents the historical V49 risk head from
contaminating the paired treatment assignment.
"""

from dataclasses import dataclass
import math
from typing import Any


ROLES = {"control", "treatment"}
_QPE_NAMES = {
    "quality": "ocrr_state::quality_value",
    "plan_inc": "ocrr_state::prospective_response_increment",
    "ego_inc": "ocrr_state::ego_reference_increment",
}


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


def _live_qpe(diag: dict[str, Any]) -> tuple[float, float, float] | None:
    """Recover the frozen Q/P/E coordinate values from V49 runtime instrumentation.

    V48/V49 already materialize the operator state in the post-selection feature
    vector as [Q, P-Q, E-P, logK].  V50 does not introduce a new feature; it only
    records those same coordinates at the actual closed-loop proposal event.
    """
    names = [str(x) for x in diag.get("_decisive_frontier_icer_scir_post_selection_value_feature_names", [])]
    vals = diag.get("_decisive_frontier_icer_scir_post_selection_value_feature", [])
    try:
        values = [float(x) for x in vals]
    except Exception:
        return None
    if len(names) != len(values):
        return None
    pos = {n: i for i, n in enumerate(names)}
    if any(n not in pos for n in _QPE_NAMES.values()):
        return None
    q = float(values[pos[_QPE_NAMES["quality"]]])
    p = q + float(values[pos[_QPE_NAMES["plan_inc"]]])
    e = p + float(values[pos[_QPE_NAMES["ego_inc"]]])
    if not all(math.isfinite(x) for x in (q, p, e)):
        return None
    return q, p, e


def apply_selected_outcome_probe(
    current_action: int,
    tournament_diagnostics: dict[str, Any],
    cfg: dict[str, Any],
    state: SelectedOutcomeProbeState,
) -> tuple[int, dict[str, Any]]:
    """Apply the V50 paired intervention contract to one planner decision.

    The live proposal is read from ``decisive_frontier_icer_scir_proposal_action``.
    That quantity is produced by the frozen RSMR selector *before* V48/V49
    post-selection retention.  In the
    V50 instrumentation config the historical risk head is therefore allowed to
    compute live Q/P/E, but its accept/veto output is not the treatment assignment:
    CONTROL returns the incumbent; TREATMENT executes that exact live RSMR winner once.
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
    # This is the action after any historical post-selection value/risk stage.
    # It is recorded for audit but is *not* called the RSMR winner in V50.
    pre_probe_selected = _as_int(diag, "decisive_frontier_icer_selected_action", int(current_action))

    qpe = _live_qpe(diag) if proposal_exists else None
    if proposal_exists and bool(pcfg.get("require_live_qpe", False)) and qpe is None:
        raise ValueError("V50 proposal event is missing live Q/P/E operator-state instrumentation")

    out_action = int(current_action)
    executed = False
    first_now = False
    if proposal_exists:
        if proposal_action < 0 or baseline_action < 0:
            raise ValueError("V50 probe saw proposal_exists without valid proposal/baseline actions")
        # Historical post-selection may only retain the same proposal or veto to
        # the incumbent.  A third action would violate the no-rerank contract.
        if pre_probe_selected not in {int(proposal_action), int(baseline_action)}:
            raise ValueError(
                "V50 probe saw a pre-probe selected action outside {live RSMR proposal, incumbent}: "
                f"selected={pre_probe_selected} proposal={proposal_action} baseline={baseline_action}"
            )
        state.proposal_event_count += 1
        if not state.first_proposal_seen:
            state.first_proposal_seen = True
            state.first_proposal_action = int(proposal_action)
            state.first_baseline_action = int(baseline_action)
            first_now = True

        if role == "control":
            out_action = int(baseline_action)
        elif not state.intervention_consumed:
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
        # For the V50 probe, the RSMR selected action is definitionally the
        # live proposal produced before historical post-selection retention.
        "rsmr_selected_action": int(proposal_action if proposal_exists else -1),
        "pre_probe_selected_action": int(pre_probe_selected),
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
    if qpe is not None:
        pdiag.update({
            "live_quality_value": float(qpe[0]),
            "live_plan_control_value": float(qpe[1]),
            "live_ego_ref_value": float(qpe[2]),
        })
    for key in (
        "v50_live_proposal_fingerprint",
        "v50_live_proposal_maneuver_id",
        "v50_live_proposal_pool_original_index",
        "v50_live_proposal_maneuver",
        "v50_live_proposal_theta",
    ):
        if key in diag:
            pdiag[key] = diag[key]
    return int(out_action), pdiag
