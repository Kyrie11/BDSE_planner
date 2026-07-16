from __future__ import annotations

from bdse.planner.evidence_atoms import hard_event_ownership


def test_hard_ownership(synthetic_sample):
    owners = hard_event_ownership(synthetic_sample.evidence_bank.atoms)
    for event, count in owners.items():
        assert count >= 1
    assert "J_hard" not in synthetic_sample.teacher.diagnostics
