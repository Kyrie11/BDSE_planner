from __future__ import annotations

import numpy as np

from bdse.planner.evidence_atoms import normalize_atom_costs, raw_local_costs


def test_atom_additivity(synthetic_sample, cfg):
    raw = raw_local_costs(synthetic_sample.evidence_bank.atoms, synthetic_sample.candidates, synthetic_sample.runtime, synthetic_sample.label_future, cfg)
    g = normalize_atom_costs(raw, synthetic_sample.evidence_bank.atoms, cfg)
    assert np.allclose(g.sum(axis=0), synthetic_sample.teacher.J_evid, atol=1e-5)
    family_sum_then_clip = np.clip(raw.sum(axis=0), 0, 1e9)
    assert family_sum_then_clip.shape == synthetic_sample.teacher.J_evid.shape
