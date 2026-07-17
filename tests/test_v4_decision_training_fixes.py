from __future__ import annotations

import argparse
import numpy as np
import torch

from bdse.model.losses import _negative_cost_logits
from bdse.experiments.train import _validation_bdse_score


def test_negative_cost_logits_preserve_argmin_under_large_scale():
    cost = torch.tensor([[10000.0, 100.0, 5000.0, 1e6]])
    valid = torch.tensor([[True, True, True, False]])
    logits = _negative_cost_logits(cost, valid, min_scale=1.0)
    assert int(torch.argmax(logits, dim=1).item()) == 1
    assert logits[0, 3].item() < -1e20


def test_validation_score_prefers_teacher_match_over_budget_matching_wrong_full():
    weak = {
        "val_teacher_action_match": 0.10,
        "val_full_interface_action_match": 0.40,
        "val_budget_vs_full_match": 0.80,
        "val_teacher_regret": 1000.0,
        "val_evidence_sufficiency": 0.1,
        "val_hard_evidence_recall": 0.1,
        "val_preserved_margin_error": 1000.0,
    }
    strong = dict(weak)
    strong["val_teacher_action_match"] = 0.20
    strong["val_full_interface_action_match"] = 0.20
    strong["val_budget_vs_full_match"] = 0.20
    assert _validation_bdse_score(strong) > _validation_bdse_score(weak)
