from __future__ import annotations

import numpy as np

from bdse.data.quality import quality_decision, runtime_interface_metrics


def test_quality_filter_rejects_runtime_interface_failures():
    cfg = {
        "preprocess": {
            "quality_filter": {
                "require_runtime_decision_sufficiency": True,
                "min_selector_value_ratio": 0.6,
                "max_candidate_log_ade_teacher": 3.5,
                "max_teacher_to_nearest_log_ade_gap": 2.0,
                "reject_missing_quality_metrics": True,
            }
        },
        "training": {"quality_filter": {"enabled": False}},
    }
    metrics = {
        "runtime_decision_sufficiency": False,
        "selector_value_ratio": 0.5,
        "candidate_log_ade_teacher": 4.0,
        "teacher_to_nearest_log_ade_gap": 2.5,
    }
    dec = quality_decision(metrics, cfg)
    assert not dec.keep
    assert "runtime_decision_insufficient" in dec.reasons
    assert "low_selector_value_ratio" in dec.reasons
    assert "poor_teacher_log_ade" in dec.reasons
    assert "teacher_far_from_log_nearest" in dec.reasons


def test_training_sample_stores_runtime_interface_quality_metrics(synthetic_sample):
    diag = synthetic_sample.teacher.diagnostics
    assert "quality_runtime_decision_sufficiency" in diag
    assert "quality_selector_value_ratio" in diag
    assert np.isfinite(float(diag["quality_selector_value_ratio"])) or float(diag["quality_selector_value_ratio"]) >= 0.0
