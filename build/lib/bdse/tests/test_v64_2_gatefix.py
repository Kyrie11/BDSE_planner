from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from bdse.model.losses import _exact_winner_flip_critical_proposal_loss


ROOT = Path(__file__).resolve().parents[2]


def test_pipeline_calibration_locals_are_nounset_safe() -> None:
    text = (ROOT / "V64_SAQA_BCC_NEXT_COMMANDS.sh").read_text(encoding="utf-8")
    assert 'local sid="$1" gpu="$2" raw=' not in text
    assert 'local sid="$1" pid="$2" log=' not in text
    subprocess.run(["bash", "-n", str(ROOT / "V64_SAQA_BCC_NEXT_COMMANDS.sh")], check=True)


def test_support_audit_treats_deployed_step0_as_hard_contract(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    names = ["legacy_anchor", "support_aware_nominal", "prefix_cache", "structural_prior"]
    base_metrics = {
        "full_interface_action_match": 0.359,
        "dense_runtime_base_contract_pass": 1.0,
        "dense_runtime_raw_query_feature_contract_available": 1.0,
        "dense_runtime_raw_query_feature_contract_pass": 1.0,
        "dense_runtime_query_score_contract_pass": 1.0,
        "dense_runtime_query_decision_match": 1.0,
    }
    for name in names:
        d = suite / name
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps(base_metrics), encoding="utf-8")
        full = 0 if name == "legacy_anchor" else 1
        bdse = 3
        row = {"scenario_token": "s", "timestamp_us": 1, "full_action": full, "bdse_action": bdse}
        (d / "metrics.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    output = tmp_path / "report.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bdse.tools.analyze_v64_support_contract_audit",
            "--suite-root",
            str(suite),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert proc.returncode == 0
    assert report["pass"] is True
    assert report["diagnostic_checks"]["support_step0_matches_legacy_dense_diagnostic"] is False
    assert report["checks"]["support_step0_matches_legacy_deployed"] is True


def test_hcbe_pushes_missed_critical_above_same_family_boundary() -> None:
    # Atom 0 is literal winner-flip critical but atom 1 is retained by HAB.
    J0 = torch.tensor([[0.0, 0.2]])
    g = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    valid = torch.tensor([[True, True]])
    active = torch.tensor([[True, True]])
    proposal_logits = torch.zeros((1, 2), requires_grad=True)
    acquisition_logits = torch.zeros((1, 2), requires_grad=True)
    deployed = torch.tensor([[False, True]])
    cfg = {
        "training": {
            "exact_winner_flip_criticality": {
                "enabled": True,
                "target_source": "model_dense",
                "positive_weight": 1.0,
                "negative_weight": 1.0,
                "rank_weight": 0.0,
                "pairwise_rank_weight": 0.0,
                "coverage_weight": 0.0,
                "exchange_rank_weight": 1.0,
                "exchange_margin": 0.2,
                "exchange_temperature": 0.25,
                "min_action_scale": 1.0,
            }
        }
    }
    loss, *_ = _exact_winner_flip_critical_proposal_loss(
        J0,
        g,
        valid,
        active,
        proposal_logits,
        deployed,
        torch.tensor([1]),
        torch.ones((1, 2)),
        cfg,
        deployment_acquisition_logits=acquisition_logits,
        family_ids=torch.tensor([[2, 2]]),
    )
    loss.backward()
    assert acquisition_logits.grad is not None
    # Gradient descent raises the missed critical score and lowers the retained
    # non-critical exchange boundary.
    assert float(acquisition_logits.grad[0, 0]) < 0.0
    assert float(acquisition_logits.grad[0, 1]) > 0.0


def test_v64_2_config_contract_passes(tmp_path: Path) -> None:
    output = tmp_path / "contract.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "bdse.tools.validate_v64_pipeline_config",
            "--train-config",
            "bdse/configs/v64_2_saqa_bcc_hcbe_train_2gpu.yaml",
            "--eval-config",
            "bdse/configs/v64_saqa_bcc_cl.yaml",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    assert json.loads(output.read_text(encoding="utf-8"))["pass"] is True
