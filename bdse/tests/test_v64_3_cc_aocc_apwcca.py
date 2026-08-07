from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from bdse.config import load_config
from bdse.model.bdse_model import BDSEModel
from bdse.model.checkpoint_contract import load_bdse_state_with_contract


ROOT = Path(__file__).resolve().parents[2]


def test_v64_3_nominal_freezes_query_extension_and_legacy_proposal() -> None:
    cfg = load_config("bdse/configs/v64_3_cc_aocc_apwcca_train_2gpu.yaml")
    trainable = set(cfg["training"]["trainable_modules"])
    assert cfg["model"]["query_extension_adapter"]["scale"] == 0.0
    assert "query_extension_proj" not in trainable
    assert "critical_proposal_adapter" in trainable
    for name in ("proposal_head", "family_head", "family_embed", "family_activity_proj", "proposal_feature_proj"):
        assert name not in trainable


def test_critical_proposal_adapter_is_step_zero_noop_and_checkpoint_optional() -> None:
    cfg = load_config("bdse/configs/v64_3_cc_aocc_apwcca_train_2gpu.yaml")
    model = BDSEModel(cfg)
    assert model.critical_proposal_adapter is not None
    last = model.critical_proposal_adapter[-1]
    assert isinstance(last, torch.nn.Linear)
    assert torch.count_nonzero(last.weight).item() == 0
    assert torch.count_nonzero(last.bias).item() == 0
    state = {
        k: v.clone()
        for k, v in model.state_dict().items()
        if not k.startswith("critical_proposal_adapter.")
    }
    report = load_bdse_state_with_contract(model, state, cfg, context="v64.3-unit")
    assert report["core_contract_pass"] is True
    assert any(k.startswith("critical_proposal_adapter.") for k in report["missing"])


def test_calibration_application_copies_beta_and_prior_radius_exactly(tmp_path: Path) -> None:
    base_cfg = {
        "selector": {
            "adverse_certificate_beta": 1.0,
            "adverse_certificate_prior_radius": 0.02,
            "adverse_certificate_epsilon": 0.0,
        },
        "runtime": {"dual_certificate": {}},
        "tournament": {"epsilon_cal": 0.123},
    }
    cal = {
        "method": "V64.3 policy-selected-top-rival split-conformal dual certificate calibration",
        "independent_calibration": True,
        "recommended_adverse_certificate_epsilon": 0.001,
        "recommended_residual_flip_epsilon": 0.2,
        "beta": 0.0,
        "prior_radius": 0.02,
    }
    cfg_path = tmp_path / "in.yaml"
    cal_path = tmp_path / "cal.json"
    out_path = tmp_path / "out.yaml"
    cfg_path.write_text(yaml.safe_dump(base_cfg), encoding="utf-8")
    cal_path.write_text(json.dumps(cal), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "bdse.tools.apply_v64_3_dual_calibration",
            "--config",
            str(cfg_path),
            "--calibration-json",
            str(cal_path),
            "--output",
            str(out_path),
        ],
        cwd=ROOT,
        check=True,
    )
    out = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert out["selector"]["adverse_certificate_beta"] == 0.0
    assert out["selector"]["adverse_certificate_prior_radius"] == 0.02
    assert out["selector"]["adverse_certificate_epsilon"] == 0.001
    assert out["tournament"]["epsilon_cal"] == 0.123


def test_v64_3_config_contract_passes(tmp_path: Path) -> None:
    out = tmp_path / "contract.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "bdse.tools.validate_v64_pipeline_config",
            "--train-config",
            "bdse/configs/v64_3_cc_aocc_apwcca_train_2gpu.yaml",
            "--eval-config",
            "bdse/configs/v64_3_cc_aocc_apwcca_cl.yaml",
            "--output",
            str(out),
        ],
        cwd=ROOT,
        check=True,
    )
    assert json.loads(out.read_text(encoding="utf-8"))["pass"] is True
