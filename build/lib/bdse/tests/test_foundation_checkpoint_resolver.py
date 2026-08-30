from pathlib import Path

import torch

from bdse.tools.resolve_foundation_checkpoint import analyze_checkpoint


def _target_state():
    return {
        "scene.weight": torch.zeros(4, 4),
        "pair_head.weight": torch.zeros(2, 4),
    }


def test_exact_v30_identity_is_safe(tmp_path: Path):
    path = tmp_path / "bdse_v30_pmvrbsr.best.pt"
    torch.save(
        {
            "model": {key: value.clone() for key, value in _target_state().items()},
            "args": {"config": "bdse/configs/v30_pmvrbsr.yaml", "output": str(path)},
            "cfg": {"model": {"pair_head_residual_over_local": False}},
            "epoch": 3,
        },
        path,
    )
    report = analyze_checkpoint(path, _target_state())
    assert report.safe_foundation
    assert report.matched_numel_fraction == 1.0


def test_algorithm_checkpoint_is_rejected_by_default(tmp_path: Path):
    path = tmp_path / "outputs_v49_dbap_exact_2gpu_v1" / "train" / "bdse_v49_dbap.best.pt"
    path.parent.mkdir(parents=True)
    torch.save(
        {
            "model": {key: value.clone() for key, value in _target_state().items()},
            "args": {"config": "bdse/configs/v49_bdse_dbap_train_2gpu.yaml", "output": str(path)},
            "cfg": {"model": {"pair_head_residual_over_local": True}},
            "epoch": 1,
        },
        path,
    )
    report = analyze_checkpoint(path, _target_state())
    assert not report.safe_foundation
    assert "algorithm_specific_checkpoint" in report.rejection_reasons
    assert "no_strong_v30_identity_evidence" in report.rejection_reasons


def test_algorithm_checkpoint_can_be_explicit_transfer_ablation(tmp_path: Path):
    path = tmp_path / "outputs_v49_dbap_exact_2gpu_v1" / "train" / "bdse_v49_dbap.best.pt"
    path.parent.mkdir(parents=True)
    torch.save(
        {
            "model": {key: value.clone() for key, value in _target_state().items()},
            "args": {"config": "bdse/configs/v49_bdse_dbap_train_2gpu.yaml", "output": str(path)},
            "cfg": {"model": {"pair_head_residual_over_local": True}},
        },
        path,
    )
    report = analyze_checkpoint(path, _target_state(), allow_algorithm_checkpoints=True)
    assert report.safe_foundation
