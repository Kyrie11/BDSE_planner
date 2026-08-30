from types import SimpleNamespace

import torch

from bdse.experiments.train import _aggregate_meters, _append_loss_meters, _make_checkpoint
from bdse.model.losses import _deployment_selector_scene_indices, _slice_scene_batch


class _DummyScaler:
    def state_dict(self):
        return {"enabled": False}


def test_rotating_exact_selector_subset_and_final_full_alignment():
    cfg = {
        "current_epoch": 2,
        "global_step": 4,
        "global_rank": 0,
        "deployment_selector_scenes_per_rank": 2,
        "deployment_selector_every_n_steps": 2,
        "deployment_selector_full_start_epoch": 1_000_000,
        "deployment_selector_full_last_n_steps": 2,
        "steps_per_epoch": 10,
        "epochs": 12,
    }
    idx = _deployment_selector_scene_indices(4, cfg, torch.device("cpu"))
    assert idx.numel() == 2
    assert len(set(idx.tolist())) == 2

    cfg["global_step"] = 5
    assert _deployment_selector_scene_indices(4, cfg, torch.device("cpu")).numel() == 0

    cfg["current_epoch"] = 11
    cfg["global_step"] = 11 * 10 + 8
    assert _deployment_selector_scene_indices(4, cfg, torch.device("cpu")).tolist() == [0, 1, 2, 3]


def test_scene_slice_only_slices_batch_leading_tensors():
    values = {
        "batched": torch.arange(12).reshape(4, 3),
        "global": torch.arange(3),
        "scalar": torch.tensor(7),
    }
    out = _slice_scene_batch(values, torch.tensor([3, 1]), batch_size=4)
    assert out["batched"].tolist() == [[9, 10, 11], [3, 4, 5]]
    assert out["global"] is values["global"]
    assert out["scalar"] is values["scalar"]


def test_loss_meters_accumulate_without_per_step_host_lists():
    meters = {}
    _append_loss_meters(meters, {"loss": torch.tensor(2.0), "aux": torch.tensor(4.0)})
    _append_loss_meters(meters, {"loss": torch.tensor(6.0), "aux": torch.tensor(8.0)})
    assert torch.is_tensor(meters["loss"][0])
    result = _aggregate_meters(meters, torch.device("cpu"), distributed=False)
    assert result == {"aux": 6.0, "loss": 4.0}


def test_mid_epoch_checkpoint_resumes_same_epoch_and_next_batch():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = SimpleNamespace(output="out/model.pt")
    checkpoint = _make_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=_DummyScaler(),
        cfg={"training": {}},
        args=args,
        epoch=3,
        metrics={},
        best_metric=None,
        best_epoch=None,
        world_size=2,
        next_batch_index=750,
    )
    assert checkpoint["next_epoch"] == 3
    assert checkpoint["next_batch_index"] == 750

    epoch_checkpoint = _make_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=_DummyScaler(),
        cfg={"training": {}},
        args=args,
        epoch=3,
        metrics={},
        best_metric=None,
        best_epoch=None,
        world_size=2,
    )
    assert epoch_checkpoint["next_epoch"] == 4
    assert epoch_checkpoint["next_batch_index"] == 0


def test_branchless_robust_loss_ignores_invalid_entries():
    from bdse.model.losses import robust_loss

    pred = torch.tensor([2.0, float("nan")], requires_grad=True)
    target = torch.tensor([1.0, 3.0])
    mask = torch.tensor([True, False])
    loss = robust_loss(pred, target, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred.grad[0])
