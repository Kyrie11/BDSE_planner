from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import BatchSampler

from bdse.experiments.train import ResumableBatchSampler
from bdse.model.losses import _packed_numpy_snapshot


def test_packed_numpy_snapshot_preserves_values_shapes_and_dtypes() -> None:
    floating = torch.tensor([[1.5, -2.0], [3.25, 4.0]], dtype=torch.float16)
    integer = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    boolean = torch.tensor([[True, False], [False, True]], dtype=torch.bool)

    snapshot = _packed_numpy_snapshot(
        {
            "floating": (floating, torch.float32),
            "integer": (integer, torch.int64),
            "boolean": (boolean, torch.bool),
            "missing": (None, torch.float32),
        }
    )

    np.testing.assert_array_equal(
        snapshot["floating"], floating.float().numpy()
    )
    np.testing.assert_array_equal(
        snapshot["integer"], integer.long().numpy()
    )
    np.testing.assert_array_equal(snapshot["boolean"], boolean.numpy())
    assert snapshot["floating"].shape == (2, 2)
    assert snapshot["floating"].dtype == np.float32
    assert snapshot["integer"].dtype == np.int64
    assert snapshot["boolean"].dtype == np.bool_
    assert snapshot["missing"] is None


def test_resumable_batch_sampler_starts_at_next_unfinished_batch() -> None:
    base = BatchSampler(range(10), batch_size=3, drop_last=False)
    sampler = ResumableBatchSampler(base)

    assert list(sampler) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert sampler.total_batches == 4

    sampler.set_start_batch(2)
    assert len(sampler) == 2
    assert list(sampler) == [[6, 7, 8], [9]]

    sampler.set_start_batch(0)
    assert len(sampler) == 4
