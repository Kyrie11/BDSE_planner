from __future__ import annotations

from dataclasses import dataclass

from bdse.data.feature_builder import cached_tracked_window


@dataclass
class _Obj:
    token: str
    x: float
    y: float
    heading: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    length: float = 4.8
    width: float = 2.0


class _ScenarioWithBrokenBulkFuture:
    log_name = "dummy_log"

    def get_time_point(self, iteration: int):
        return 1_000_000 + iteration * 100_000

    def get_future_tracked_objects(self, iteration: int, time_horizon: float, num_samples: int):
        def _broken_generator():
            raise AttributeError("'NoneType' object has no attribute 'hex'")
            yield None

        return _broken_generator()

    def get_tracked_objects_at_iteration(self, iteration: int):
        return [_Obj(token=f"agent-{iteration}", x=float(iteration), y=0.0)]


def test_future_tracked_window_falls_back_when_bulk_next_token_is_none():
    cfg = {"preprocess": {"temporal_frame_cache": False}}
    frames, stats = cached_tracked_window(
        _ScenarioWithBrokenBulkFuture(),
        iteration=0,
        cfg=cfg,
        direction="future",
        time_horizon=0.3,
        num_samples=3,
        step_s=0.1,
    )

    assert len(frames) == 3
    assert [frame.tokens[0] for frame in frames] == ["agent-1", "agent-2", "agent-3"]
    assert stats["bulk_call"] == 1
    assert stats["bulk_individual_fallback"] == 1
    assert "NoneType" in stats["bulk_iteration_error"]
