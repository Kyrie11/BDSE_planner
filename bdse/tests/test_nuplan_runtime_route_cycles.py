from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bdse.data import nuplan_runtime_adapter as adapter


@dataclass
class _Point:
    x: float
    y: float
    heading: float = 0.0


class _Baseline:
    def __init__(self, pts):
        self.discrete_path = [_Point(float(x), float(y)) for x, y in pts]


class _Edge:
    def __init__(self, edge_id: str, pts):
        self.id = edge_id
        self.baseline_path = _Baseline(pts)
        self.outgoing_edges = []

    def get_roadblock_id(self):
        return self.id.split(":")[0]


class _RoadBlock:
    def __init__(self, rid: str, edges):
        self.id = rid
        self.interior_edges = list(edges)
        self.outgoing_edges = []
        self.incoming_edges = []


class _Init:
    def __init__(self, ids):
        self.route_roadblock_ids = ids
        self.map_api = object()
        self.mission_goal = None


def test_object_baseline_polylines_ignores_cyclic_graph_neighbors():
    a = _Edge("rb0:a", [(0, 0), (5, 0)])
    b = _Edge("rb1:b", [(5, 0), (10, 0)])
    a.outgoing_edges = [b]
    b.outgoing_edges = [a]
    rb0 = _RoadBlock("rb0", [a])

    polylines = adapter._object_baseline_polylines(rb0)

    assert len(polylines) == 1
    assert np.allclose(polylines[0], np.array([[0, 0], [5, 0]], dtype=np.float32))


def test_route_from_map_api_stitches_route_without_following_cycles(monkeypatch):
    a = _Edge("rb0:a", [(0, 0), (5, 0)])
    b = _Edge("rb1:b", [(5, 0), (10, 0)])
    c = _Edge("rb2:c", [(10, 0), (15, 0)])
    a.outgoing_edges = [b]
    b.outgoing_edges = [c, a]  # cycle back to a must not recurse forever
    c.outgoing_edges = [b]
    roadblocks = {
        "rb0": _RoadBlock("rb0", [a]),
        "rb1": _RoadBlock("rb1", [b]),
        "rb2": _RoadBlock("rb2", [c]),
    }

    monkeypatch.setattr(adapter, "_get_map_object_any_layer", lambda _map_api, rid: roadblocks.get(str(rid)))
    route = adapter._route_from_map_api(_Init(["rb0", "rb1", "rb2"]), np.array([0, 0, 0, 0, 0], dtype=np.float32), {"runtime": {"max_route_points": 64}})

    assert route.shape[0] >= 4
    assert np.allclose(route[:4], np.array([[0, 0], [5, 0], [10, 0], [15, 0]], dtype=np.float32))
