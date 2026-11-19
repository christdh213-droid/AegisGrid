"""Acceptance checks for anonymous occupancy routing."""
import numpy as np
from scipy.ndimage import distance_transform_edt

from src.config import PipelineConfig
from src.path_planner import (
    OccupancyPlanner, fixed_sample, inflated, leaf_adjacency, occupancy_masks,
    route_clear, segment_cells, snap,
)
from src.stream_protocol import decode_frame
from src.stream_runtime import StreamState


def grid_fixture():
    cfg = PipelineConfig(map_size_m=12.)
    y, x = np.indices((cfg.grid_dim, cfg.grid_dim))
    ids = np.arange(x.size).reshape(x.shape)
    return cfg, ids, {
        "x_idx": x.ravel(), "y_idx": y.ravel(),
        "level": np.zeros(x.size, dtype=np.uint8),
        "flags": np.full(x.size, 8, dtype=np.uint8),
    }


def test_csr_uses_shared_cardinal_edges_without_duplicates():
    ids = np.array([[0, 0, 1], [0, 0, 2], [3, 3, 2]])
    graph = leaf_adjacency(ids, 4)
    actual = [graph.neighbours[graph.offsets[i]:graph.offsets[i+1]].tolist() for i in range(4)]
    assert actual == [[1, 2, 3], [0, 2], [0, 1, 3], [0, 2]]


def test_clearance_stencil_matches_euclidean_dilation():
    mask = np.random.default_rng(42).random((19, 23)) < .08
    for radius in (.9, 1.):
        expected = distance_transform_edt(~mask, sampling=.25) <= radius + .25 / np.sqrt(2)
        np.testing.assert_array_equal(inflated(mask, radius, .25), expected)


def test_path_routes_around_inflated_hazard_and_anonymous_dynamic_occupancy():
    cfg, ids, columns = grid_fixture()
    flags = columns["flags"]
    flags[ids[18:21, 30]] = 8 | (2 << 1)
    planner = OccupancyPlanner(cfg)
    first = planner.update(ids, columns, (4., 0.), now=0.)
    assert first["path_status"] == "OPTIMAL"
    assert first["adaptive_nodes_expanded"] > 0
    # Insert occupied cells on the current route before the periodic deadline.
    flags[ids[23:26, 30:32]] = 9
    second = planner.update(ids, columns, (4., 0.), now=.05)
    blocked, caution = occupancy_masks(ids, flags, cfg)
    assert second["path_status"] == "REROUTING"
    assert not route_clear(first["planned_path"], blocked, .25, cfg.half_size)
    assert route_clear(second["planned_path"], blocked, .25, cfg.half_size)
    assert planner.update(ids, columns, (4., 0.), now=1.2)["path_status"] == "OPTIMAL"
    fixed_nodes, fixed_ms = fixed_sample(blocked, caution, (24,24), (24,40), .25)
    assert fixed_nodes > 0 and fixed_ms > 0


def test_snap_clamps_then_snaps_and_reports_blocked():
    cfg, ids, columns = grid_fixture()
    allowed = np.zeros(ids.shape, dtype=bool)
    allowed[-2, -2] = True
    assert snap((500., 500.), allowed, .25, 6., 3.)[0] == (46, 46)
    assert snap((0., 0.), allowed, .25, 6., 3.) is None
    planner = OccupancyPlanner(cfg)
    first = planner.update(ids, columns, (4, 0), now=0)
    columns["flags"][ids[:, 31:]] = 8 | (3 << 1)
    second = planner.update(ids, columns, (99, 0), now=.05)
    assert first["path_status"] == "OPTIMAL"
    assert second["path_status"] == "BLOCKED"
    assert second["planned_path"] == first["planned_path"]
    assert second["path_reason"]


def test_supercover_rejects_corner_cut_and_mask_boundary():
    blocked = np.zeros((3, 3), dtype=bool)
    blocked[0, 1] = True
    assert not route_clear([(0.5, 0.5), (1.5, 1.5)], blocked, 1., 0.)
    assert not route_clear([(1., 0.5), (1., 1.5)], blocked, 1., 0.)
    assert route_clear([(0.5, 0.5), (0.5, 1.5)], blocked, 1., 0.)
    rng = np.random.default_rng(42)
    blocked = rng.random((12, 12)) < .12
    for path in rng.integers(0, 24, size=(60, 4, 2)) / 2:
        expected = all(not blocked[segment_cells(a, b, blocked.shape, 1., 0.)].any()
                       for a, b in zip(path, path[1:]))
        assert route_clear(path, blocked, 1., 0.) == expected


def test_real_stream_default_reachable_and_goal_control_replans_while_paused():
    state = StreamState()
    state.execute({"action": "reset"})
    extra = decode_frame(state.latest_binary)["metadata"]["extras"]
    assert extra["path_status"] == "OPTIMAL"
    assert len(extra["planned_path"]) >= 2
    assert np.linalg.norm(np.subtract(extra["goal_xy"], state.config.planner_default_goal)) <= 3.
    reply = state.execute({"action": "set_goal", "goal_xy": [-5, 5]})
    assert not reply["playing"]
    extra = decode_frame(state.latest_binary)["metadata"]["extras"]
    assert extra["runtime_config"]["goal_xy"] == [-5, 5]
    assert extra["path_status"] == "OPTIMAL"
    assert np.linalg.norm(np.subtract(extra["goal_xy"], [-5, 5])) <= 3.
