"""Deterministic governor policy and live profile integration checks."""
import numpy as np
import pytest

from src.compute_governor import ComputeGovernor
from src.config import GOVERNOR_PROFILES, PipelineConfig
from src.mapping_engine import AdaptiveQuadtreeMap, FixedGridMap
from src.stream_protocol import decode_frame
from src.stream_runtime import StreamState


def test_spike_adapts_and_converges_without_oscillation():
    cfg = PipelineConfig()
    governor = ComputeGovernor(cfg)
    mode = "CPU Load Spike Stress Test"
    prior = governor.select(mode)
    governor.observe(10., 0., prior, 2400, 12.)
    settings = []
    for frame in range(30):
        active = governor.select(mode)
        # With more merging, simulated remaining work falls into the target band.
        load = 32. - 16. * governor.fraction
        result = governor.observe(load, 0., active, 2400, 12.)
        settings.append(governor.select(mode).height)
        if frame == 2:
            assert settings[-1] > prior.height
    assert result["governed_latency_ms"] <= 18.
    assert max(settings[-8:]) - min(settings[-8:]) <= .05 * (.28 - .12)
    assert settings == sorted(settings)
    assert result["radio_mb_s_fixed"] == 256 * 256 * 12 / 1e6
    assert result["radio_mb_s_adaptive"] == 2400 * 10 * 12 / 1e6


def test_persistent_overload_locks_and_reduces_only_regular_replanning():
    governor = ComputeGovernor(PipelineConfig())
    mode = "Low-Power UGV Node"
    for _ in range(30):
        active = governor.select(mode)
        result = governor.observe(15., 0., active, 2400, None)
    assert result["governor_state"] == "BUDGET_LOCKED"
    assert active.height == pytest.approx(.28)
    assert active.gradient == pytest.approx(.22)
    assert active.replan_interval_s == pytest.approx(.4)
    assert result["radio_mb_s_fixed"] is None
    nominal = governor.select("Unrestricted")
    assert nominal.height == .12 and nominal.replan_interval_s == .1


def test_stress_busy_loop_scales_with_leaves():
    ticks = iter(i / 100_000 for i in range(10_000))
    governor = ComputeGovernor(PipelineConfig(), clock=lambda: next(ticks))
    governor.select("CPU Load Spike Stress Test")
    assert governor.inject_stress(1200) == pytest.approx(6., abs=.03)
    assert governor.inject_stress(2400) == pytest.approx(12., abs=.03)


def test_previous_occupancy_buffer_and_high_span_keep_l0():
    cfg = PipelineConfig(map_size_m=4.)
    governor = ComputeGovernor(cfg)
    active = governor.select("Low-Power UGV Node")
    fixed = FixedGridMap(cfg)
    fixed.update(np.array([[1.125, 1.125, 0.], [1.125, 1.125, .5]]))
    previous = np.zeros((cfg.grid_dim, cfg.grid_dim), dtype=bool)
    previous[4, 4] = True
    options = governor.map_options(fixed, previous, active)
    expected = np.zeros_like(previous)
    expected[2:7, 2:7] = True
    expected[12, 12] = True
    np.testing.assert_array_equal(options["protected_mask"], expected)
    adaptive = AdaptiveQuadtreeMap(cfg)
    adaptive.build(fixed, np.zeros_like(previous), **options)
    assert (adaptive.leaf_arrays["level"][adaptive.leaf_id_grid[expected]] == 0).all()


@pytest.mark.parametrize("mode", GOVERNOR_PROFILES)
def test_profile_wire_data_and_golden_metrics(mode):
    state = StreamState()
    state.execute({"action": "set_profile", "profile": mode})
    for _ in range(13):
        state.execute({"action": "step"})
        metrics = state.current_frame.metrics
        assert metrics.obstacle_rmse_m <= .05
        assert metrics.ghost_cells_detected == 0
        if mode == "Unrestricted":
            assert metrics.cell_reduction_pct >= 65
            assert metrics.memory_saved_pct >= 60
    extra = decode_frame(state.latest_binary)["metadata"]["extras"]
    assert extra["governor_mode"] == mode
    assert extra["target_budget_ms"] == GOVERNOR_PROFILES[mode].budget_ms
    assert extra["radio_mb_s_adaptive"] < extra["radio_mb_s_fixed"]
