from dataclasses import replace

import numpy as np

from src.config import PipelineConfig, RuntimeConfig
from src.data_source import SyntheticLidarStream
from src.pipeline import PerceptionPipeline
from src.plugins import TraversabilityRiskHook, tilt_rotation
from src.stream_protocol import decode_frame
from src.stream_runtime import StreamState


def test_tilt_rotation_and_compensation():
    config = PipelineConfig()
    clean = SyntheticLidarStream(config)
    tilted = SyntheticLidarStream(config)
    runtime = RuntimeConfig(simulate_hill_tilt=True, tilt_mode="Manual", pitch_deg=8, roll_deg=8)
    tilted.runtime_config = runtime
    rotation = tilt_rotation(8, 8)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
    for _ in range(13):
        expected, _ = next(clean)
        points, metadata = next(tilted)
    assert np.max(np.abs(points[:31500, 2])) > 1
    pipeline = PerceptionPipeline(config, perception=True)
    enabled = pipeline.process_frame(points, metadata={**metadata, "runtime_config": runtime})
    np.testing.assert_allclose(points @ rotation, expected, atol=1e-9)
    assert enabled.extras["tilt_corrected"]
    assert enabled.metrics.obstacle_rmse_m <= .05
    pipeline.reset()
    disabled = pipeline.process_frame(points, metadata={
        **metadata, "runtime_config": replace(runtime, imu_levelling_enabled=False),
    })
    assert not disabled.extras["tilt_corrected"]
    assert disabled.extras["hazard_pct"] > 0


def test_dust_quality_gates_and_wire_risk():
    state = StreamState()
    state.execute({"action": "configure", "settings": {"outlier_ratio": .15}})
    for _ in range(8):
        state.execute({"action": "step"})
        frame = state.current_frame
        assert frame.extras["noise_recall_pct"] >= 95
        assert frame.extras["true_points_lost_pct"] <= 1
        assert frame.metrics.ghost_cells_detected == 0
        assert frame.metrics.obstacle_rmse_m <= .05
    wire = decode_frame(state.latest_binary)
    pct = [frame.extras[f"{name}_pct"] for name in ("safe", "caution", "hazard", "unknown")]
    np.testing.assert_allclose(sum(pct), 100.)
    risk = (wire["cells"]["flags"] >> 1) & 3
    assert [TraversabilityRiskHook.LABELS[i] for i in risk] == [
        cell.traversability for cell in frame.adaptive_cells
    ]
    state.execute({"action": "configure", "settings": {"sor_enabled": False}})
    state.execute({"action": "step"})
    assert state.current_frame.extras["noise_recall_pct"] == 0
    assert state.current_frame.extras["filtered_noise_points"] == 0


def test_risk_boundaries_and_precedence():
    cfg = PipelineConfig()
    spans = np.array([.12, .120001, .35, .350001, .0, .0, .0, .0, .8])
    steps = np.array([.15, .0, .0, .0, .150001, .35, .350001, .0, .8])
    dynamic = np.array([False] * 7 + [True, True])
    supported = np.array([True] * 8 + [False])
    assert TraversabilityRiskHook.classify(spans, steps, dynamic, supported, cfg).tolist() == [
        0, 1, 1, 2, 1, 1, 2, 2, 3,
    ]


def test_neighbourhood_step_matches_supported_eight_neighbours():
    rng = np.random.default_rng(42)
    height = rng.normal(size=(9, 13)).astype(np.float32)
    supported = rng.random(height.shape) > .3
    expected = np.zeros_like(height)
    for y, x in np.argwhere(supported):
        region = (slice(max(0, y - 1), y + 2), slice(max(0, x - 1), x + 2))
        expected[y, x] = np.max(np.abs(height[region][supported[region]] - height[y, x]))
    np.testing.assert_array_equal(
        TraversabilityRiskHook.neighbourhood_step(height, supported), expected
    )
