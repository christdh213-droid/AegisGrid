"""
PerceptionPipeline — orchestrates the full per-frame processing chain.

Hooks are injected at construction; Codex swaps in real implementations
without modifying this file.
"""
from __future__ import annotations

import time
from typing import Generator, Optional

import numpy as np

from src.config import PipelineConfig, RuntimeConfig
from src.plugins import DustOutlierFilterHook, ImuTiltCompensationHook, TraversabilityRiskHook
from src.schemas import AdaptiveCell, FrameOutput, default_frame_extras
from src.hooks import (
    DefaultPreProcessHook,
    DefaultPostMapHook,
    DefaultStreamCallbackHook,
)
from src.mapping_engine import FixedGridMap, AdaptiveQuadtreeMap
from src.dynamic_tracker import DynamicTracker
from src.benchmark import FrameTimer, build_frame_metrics


class PerceptionPipeline:
    """
    Full adaptive LiDAR perception pipeline.

    Usage:
        pipe = PerceptionPipeline(config)
        for frame in pipe.run_stream(stream, num_frames=50):
            print(frame.metrics.to_dict())
    """

    def __init__(
        self,
        config: PipelineConfig,
        pre_hook=None,
        post_hook=None,
        stream_callback=None,
        source_hook=None,
        pre_map_hook=None,
        post_track_hook=None,
        post_frame_hook=None,
        perception=False,
    ):
        self.cfg = config
        self.pre_hook = pre_hook or DefaultPreProcessHook()
        self.post_hook = post_hook or DefaultPostMapHook()
        self.stream_callback = stream_callback or DefaultStreamCallbackHook()
        self.source_hook = source_hook
        self.pre_map_hook = pre_map_hook
        self.post_track_hook = post_track_hook
        self.post_frame_hook = post_frame_hook
        self.perception = perception
        self.imu = ImuTiltCompensationHook()
        self.dust_filter = DustOutlierFilterHook()
        self.risk = TraversabilityRiskHook()

        # Internal state
        self.fixed_map = FixedGridMap(config)
        self.adaptive_map = AdaptiveQuadtreeMap(config)
        self.tracker = DynamicTracker(config)

        self._frame_id = 0
        self._time = 0.0

    def reset(self) -> None:
        """Reset all internal state for replay / dataset switch."""
        self.fixed_map = FixedGridMap(self.cfg)
        self.adaptive_map = AdaptiveQuadtreeMap(self.cfg)
        self.tracker = DynamicTracker(self.cfg)
        self._frame_id = 0
        self._time = 0.0

    def process_frame(
        self,
        points: np.ndarray,
        dt: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> FrameOutput:
        """Process a single frame of Nx3 LiDAR points."""
        if dt is None:
            dt = self.cfg.dt
        if metadata is None:
            metadata = {}

        timer = FrameTimer()
        t_start = time.perf_counter()
        extras = default_frame_extras()

        if self.source_hook:
            points, metadata = self.source_hook(points, metadata)

        if self.perception:
            metadata = metadata.copy()
            truth = metadata.pop("is_injected_noise", None)
            runtime = metadata.get("runtime_config") or RuntimeConfig()
            points, imu_extras = self.imu(points, metadata, runtime)
            extras.update(imu_extras)
            start = time.perf_counter()
            clean_simulation = (
                metadata.get("source") == "synthetic"
                and metadata.get("simulated_outlier_ratio") == 0
            )
            run_filter = runtime.sor_enabled and not clean_simulation
            keep = self.dust_filter(points, self.cfg) if run_filter else np.ones(len(points), dtype=bool)
            extras["sor_ms"] = (time.perf_counter() - start) * 1000 if run_filter else None
            removed = ~keep
            removed_count = int(removed.sum())
            extras["filtered_noise_points"] = removed_count
            extras["noise_removal_pct"] = 100.0 * removed_count / len(points) if len(points) else 0.0
            if truth is not None:
                if truth.any():
                    extras["noise_recall_pct"] = 100.0 * np.count_nonzero(removed & truth) / truth.sum()
                if not truth.all():
                    extras["true_points_lost_pct"] = 100.0 * np.count_nonzero(removed & ~truth) / np.count_nonzero(~truth)
            if removed_count:
                points = points[keep]

        # ── 0. Pre-process hook (Codex: SOR / IMU levelling) ─────────────
        points, hook_extras = self.pre_hook(points, metadata)
        if hook_extras:
            extras.update(hook_extras)

        # Handle empty point cloud
        if points.shape[0] == 0:
            points = np.empty((0, 3), dtype=np.float64)

        # ── 1. Fixed Grid rasterisation ──────────────────────────────────
        timer.start("fixed")
        self.fixed_map.reset()
        self.fixed_map.update(points)
        timer.stop("fixed")

        map_options = self.pre_map_hook(self.fixed_map, metadata.get('runtime_config')) if self.pre_map_hook else {}

        # ── 2. Dynamic detection ─────────────────────────────────────────
        timer.start("dynamic")
        dynamic_mask, tracks, ghost_count = self.tracker.process(
            self.fixed_map.max_z,
            self.fixed_map.count,
            self.fixed_map.mean_z,
            dt,
        )
        timer.stop("dynamic")

        # ── 3. Adaptive quadtree build ───────────────────────────────────
        timer.start("adaptive")
        cells = self.adaptive_map.build(self.fixed_map, dynamic_mask, **map_options)
        timer.stop("adaptive")

        # ── 4. Post-map hook (Codex: traversability) ─────────────────────
        cells, post_extras = self.post_hook(cells, self.cfg)
        if post_extras:
            extras.update(post_extras)
        if self.post_track_hook:
            cells, track_extras = self.post_track_hook(cells, tracks, self.cfg)
            extras.update(track_extras)

        # ── 5. Metrics ───────────────────────────────────────────────────
        total_ms = (time.perf_counter() - t_start) * 1000.0
        metrics = build_frame_metrics(
            timer, self.fixed_map, self.adaptive_map, ghost_count, total_ms,
        )

        # ── 6. Assemble output ───────────────────────────────────────────
        frame = FrameOutput(
            frame_id=self._frame_id,
            timestamp=self._time,
            config_snapshot=self.cfg.to_dict(),
            adaptive_cells=cells,
            dynamic_tracks=tracks,
            metrics=metrics,
            fixed_grid_shape=(self.cfg.grid_dim, self.cfg.grid_dim),
            extras=extras,
        )

        if self.perception:
            frame.extras.update(self.risk(frame, self))

        # ── 7. Stream callback (Codex: WebSocket / UI push) ─────────────
        if self.post_frame_hook:
            frame.extras.update(self.post_frame_hook(frame, self) or {})
        # Avoid constructing thousands of dictionaries for a no-op callback.
        if type(self.stream_callback) is not DefaultStreamCallbackHook:
            self.stream_callback(frame.to_dict())

        self._frame_id += 1
        self._time += dt

        return frame

    def run_stream(
        self,
        stream,
        num_frames: int,
    ) -> Generator[FrameOutput, None, None]:
        """
        Process `num_frames` from an iterable stream, yielding FrameOutputs.

        Args:
            stream: Iterable yielding (points_xyz: np.ndarray, metadata: dict).
            num_frames: Number of frames to process.

        Yields:
            FrameOutput for each processed frame.
        """
        for i, (points, metadata) in enumerate(stream):
            if i >= num_frames:
                break
            dt = metadata.get("dt", self.cfg.dt)
            yield self.process_frame(points, dt=dt, metadata=metadata)

