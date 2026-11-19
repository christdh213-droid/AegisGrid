"""
PerceptionPipeline — orchestrates the full per-frame processing chain.

Hooks are injected at construction; Codex swaps in real implementations
without modifying this file.
"""
from __future__ import annotations

import time
from typing import Generator, Optional

import numpy as np

from src.config import PipelineConfig
from src.schemas import AdaptiveCell, FrameOutput
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
    ):
        self.cfg = config
        self.pre_hook = pre_hook or DefaultPreProcessHook()
        self.post_hook = post_hook or DefaultPostMapHook()
        self.stream_callback = stream_callback or DefaultStreamCallbackHook()

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
        extras: dict = {}

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

        # ── 2. Dynamic detection ─────────────────────────────────────────
        timer.start("dynamic")
        dynamic_mask, tracks, ghost_count = self.tracker.process(
            self.fixed_map.max_z.copy(),
            self.fixed_map.count.copy(),
            self.fixed_map.mean_z.copy(),
            dt,
        )
        timer.stop("dynamic")

        # ── 3. Adaptive quadtree build ───────────────────────────────────
        timer.start("adaptive")
        cells = self.adaptive_map.build(self.fixed_map, dynamic_mask)
        timer.stop("adaptive")

        # ── 4. Post-map hook (Codex: traversability) ─────────────────────
        cells, post_extras = self.post_hook(cells, self.cfg)
        if post_extras:
            extras.update(post_extras)

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

        # ── 7. Stream callback (Codex: WebSocket / UI push) ─────────────
        frame_dict = frame.to_dict()
        self.stream_callback(frame_dict)

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

