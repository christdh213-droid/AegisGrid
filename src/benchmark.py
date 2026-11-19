"""
Benchmark utilities: per-frame and aggregate metric computation.
"""
from __future__ import annotations

import time
from typing import List

import numpy as np

from src.config import PipelineConfig
from src.mapping_engine import FixedGridMap, AdaptiveQuadtreeMap
from src.schemas import FrameMetrics


def compute_obstacle_rmse(
    fixed: FixedGridMap,
    adaptive: AdaptiveQuadtreeMap,
) -> float:
    """
    RMSE of max_z between Fixed and Adaptive across obstacle cells only.

    Obstacle cells: span_z > 0.20 m or max_z > 0.30 m in the fixed grid.
    """
    occupied = fixed.count > 0
    span = fixed.span_z
    obs_mask = occupied & ((span > 0.20) | (fixed.max_z > 0.30))

    if not np.any(obs_mask):
        return 0.0

    dense_adaptive = adaptive.reconstruct_dense_max_z()
    fixed_vals = fixed.max_z[obs_mask].astype(np.float64)
    adaptive_vals = dense_adaptive[obs_mask].astype(np.float64)

    # NaN in adaptive → cell wasn't covered (shouldn't happen but be safe)
    valid = ~np.isnan(adaptive_vals)
    if not np.any(valid):
        return 0.0

    diff = fixed_vals[valid] - adaptive_vals[valid]
    return float(np.sqrt(np.mean(diff ** 2)))


class FrameTimer:
    """Context-manager style timer for sub-step latency measurement."""

    def __init__(self):
        self._starts: dict = {}
        self._durations: dict = {}

    def start(self, name: str) -> None:
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        elapsed = (time.perf_counter() - self._starts[name]) * 1000.0
        self._durations[name] = elapsed
        return elapsed

    def get_ms(self, name: str) -> float:
        return self._durations.get(name, 0.0)


def build_frame_metrics(
    timer: FrameTimer,
    fixed: FixedGridMap,
    adaptive: AdaptiveQuadtreeMap,
    ghost_count: int,
    total_ms: float,
) -> FrameMetrics:
    """Assemble a FrameMetrics from the timer and map states."""
    fixed_active = fixed.active_cells
    adaptive_leaves = len(adaptive.leaves)

    cell_reduction = (
        (1.0 - adaptive_leaves / fixed_active) * 100.0
        if fixed_active > 0 else 0.0
    )

    fixed_mem = fixed.memory_kb()
    adaptive_mem = adaptive.memory_kb()
    mem_saved = (
        (1.0 - adaptive_mem / fixed_mem) * 100.0
        if fixed_mem > 0 else 0.0
    )

    rmse = compute_obstacle_rmse(fixed, adaptive)
    fps = 1000.0 / total_ms if total_ms > 0 else 0.0

    return FrameMetrics(
        fixed_latency_ms=timer.get_ms("fixed"),
        adaptive_latency_ms=timer.get_ms("adaptive"),
        total_pipeline_ms=total_ms,
        fps=fps,
        fixed_active_cells=fixed_active,
        adaptive_leaf_cells=adaptive_leaves,
        cell_reduction_pct=cell_reduction,
        fixed_memory_kb=fixed_mem,
        adaptive_memory_kb=adaptive_mem,
        memory_saved_pct=mem_saved,
        obstacle_rmse_m=rmse,
        ghost_cells_detected=ghost_count,
    )


def aggregate_metrics(all_metrics: List[FrameMetrics]) -> dict:
    """Compute summary statistics over a list of per-frame metrics."""
    if not all_metrics:
        return {}

    def _avg(values):
        return float(np.mean(values))

    def _med(values):
        return float(np.median(values))

    n = len(all_metrics)
    return {
        "num_frames": n,
        "avg_total_pipeline_ms": _avg([m.total_pipeline_ms for m in all_metrics]),
        "median_total_pipeline_ms": _med([m.total_pipeline_ms for m in all_metrics]),
        "avg_fps": _avg([m.fps for m in all_metrics]),
        "avg_fixed_latency_ms": _avg([m.fixed_latency_ms for m in all_metrics]),
        "avg_adaptive_latency_ms": _avg([m.adaptive_latency_ms for m in all_metrics]),
        "avg_cell_reduction_pct": _avg([m.cell_reduction_pct for m in all_metrics]),
        "avg_memory_saved_pct": _avg([m.memory_saved_pct for m in all_metrics]),
        "avg_obstacle_rmse_m": _avg([m.obstacle_rmse_m for m in all_metrics]),
        "max_obstacle_rmse_m": float(max(m.obstacle_rmse_m for m in all_metrics)),
        "total_ghost_cells": int(sum(m.ghost_cells_detected for m in all_metrics)),
        "avg_fixed_active_cells": _avg([m.fixed_active_cells for m in all_metrics]),
        "avg_adaptive_leaf_cells": _avg([m.adaptive_leaf_cells for m in all_metrics]),
        "avg_fixed_memory_kb": _avg([m.fixed_memory_kb for m in all_metrics]),
        "avg_adaptive_memory_kb": _avg([m.adaptive_memory_kb for m in all_metrics]),
    }

