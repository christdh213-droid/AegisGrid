"""
Clean dataclass schemas for every pipeline output.

Every `.to_dict()` returns native Python types (float, int, str, list, dict)
so `json.dumps()` never chokes on numpy scalars.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


def _float(v) -> float:
    """Coerce any numeric (including np scalars) to native float."""
    return float(v)


def _int(v) -> int:
    return int(v)


def _str(v) -> str:
    return str(v)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AdaptiveCell — one leaf node in the variable-resolution quadtree
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class AdaptiveCell:
    cell_id: int
    x: float            # centre X (metres)
    y: float            # centre Y (metres)
    size: float          # edge length (metres)
    level: int           # pyramid level (0 = finest 0.25 m)
    min_z: float
    max_z: float
    mean_z: float
    span_z: float
    point_count: int
    is_dynamic: bool
    traversability: str = "UNKNOWN"   # Reserved for Codex Phase-2

    def to_dict(self) -> dict:
        return {
            "cell_id":        _int(self.cell_id),
            "x":              _float(self.x),
            "y":              _float(self.y),
            "size":           _float(self.size),
            "level":          _int(self.level),
            "min_z":          _float(self.min_z),
            "max_z":          _float(self.max_z),
            "mean_z":         _float(self.mean_z),
            "span_z":         _float(self.span_z),
            "point_count":    _int(self.point_count),
            "is_dynamic":     bool(self.is_dynamic),
            "traversability": _str(self.traversability),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DynamicTrack — one persistent tracked moving object
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class DynamicTrack:
    track_id: int
    x: float
    y: float
    z_max: float
    vx: float
    vy: float
    speed: float
    width: float
    length: float
    pred_x: float
    pred_y: float
    age: int
    classification: str   # "VEHICLE" or "PEDESTRIAN"

    def to_dict(self) -> dict:
        return {
            "track_id":       _int(self.track_id),
            "x":              _float(self.x),
            "y":              _float(self.y),
            "z_max":          _float(self.z_max),
            "vx":             _float(self.vx),
            "vy":             _float(self.vy),
            "speed":          _float(self.speed),
            "width":          _float(self.width),
            "length":         _float(self.length),
            "pred_x":         _float(self.pred_x),
            "pred_y":         _float(self.pred_y),
            "age":            _int(self.age),
            "classification": _str(self.classification),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FrameMetrics — per-frame performance & quality telemetry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class FrameMetrics:
    fixed_latency_ms: float
    adaptive_latency_ms: float
    total_pipeline_ms: float
    fps: float
    fixed_active_cells: int
    adaptive_leaf_cells: int
    cell_reduction_pct: float
    fixed_memory_kb: float
    adaptive_memory_kb: float
    memory_saved_pct: float
    obstacle_rmse_m: float
    ghost_cells_detected: int

    def to_dict(self) -> dict:
        return {
            "fixed_latency_ms":    _float(self.fixed_latency_ms),
            "adaptive_latency_ms": _float(self.adaptive_latency_ms),
            "total_pipeline_ms":   _float(self.total_pipeline_ms),
            "fps":                 _float(self.fps),
            "fixed_active_cells":  _int(self.fixed_active_cells),
            "adaptive_leaf_cells": _int(self.adaptive_leaf_cells),
            "cell_reduction_pct":  _float(self.cell_reduction_pct),
            "fixed_memory_kb":     _float(self.fixed_memory_kb),
            "adaptive_memory_kb":  _float(self.adaptive_memory_kb),
            "memory_saved_pct":    _float(self.memory_saved_pct),
            "obstacle_rmse_m":     _float(self.obstacle_rmse_m),
            "ghost_cells_detected": _int(self.ghost_cells_detected),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FrameOutput — complete per-frame pipeline output
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class FrameOutput:
    frame_id: int
    timestamp: float
    config_snapshot: dict
    adaptive_cells: List[AdaptiveCell]
    dynamic_tracks: List[DynamicTrack]
    metrics: FrameMetrics
    fixed_grid_shape: Tuple[int, int]
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "frame_id":        _int(self.frame_id),
            "timestamp":       _float(self.timestamp),
            "config_snapshot":  self.config_snapshot,
            "adaptive_cells":  [c.to_dict() for c in self.adaptive_cells],
            "dynamic_tracks":  [t.to_dict() for t in self.dynamic_tracks],
            "metrics":         self.metrics.to_dict(),
            "fixed_grid_shape": [_int(d) for d in self.fixed_grid_shape],
            "extras":          self.extras,
        }

