"""
Codex Plugin Contract — typed hook interfaces and default no-op implementations.

Phase-2 (Codex) injects real implementations *without* touching core pipeline code.
"""
from __future__ import annotations

from typing import Protocol, List, Tuple, Dict, Any, runtime_checkable

import numpy as np

from src.schemas import AdaptiveCell
from src.config import PipelineConfig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Protocol interfaces
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@runtime_checkable
class PreProcessHook(Protocol):
    """
    # CODEX_HOOK: PRE_PROCESS
    Plug in Dust/Rain/Smoke Statistical Outlier Removal (SOR)
    and IMU pitch/roll ground-levelling here.

    Args:
        points:   Nx3 float64 array of (x, y, z) LiDAR points.
        metadata: Dict with optional keys like 'imu_pitch', 'imu_roll'.

    Returns:
        (filtered_points, hook_extras) — hook_extras is merged into
        FrameOutput.extras so the UI can display filter statistics.
    """
    def __call__(
        self, points: np.ndarray, metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]: ...


@runtime_checkable
class PostMapHook(Protocol):
    """
    # CODEX_HOOK: POST_MAP
    Plug in Off-Road Slope/Roughness Traversability Classification
    (SAFE / CAUTION / HAZARD) here.

    Args:
        cells:  List of AdaptiveCell leaf nodes from the quadtree.
        config: Current PipelineConfig snapshot.

    Returns:
        (annotated_cells, hook_extras) — each cell's .traversability
        field should be set to "SAFE", "CAUTION", or "HAZARD".
    """
    def __call__(
        self, cells: List[AdaptiveCell], config: PipelineConfig
    ) -> Tuple[List[AdaptiveCell], Dict[str, Any]]: ...


@runtime_checkable
class StreamCallbackHook(Protocol):
    """
    # CODEX_HOOK: STREAM_CALLBACK
    Plug in live WebSocket / Streamlit / PyQt UI frame push here.

    Args:
        frame_dict: The complete FrameOutput.to_dict() payload,
                    ready for JSON serialisation and streaming.
    """
    def __call__(self, frame_dict: Dict[str, Any]) -> None: ...


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Default no-op implementations (pass-through)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DefaultPreProcessHook:
    """No-op: returns points and metadata unchanged."""
    def __call__(
        self, points: np.ndarray, metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        return points, {}


class DefaultPostMapHook:
    """No-op: returns cells unchanged."""
    def __call__(
        self, cells: List[AdaptiveCell], config: PipelineConfig
    ) -> Tuple[List[AdaptiveCell], Dict[str, Any]]:
        return cells, {}


class DefaultStreamCallbackHook:
    """No-op: silently discards the frame dict."""
    def __call__(self, frame_dict: Dict[str, Any]) -> None:
        pass


class SourceHook(Protocol):
    """# CODEX_HOOK: SOURCE -- sensor physics, before PRE_PROCESS."""
    def __call__(self, points: np.ndarray, metadata: dict) -> tuple[np.ndarray, dict]: ...


class PreMapHook(Protocol):
    """# CODEX_HOOK: PRE_MAP -- return per-build keyword options (no-op default)."""
    def __call__(self, fixed, runtime_config) -> dict: ...


class PostTrackHook(Protocol):
    """# CODEX_HOOK: POST_TRACK -- annotate cells after map and tracking."""
    def __call__(self, cells, tracks, config) -> tuple[list[AdaptiveCell], dict]: ...


class PostFrameHook(Protocol):
    """# CODEX_HOOK: POST_FRAME -- planner/observer, before serialization."""
    def __call__(self, frame, pipeline) -> dict: ...

