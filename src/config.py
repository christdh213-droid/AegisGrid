"""
Centralized pipeline configuration dataclass.

All tunable parameters live here so that Codex can directly bind UI sliders
to config fields without hunting through scattered magic numbers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict


@dataclass
class PipelineConfig:
    """Master configuration for the adaptive LiDAR perception pipeline."""

    # ── Grid bounds & resolution ──────────────────────────────────────────
    map_size_m: float = 64.0          # Total map extent (square), metres
    finest_res_m: float = 0.25        # Finest cell resolution, metres
    max_quadtree_level: int = 4       # Pyramid depth (L0=0.25m … L4=4.0m)

    # ── Adaptive merge thresholds ─────────────────────────────────────────
    merge_height_threshold: float = 0.12   # Max vertical span for merge (m)
    merge_gradient_threshold: float = 0.10 # Max neighbour height diff (m)
    min_points_per_cell: int = 1           # Min points to consider a cell valid

    # ── Dynamic perception ────────────────────────────────────────────────
    dynamic_height_residual_m: float = 0.30  # Height above static ref → dynamic
    min_cluster_cells: int = 4               # Min cells for a valid cluster
    min_component_area_m2: float = 0.15
    max_component_extent_m: float = 8.0
    elongated_extent_m: float = 5.0
    max_elongated_aspect: float = 6.0
    max_cluster_cells: int = 400             # Max cells for a valid cluster
    track_association_dist_m: float = 2.5    # Max match distance (Hungarian)
    track_max_missed_frames: int = 3         # Frames before track deletion
    prediction_horizon_s: float = 1.5        # Future position prediction (s)

    # ── Static reference EMA ──────────────────────────────────────────────
    static_ema_alpha: float = 0.05           # EMA learning rate for static ref

    # ── Simulation / synthetic data ───────────────────────────────────────
    points_per_frame: int = 45000
    sensor_noise_std: float = 0.02
    dust_voxel_m: float = 0.5
    dust_min_elevation_m: float = 0.38
    dust_near_neighbors: int = 10
    dust_far_neighbors: int = 9
    caution_span_m: float = 0.12
    caution_step_m: float = 0.15
    hazard_span_m: float = 0.35
    hazard_step_m: float = 0.35
    planner_terrain_clearance_m: float = 0.9
    planner_dynamic_clearance_m: float = 1.0
    planner_default_goal: tuple[float, float] = (-10.0, 8.0)
    dt: float = 0.1
    seed: int = 42

    # ── Derived (computed on post_init) ───────────────────────────────────
    grid_dim: int = field(init=False)
    half_size: float = field(init=False)

    def __post_init__(self) -> None:
        self.grid_dim = int(self.map_size_m / self.finest_res_m)   # 256
        self.half_size = self.map_size_m / 2.0                      # 32.0

    # ── Serialization helpers ─────────────────────────────────────────────
    def to_dict(self) -> dict:
        """Return a JSON-safe plain-Python dictionary."""
        d = asdict(self)
        # asdict already produces native Python types for simple dataclasses
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        # Drop derived fields that are computed in __post_init__
        d = {k: v for k, v in d.items() if k not in ("grid_dim", "half_size")}
        return cls(**d)


@dataclass(frozen=True)
class GovernorProfile:
    budget_ms: float
    cpu_scale: float
    height_ceiling_m: float | None
    gradient_ceiling_m: float | None
    stress_ms: float = 0.0


GOVERNOR_PROFILES = {
    "Unrestricted": GovernorProfile(33., 1., None, None),
    "Tactical Edge ECU": GovernorProfile(18., 2., .24, .18),
    "Low-Power UGV Node": GovernorProfile(12., 3., .28, .22),
    "CPU Load Spike Stress Test": GovernorProfile(18., 1., .28, .22, 12.),
}


@dataclass(frozen=True)
class RuntimeConfig:
    """Immutable between-frame settings; module switches are scaffolding in Session 1.

    goal_xy=None defers reachable-goal selection to the planner session.
    """
    outlier_ratio: float = 0.0
    sor_enabled: bool = True
    imu_levelling_enabled: bool = True
    simulate_hill_tilt: bool = False
    tilt_mode: str = "Auto"
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    imu_noise_deg: float = 0.0
    planner_enabled: bool = True
    goal_xy: tuple[float, float] | None = None
    governor_profile: str = "Unrestricted"
    view_mode: str = "Split Wipe"
    speed: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

