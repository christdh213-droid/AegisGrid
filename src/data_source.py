"""
LiDAR data sources: file loader and synthetic stream generator.

SyntheticLidarStream produces realistic 64 m × 64 m scenes with undulating
terrain, static obstacles, and 4 dynamic actors (2 vehicles + 2 pedestrians).
"""
from __future__ import annotations

import csv
import os
from typing import Dict, Any, Iterator, Optional, Tuple

import numpy as np

from src.config import PipelineConfig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# File Loader
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LidarFileLoader:
    """
    Loads a single point-cloud file and returns (Nx3 XYZ, metadata dict).
    Supported: KITTI .bin (float32, Nx4), .npy (Nx3 or Nx4), .csv (x,y,z).
    """

    def __init__(self, filepath: str, auto_ground_shift: bool = True):
        self.filepath = filepath
        self.auto_ground_shift = auto_ground_shift

    def load(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        ext = os.path.splitext(self.filepath)[1].lower()
        if ext == ".bin":
            pts = np.fromfile(self.filepath, dtype=np.float32).reshape(-1, 4)
            xyz = pts[:, :3].astype(np.float64)
        elif ext == ".npy":
            pts = np.load(self.filepath)
            xyz = pts[:, :3].astype(np.float64)
        elif ext == ".csv":
            rows = []
            with open(self.filepath, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    rows.append([float(v) for v in row[:3]])
            xyz = np.array(rows, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported point cloud format: {ext}")

        metadata: Dict[str, Any] = {
            "source": self.filepath,
            "original_shape": xyz.shape,
        }

        # Optional ground-height normalisation (e.g. KITTI sensor at ~1.73 m)
        if self.auto_ground_shift and xyz.shape[0] > 100:
            z_10th = np.percentile(xyz[:, 2], 10)
            if abs(z_10th) > 0.5:
                xyz[:, 2] -= z_10th
                metadata["z_shift_applied"] = float(z_10th)

        return xyz, metadata

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Dict[str, Any]]]:
        yield self.load()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Synthetic Stream
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SyntheticLidarStream:
    """
    Generates realistic synthetic 64 m × 64 m LiDAR point clouds at ~10 Hz.

    Features:
      • Gently undulating terrain with road crown
      • Static obstacles (barriers, walls, boulders, pillars)
      • 4 dynamic actors (2 vehicles + 2 pedestrians) moving smoothly
      • Optional dust/rain outlier injection for Codex filter testing
    """

    def __init__(
        self,
        config: PipelineConfig,
        outlier_ratio: float = 0.0,
    ):
        self.cfg = config
        self.outlier_ratio = outlier_ratio
        self.rng = np.random.default_rng(config.seed)
        self.frame_idx = 0
        self.time = 0.0
        self.half = config.half_size  # 32.0

        # ── Static obstacles (centre_x, centre_y, half_w, half_l, height) ──
        self.static_obstacles = [
            # Concrete barriers along road edge
            (10.0, -5.0, 3.0, 0.3, 0.9),
            (10.0,  5.0, 3.0, 0.3, 0.9),
            (-8.0, -12.0, 2.0, 0.3, 0.8),
            # Road-side walls
            (0.0,  20.0, 8.0, 0.4, 2.0),
            (0.0, -20.0, 8.0, 0.4, 2.0),
            # Boulders / rocks
            (-15.0, 10.0, 1.0, 1.0, 0.6),
            (18.0, -15.0, 0.8, 0.8, 0.5),
            (-20.0, -18.0, 1.2, 1.2, 0.7),
            # Tree trunks / pillars
            (5.0, 15.0, 0.4, 0.4, 3.0),
            (-12.0, -8.0, 0.4, 0.4, 3.5),
            (22.0, 8.0, 0.3, 0.3, 2.8),
        ]

        # ── Dynamic actors ────────────────────────────────────────────────
        # (start_x, start_y, vx, vy, half_w, half_l, height, type)
        self.actors = [
            # Vehicle 1: moving +X
            {"x": -25.0, "y": -2.0, "vx": 7.0, "vy": 0.3,
             "hw": 0.9, "hl": 2.1, "h": 1.6, "type": "VEHICLE"},
            # Vehicle 2: moving -X
            {"x": 25.0, "y": 3.0, "vx": -8.0, "vy": -0.2,
             "hw": 0.9, "hl": 2.1, "h": 1.5, "type": "VEHICLE"},
            # Pedestrian 1: moving +Y
            {"x": -5.0, "y": -20.0, "vx": 0.2, "vy": 1.4,
             "hw": 0.35, "hl": 0.35, "h": 1.7, "type": "PEDESTRIAN"},
            # Pedestrian 2: moving diagonally
            {"x": 12.0, "y": -15.0, "vx": -0.8, "vy": 1.1,
             "hw": 0.35, "hl": 0.35, "h": 1.6, "type": "PEDESTRIAN"},
        ]

    def _terrain_z(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Gently undulating terrain with road crown."""
        # Broad undulation
        z = 0.08 * np.sin(0.15 * x) * np.cos(0.12 * y)
        # Road crown (slight convexity near y=0)
        z += 0.04 * np.exp(-0.01 * y ** 2)
        return z

    def _generate_frame(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        cfg = self.cfg
        dt = cfg.dt
        n_pts = cfg.points_per_frame

        # ── 1. Ground points (~70 %) ──────────────────────────────────────
        n_ground = int(n_pts * 0.70)
        gx = self.rng.uniform(-self.half, self.half, n_ground)
        gy = self.rng.uniform(-self.half, self.half, n_ground)
        gz = self._terrain_z(gx, gy) + self.rng.normal(0, cfg.sensor_noise_std, n_ground)
        ground_pts = np.column_stack([gx, gy, gz])

        # ── 2. Static obstacle points (~15 %) ────────────────────────────
        n_obs = int(n_pts * 0.15)
        obs_pts_list = []
        pts_per_obs = n_obs // max(len(self.static_obstacles), 1)
        for cx, cy, hw, hl, h in self.static_obstacles:
            n = pts_per_obs
            ox = self.rng.uniform(cx - hw, cx + hw, n)
            oy = self.rng.uniform(cy - hl, cy + hl, n)
            oz_base = self._terrain_z(ox, oy)
            oz = oz_base + self.rng.uniform(0.0, h, n)
            oz += self.rng.normal(0, cfg.sensor_noise_std, n)
            obs_pts_list.append(np.column_stack([ox, oy, oz]))
        static_pts = np.vstack(obs_pts_list) if obs_pts_list else np.empty((0, 3))

        # ── 3. Dynamic actor points (~15 %) ──────────────────────────────
        n_dyn = n_pts - n_ground - static_pts.shape[0]
        pts_per_actor = n_dyn // max(len(self.actors), 1)
        dyn_pts_list = []
        gt_actors = []
        for actor in self.actors:
            # Update position with wrap-around
            actor["x"] += actor["vx"] * dt
            actor["y"] += actor["vy"] * dt
            # Wrap to keep inside map
            for axis in ("x", "y"):
                if actor[axis] > self.half - 1:
                    actor[axis] = -self.half + 2
                elif actor[axis] < -self.half + 1:
                    actor[axis] = self.half - 2

            ax = self.rng.uniform(
                actor["x"] - actor["hl"],
                actor["x"] + actor["hl"],
                pts_per_actor,
            )
            ay = self.rng.uniform(
                actor["y"] - actor["hw"],
                actor["y"] + actor["hw"],
                pts_per_actor,
            )
            az_base = self._terrain_z(ax, ay)
            az = az_base + self.rng.uniform(0.0, actor["h"], pts_per_actor)
            az += self.rng.normal(0, cfg.sensor_noise_std, pts_per_actor)
            dyn_pts_list.append(np.column_stack([ax, ay, az]))

            gt_actors.append({
                "type": actor["type"],
                "x": float(actor["x"]),
                "y": float(actor["y"]),
                "vx": float(actor["vx"]),
                "vy": float(actor["vy"]),
                "half_w": float(actor["hw"]),
                "half_l": float(actor["hl"]),
                "height": float(actor["h"]),
            })

        dyn_pts = np.vstack(dyn_pts_list) if dyn_pts_list else np.empty((0, 3))

        # ── 4. Combine ───────────────────────────────────────────────────
        all_pts = np.vstack([ground_pts, static_pts, dyn_pts])

        # ── 5. Optional dust/outlier injection ───────────────────────────
        if self.outlier_ratio > 0:
            n_outliers = int(all_pts.shape[0] * self.outlier_ratio)
            ox = self.rng.uniform(-self.half, self.half, n_outliers)
            oy = self.rng.uniform(-self.half, self.half, n_outliers)
            oz = self.rng.uniform(-2.0, 8.0, n_outliers)
            outliers = np.column_stack([ox, oy, oz])
            all_pts = np.vstack([all_pts, outliers])

        # ── 6. Clip to map bounds ────────────────────────────────────────
        mask = (
            (all_pts[:, 0] >= -self.half) & (all_pts[:, 0] < self.half) &
            (all_pts[:, 1] >= -self.half) & (all_pts[:, 1] < self.half)
        )
        all_pts = all_pts[mask]

        # Simulated IMU angles (small realistic values)
        imu_pitch = float(0.5 * np.sin(0.3 * self.time))
        imu_roll = float(0.3 * np.cos(0.2 * self.time))

        metadata: Dict[str, Any] = {
            "frame_id": self.frame_idx,
            "timestamp": self.time,
            "imu_pitch": imu_pitch,
            "imu_roll": imu_roll,
            "ground_truth_actors": gt_actors,
            "source": "synthetic",
        }

        return all_pts.astype(np.float64), metadata

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Dict[str, Any]]]:
        return self

    def __next__(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        pts, meta = self._generate_frame()
        self.frame_idx += 1
        self.time += self.cfg.dt
        return pts, meta

    def reset(self) -> None:
        """Reset stream to initial state for replay."""
        self.__init__(self.cfg, self.outlier_ratio)

