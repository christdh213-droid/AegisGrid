"""
LiDAR data sources: file loader and synthetic stream generator.

SyntheticLidarStream produces a metric urban intersection with flat roads,
curbs, buildings and four dynamic actors (two vehicles and two pedestrians).
"""
from __future__ import annotations

import csv
import os
from typing import Dict, Any, Iterator, Optional, Tuple

import numpy as np
from src.plugins import tilt_rotation

from src.config import PipelineConfig


def _rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def scene_layout():
    """Single metric geometry authority for synthetic surfaces and cartography.

    XY footprints are ground-plane coordinates, Z is metres above flat road.
    Markings are zero-height painted geometry, not obstacle returns.
    """
    roads = [_rect(-32, -5, 32, 5), _rect(-5, -32, 5, 32)]
    sidewalks = []
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        arc = [[round(8-3*np.sin(a), 2), round(8-3*np.cos(a), 2)] for a in np.linspace(0, np.pi/2, 13)]
        corner = [[5, 5], *arc]
        roads.append([[sx*x, sy*y] for x, y in corner])
        pavement = [[5, 32], [7.5, 32], [7.5, 9], [9, 7.5], [32, 7.5], [32, 5], *arc]
        sidewalks.append({'polygon': [[sx*x, sy*y] for x,y in pavement], 'height': .16})
    # Flat, open access to the existing (-10,8) goal and (-5,5) test waypoint.
    courtyards = [_rect(-17, 6, -7, 12), _rect(-9, 3.5, -3, 10.5), _rect(10, -28, 26, -20)]
    structures = [
        {'kind':'building','height':3.6,'polygon':[[-28,13],[-18,13],[-18,18],[-22,18],[-22,28],[-28,28]]},
        {'kind':'building','height':4.2,'polygon':[[12,12],[28,12],[28,20],[23,20],[23,28],[14,28],[14,22],[12,22]]},
        {'kind':'building','height':2.4,'polygon':[[-28,-28],[-12,-28],[-12,-21],[-17,-21],[-17,-14],[-22,-14],[-22,-17],[-28,-17]]},
        {'kind':'building','height':3.2,'polygon':[[14,-18],[28,-18],[28,-10],[23,-10],[23,-11],[21,-11],[21,-10],[14,-10]]},
    ]
    # Set back the buildings from generous, flat compound courtyards.
    for structure in structures:
        polygon=np.asarray(structure['polygon'],dtype=float)
        centre=(polygon.min(axis=0)+polygon.max(axis=0))/2
        structure['polygon']=(np.round((centre+(polygon-centre)*.60)*2)/2).tolist()
    for x0,y0,x1,y1 in [(-30,20,-29.5,24),(29.5,20,30,24),(-24,-29,-20,-28.5),(20,-30,24,-29.5)]:
        structures.append({'kind':'wall','height':2.4,'polygon':_rect(x0,y0,x1,y1)})
    for x0,y0,x1,y1 in [(15,-.25,21,.25),(-23,-.25,-17,.25)]:
        structures.append({'kind':'barrier','height':.9,'polygon':_rect(x0,y0,x1,y1)})
    for x,y in [(-10,-12),(-10,-18),(10,17),(10,24)]:
        structures.append({'kind':'parked_car','height':1.5,'polygon':_rect(x-.9,y-2.1,x+.9,y+2.1)})
    for x,y in [(-7.1,-17),(-7.1,20),(7.1,-22),(7.1,17)]:
        structures.append({'kind':'lamp','height':3.5,'polygon':_rect(x-.12,y-.12,x+.12,y+.12)})
    crosswalks=[]
    for sign in (-1,1):
        for pos in np.arange(-4.5,5,.9):
            crosswalks.append(_rect(float(pos),sign*6-.75,float(pos+.45),sign*6+.75))
            crosswalks.append(_rect(sign*6-.75,float(pos),sign*6+.75,float(pos+.45)))
    return dict(version=1, bounds=[-32,-32,32,32], roads=roads, courtyards=courtyards,
                sidewalks=sidewalks, curbs=sidewalks, structures=structures,
                crosswalks=crosswalks, road_height=0., paint_height=0.,
                ground_goal=[-10,8], coordinate_unit='metres')


def _inside_polygon(x, y, polygon):
    inside = np.zeros(np.shape(x), dtype=bool)
    for (ax, ay), (bx, by) in zip(polygon, polygon[1:]+polygon[:1]):
        if ay == by:
            continue
        inside ^= ((ay > y) != (by > y)) & (x < (bx-ax)*(y-ay)/(by-ay)+ax)
    return inside


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
        self.weather_rng = np.random.default_rng(config.seed + 1)
        self.imu_rng = np.random.default_rng(config.seed + 2)
        self.runtime_config = None
        self.tilt_started_at = None
        self.frame_idx = 0
        self.time = 0.0
        self.half = config.half_size  # 32.0

        self.layout = scene_layout()
        self._static_points = None
        self._ground_points = None

        # ── Dynamic actors ────────────────────────────────────────────────
        # (start_x, start_y, vx, vy, half_w, half_l, height, type)
        self.actors = [
            # Vehicle 1: moving +X
            {"x": -25.0, "y": -2.5, "vx": 7.0, "vy": 0.0,
             "hw": 0.9, "hl": 2.1, "h": 1.6, "type": "VEHICLE"},
            # Vehicle 2: moving -X
            {"x": 25.0, "y": 2.5, "vx": -8.0, "vy": 0.0,
             "hw": 0.9, "hl": 2.1, "h": 1.5, "type": "VEHICLE"},
            # Pedestrian 1: moving +Y
            {"x": -6.25, "y": -20.0, "vx": 0.0, "vy": 1.4,
             "hw": 0.35, "hl": 0.35, "h": 1.7, "type": "PEDESTRIAN"},
            # Pedestrian 2: moving diagonally
            {"x": 6.25, "y": 15.0, "vx": 0.0, "vy": -1.1,
             "hw": 0.35, "hl": 0.35, "h": 1.6, "type": "PEDESTRIAN"},
        ]

    def _terrain_z(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Road/courtyard grade and raised sidewalk surfaces."""
        z = np.zeros(np.shape(x), dtype=np.float64)
        for sidewalk in self.layout['sidewalks']:
            z[_inside_polygon(x, y, sidewalk['polygon'])] = sidewalk['height']
        for polygon in self.layout['courtyards']:
            z[_inside_polygon(x, y, polygon)] = 0.
        return z

    def _surface_z(self, x, y):
        z = self._terrain_z(x, y)
        for structure in self.layout['structures']:
            mask = _inside_polygon(x, y, structure['polygon'])
            z[mask] = structure['height']
        return z

    def _static_scan(self, n_ground, n_obs):
        # Repeatable scan lattice: fixed surface sampling prevents static facades
        # flickering into foreground detections. Noise is seeded once per reset.
        side = int(np.ceil(np.sqrt(n_ground)))
        axis = np.linspace(-self.half+.08, self.half-.08, side)
        xx, yy = np.meshgrid(axis, axis)
        xy = np.column_stack((xx.ravel(), yy.ravel()))
        xy = xy[np.linspace(0, len(xy)-1, n_ground).astype(int)]
        z = self._surface_z(xy[:,0], xy[:,1])
        z += np.clip(self.rng.normal(0,self.cfg.sensor_noise_std,n_ground),-.01,.01)
        self._ground_points = np.column_stack((xy,z))
        edges=[]
        for structure in self.layout['structures']:
            polygon=structure['polygon']
            for a,b in zip(polygon,polygon[1:]+polygon[:1]):
                length=float(np.linalg.norm(np.subtract(b,a)))
                edges.append((a,b,structure['height'],length))
        total=sum(e[3]*e[2] for e in edges)
        surfaces=[]
        for a,b,h,length in edges:
            n=max(8,int(n_obs*length*h/total))
            u=self.rng.uniform(.001,.999,n)
            x=a[0]+(b[0]-a[0])*u; y=a[1]+(b[1]-a[1])*u
            z=self.rng.uniform(.05,h,n)
            surfaces.append(np.column_stack((x,y,z)))
        # Curb faces are actual 16 cm geometry, not a painted height effect.
        for sidewalk in self.layout['sidewalks']:
            polygon=sidewalk['polygon']
            for a,b in zip(polygon,polygon[1:]+polygon[:1]):
                length=np.linalg.norm(np.subtract(b,a)); u=np.arange(.02,length,.12)/max(length,.01)
                x=a[0]+(b[0]-a[0])*u; y=a[1]+(b[1]-a[1])*u
                keep=np.ones(len(x),dtype=bool)
                for courtyard in self.layout['courtyards']:
                    keep &= ~_inside_polygon(x,y,courtyard)
                x,y=x[keep],y[keep]
                surfaces.append(np.column_stack((x,y,np.full(len(x),.16))))

        self._static_points = np.vstack(surfaces)

    def _generate_frame(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        cfg = self.cfg
        dt = cfg.dt
        n_pts = cfg.points_per_frame

        n_ground = int(n_pts * .70)
        if self._static_points is None:
            self._static_scan(n_ground, int(n_pts*.15))
        ground_pts = self._ground_points
        static_pts = self._static_points
        n_dyn = int(n_pts*.15)
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
        runtime = self.runtime_config
        ratio = runtime.outlier_ratio if runtime is not None else self.outlier_ratio
        n_real = len(all_pts)
        if ratio > 0:
            n_outliers = int(n_real * ratio)
            rng = self.weather_rng
            ox = rng.uniform(-self.half, self.half, n_outliers)
            oy = rng.uniform(-self.half, self.half, n_outliers)
            # A loose drifting band within distributed airborne returns.
            band = rng.random(n_outliers) < .08
            oy[band] = rng.normal(10 * np.sin(.4 * self.time), 5., band.sum())
            surface = self._surface_z(ox, oy)
            for dx,dy in ((-.6,0),(.6,0),(0,-.6),(0,.6)):
                surface = np.maximum(surface, self._surface_z(ox+dx, oy+dy))
            oz = surface + rng.uniform(.5, 4., n_outliers)
            outliers = np.column_stack([ox, oy, oz])
            all_pts = np.vstack([all_pts, outliers])

        # ── 6. Clip to map bounds ────────────────────────────────────────
        mask = (
            (all_pts[:, 0] >= -self.half) & (all_pts[:, 0] < self.half) &
            (all_pts[:, 1] >= -self.half) & (all_pts[:, 1] < self.half)
        )
        all_pts = all_pts[mask]
        noise_truth = (np.arange(len(mask)) >= n_real)[mask]

        imu_pitch = imu_roll = 0.0
        tilt = runtime is not None and runtime.simulate_hill_tilt
        if tilt:
            if self.tilt_started_at is None:
                self.tilt_started_at = self.time
            ease = min(1.0, self.time - self.tilt_started_at)
            if runtime.tilt_mode == "Manual":
                pitch, roll = runtime.pitch_deg, runtime.roll_deg
            else:
                pitch = 6 * np.sin(2 * np.pi * .20 * self.time)
                roll = 4 * np.sin(2 * np.pi * .13 * self.time + .7)
            imu_pitch, imu_roll = float(ease * pitch), float(ease * roll)
            all_pts = all_pts @ tilt_rotation(imu_pitch, imu_roll).T
            if runtime.imu_noise_deg:
                noise = self.imu_rng.normal(0., runtime.imu_noise_deg, 2)
                imu_pitch += float(noise[0])
                imu_roll += float(noise[1])
        else:
            self.tilt_started_at = None

        metadata: Dict[str, Any] = {
            "frame_id": self.frame_idx,
            "timestamp": self.time,
            "imu_pitch": imu_pitch,
            "imu_roll": imu_roll,
            "sensor_tilt_applied": tilt,
            "simulated_outlier_ratio": ratio,
            "is_injected_noise": noise_truth,
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

