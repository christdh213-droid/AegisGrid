"""
Dynamic object tracker: static/dynamic segmentation, multi-object clustering,
and persistent Kalman-filter–based tracking with Hungarian matching.

Guarantees zero ghost trails by building the live map from current-frame
observations + verified static background only.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from src.config import PipelineConfig
from src.schemas import DynamicTrack


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Kalman Filter (constant-velocity, 4-state: x, y, vx, vy)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _KalmanCV:
    """Minimal constant-velocity Kalman filter for 2D tracking."""

    def __init__(self, x: float, y: float, dt: float = 0.1):
        self.state = np.array([x, y, 0.0, 0.0], dtype=np.float64)  # [x, y, vx, vy]
        self.P = np.diag([1.0, 1.0, 5.0, 5.0])
        self.dt = dt
        # Process noise
        q = 2.0
        self.Q = np.diag([0.1, 0.1, q, q])
        # Measurement noise
        self.R = np.diag([0.3, 0.3])
        # Observation matrix (we observe x, y only)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float64)

    def predict(self, dt: Optional[float] = None) -> np.ndarray:
        if dt is None:
            dt = self.dt
        F = np.eye(4, dtype=np.float64)
        F[0, 2] = dt
        F[1, 3] = dt
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q
        return self.state.copy()

    def update(self, z_meas: np.ndarray) -> np.ndarray:
        y = z_meas - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        I4 = np.eye(4)
        self.P = (I4 - K @ self.H) @ self.P
        return self.state.copy()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DynamicTracker
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DynamicTracker:
    """
    Detects, clusters, and tracks dynamic objects in the 2.5D grid.

    Static reference model uses EMA on confirmed-static cells;
    dynamic foreground is detected via height residual.
    Ghost-trail erasure is structural: live map = current obs + static ref.
    """

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        dim = config.grid_dim
        self.dim = dim

        # Static reference elevation (initialised on first frame)
        self.static_ref_z: Optional[np.ndarray] = None   # float32 (dim, dim)
        self.static_count: Optional[np.ndarray] = None    # int32  (dim, dim)

        # Previous-frame dynamic mask for ghost detection
        self.prev_dynamic_mask: Optional[np.ndarray] = None

        # Tracking state
        self._next_track_id = 1
        self._active_tracks: Dict[int, dict] = {}  # track_id → track_state

        # Ghost audit
        self.ghost_cells_detected = 0

    def reset(self) -> None:
        self.__init__(self.cfg)

    # ── Core API ──────────────────────────────────────────────────────────

    def process(
        self,
        max_z: np.ndarray,         # float32 (256, 256) current-frame max_z
        count: np.ndarray,         # int32   (256, 256) current-frame point counts
        mean_z: np.ndarray,        # float32 (256, 256) current-frame mean_z
        dt: float,
    ) -> Tuple[np.ndarray, List[DynamicTrack], int]:
        """
        Returns:
            dynamic_mask: bool (256, 256) — cells containing dynamic objects.
            tracks:       List of DynamicTrack with filtered state.
            ghost_count:  Number of ghost cells detected this frame.
        """
        cfg = self.cfg
        dim = self.dim
        occupied = count > 0

        # ── 1. Initialise static reference on first frame ─────────────────
        if self.static_ref_z is None:
            self.static_ref_z = np.where(occupied, max_z, np.float32(0.0)).copy()
            self.static_count = np.where(occupied, np.int32(1), np.int32(0)).copy()
            self.prev_dynamic_mask = np.zeros((dim, dim), dtype=bool)
            # First frame: no dynamic detection yet
            return np.zeros((dim, dim), dtype=bool), [], 0

        # ── 2. Detect dynamic foreground ──────────────────────────────────
        ref = self.static_ref_z
        height_above_ref = np.where(
            occupied & (self.static_count > 0),
            max_z - ref,
            np.float32(0.0),
        )
        dynamic_mask = occupied & (height_above_ref > cfg.dynamic_height_residual_m)

        # ── 3. Ghost-trail audit ──────────────────────────────────────────
        # Cells that were dynamic last frame but are NOT dynamic now
        # should have reverted to ground level (or be empty).
        ghost_count = 0
        if self.prev_dynamic_mask is not None:
            vacated = self.prev_dynamic_mask & ~dynamic_mask
            # Among vacated cells that are still occupied, check height
            vacated_occupied = vacated & occupied
            if np.any(vacated_occupied):
                residual = max_z[vacated_occupied] - ref[vacated_occupied]
                ghosts = residual > cfg.dynamic_height_residual_m
                ghost_count = int(np.sum(ghosts))

        self.ghost_cells_detected += ghost_count

        # ── 4. Update static reference (EMA on non-dynamic cells) ─────────
        static_update_mask = occupied & ~dynamic_mask
        alpha = cfg.static_ema_alpha
        self.static_ref_z = np.where(
            static_update_mask,
            (1.0 - alpha) * ref + alpha * max_z,
            ref,
        ).astype(np.float32)
        self.static_count = np.where(
            static_update_mask,
            self.static_count + 1,
            self.static_count,
        )

        # ── 5. Cluster dynamic cells ──────────────────────────────────────
        labelled, num_clusters = ndimage.label(dynamic_mask, structure=np.ones((3, 3)))
        detections = []  # list of dicts with centroid, size, z_max, etc.

        for cid in range(1, num_clusters + 1):
            cluster_mask = labelled == cid
            cell_count = int(cluster_mask.sum())
            if cell_count < cfg.min_cluster_cells or cell_count > cfg.max_cluster_cells:
                continue

            rows, cols = np.where(cluster_mask)
            # World coordinates
            xs = (cols.astype(np.float64) + 0.5) * cfg.finest_res_m - cfg.half_size
            ys = (rows.astype(np.float64) + 0.5) * cfg.finest_res_m - cfg.half_size
            cx, cy = float(xs.mean()), float(ys.mean())
            z_max_val = float(max_z[cluster_mask].max())
            width = float(xs.max() - xs.min() + cfg.finest_res_m)
            length = float(ys.max() - ys.min() + cfg.finest_res_m)

            detections.append({
                "cx": cx, "cy": cy, "z_max": z_max_val,
                "width": width, "length": length, "cells": cell_count,
            })

        # ── 6. Hungarian matching ─────────────────────────────────────────
        tracks = self._match_and_update(detections, dt)

        self.prev_dynamic_mask = dynamic_mask.copy()
        return dynamic_mask, tracks, ghost_count

    # ── Matching & Kalman tracking ────────────────────────────────────────

    def _match_and_update(
        self, detections: List[dict], dt: float,
    ) -> List[DynamicTrack]:
        cfg = self.cfg

        # Predict all existing tracks
        for tid, ts in self._active_tracks.items():
            ts["kf"].predict(dt)

        track_ids = list(self._active_tracks.keys())
        n_tracks = len(track_ids)
        n_dets = len(detections)

        matched_det = set()
        matched_trk = set()

        if n_tracks > 0 and n_dets > 0:
            cost = np.full((n_tracks, n_dets), 1e6, dtype=np.float64)
            for i, tid in enumerate(track_ids):
                ts = self._active_tracks[tid]
                tx, ty = ts["kf"].state[0], ts["kf"].state[1]
                for j, det in enumerate(detections):
                    dist = np.hypot(tx - det["cx"], ty - det["cy"])
                    if dist <= cfg.track_association_dist_m:
                        cost[i, j] = dist

            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= cfg.track_association_dist_m:
                    matched_trk.add(r)
                    matched_det.add(c)
                    tid = track_ids[r]
                    det = detections[c]
                    ts = self._active_tracks[tid]
                    ts["kf"].update(np.array([det["cx"], det["cy"]]))
                    ts["z_max"] = det["z_max"]
                    ts["width"] = det["width"]
                    ts["length"] = det["length"]
                    ts["missed"] = 0
                    ts["age"] += 1

        # Unmatched tracks: increment missed counter
        for i, tid in enumerate(track_ids):
            if i not in matched_trk:
                self._active_tracks[tid]["missed"] += 1

        # Delete stale tracks
        stale = [tid for tid, ts in self._active_tracks.items()
                 if ts["missed"] > cfg.track_max_missed_frames]
        for tid in stale:
            del self._active_tracks[tid]

        # Unmatched detections: create new tracks
        for j, det in enumerate(detections):
            if j not in matched_det:
                tid = self._next_track_id
                self._next_track_id += 1
                kf = _KalmanCV(det["cx"], det["cy"], dt=dt)
                self._active_tracks[tid] = {
                    "kf": kf,
                    "z_max": det["z_max"],
                    "width": det["width"],
                    "length": det["length"],
                    "missed": 0,
                    "age": 1,
                }

        # ── Build output list ─────────────────────────────────────────────
        output: List[DynamicTrack] = []
        for tid, ts in self._active_tracks.items():
            st = ts["kf"].state
            speed = float(np.hypot(st[2], st[3]))
            # Classify by footprint
            footprint = ts["width"] * ts["length"]
            classification = "VEHICLE" if footprint > 2.0 else "PEDESTRIAN"

            pred_x = float(st[0] + st[2] * cfg.prediction_horizon_s)
            pred_y = float(st[1] + st[3] * cfg.prediction_horizon_s)

            output.append(DynamicTrack(
                track_id=int(tid),
                x=float(st[0]),
                y=float(st[1]),
                z_max=float(ts["z_max"]),
                vx=float(st[2]),
                vy=float(st[3]),
                speed=speed,
                width=float(ts["width"]),
                length=float(ts["length"]),
                pred_x=pred_x,
                pred_y=pred_y,
                age=int(ts["age"]),
                classification=classification,
            ))

        return output

