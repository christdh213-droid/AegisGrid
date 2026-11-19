"""LiDAR preprocessing and terrain risk for the synthetic viewer."""
import time

import numpy as np
from scipy import ndimage


def tilt_rotation(pitch_deg, roll_deg):
    """Column-vector rotation about Y after rotation about X."""
    pitch, roll = np.deg2rad([pitch_deg, roll_deg])
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array([[cp, sp * sr, sp * cr], [0, cr, -sr], [-sp, cp * sr, cp * cr]])


class ImuTiltCompensationHook:
    def __call__(self, points, metadata, runtime):
        if metadata.get("source") != "synthetic":
            return points, {}
        pitch = metadata.get("imu_pitch", 0.0)
        roll = metadata.get("imu_roll", 0.0)
        applied = metadata.get("sensor_tilt_applied", False) and runtime.imu_levelling_enabled
        if applied:
            points = points @ tilt_rotation(pitch, roll)
        return points, {"imu_pitch_deg": pitch, "imu_roll_deg": roll, "tilt_corrected": applied}


class DustOutlierFilterHook:
    def __call__(self, points, cfg):
        """Return a keep mask, using occupancy alone to classify elevated returns."""
        if not len(points):
            return np.ones(0, dtype=bool)
        # Ignore low returns when estimating airborne density. A dense road
        # surface must not make isolated points above it look like obstacles.
        elevated = np.flatnonzero(points[:, 2] >= cfg.dust_min_elevation_m)
        keep = np.ones(len(points), dtype=bool)
        if not len(elevated):
            return keep
        sample = points[elevated]
        axes = [np.floor(sample[:, axis] / cfg.dust_voxel_m).astype(np.int64) for axis in range(3)]
        for axis in axes:
            axis -= axis.min()
        dims = tuple(int(axis.max()) + 1 for axis in axes)
        # Prevent an isolated extreme return from allocating a huge dense grid.
        if np.prod(dims, dtype=np.float64) > 4_000_000:
            return keep
        x, y, z = axes
        flat = (x * dims[1] + y) * dims[2] + z
        occupancy = np.bincount(flat, minlength=int(np.prod(dims))).astype(np.int32).reshape(dims)
        for axis in range(3):
            occupancy = ndimage.correlate1d(occupancy, [1, 1, 1], axis=axis, mode="constant")
        radius_sq = sample[:, 0] ** 2 + sample[:, 1] ** 2
        minimum = np.where(radius_sq > 20 ** 2, cfg.dust_far_neighbors, cfg.dust_near_neighbors)
        keep[elevated] = occupancy.ravel()[flat] >= minimum
        return keep


class TraversabilityRiskHook:
    LABELS = ("SAFE", "CAUTION", "HAZARD", "UNKNOWN")

    @staticmethod
    def classify(span, step, dynamic, supported, cfg):
        risk = np.zeros(len(span), dtype=np.uint8)
        risk[(span > cfg.caution_span_m) | (step > cfg.caution_step_m)] = 1
        risk[dynamic | (span > cfg.hazard_span_m) | (step > cfg.hazard_step_m)] = 2
        risk[~supported] = 3
        return risk

    def __call__(self, frame, pipeline):
        start = time.perf_counter()
        adaptive = pipeline.adaptive_map
        columns = adaptive.leaf_arrays
        height = adaptive.reconstruct_dense_max_z()
        leaf_ids = adaptive.leaf_id_grid
        flags = columns["flags"]
        supported = (flags & 8) != 0
        valid = supported[leaf_ids]
        step = np.zeros_like(height)
        for first, second in (
            (np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:]),
            (np.s_[:-1, :-1], np.s_[1:, 1:]), (np.s_[1:, :-1], np.s_[:-1, 1:]),
        ):
            diff = np.where(valid[first] & valid[second], np.abs(height[first] - height[second]), 0)
            np.maximum(step[first], diff, out=step[first])
            np.maximum(step[second], diff, out=step[second])
        leaf_step = np.zeros(len(flags), dtype=np.float32)
        np.maximum.at(leaf_step, leaf_ids.ravel(), step.ravel())
        cells = frame.adaptive_cells
        risk = self.classify(columns["span_z"], leaf_step, (flags & 1) != 0, supported, pipeline.cfg)
        columns["flags"] = (flags & 0b11111001) | (risk << 1)
        # Materialize string labels for legacy JSON and existing cell readers.
        labels = np.asarray(self.LABELS, dtype=object)[risk].tolist()
        for cell, label in zip(cells, labels):
            cell.traversability = label
        percentages = np.bincount(risk, minlength=4) * (100.0 / len(risk))
        extras = {f"{label.lower()}_pct": float(value) for label, value in zip(self.LABELS, percentages)}
        extras["traversability_ms"] = (time.perf_counter() - start) * 1000
        return extras
