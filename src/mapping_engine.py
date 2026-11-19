"""
Vectorized 2.5D mapping engine: FixedGridMap and AdaptiveQuadtreeMap.

The AdaptiveQuadtreeMap uses a **bottom-up NumPy pyramid** — no recursive
per-point insertion.  Rasterize → aggregate → merge upward → extract leaves.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.config import PipelineConfig
from src.schemas import AdaptiveCell


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FixedGridMap — baseline uniform 256×256 (0.25 m) elevation grid
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FixedGridMap:
    """
    Dense 256×256 2.5D elevation grid over [-32 m, +32 m] × [-32 m, +32 m].

    Stores per-cell: min_z, max_z, sum_z, count  →  derived mean_z, span_z.
    All operations are pure vectorised NumPy.
    """

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        dim = config.grid_dim                       # 256
        self.dim = dim
        self.min_z  = np.full((dim, dim), np.inf,  dtype=np.float32)
        self.max_z  = np.full((dim, dim), -np.inf, dtype=np.float32)
        self.sum_z  = np.zeros((dim, dim), dtype=np.float64)
        self.count  = np.zeros((dim, dim), dtype=np.int32)

    # ── Public API ────────────────────────────────────────────────────────

    def update(self, points: np.ndarray) -> None:
        """Rasterise Nx3 points into the grid (vectorised)."""
        if points.shape[0] == 0:
            return
        cfg = self.cfg
        dim = self.dim

        # Map world coords → grid indices
        col = np.floor((points[:, 0] + cfg.half_size) / cfg.finest_res_m).astype(np.intp)
        row = np.floor((points[:, 1] + cfg.half_size) / cfg.finest_res_m).astype(np.intp)
        z   = points[:, 2]

        # Clip to valid range
        valid = (col >= 0) & (col < dim) & (row >= 0) & (row < dim)
        col, row, z = col[valid], row[valid], z[valid]

        flat = row * dim + col
        z32  = z.astype(np.float32)

        np.minimum.at(self.min_z.ravel(), flat, z32)
        np.maximum.at(self.max_z.ravel(), flat, z32)
        np.add.at(self.sum_z.ravel(), flat, z)
        np.add.at(self.count.ravel(), flat, 1)

    def reset(self) -> None:
        dim = self.dim
        self.min_z[:] = np.inf
        self.max_z[:] = -np.inf
        self.sum_z[:] = 0.0
        self.count[:] = 0

    @property
    def mean_z(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.where(self.count > 0, self.sum_z / self.count, 0.0)
        return m.astype(np.float32)

    @property
    def span_z(self) -> np.ndarray:
        s = np.where(self.count > 0, self.max_z - self.min_z, 0.0)
        return s.astype(np.float32)

    @property
    def active_cells(self) -> int:
        return int(np.count_nonzero(self.count))

    def memory_kb(self) -> float:
        """Exact structured memory footprint in KiB."""
        total = (self.min_z.nbytes + self.max_z.nbytes +
                 self.sum_z.nbytes + self.count.nbytes)
        return total / 1024.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AdaptiveQuadtreeMap — vectorised bottom-up pyramid
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Compact leaf struct: 36 bytes
# float32 × 7 (x, y, size, min_z, max_z, mean_z, span_z) = 28
# uint16 count = 2, uint8 level = 1, uint8 flags = 1, uint32 id = 4  → 36
LEAF_STRUCT_BYTES = 36


class AdaptiveQuadtreeMap:
    """
    Vectorised bottom-up quadtree pyramid for adaptive-resolution 2.5D mapping.

    Pyramid levels (power-of-2):
        L0: 0.25 m  256×256  (base = FixedGridMap output)
        L1: 0.50 m  128×128
        L2: 1.00 m   64×64
        L3: 2.00 m   32×32
        L4: 4.00 m   16×16
    """

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.leaves: List[AdaptiveCell] = []
        self._dense_max_z: Optional[np.ndarray] = None

    def build(
        self,
        fixed: FixedGridMap,
        dynamic_mask: np.ndarray,
    ) -> List[AdaptiveCell]:
        """
        Build the adaptive quadtree from a populated FixedGridMap.

        Args:
            fixed:        Populated FixedGridMap (256×256).
            dynamic_mask: bool array (256×256) — True where dynamic objects are.

        Returns:
            List of AdaptiveCell leaf nodes covering the entire 64 m scene.
        """
        cfg = self.cfg
        dim = cfg.grid_dim           # 256
        max_level = cfg.max_quadtree_level  # 4

        # ── Build pyramid arrays bottom-up ────────────────────────────────
        # Level 0: from fixed grid
        valid0 = fixed.count > 0
        min_z0 = np.where(valid0, fixed.min_z, np.float32(0.0)).astype(np.float32)
        max_z0 = np.where(valid0, fixed.max_z, np.float32(0.0)).astype(np.float32)
        mean_z0 = fixed.mean_z.copy()
        count0 = fixed.count.copy()
        dyn0 = dynamic_mask.astype(bool).copy()

        pyr_min_z = [min_z0]
        pyr_max_z = [max_z0]
        pyr_mean_z = [mean_z0]
        pyr_count = [count0]
        pyr_dyn = [dyn0]
        pyr_valid = [valid0.copy()]

        for lev in range(1, max_level + 1):
            cur_dim = dim >> lev
            prev_min = pyr_min_z[lev - 1]
            prev_max = pyr_max_z[lev - 1]
            prev_mean = pyr_mean_z[lev - 1]
            prev_cnt = pyr_count[lev - 1]
            prev_dyn = pyr_dyn[lev - 1]
            prev_val = pyr_valid[lev - 1]

            # Reshape to (cur_dim, 2, cur_dim, 2) for 2×2 block access
            blk_min  = prev_min.reshape(cur_dim, 2, cur_dim, 2)
            blk_max  = prev_max.reshape(cur_dim, 2, cur_dim, 2)
            blk_mean = prev_mean.reshape(cur_dim, 2, cur_dim, 2)
            blk_cnt  = prev_cnt.reshape(cur_dim, 2, cur_dim, 2)
            blk_dyn  = prev_dyn.reshape(cur_dim, 2, cur_dim, 2)
            blk_val  = prev_val.reshape(cur_dim, 2, cur_dim, 2)

            any_dyn = blk_dyn.any(axis=(1, 3))
            any_val = blk_val.any(axis=(1, 3))
            total_cnt = blk_cnt.sum(axis=(1, 3))

            # Min/max across valid children only
            blk_min_m = np.where(blk_val, blk_min, np.float32(np.inf))
            blk_max_m = np.where(blk_val, blk_max, np.float32(-np.inf))
            cur_min = blk_min_m.min(axis=(1, 3))
            cur_max = blk_max_m.max(axis=(1, 3))

            # Weighted mean (safe division)
            weighted = (blk_mean * blk_cnt).sum(axis=(1, 3))
            with np.errstate(divide="ignore", invalid="ignore"):
                cur_mean = np.where(total_cnt > 0, weighted / total_cnt, 0.0).astype(np.float32)

            cur_min = np.where(any_val, cur_min, np.float32(0.0))
            cur_max = np.where(any_val, cur_max, np.float32(0.0))

            pyr_min_z.append(cur_min.astype(np.float32))
            pyr_max_z.append(cur_max.astype(np.float32))
            pyr_mean_z.append(cur_mean)
            pyr_count.append(total_cnt)
            pyr_dyn.append(any_dyn)
            pyr_valid.append(any_val)

        # ── Decide which cells merge: bottom-up "can_merge" ──────────────
        # can_merge[lev] = bool (dim>>lev, dim>>lev): can the 2×2 block at
        # level lev-1 merge into this parent at level lev?
        can_merge = [None]  # level 0 has no merge concept (it IS the base)

        for lev in range(1, max_level + 1):
            cur_dim = dim >> lev
            cm = np.ones((cur_dim, cur_dim), dtype=bool)

            # Rule 1: no dynamic children
            cm &= ~pyr_dyn[lev]

            # Rule 2: combined span ≤ threshold (or entirely empty)
            span = pyr_max_z[lev] - pyr_min_z[lev]
            cm &= (~pyr_valid[lev]) | (span <= cfg.merge_height_threshold)

            # Rule 3: gradient between sibling children ≤ threshold
            prev_mean = pyr_mean_z[lev - 1]
            prev_val = pyr_valid[lev - 1]
            # children: (cur_dim, 2, cur_dim, 2) → reorder to (cur_dim, cur_dim, 2, 2) → flatten last two to 4
            children_mean = prev_mean.reshape(cur_dim, 2, cur_dim, 2).transpose(0, 2, 1, 3).reshape(cur_dim, cur_dim, 4)
            children_valid = prev_val.reshape(cur_dim, 2, cur_dim, 2).transpose(0, 2, 1, 3).reshape(cur_dim, cur_dim, 4)

            # Vectorised max gradient across all 6 pairs of 4 children
            max_grad = np.zeros((cur_dim, cur_dim), dtype=np.float32)
            for i in range(4):
                for j in range(i + 1, 4):
                    both = children_valid[:, :, i] & children_valid[:, :, j]
                    diff = np.abs(children_mean[:, :, i] - children_mean[:, :, j])
                    max_grad = np.maximum(max_grad, np.where(both, diff, 0.0))

            cm &= (max_grad <= cfg.merge_gradient_threshold)

            can_merge.append(cm)

        # ── Top-down leaf extraction with proper coverage tracking ────────
        absorbed = [np.zeros((dim >> l, dim >> l), dtype=bool)
                    for l in range(max_level + 1)]
        is_leaf = [np.zeros((dim >> l, dim >> l), dtype=bool)
                   for l in range(max_level + 1)]

        for lev in range(max_level, 0, -1):
            leaf_here = can_merge[lev] & ~absorbed[lev]
            is_leaf[lev] = leaf_here

            expanded = leaf_here
            for child_lev in range(lev - 1, -1, -1):
                expanded = expanded.repeat(2, axis=0).repeat(2, axis=1)
                absorbed[child_lev] |= expanded

        is_leaf[0] = ~absorbed[0]
        self._is_leaf = is_leaf
        self._pyr_max_z = pyr_max_z
        self._dense_max_z = None  # Invalidate cache

        # ── Extract leaf cells (vectorised per level) ─────────────────────
        all_cell_ids = []
        all_xs = []
        all_ys = []
        all_sizes = []
        all_levels = []
        all_min_z = []
        all_max_z = []
        all_mean_z = []
        all_span_z = []
        all_counts = []
        all_dyns = []

        cell_id_counter = 0

        for lev in range(max_level + 1):
            cur_dim_l = dim >> lev
            cell_size = cfg.finest_res_m * (1 << lev)
            indices = np.argwhere(is_leaf[lev])  # (N, 2) → row, col

            if indices.shape[0] == 0:
                continue

            rows = indices[:, 0]
            cols = indices[:, 1]
            n = len(rows)

            xs = (cols.astype(np.float64) + 0.5) * cell_size - cfg.half_size
            ys = (rows.astype(np.float64) + 0.5) * cell_size - cfg.half_size

            ids = np.arange(cell_id_counter, cell_id_counter + n)
            cell_id_counter += n

            all_cell_ids.append(ids)
            all_xs.append(xs)
            all_ys.append(ys)
            all_sizes.append(np.full(n, cell_size))
            all_levels.append(np.full(n, lev, dtype=np.int32))
            all_min_z.append(pyr_min_z[lev][rows, cols].astype(np.float64))
            all_max_z.append(pyr_max_z[lev][rows, cols].astype(np.float64))
            all_mean_z.append(pyr_mean_z[lev][rows, cols].astype(np.float64))
            counts_arr = pyr_count[lev][rows, cols]
            all_counts.append(counts_arr)
            all_dyns.append(pyr_dyn[lev][rows, cols])

            # Span: max_z - min_z where count > 0
            mn = pyr_min_z[lev][rows, cols].astype(np.float64)
            mx = pyr_max_z[lev][rows, cols].astype(np.float64)
            sp = np.where(counts_arr > 0, mx - mn, 0.0)
            all_span_z.append(sp)

        # Concatenate all
        if all_cell_ids:
            cat_ids = np.concatenate(all_cell_ids).tolist()
            cat_xs = np.concatenate(all_xs).tolist()
            cat_ys = np.concatenate(all_ys).tolist()
            cat_sizes = np.concatenate(all_sizes).tolist()
            cat_levels = np.concatenate(all_levels).tolist()
            cat_min_z = np.concatenate(all_min_z).tolist()
            cat_max_z = np.concatenate(all_max_z).tolist()
            cat_mean_z = np.concatenate(all_mean_z).tolist()
            cat_span_z = np.concatenate(all_span_z).tolist()
            cat_counts = np.concatenate(all_counts).tolist()
            cat_dyns = np.concatenate(all_dyns).tolist()

            leaves: List[AdaptiveCell] = [
                AdaptiveCell(i, x, y, s, l, mn, mx, me, sp, c, d)
                for i, x, y, s, l, mn, mx, me, sp, c, d in zip(
                    cat_ids, cat_xs, cat_ys, cat_sizes, cat_levels,
                    cat_min_z, cat_max_z, cat_mean_z, cat_span_z,
                    cat_counts, cat_dyns,
                )
            ]
        else:
            leaves = []

        self.leaves = leaves
        return leaves

    def reconstruct_dense_max_z(self) -> np.ndarray:
        """Return dense 256×256 max_z array for benchmarking (vectorised)."""
        if self._dense_max_z is not None:
            return self._dense_max_z

        if not hasattr(self, "_is_leaf") or not hasattr(self, "_pyr_max_z"):
            raise RuntimeError("Call build() first")

        cfg = self.cfg
        dim = cfg.grid_dim
        max_level = cfg.max_quadtree_level

        dense = np.full((dim, dim), np.nan, dtype=np.float32)
        for lev in range(max_level, -1, -1):
            scale = 1 << lev
            leaf_mask = self._is_leaf[lev]
            if not np.any(leaf_mask):
                continue
            leaf_vals = np.where(leaf_mask, self._pyr_max_z[lev], np.float32(np.nan))
            upsampled = leaf_vals.repeat(scale, axis=0).repeat(scale, axis=1) if scale > 1 else leaf_vals
            valid = ~np.isnan(upsampled)
            dense[valid] = upsampled[valid]

        self._dense_max_z = dense
        return self._dense_max_z

    def memory_kb(self) -> float:
        """Compact binary memory: num_leaves × LEAF_STRUCT_BYTES."""
        return len(self.leaves) * LEAF_STRUCT_BYTES / 1024.0
