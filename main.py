#!/usr/bin/env python3
"""
SIH26053 — Adaptive Variable Resolution 2.5D LiDAR Mapping Engine
Main CLI entry-point: benchmark, visualise, and export.

Usage:
    python main.py --frames 50 --save-plot --save-json
    python main.py --frames 10 --source file --file data/scene.npy
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure UTF-8 output on Windows consoles if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import time

# Agg backend MUST be set before any pyplot import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np

# ── Project imports ──────────────────────────────────────────────────────
from src.config import PipelineConfig
from src.data_source import SyntheticLidarStream, LidarFileLoader
from src.pipeline import PerceptionPipeline
from src.benchmark import aggregate_metrics
from src.schemas import FrameOutput


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SIH26053 Adaptive 2.5D LiDAR Mapping — Benchmark & Visualisation",
    )
    p.add_argument("--frames", type=int, default=50, help="Number of frames to process")
    p.add_argument("--source", choices=["synthetic", "file"], default="synthetic")
    p.add_argument("--file", type=str, default=None, help="Path to point cloud file")
    p.add_argument("--save-plot", action="store_true", help="Save verification plot")
    p.add_argument("--save-json", action="store_true", help="Save JSON outputs")
    p.add_argument("--output-dir", type=str, default="output", help="Output directory")
    return p.parse_args()


def run_benchmark(args: argparse.Namespace) -> None:
    config = PipelineConfig()
    pipe = PerceptionPipeline(config)

    # ── Data source ──────────────────────────────────────────────────────
    if args.source == "file" and args.file:
        loader = LidarFileLoader(args.file)
        stream = loader
    else:
        stream = SyntheticLidarStream(config)

    # ── Process frames ───────────────────────────────────────────────────
    all_frames = []
    all_metrics = []
    print(f"\n{'='*72}")
    print(f"  SIH26053 - Adaptive 2.5D LiDAR Perception Engine (Phase 1)")
    print(f"  Processing {args.frames} frames ...")
    print(f"{'='*72}\n")

    t_total_start = time.perf_counter()
    for frame in pipe.run_stream(stream, num_frames=args.frames):
        all_frames.append(frame)
        all_metrics.append(frame.metrics)
        if (frame.frame_id + 1) % 10 == 0 or frame.frame_id == 0:
            m = frame.metrics
            print(
                f"  Frame {frame.frame_id:3d}  |  "
                f"Pipeline {m.total_pipeline_ms:6.1f} ms  |  "
                f"Cells {m.adaptive_leaf_cells:5d} ({m.cell_reduction_pct:5.1f}% reduction)  |  "
                f"Mem {m.adaptive_memory_kb:7.1f} KB ({m.memory_saved_pct:5.1f}% saved)  |  "
                f"RMSE {m.obstacle_rmse_m:.4f} m  |  "
                f"Ghosts {m.ghost_cells_detected}"
            )
    t_total = time.perf_counter() - t_total_start

    # ── Summary ──────────────────────────────────────────────────────────
    summary = aggregate_metrics(all_metrics)
    summary["wall_clock_total_s"] = round(t_total, 3)

    print(f"\n{'-'*72}")
    print(f"  BENCHMARK SUMMARY  ({summary['num_frames']} frames)")
    print(f"{'-'*72}")
    print(f"  Avg Pipeline Latency : {summary['avg_total_pipeline_ms']:.2f} ms")
    print(f"  Median Latency       : {summary['median_total_pipeline_ms']:.2f} ms")
    print(f"  Avg FPS              : {summary['avg_fps']:.1f}")
    print(f"  Avg Cell Reduction   : {summary['avg_cell_reduction_pct']:.1f}%")
    print(f"  Avg Memory Saved     : {summary['avg_memory_saved_pct']:.1f}%")
    print(f"  Avg Obstacle RMSE    : {summary['avg_obstacle_rmse_m']:.4f} m")
    print(f"  Max Obstacle RMSE    : {summary['max_obstacle_rmse_m']:.4f} m")
    print(f"  Total Ghost Cells    : {summary['total_ghost_cells']}")
    print(f"  Wall-clock Total     : {t_total:.2f} s")
    print(f"{'-'*72}\n")

    # ── Save outputs ─────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    if args.save_json:
        # Sample frame (frame 25 or last)
        sample_idx = min(24, len(all_frames) - 1)
        sample_frame = all_frames[sample_idx]
        sample_path = os.path.join(args.output_dir, "sample_frame.json")
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(sample_frame.to_dict(), f, indent=2)
        # Validate round-trip
        with open(sample_path, "r", encoding="utf-8") as f:
            json.loads(f.read())
        print(f"  [OK] Saved {sample_path}")

        # Benchmark summary
        summary_path = os.path.join(args.output_dir, "benchmark_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"  [OK] Saved {summary_path}")

    if args.save_plot:
        _generate_verification_plot(all_frames, pipe, config, args.output_dir)


def _generate_verification_plot(
    all_frames: list,
    pipe: PerceptionPipeline,
    config: PipelineConfig,
    output_dir: str,
) -> None:
    """Generate the 2×2 high-DPI verification plot."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=150)
    half = config.half_size

    # Use a mid-run frame for the spatial plots
    plot_idx = min(24, len(all_frames) - 1)
    frame = all_frames[plot_idx]

    # ── Top-Left: Fixed 0.25m 2.5D Elevation Map ─────────────────────────
    ax = axes[0, 0]
    fixed_max_z = pipe.fixed_map.max_z.copy()
    fixed_max_z_display = np.where(pipe.fixed_map.count > 0, fixed_max_z, np.nan)
    im0 = ax.imshow(
        fixed_max_z_display,
        extent=[-half, half, -half, half],
        origin="lower", cmap="terrain", vmin=-0.5, vmax=3.0,
    )
    ax.set_title(f"Fixed Grid 0.25m — Elevation (Frame {plot_idx})", fontsize=11)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.colorbar(im0, ax=ax, label="max_z (m)", shrink=0.8)

    # ── Top-Right: Adaptive Quadtree Map ──────────────────────────────────
    ax = axes[0, 1]
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")

    patches = []
    colors = []
    for cell in frame.adaptive_cells:
        x0 = cell.x - cell.size / 2
        y0 = cell.y - cell.size / 2
        rect = mpatches.Rectangle((x0, y0), cell.size, cell.size)
        patches.append(rect)
        colors.append(cell.max_z if cell.point_count > 0 else np.nan)

    colors_arr = np.array(colors, dtype=np.float64)
    valid_colors = ~np.isnan(colors_arr)
    if np.any(valid_colors):
        colors_arr[~valid_colors] = 0.0
    pc = PatchCollection(patches, cmap="terrain", edgecolors="black",
                         linewidths=0.15, alpha=0.9)
    pc.set_array(colors_arr)
    pc.set_clim(-0.5, 3.0)
    ax.add_collection(pc)
    ax.set_title(
        f"Adaptive Quadtree — {len(frame.adaptive_cells)} cells (Frame {plot_idx})",
        fontsize=11,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.colorbar(pc, ax=ax, label="max_z (m)", shrink=0.8)

    # ── Bottom-Left: Dynamic Perception View ──────────────────────────────
    ax = axes[1, 0]
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")

    # Static cells in muted grey
    for cell in frame.adaptive_cells:
        if not cell.is_dynamic and cell.point_count > 0:
            x0 = cell.x - cell.size / 2
            y0 = cell.y - cell.size / 2
            rect = mpatches.Rectangle(
                (x0, y0), cell.size, cell.size,
                facecolor="#cccccc", edgecolor="#999999", linewidth=0.1,
            )
            ax.add_patch(rect)

    # Dynamic cells in red
    for cell in frame.adaptive_cells:
        if cell.is_dynamic:
            x0 = cell.x - cell.size / 2
            y0 = cell.y - cell.size / 2
            rect = mpatches.Rectangle(
                (x0, y0), cell.size, cell.size,
                facecolor="#ff4444", edgecolor="red", linewidth=0.3,
            )
            ax.add_patch(rect)

    # Tracked objects with bounding boxes + velocity arrows
    for track in frame.dynamic_tracks:
        # Bounding box
        bx = track.x - track.length / 2
        by = track.y - track.width / 2
        rect = mpatches.Rectangle(
            (bx, by), track.length, track.width,
            fill=False, edgecolor="lime", linewidth=2,
        )
        ax.add_patch(rect)
        # ID label
        ax.text(track.x, track.y + track.width / 2 + 0.5,
                f"ID:{track.track_id} {track.classification[:3]}",
                fontsize=7, color="lime", ha="center", fontweight="bold")
        # Velocity arrow
        if track.speed > 0.3:
            ax.annotate(
                "", xy=(track.pred_x, track.pred_y),
                xytext=(track.x, track.y),
                arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5),
            )

    ax.set_title(f"Dynamic Perception — {len(frame.dynamic_tracks)} tracks", fontsize=11)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_facecolor("#222222")

    # ── Bottom-Right: Performance Dashboard ───────────────────────────────
    ax = axes[1, 1]
    n = len(all_frames)
    frame_ids = [f.frame_id for f in all_frames]

    # Latency
    fixed_lat = [f.metrics.fixed_latency_ms for f in all_frames]
    adaptive_lat = [f.metrics.adaptive_latency_ms for f in all_frames]
    total_lat = [f.metrics.total_pipeline_ms for f in all_frames]

    ax.plot(frame_ids, total_lat, "r-", linewidth=1.2, label="Total Pipeline (ms)")
    ax.plot(frame_ids, fixed_lat, "b--", linewidth=0.8, label="Fixed Grid (ms)")
    ax.plot(frame_ids, adaptive_lat, "g--", linewidth=0.8, label="Adaptive Build (ms)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Latency (ms)", color="red")
    ax.tick_params(axis="y", labelcolor="red")
    ax.legend(loc="upper left", fontsize=7)

    ax2 = ax.twinx()
    cells_fixed = [f.metrics.fixed_active_cells for f in all_frames]
    cells_adaptive = [f.metrics.adaptive_leaf_cells for f in all_frames]
    ax2.plot(frame_ids, cells_fixed, "b-", linewidth=0.8, alpha=0.5, label="Fixed Cells")
    ax2.plot(frame_ids, cells_adaptive, "g-", linewidth=0.8, alpha=0.5, label="Adaptive Cells")
    ax2.set_ylabel("Active Cells", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.legend(loc="upper right", fontsize=7)

    ax.set_title("Performance: Latency & Cell Count", fontsize=11)

    fig.suptitle(
        "SIH26053 - Adaptive Variable Resolution 2.5D LiDAR Mapping (Phase 1)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plot_path = os.path.join(output_dir, "verification_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved {plot_path}")


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)

