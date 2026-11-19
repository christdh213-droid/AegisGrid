#!/usr/bin/env python3
"""
SIH26053 — Adaptive Variable Resolution 2.5D LiDAR Mapping
Interactive Live Prototype Dashboard (FastAPI + HTML5 Canvas)

Launch via:
    python prototype_app.py --port 8000
    or
    uvicorn prototype_app:app --port 8000
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import asynccontextmanager
import json
import os
import sys
import threading
import time
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

# ── Core Backend Imports ──────────────────────────────────────────────────
from src.config import PipelineConfig
from src.data_source import SyntheticLidarStream
from src.pipeline import PerceptionPipeline
from src.schemas import FrameOutput

# ── FastAPI App Setup ─────────────────────────────────────────────────────
app = FastAPI(
    title="SIH26053 DRDO - Adaptive LiDAR Perception Engine",
    description="Interactive Live Split-Screen 2.5D LiDAR Perception Dashboard",
    version="1.0.0",
)


class DashboardState:
    """Encapsulates backend pipeline and stream state for live interaction."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.pipeline = PerceptionPipeline(self.config)
        self.stream = SyntheticLidarStream(self.config)
        self.is_playing = False
        self.target_fps = 10.0
        self.current_frame: Optional[FrameOutput] = None
        self.latest_payload: Optional[dict] = None
        self.lock = threading.Lock()

        # Generate initial frame 0
        self.step_frame()

    def reset(self) -> dict:
        """Reset stream and pipeline back to frame 0."""
        self.pipeline.reset()
        self.stream.reset()
        return self.step_frame()

    def step_frame(self) -> dict:
        """Advance one frame and build the serialized payload."""
        pts, meta = next(self.stream)
        frame = self.pipeline.process_frame(pts, metadata=meta)
        self.current_frame = frame

        # Convert to dictionary (safe native Python types)
        d = frame.to_dict()

        # Encode 256x256 fixed grid into compact base64 uint8:
        # [-0.5m, 4.0m] mapped to [1, 255]; 0 = empty cell
        fixed = self.pipeline.fixed_map
        occupied = fixed.count > 0
        norm = np.clip(
            (fixed.max_z - (-0.5)) / 4.5 * 254.0 + 1.0, 1.0, 255.0
        ).astype(np.uint8)
        norm[~occupied] = 0
        d["fixed_grid_b64"] = base64.b64encode(norm.tobytes()).decode("ascii")

        # Level breakdown counts for live status HUD
        level_counts = [0, 0, 0, 0, 0]
        for c in frame.adaptive_cells:
            if 0 <= c.level <= 4:
                level_counts[c.level] += 1
        d["level_counts"] = level_counts

        self.latest_payload = d
        return d


# Session 1 binary service; the original DashboardState remains for baseline recon.
from pathlib import Path
from fastapi import Request, HTTPException
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from src.stream_runtime import StreamState
from src.stream_protocol import HEADER
import contextlib

state = StreamState()
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(state.run())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

app.router.lifespan_context = lifespan

@app.get("/healthz", include_in_schema=False)
async def healthz():
    return {"status": "ok"}

async def control(message):
    try:
        ack = await asyncio.to_thread(state.execute, message)
    except (ValueError, TypeError) as error:
        raise HTTPException(400, str(error)) from error
    if message.get("action") in ("step", "reset"):
        state.publish()
    return ack

async def frame_response(request):
    if "application/octet-stream" in request.headers.get("accept", ""):
        if state.current_frame is None:
            await asyncio.to_thread(state.tick)
        payload = state.latest_binary
        return Response(payload, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})
    return JSONResponse(await asyncio.to_thread(state.legacy_payload))

@app.get("/api/frame")
async def get_frame(request: Request):
    return await frame_response(request)

@app.post("/api/control")
async def configure(request: Request):
    try:
        message = await request.json()
    except ValueError as error:
        raise HTTPException(400, "Invalid JSON") from error
    return await control(message)

@app.post("/api/step")
async def step_frame(request: Request):
    await control({"action": "step"})
    return await frame_response(request)

@app.post("/api/reset")
async def reset_simulation(request: Request):
    await control({"action": "reset"})
    return await frame_response(request)

@app.post("/api/play")
async def play_simulation():
    return await control({"action": "play"})

@app.post("/api/pause")
async def pause_simulation():
    return await control({"action": "pause"})

@app.post("/api/speed")
async def set_speed(fps: float):
    return await control({"action": "set_speed", "fps": fps})

@app.post("/api/upload")
async def upload_stub():
    raise HTTPException(501, "Upload parsing is scheduled for Session 6")

@app.get("/api/stream-stats")
async def stream_statistics():
    return await asyncio.to_thread(state.statistics)

@app.websocket("/ws/stream")
async def binary_stream(ws: WebSocket):
    await socket_session(ws, binary=True)

@app.websocket("/ws")
async def legacy_stream(ws: WebSocket):
    await socket_session(ws, binary=False)

async def socket_session(ws, binary):
    await ws.accept()
    if state.current_frame is None:
        await asyncio.to_thread(state.tick)
    queue = asyncio.Queue(maxsize=1)
    state.subscribers.add(queue)
    queue.put_nowait((state.revision, state.latest_binary))
    send_lock = asyncio.Lock()
    async def send_frames():
        while True:
            _, payload = await queue.get()
            if binary:
                frame_id = HEADER.unpack_from(payload)[3]
                # Pack timing is captured for this frame, independently of delivery.
                pack_ms = state.pack_by_frame.get(frame_id, 0.)
                async with send_lock:
                    started = time.perf_counter()
                    await ws.send_bytes(payload)
                    elapsed = (time.perf_counter() - started) * 1000
                await asyncio.to_thread(state.record_send, frame_id, pack_ms, elapsed)
            else:
                data = await asyncio.to_thread(state.legacy_payload)
                async with send_lock:
                    await ws.send_json(data)
    sender = asyncio.create_task(send_frames())
    try:
        while True:
            try:
                message = await ws.receive_json()
                ack = await control(message)
                if binary:
                    async with send_lock:
                        await ws.send_json(ack)
            except (HTTPException, ValueError, TypeError) as error:
                async with send_lock:
                    await ws.send_json({"type": "error", "message": str(getattr(error, "detail", error))})
    except WebSocketDisconnect:
        pass
    finally:
        state.subscribers.discard(queue)
        sender.cancel()
        with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
            await sender


# ── Self-Contained HTML5 Canvas Frontend ──────────────────────────────────

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>SIH26053 (DRDO) — Adaptive Variable-Resolution 2.5D LiDAR Perception Engine</title>
<style>
  :root {
    --bg-dark: #070b12;
    --card-bg: #0f172a;
    --card-border: #1e293b;
    --accent-cyan: #00e5ff;
    --accent-green: #00ff66;
    --accent-amber: #ffaa00;
    --accent-red: #ff3344;
    --accent-blue: #1e3a8a;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background-color: var(--bg-dark);
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    padding: 16px 24px;
    min-height: 100vh;
  }

  /* Header */
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--card-border);
    padding-bottom: 12px;
    margin-bottom: 16px;
  }
  .title-group h1 {
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .drdo-tag {
    background: linear-gradient(135deg, #0052cc, #00e5ff);
    color: #fff;
    font-size: 0.7rem;
    font-weight: 800;
    padding: 2px 8px;
    border-radius: 4px;
    letter-spacing: 1px;
  }
  .title-group p {
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin-top: 2px;
  }
  .status-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
  }
  .status-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: var(--accent-amber);
    box-shadow: 0 0 8px var(--accent-amber);
    transition: all 0.3s ease;
  }
  .status-dot.live {
    background: var(--accent-green);
    box-shadow: 0 0 10px var(--accent-green);
  }

  /* Metric HUD Cards */
  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
    margin-bottom: 16px;
  }
  .metric-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 10px 14px;
    position: relative;
    overflow: hidden;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: var(--card-border);
  }
  .metric-card.accent-cyan::before { background: var(--accent-cyan); }
  .metric-card.accent-green::before { background: var(--accent-green); }
  .metric-card.accent-amber::before { background: var(--accent-amber); }
  .metric-card.accent-red::before { background: var(--accent-red); }

  .metric-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .metric-value {
    font-size: 1.45rem;
    font-weight: 800;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', Consolas, monospace;
    margin: 4px 0 2px 0;
  }
  .metric-sub {
    font-size: 0.72rem;
    color: var(--text-secondary);
  }

  /* Control Toolbar */
  .control-bar {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 10px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    gap: 16px;
    flex-wrap: wrap;
  }
  .btn-group {
    display: flex;
    gap: 8px;
    align-items: center;
  }
  button {
    background: #1e293b;
    color: var(--text-primary);
    border: 1px solid #334155;
    padding: 7px 16px;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: all 0.2s ease;
  }
  button:hover {
    background: #334155;
    border-color: var(--accent-cyan);
    color: #fff;
  }
  button.btn-primary {
    background: #0284c7;
    border-color: #38bdf8;
    color: #fff;
  }
  button.btn-primary:hover {
    background: #0369a1;
  }
  button.btn-pause {
    background: #b45309;
    border-color: #f59e0b;
  }

  .slider-group {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.8rem;
    color: var(--text-secondary);
  }
  input[type=range] {
    accent-color: var(--accent-cyan);
    cursor: pointer;
  }

  .toggle-group {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 0.8rem;
    color: var(--text-secondary);
  }
  select {
    background: #1e293b;
    color: var(--text-primary);
    border: 1px solid #334155;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 0.8rem;
    outline: none;
    cursor: pointer;
  }

  /* Dual Canvas Split View */
  .split-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
  }
  .canvas-container {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    align-items: center;
    position: relative;
  }
  .canvas-header {
    width: 100%;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }
  .canvas-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: 0.5px;
  }
  .canvas-badge {
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
    background: #1e293b;
    color: var(--accent-cyan);
  }
  .viewport {
    width: 100%;
    max-width: 540px;
    aspect-ratio: 1 / 1;
    background: #000;
    border: 1px solid #334155;
    border-radius: 4px;
    position: relative;
    overflow: hidden;
  }
  canvas {
    width: 100%;
    height: 100%;
    display: block;
    image-rendering: pixelated;
  }
  .canvas-overlay-info {
    position: absolute;
    bottom: 6px;
    left: 6px;
    background: rgba(15, 23, 42, 0.85);
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', Consolas, monospace;
    color: #e2e8f0;
    pointer-events: none;
    border: 1px solid #334155;
  }

  /* Quadtree Hierarchy Breakdown Bar */
  .hierarchy-bar {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 10px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.8rem;
    color: var(--text-secondary);
  }
  .levels-list {
    display: flex;
    gap: 16px;
    align-items: center;
  }
  .level-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 0.75rem;
  }
  .level-chip {
    width: 12px;
    height: 12px;
    border-radius: 2px;
  }
  .chip-l0 { background: #ff3344; }
  .chip-l1 { background: #ffaa00; }
  .chip-l2 { background: #a6e22e; }
  .chip-l3 { background: #00e5ff; }
  .chip-l4 { background: #1e3a8a; }

  /* Tooltip HUD */
  #tooltip {
    position: fixed;
    display: none;
    background: rgba(15, 23, 42, 0.95);
    border: 1px solid var(--accent-cyan);
    color: #f8fafc;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', Consolas, monospace;
    pointer-events: none;
    z-index: 1000;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
  }
</style>
</head>
<body>

<header>
  <div class="title-group">
    <h1>
      <span class="drdo-tag">DRDO SIH26053</span>
      Adaptive Variable-Resolution 2.5D LiDAR Perception Engine
    </h1>
    <p>Real-Time Vectorized Quadtree Pyramid · Multi-Actor Dynamic Tracking · Instant Zero-Ghost Reversion</p>
  </div>
  <div class="status-badge">
    <div id="statusDot" class="status-dot"></div>
    <span id="statusText">PAUSED (Frame 0)</span>
  </div>
</header>

<!-- Metrics HUD -->
<section class="metrics-grid">
  <div class="metric-card accent-cyan">
    <div class="metric-label">Throughput / FPS</div>
    <div id="metricFps" class="metric-value">--</div>
    <div class="metric-sub">Real-Time Target: &gt;20 FPS</div>
  </div>
  <div class="metric-card accent-green">
    <div class="metric-label">Pipeline Latency</div>
    <div id="metricLatency" class="metric-value">-- ms</div>
    <div id="metricLatencySub" class="metric-sub">Fixed: -- | Adapt: --</div>
  </div>
  <div class="metric-card accent-cyan">
    <div class="metric-label">Cell Reduction</div>
    <div id="metricReduction" class="metric-value">-- %</div>
    <div id="metricReductionSub" class="metric-sub">Leaves: -- / 65,536</div>
  </div>
  <div class="metric-card accent-green">
    <div class="metric-label">Memory Saved</div>
    <div id="metricMemory" class="metric-value">-- %</div>
    <div id="metricMemorySub" class="metric-sub">1280 KB &rarr; -- KB</div>
  </div>
  <div class="metric-card accent-amber">
    <div class="metric-label">Obstacle RMSE</div>
    <div id="metricRmse" class="metric-value">-- m</div>
    <div class="metric-sub">Gate: &le; 0.0500 m</div>
  </div>
  <div class="metric-card accent-cyan">
    <div class="metric-label">Ghost Trails</div>
    <div id="metricGhosts" class="metric-value">0</div>
    <div class="metric-sub">Audit: Zero Reversion Lag</div>
  </div>
</section>

<!-- Controls Bar -->
<section class="control-bar">
  <div class="btn-group">
    <button id="btnPlayPause" class="btn-primary" onclick="togglePlay()">
      <span id="playIcon">&#9654;</span> <span id="playLabel">Play</span>
    </button>
    <button onclick="stepSimulation()">&#9197; Step</button>
    <button onclick="resetSimulation()">&#8634; Reset</button>
  </div>

  <div class="slider-group">
    <span>Simulation Speed:</span>
    <input id="speedSlider" type="range" min="1" max="30" value="10" oninput="changeSpeed(this.value)"/>
    <span id="speedVal" style="font-family: monospace; font-weight: bold; width: 45px;">10 FPS</span>
  </div>

  <div class="toggle-group">
    <span>Color Mode:</span>
    <select id="colorModeSelect" onchange="renderCurrentFrame()">
      <option value="level">Quadtree Resolution Level (L0-L4)</option>
      <option value="elevation">Elevation Height (Z)</option>
    </select>
  </div>

  <div class="toggle-group">
    <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
      <input type="checkbox" id="chkShowBorders" checked onchange="renderCurrentFrame()"/> Cell Borders
    </label>
    <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
      <input type="checkbox" id="chkShowTracks" checked onchange="renderCurrentFrame()"/> Dynamic Bounding Boxes &amp; Arrows
    </label>
  </div>
</section>

<!-- Split-Screen Canvases -->
<section class="split-grid">
  <div class="canvas-container">
    <div class="canvas-header">
      <div class="canvas-title">BASELINE FIXED 0.25m GRID</div>
      <div class="canvas-badge">256 &times; 256 (65,536 Cells)</div>
    </div>
    <div class="viewport">
      <canvas id="leftCanvas" width="512" height="512"></canvas>
      <div class="canvas-overlay-info">X: [-32m, +32m] &nbsp;|&nbsp; Y: [-32m, +32m]</div>
    </div>
  </div>

  <div class="canvas-container">
    <div class="canvas-header">
      <div class="canvas-title">ADAPTIVE VARIABLE-RESOLUTION QUADTREE</div>
      <div id="adaptiveBadge" class="canvas-badge">-- Active Leaves</div>
    </div>
    <div class="viewport">
      <canvas id="rightCanvas" width="512" height="512"></canvas>
      <div id="rightOverlayInfo" class="canvas-overlay-info">Tracks: -- &nbsp;|&nbsp; Reduction: --%</div>
    </div>
  </div>
</section>

<!-- Hierarchy Breakdown Bar -->
<section class="hierarchy-bar">
  <span style="font-weight: 600; color: var(--text-primary);">Pyramid Hierarchy Breakdown:</span>
  <div class="levels-list">
    <div class="level-pill"><span class="level-chip chip-l0"></span> L0 (0.25m Obstacles): <b id="cntL0">--</b></div>
    <div class="level-pill"><span class="level-chip chip-l1"></span> L1 (0.50m): <b id="cntL1">--</b></div>
    <div class="level-pill"><span class="level-chip chip-l2"></span> L2 (1.00m): <b id="cntL2">--</b></div>
    <div class="level-pill"><span class="level-chip chip-l3"></span> L3 (2.00m Ground): <b id="cntL3">--</b></div>
    <div class="level-pill"><span class="level-chip chip-l4"></span> L4 (4.00m Far Plain): <b id="cntL4">--</b></div>
  </div>
  <div style="font-size: 0.72rem; color: var(--text-muted);">Hover cursor over right map for cell inspection</div>
</section>

<div id="tooltip"></div>

<script>
  let isPlaying = false;
  let currentFrameData = null;
  let ws = null;
  const MAP_SIZE = 64.0;
  const HALF_SIZE = 32.0;

  // Level Colors
  const LEVEL_COLORS = [
    '#ff3344', // L0 (0.25m) red
    '#ffaa00', // L1 (0.50m) amber
    '#a6e22e', // L2 (1.00m) lime
    '#00e5ff', // L3 (2.00m) cyan
    '#1e3a8a', // L4 (4.00m) deep blue
  ];

  // Canvas elements & contexts
  const leftCanvas = document.getElementById('leftCanvas');
  const leftCtx = leftCanvas.getContext('2d');
  const rightCanvas = document.getElementById('rightCanvas');
  const rightCtx = rightCanvas.getContext('2d');
  const tooltip = document.getElementById('tooltip');

  // Colormap helper for Elevation mode (Terrain: dark blue -> teal -> green -> yellow -> red)
  function elevationToRgb(z) {
    // Normalise [-0.5m, 3.5m] -> [0, 1]
    const t = Math.max(0, Math.min(1, (z - (-0.5)) / 4.0));
    let r = 0, g = 0, b = 0;
    if (t < 0.2) {
      const u = t / 0.2;
      r = Math.round(15 + u * 20);
      g = Math.round(30 + u * 100);
      b = Math.round(100 + u * 155);
    } else if (t < 0.4) {
      const u = (t - 0.2) / 0.2;
      r = Math.round(35 - u * 35);
      g = Math.round(130 + u * 80);
      b = Math.round(255 - u * 100);
    } else if (t < 0.7) {
      const u = (t - 0.4) / 0.3;
      r = Math.round(u * 220);
      g = Math.round(210 + u * 30);
      b = Math.round(155 - u * 155);
    } else {
      const u = (t - 0.7) / 0.3;
      r = Math.round(220 + u * 35);
      g = Math.round(240 - u * 180);
      b = Math.round(u * 50);
    }
    return [r, g, b];
  }

  // World coordinates (x, y in [-32, +32]) to canvas pixels (512x512)
  function worldToCanvas(wx, wy) {
    const px = ((wx + HALF_SIZE) / MAP_SIZE) * 512;
    const py = 512 - ((wy + HALF_SIZE) / MAP_SIZE) * 512; // Invert Y
    return [px, py];
  }

  function canvasToWorld(cx, cy) {
    const wx = (cx / 512) * MAP_SIZE - HALF_SIZE;
    const wy = ((512 - cy) / 512) * MAP_SIZE - HALF_SIZE;
    return [wx, wy];
  }

  // ── Render Frame ────────────────────────────────────────────────────────
  function renderCurrentFrame() {
    if (!currentFrameData) return;
    const d = currentFrameData;
    const mode = document.getElementById('colorModeSelect').value;
    const showBorders = document.getElementById('chkShowBorders').checked;
    const showTracks = document.getElementById('chkShowTracks').checked;

    // ── 1. Update Metrics HUD ─────────────────────────────────────────────
    const m = d.metrics;
    document.getElementById('metricFps').innerText = m.fps.toFixed(1);
    document.getElementById('metricLatency').innerText = m.total_pipeline_ms.toFixed(1) + ' ms';
    document.getElementById('metricLatencySub').innerText =
      `Fixed: ${m.fixed_latency_ms.toFixed(1)}ms | Adapt: ${m.adaptive_latency_ms.toFixed(1)}ms`;
    document.getElementById('metricReduction').innerText = m.cell_reduction_pct.toFixed(1) + ' %';
    document.getElementById('metricReductionSub').innerText =
      `${m.adaptive_leaf_cells} / ${m.fixed_active_cells || 65536} cells`;
    document.getElementById('metricMemory').innerText = m.memory_saved_pct.toFixed(1) + ' %';
    document.getElementById('metricMemorySub').innerText =
      `${m.fixed_memory_kb.toFixed(0)} KB \u2192 ${m.adaptive_memory_kb.toFixed(1)} KB`;
    document.getElementById('metricRmse').innerText = m.obstacle_rmse_m.toFixed(4) + ' m';
    document.getElementById('metricGhosts').innerText = m.ghost_cells_detected;

    document.getElementById('adaptiveBadge').innerText = `${d.adaptive_cells.length} Active Leaves`;
    document.getElementById('rightOverlayInfo').innerText =
      `Tracks: ${d.dynamic_tracks.length} | Reduction: ${m.cell_reduction_pct.toFixed(1)}%`;

    // Hierarchy counts
    if (d.level_counts) {
      document.getElementById('cntL0').innerText = d.level_counts[0];
      document.getElementById('cntL1').innerText = d.level_counts[1];
      document.getElementById('cntL2').innerText = d.level_counts[2];
      document.getElementById('cntL3').innerText = d.level_counts[3];
      document.getElementById('cntL4').innerText = d.level_counts[4];
    }

    // Status badge
    document.getElementById('statusText').innerText =
      (isPlaying ? 'STREAMING' : 'PAUSED') + ` (Frame ${d.frame_id})`;

    // ── 2. Render Left Canvas (Fixed 0.25m Grid) ──────────────────────────
    leftCtx.fillStyle = '#06090e';
    leftCtx.fillRect(0, 0, 512, 512);

    if (d.fixed_grid_b64) {
      const raw = atob(d.fixed_grid_b64);
      const imgData = leftCtx.createImageData(256, 256);
      const dataBuf = imgData.data;

      for (let i = 0; i < 65536; i++) {
        const val = raw.charCodeAt(i);
        const p = i * 4;
        if (val === 0) {
          dataBuf[p] = 6;
          dataBuf[p+1] = 9;
          dataBuf[p+2] = 14;
          dataBuf[p+3] = 255;
        } else {
          if (mode === 'level') {
            // In level mode: fixed grid is 100% L0 (0.25m) red
            dataBuf[p] = 255;
            dataBuf[p+1] = 51;
            dataBuf[p+2] = 68;
            dataBuf[p+3] = 255;
          } else {
            // Elevation mode
            const zNorm = -0.5 + ((val - 1) / 254.0) * 4.5;
            const [r, g, b] = elevationToRgb(zNorm);
            dataBuf[p] = r;
            dataBuf[p+1] = g;
            dataBuf[p+2] = b;
            dataBuf[p+3] = 255;
          }
        }
      }

      // Draw 256x256 buffer scaled to 512x512 with crisp pixels
      createImageBitmap(imgData).then(bmp => {
        leftCtx.imageSmoothingEnabled = false;
        leftCtx.drawImage(bmp, 0, 0, 512, 512);
        drawGridAxes(leftCtx);
      });
    }

    // ── 3. Render Right Canvas (Adaptive Quadtree Map) ─────────────────────
    rightCtx.fillStyle = '#06090e';
    rightCtx.fillRect(0, 0, 512, 512);

    const cells = d.adaptive_cells;
    const scale = 512.0 / MAP_SIZE;

    // Draw leaf cells
    for (let i = 0; i < cells.length; i++) {
      const c = cells[i];
      // Top-left in world coords
      const wx0 = c.x - c.size / 2.0;
      const wy1 = c.y + c.size / 2.0; // Top edge
      const px = ((wx0 + HALF_SIZE) / MAP_SIZE) * 512;
      const py = 512 - ((wy1 + HALF_SIZE) / MAP_SIZE) * 512;
      const pw = c.size * scale;
      const ph = c.size * scale;

      if (c.point_count === 0) {
        rightCtx.fillStyle = '#06090e';
      } else if (c.is_dynamic) {
        rightCtx.fillStyle = '#ff2233'; // Vibrant dynamic cell indicator
      } else if (mode === 'level') {
        rightCtx.fillStyle = LEVEL_COLORS[c.level] || '#fff';
      } else {
        const [r, g, b] = elevationToRgb(c.max_z);
        rightCtx.fillStyle = `rgb(${r},${g},${b})`;
      }

      rightCtx.fillRect(px, py, pw, ph);

      if (showBorders && c.size >= 0.5) {
        rightCtx.strokeStyle = 'rgba(0, 0, 0, 0.45)';
        rightCtx.lineWidth = c.size >= 2.0 ? 1.0 : 0.5;
        rightCtx.strokeRect(px, py, pw, ph);
      }
    }

    // Draw Dynamic Tracks, Bounding Boxes & 1.5s Velocity Arrows
    if (showTracks && d.dynamic_tracks) {
      for (let i = 0; i < d.dynamic_tracks.length; i++) {
        const trk = d.dynamic_tracks[i];
        const [cx, cy] = worldToCanvas(trk.x, trk.y);
        const bw = trk.length * scale;
        const bh = trk.width * scale;

        // Bounding box
        rightCtx.strokeStyle = '#00ff66';
        rightCtx.lineWidth = 2.0;
        rightCtx.strokeRect(cx - bw / 2, cy - bh / 2, bw, bh);

        // ID tag & speed
        rightCtx.fillStyle = 'rgba(15, 23, 42, 0.85)';
        const tag = `ID:${trk.track_id} ${trk.classification.substring(0,3)} ${trk.speed.toFixed(1)}m/s`;
        rightCtx.font = 'bold 9px monospace';
        const txtWidth = rightCtx.measureText(tag).width;
        rightCtx.fillRect(cx - txtWidth/2 - 3, cy - bh/2 - 14, txtWidth + 6, 12);
        rightCtx.fillStyle = '#00ff66';
        rightCtx.fillText(tag, cx - txtWidth/2, cy - bh/2 - 4);

        // Velocity prediction arrow (1.5s horizon)
        if (trk.speed > 0.3) {
          const [px1, py1] = worldToCanvas(trk.pred_x, trk.pred_y);
          drawArrow(rightCtx, cx, cy, px1, py1, '#ffea00');
        }
      }
    }

    drawGridAxes(rightCtx);
  }

  // Draw arrow helper
  function drawArrow(ctx, fromx, fromy, tox, toy, color) {
    const headlen = 8;
    const dx = tox - fromx;
    const dy = toy - fromy;
    const angle = Math.atan2(dy, dx);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;

    ctx.beginPath();
    ctx.moveTo(fromx, fromy);
    ctx.lineTo(tox, toy);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(tox, toy);
    ctx.lineTo(tox - headlen * Math.cos(angle - Math.PI / 6), toy - headlen * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(tox - headlen * Math.cos(angle + Math.PI / 6), toy - headlen * Math.sin(angle + Math.PI / 6));
    ctx.lineTo(tox, toy);
    ctx.fill();
  }

  // Grid axes overlay
  function drawGridAxes(ctx) {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.lineWidth = 1;
    // Crosshair at origin (0, 0)
    ctx.beginPath();
    ctx.moveTo(256, 0); ctx.lineTo(256, 512);
    ctx.moveTo(0, 256); ctx.lineTo(512, 256);
    ctx.stroke();

    // Corner markings
    ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.font = '8px monospace';
    ctx.fillText('-32m', 4, 508);
    ctx.fillText('+32m', 484, 508);
    ctx.fillText('+32m', 4, 12);
    ctx.fillText('0,0', 260, 252);
  }

  // ── WebSocket Connection ────────────────────────────────────────────────
  function connectWebSocket() {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${window.location.host}/ws`);

    ws.onopen = () => {
      document.getElementById('statusDot').classList.add('live');
    };

    ws.onmessage = (event) => {
      try {
        currentFrameData = JSON.parse(event.data);
        renderCurrentFrame();
      } catch (e) {
        console.error('Error parsing frame:', e);
      }
    };

    ws.onclose = () => {
      document.getElementById('statusDot').classList.remove('live');
      setTimeout(connectWebSocket, 1500);
    };
  }

  // ── User Interaction Controls ───────────────────────────────────────────
  function togglePlay() {
    isPlaying = !isPlaying;
    const btn = document.getElementById('btnPlayPause');
    const icon = document.getElementById('playIcon');
    const label = document.getElementById('playLabel');

    if (isPlaying) {
      btn.className = 'btn-pause';
      icon.innerHTML = '&#10074;&#10074;';
      label.innerText = 'Pause';
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: 'play' }));
      } else {
        fetch('/api/play', { method: 'POST' });
      }
    } else {
      btn.className = 'btn-primary';
      icon.innerHTML = '&#9654;';
      label.innerText = 'Play';
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: 'pause' }));
      } else {
        fetch('/api/pause', { method: 'POST' });
      }
    }
  }

  function stepSimulation() {
    if (isPlaying) togglePlay();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'step' }));
    } else {
      fetch('/api/step', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
          currentFrameData = data;
          renderCurrentFrame();
        });
    }
  }

  function resetSimulation() {
    if (isPlaying) togglePlay();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'reset' }));
    } else {
      fetch('/api/reset', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
          currentFrameData = data;
          renderCurrentFrame();
        });
    }
  }

  function changeSpeed(fps) {
    document.getElementById('speedVal').innerText = `${fps} FPS`;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'set_speed', fps: parseFloat(fps) }));
    } else {
      fetch(`/api/speed?fps=${fps}`, { method: 'POST' });
    }
  }

  // Keyboard shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
      e.preventDefault();
      togglePlay();
    } else if (e.code === 'ArrowRight') {
      e.preventDefault();
      stepSimulation();
    } else if (e.key === 'r' || e.key === 'R') {
      resetSimulation();
    }
  });

  // Hover Tooltip on Right Canvas
  rightCanvas.addEventListener('mousemove', (e) => {
    if (!currentFrameData || !currentFrameData.adaptive_cells) return;
    const rect = rightCanvas.getBoundingClientRect();
    const cx = (e.clientX - rect.left) * (512 / rect.width);
    const cy = (e.clientY - rect.top) * (512 / rect.height);
    const [wx, wy] = canvasToWorld(cx, cy);

    // Find cell containing (wx, wy)
    const cells = currentFrameData.adaptive_cells;
    let found = null;
    for (let i = 0; i < cells.length; i++) {
      const c = cells[i];
      const h = c.size / 2.0;
      if (wx >= c.x - h && wx < c.x + h && wy >= c.y - h && wy < c.y + h) {
        found = c;
        break;
      }
    }

    if (found) {
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX + 14) + 'px';
      tooltip.style.top = (e.clientY + 14) + 'px';
      tooltip.innerHTML = `
        <b>Cell #${found.cell_id}</b> [Level ${found.level} &mdash; ${found.size.toFixed(2)}m]<br/>
        Pos: (${found.x.toFixed(2)}m, ${found.y.toFixed(2)}m)<br/>
        max_z: ${found.max_z.toFixed(2)}m &nbsp;|&nbsp; span_z: ${found.span_z.toFixed(2)}m<br/>
        Points: ${found.point_count} &nbsp;|&nbsp; Dynamic: <b style="color:${found.is_dynamic ? '#ff3344':'#00ff66'}">${found.is_dynamic ? 'YES' : 'NO'}</b>
      `;
    } else {
      tooltip.style.display = 'none';
    }
  });

  rightCanvas.addEventListener('mouseleave', () => {
    tooltip.style.display = 'none';
  });

  // Start initial frame fetch & connect WebSocket
  fetch('/api/frame')
    .then(r => r.json())
    .then(d => {
      currentFrameData = d;
      renderCurrentFrame();
      connectWebSocket();
    })
    .catch(() => connectWebSocket());

</script>
</body>
</html>
"""


@app.get("/legacy", response_class=HTMLResponse)
async def get_index():
    """Serves the standalone single-file HTML5 Canvas dashboard."""
    return HTML_CONTENT


@app.get("/")
async def spatial_viewer():
    return FileResponse(STATIC_DIR / "index.html")


def main():
    parser = argparse.ArgumentParser(
        description="SIH26053 DRDO - Adaptive LiDAR Perception Interactive Prototype"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=os.environ.get("PORT", "8000"), help="Port to bind")
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  SIH26053 (DRDO) — Adaptive 2.5D LiDAR Perception Engine")
    print(f"  Interactive Prototype Dashboard")
    print(f"{'='*72}")
    print(f"  Local Web UI URL: http://{args.host}:{args.port}")
    print(f"  WebSocket Channel: ws://{args.host}:{args.port}/ws")
    print(f"  Press Ctrl+C to terminate.")
    print(f"{'='*72}\n")

    # Binary raster/SoA is already compact; deflate adds avoidable send latency.
    uvicorn.run(app, host=args.host, port=args.port, log_level="info",
                ws_per_message_deflate=False)


if __name__ == "__main__":
    main()
