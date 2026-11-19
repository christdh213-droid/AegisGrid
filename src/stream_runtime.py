"""Single pipeline owner; locks serialize controls and frames at frame boundaries."""
from __future__ import annotations
import asyncio
from dataclasses import replace, fields
import math
import os
import threading
import time
from collections import deque

from src.config import PipelineConfig, RuntimeConfig
from src.data_source import SyntheticLidarStream
from src.pipeline import PerceptionPipeline
from src.path_planner import FixedGridComparison, OccupancyPlanner
from src.hooks import DefaultPostMapHook
from src.stream_protocol import encode_frame, quantize_elevation

PROFILES = ('Unrestricted', 'Tactical Edge ECU', 'Low-Power UGV Node', 'CPU Load Spike Stress Test')
MODES = ('Split Wipe', 'Elevation', 'Quadtree', 'Traversability')


def updated_config(config, updates):
    """Validate whole command before committing; reject unknown or nonfinite values."""
    if not isinstance(updates, dict):
        raise ValueError('settings must be an object')
    defaults = RuntimeConfig()
    for key, value in updates.items():
        if key not in {f.name for f in fields(defaults)}:
            raise ValueError(f'Unknown setting: {key}')
        if isinstance(getattr(defaults, key), bool) and type(value) is not bool:
            raise ValueError(f'{key} must be boolean')
        ranges = {'outlier_ratio': (0, .15), 'speed': (.5, 2), 'pitch_deg': (-8, 8), 'roll_deg': (-8, 8), 'imu_noise_deg': (0, 5)}
        if key in ranges:
            lo, hi = ranges[key]
            if type(value) not in (float, int) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f'{key} must be in [{lo}, {hi}]')
        if key == 'governor_profile' and value not in PROFILES:
            raise ValueError('Unknown governor profile')
        if key == 'view_mode' and value not in MODES:
            raise ValueError('Unknown view mode')
        if key == 'tilt_mode' and value not in ('Auto', 'Manual'):
            raise ValueError('tilt_mode must be Auto or Manual')
        if key == 'goal_xy' and value is not None:
            if not isinstance(value, (list, tuple)) or len(value) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
                raise ValueError('goal_xy must be two finite coordinates')
            updates = {**updates, 'goal_xy': tuple(max(-32., min(31.999, v)) for v in value)}
    return replace(config, **updates)


class StreamState:
    def __init__(self):
        self.max_fps = float(os.environ.get("MAX_FPS", "30"))
        if not math.isfinite(self.max_fps) or self.max_fps < 1:
            raise ValueError("MAX_FPS must be a finite number of at least 1")
        self.disable_process_pool = os.environ.get("DISABLE_PROCESS_POOL", "0").strip().lower() in {
            "1", "true", "yes", "on",
        }
        self.lock = threading.RLock()
        self.config = PipelineConfig()
        self.runtime = RuntimeConfig()
        self.pipeline = PerceptionPipeline(self.config, perception=True)
        self.planner = OccupancyPlanner(self.config)
        self.stream = SyntheticLidarStream(self.config)
        self.is_playing = False
        self.target_fps = min(30.0, self.max_fps)
        self.current_frame = None
        self.latest_binary = None
        self.revision = 0
        self.published_revision = -1
        self.completions = deque(maxlen=120)
        self.pack_samples = deque(maxlen=300)
        self.pack_by_frame = {}
        self.send_samples = deque(maxlen=300)
        self.previous_encode = None
        self.previous_encode_id = None
        self.subscribers = set()

    def _step(self):
        snapshot = self.runtime  # frozen, remains valid for the whole frame
        self.stream.runtime_config = snapshot
        points, metadata = next(self.stream)
        metadata['runtime_config'] = snapshot
        frame = self.pipeline.process_frame(points, metadata=metadata)
        if snapshot.planner_enabled:
            adaptive = self.pipeline.adaptive_map
            frame.extras.update(self.planner.update(adaptive.leaf_id_grid, adaptive.leaf_arrays, snapshot.goal_xy))
        frame.extras.update(runtime_config=snapshot.to_dict(), server_fps=self.server_fps(),
                            encode_ms=self.previous_encode, encode_frame_id=self.previous_encode_id)
        start = time.perf_counter()
        columns = self.pipeline.adaptive_map.leaf_arrays
        # UNKNOWN flags are already vectorized by the mapper; classifier hooks
        # use live AdaptiveCell labels instead, so future risk updates propagate.
        flags = columns.get('flags') if type(self.pipeline.post_hook) is DefaultPostMapHook and self.pipeline.post_track_hook is None and self.pipeline.post_frame_hook is None else None
        payload = encode_frame(frame, self.pipeline.fixed_map, leaf_arrays=columns, flags=flags)
        pack_ms = (time.perf_counter() - start) * 1000
        self.pack_samples.append(pack_ms)
        self.pack_by_frame[frame.frame_id] = pack_ms
        if len(self.pack_by_frame) > 300:
            del self.pack_by_frame[next(iter(self.pack_by_frame))]
        self.current_frame, self.latest_binary = frame, payload
        self.completions.append(time.perf_counter())
        self.revision += 1
        return payload

    def server_fps(self):
        if len(self.completions) < 2:
            return None
        elapsed = self.completions[-1] - self.completions[0]
        return (len(self.completions) - 1) / elapsed if elapsed > 0 else None

    def execute(self, message):
        with self.lock:
            if not isinstance(message, dict):
                raise ValueError('Control must be a JSON object')
            action = message.get('action')
            if action == 'reset':
                self.runtime = RuntimeConfig()
                self.pipeline.reset()
                self.planner.reset()
                self.stream.reset()
                self.is_playing = False
                self.target_fps = min(30.0, self.max_fps)
                self.completions.clear()
                self.previous_encode = self.previous_encode_id = None
                self._step()
            elif action == 'step':
                self._step()
            elif action == 'play':
                self.is_playing = True
            elif action == 'pause':
                self.is_playing = False
            elif action == 'set_speed':
                fps = message.get('fps')
                if type(fps) not in (float, int) or not math.isfinite(fps):
                    raise ValueError('fps must be finite')
                self.target_fps = max(1., min(30., self.max_fps, float(fps)))
            elif action in ('configure', 'set_goal', 'set_profile'):
                updates = message.get('settings', {}) if action == 'configure' else ({'goal_xy': message.get('goal_xy')} if action == 'set_goal' else {'governor_profile': message.get('profile')})
                previous = self.runtime
                self.runtime = updated_config(previous, updates)
                if previous.planner_enabled != self.runtime.planner_enabled:
                    self.planner.reset()
                if not self.is_playing and (previous.goal_xy != self.runtime.goal_xy
                        or previous.planner_enabled != self.runtime.planner_enabled):
                    self._step()
            else:
                raise ValueError('Unknown action')
            if self.current_frame is None:
                self._step()
            return dict(type='control_ack', action=action, playing=self.is_playing,
                        config=self.runtime.to_dict(), fps=self.target_fps,
                        modules_active=True)

    def open_comparison(self):
        if not self.disable_process_pool:
            self.planner.comparison = FixedGridComparison()

    def close_comparison(self):
        with self.lock:
            if self.planner.comparison is not None:
                self.planner.comparison.close()
                self.planner.comparison = None

    def tick(self):
        with self.lock:
            if self.is_playing or self.current_frame is None:
                self._step()
            return self.revision, self.latest_binary

    def legacy_payload(self):
        """Compatibility only: JSON requested by /legacy and existing API clients."""
        import base64
        with self.lock:
            if self.current_frame is None:
                self._step()
            data = self.current_frame.to_dict()
            fixed = self.pipeline.fixed_map
            # Preserve the original legacy quantisation range.
            import numpy as np
            raster = quantize_elevation(fixed.max_z, fixed.count > 0)
            occupied = fixed.count > 0
            raster[occupied] = np.clip((fixed.max_z[occupied] + .5) / 4.5 * 254 + 1, 1, 255).astype('u1')
            data['fixed_grid_b64'] = base64.b64encode(raster.tobytes()).decode('ascii')
            data['level_counts'] = [sum(c.level == level for c in self.current_frame.adaptive_cells) for level in range(5)]
            return data

    def publish(self):
        """Called only on the event loop. Every subscriber owns a one-frame slot."""
        if self.revision <= self.published_revision:
            return
        self.published_revision = self.revision
        for queue in tuple(self.subscribers):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait((self.revision, self.latest_binary))

    async def run(self):
        deadline = asyncio.get_running_loop().time()
        while True:
            await asyncio.to_thread(self.tick)
            self.publish()
            interval = 1 / min(self.max_fps, self.target_fps * self.runtime.speed)
            deadline += interval
            now = asyncio.get_running_loop().time()
            if deadline < now:
                deadline = now  # no catch-up burst
            await asyncio.sleep(max(.001, deadline - now))

    def record_send(self, frame_id, pack_ms, send_ms):
        with self.lock:
            self.send_samples.append(dict(frame_id=frame_id, pack_ms=pack_ms, send_ms=send_ms, encode_ms=pack_ms + send_ms))
            self.previous_encode = pack_ms + send_ms
            self.previous_encode_id = frame_id

    def statistics(self):
        """Measured rolling samples; per-client send includes ASGI back-pressure."""
        import numpy as np
        with self.lock:
            def summary(samples):
                return None if not samples else dict(count=len(samples), p50_ms=float(np.percentile(samples, 50)), p95_ms=float(np.percentile(samples, 95)))
            return dict(pack=summary(list(self.pack_samples)),
                        pack_plus_send=summary([s['encode_ms'] for s in self.send_samples]),
                        send=summary([s['send_ms'] for s in self.send_samples]),
                        server_fps=self.server_fps(), payload_bytes=len(self.latest_binary) if self.latest_binary else None)
