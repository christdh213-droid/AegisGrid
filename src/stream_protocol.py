"""Versioned little-endian binary LiDAR stream. See HANDOFF for wire layout."""
from __future__ import annotations
import json
import struct
import time
import numpy as np

MAGIC = b'L25D'
VERSION = 1
HEADER = struct.Struct('<4sHHIdIIIII')
Z_MIN = -2.0
Z_MAX = 6.0
Z_STEP = (Z_MAX - Z_MIN) / 254.0
CELL_BYTES = 10
RISK = {'SAFE': 0, 'CAUTION': 1, 'HAZARD': 2, 'UNKNOWN': 3}


def quantize_elevation(z, occupied):
    """0=no data; 1..255 cover [-2,6] in 254 intervals; clip out-of-range."""
    safe = np.where(occupied, z, Z_MIN)
    return np.where(occupied, np.rint((np.clip(safe, Z_MIN, Z_MAX) - Z_MIN) / Z_STEP) + 1, 0).astype('u1')


def encode_frame(frame, fixed, *, server_time_ms=None, leaf_arrays=None, flags=None):
    """Pack raster + 10-byte/leaf SoA + small JSON + reserved extension section."""
    raster = quantize_elevation(fixed.max_z, fixed.count > 0).tobytes()
    cells = frame.adaptive_cells
    n = len(cells)
    cfg = frame.config_snapshot
    res, half = cfg['finest_res_m'], cfg['half_size']
    if leaf_arrays:
        x, y, levels = (leaf_arrays[key] for key in ('x_idx', 'y_idx', 'level'))
        z_min, z_max = leaf_arrays['z_min_cm'], leaf_arrays['z_max_cm']
    else:
        x = np.fromiter((round((c.x - c.size / 2 + half) / res) for c in cells), dtype='<u2', count=n)
        y = np.fromiter((round((c.y - c.size / 2 + half) / res) for c in cells), dtype='<u2', count=n)
        levels = np.fromiter((c.level for c in cells), dtype='u1', count=n)
        mn = np.fromiter((c.min_z for c in cells), dtype=np.float64, count=n)
        mx = np.fromiter((c.max_z for c in cells), dtype=np.float64, count=n)
        z_min = np.clip(np.rint(mn * 100), -32768, 32767).astype('<i2')
        z_max = np.clip(np.rint(mx * 100), -32768, 32767).astype('<i2')
    if flags is None:
        flags = np.fromiter((int(c.is_dynamic) | (RISK.get(c.traversability, 3) << 1) | (int(c.point_count > 0) << 3) for c in cells), dtype='u1', count=n)
    adaptive = b''.join(a.tobytes() for a in (x, y, levels, flags, z_min, z_max))
    metadata = json.dumps(dict(protocol_version=VERSION, timestamp=frame.timestamp,
                               dynamic_tracks=[t.to_dict() for t in frame.dynamic_tracks],
                               metrics=frame.metrics.to_dict(), extras=frame.extras,
                               grid=dict(dim=fixed.dim, resolution_m=res, half_size_m=half,
                                         z_min_m=Z_MIN, z_step_m=Z_STEP)),
                          separators=(',', ':'), allow_nan=False).encode('utf-8')
    header = HEADER.pack(MAGIC, VERSION, HEADER.size, frame.frame_id,
                         time.time() * 1000 if server_time_ms is None else server_time_ms,
                         len(raster), len(adaptive), len(metadata), 0, n)
    return header + raster + adaptive + metadata


def decode_frame(data):
    """Strict reference decoder used by protocol tests and diagnostics."""
    if len(data) < HEADER.size:
        raise ValueError('Truncated header')
    magic, version, header_size, frame_id, timestamp, nr, na, nj, ne, n = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION or header_size != HEADER.size:
        raise ValueError('Unsupported protocol')
    if len(data) != header_size + nr + na + nj + ne or na != n * CELL_BYTES:
        raise ValueError('Invalid section lengths')
    offset = header_size
    raster = np.frombuffer(data, dtype='u1', count=nr, offset=offset)
    offset += nr
    arrays = {}
    for name, dtype in [('x_idx', '<u2'), ('y_idx', '<u2'), ('level', 'u1'), ('flags', 'u1'), ('z_min', '<i2'), ('z_max', '<i2')]:
        arrays[name] = np.frombuffer(data, dtype=dtype, count=n, offset=offset)
        offset += arrays[name].nbytes
    return dict(frame_id=frame_id, server_time_ms=timestamp, baseline=raster, cells=arrays,
                metadata=json.loads(data[offset:offset + nj]), extension=data[offset + nj:])
