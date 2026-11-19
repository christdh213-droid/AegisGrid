"""Bounded, in-memory static scan input. No client filenames reach disk."""
import io
from pathlib import PurePath
import numpy as np

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_UPLOAD_POINTS = 1_000_000


def parse_scan(data, filename):
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Scan exceeds 50 MiB limit")
    suffix = PurePath(filename).suffix.lower()
    if suffix == '.npy':
        source = io.BytesIO(data)
        version = np.lib.format.read_magic(source)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(source)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(source)
        else:
            raise ValueError("Unsupported NPY version")
        validate_layout(shape, dtype)
        if int(np.prod(shape)) * dtype.itemsize > MAX_UPLOAD_BYTES:
            raise ValueError("NPY array exceeds byte limit")
        source.seek(0)
        points = np.load(source, allow_pickle=False)
    elif suffix == '.bin':
        if len(data) % 16:
            raise ValueError("BIN requires little-endian float32 XYZI records")
        if len(data) // 16 > MAX_UPLOAD_POINTS:
            raise ValueError("Scan exceeds point limit")
        points = np.frombuffer(data, dtype='<f4').reshape(-1, 4)
    elif suffix == '.csv':
        text = data.decode('utf-8-sig')
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Empty scan")
        header = [v.strip().lower() for v in lines[0].split(',')]
        if header in (['x', 'y', 'z'], ['x', 'y', 'z', 'intensity']):
            lines = lines[1:]
        if len(lines) > MAX_UPLOAD_POINTS:
            raise ValueError("Scan exceeds point limit")
        points = np.loadtxt(io.StringIO('\n'.join(lines)), delimiter=',', ndmin=2)
    else:
        raise ValueError("Choose a .bin, .npy or .csv scan")
    validate_layout(points.shape, points.dtype)
    if not np.isfinite(points).all():
        raise ValueError("All scan values must be finite")
    points = points[:, :3].astype(np.float64)
    points = points[(np.abs(points[:, :2]) <= 32).all(axis=1)]
    if not len(points):
        raise ValueError("No points inside [-32, 32] m")
    # +32 belongs to the last half-open cell, not an out-of-bounds index.
    points[:, :2] = np.minimum(points[:, :2], np.nextafter(32., 0.))
    ground = float(np.percentile(points[:, 2], 10))
    points[:, 2] -= ground
    if np.max(np.abs(points[:, 2])) > 1000:
        raise ValueError("Scan elevation range exceeds 1000 m")
    return points, ground


def validate_layout(shape, dtype):
    if len(shape) != 2 or shape[1] not in (3, 4) or not 0 < shape[0] <= MAX_UPLOAD_POINTS:
        raise ValueError("Scan must be Nx3 or Nx4, with 1 to 1,000,000 points")
    if dtype.kind not in 'fiu' or dtype.hasobject:
        raise ValueError("Scan must use a real numeric dtype; pickle is forbidden")


class StaticOccupancy:
    """Tracker adapter: one static scan has no temporal/dynamic state."""
    def __init__(self, dim):
        self.prev_dynamic_mask = np.zeros((dim, dim), dtype=bool)

    def process(self, *_args):
        return self.prev_dynamic_mask, [], 0
