"""Quadtree routing over current anonymous occupancy and terrain labels."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import heapq
import math
import time

import numpy as np


@dataclass(frozen=True)
class LeafGraph:
    offsets: np.ndarray
    neighbours: np.ndarray


def leaf_adjacency(ids, count):
    """Scan each cardinal boundary once, then sort and deduplicate directed keys."""
    edges = []
    for a, b in ((ids[:, :-1], ids[:, 1:]), (ids[:-1, :], ids[1:, :])):
        boundary = (a != b) & (a >= 0) & (b >= 0)
        left, right = a[boundary].astype(np.int64), b[boundary].astype(np.int64)
        edges.extend((left * count + right, right * count + left))
    keys = np.unique(np.concatenate(edges))
    offsets = np.r_[0, np.cumsum(np.bincount(keys // count, minlength=count))]
    return LeafGraph(offsets, (keys % count).astype(np.int32))


def inflated(mask, clearance, resolution):
    if not mask.any():
        return mask.copy()
    # Measure from the occupied cell edge, conservatively rounding the square
    # footprint to its circumscribed radius in the distance field.
    radius = clearance / resolution + 1 / math.sqrt(2)
    extent = int(radius)
    rows, cols = mask.shape
    horizontal = mask.copy()
    output = np.zeros_like(mask)
    half_width = 0
    # Each disk row is a horizontal interval. Accumulate those intervals from
    # narrowest to widest and copy them to the two corresponding row offsets.
    for dy in range(min(extent, rows - 1), -1, -1):
        target = min(cols - 1, int(math.sqrt(radius * radius - dy * dy)))
        while half_width < target:
            half_width += 1
            horizontal[:, half_width:] |= mask[:, :-half_width]
            horizontal[:, :-half_width] |= mask[:, half_width:]
        if dy:
            output[dy:] |= horizontal[:-dy]
            output[:-dy] |= horizontal[dy:]
        else:
            output |= horizontal
    return output


def occupancy_masks(ids, flags, cfg):
    risk = ((flags >> 1) & 3)[ids]
    anonymous_dynamic = ((flags & 1) != 0)[ids]
    blocked = inflated(risk >= 2, cfg.planner_terrain_clearance_m, cfg.finest_res_m)
    blocked |= inflated(anonymous_dynamic, cfg.planner_dynamic_clearance_m, cfg.finest_res_m)
    return blocked, risk == 1


def snap(point, allowed, resolution, half, radius):
    """Snap a clamped point to the nearest allowed cell centre within radius."""
    point = np.clip(point, -half, np.nextafter(half, -np.inf))
    col, row = np.clip(np.floor((point + half) / resolution).astype(int), 0, allowed.shape[0] - 1)
    window = min(max(allowed.shape), math.ceil(radius / resolution) + 1)
    r0, c0 = max(0, row - window), max(0, col - window)
    rows, cols = np.nonzero(allowed[r0:row + window + 1, c0:col + window + 1])
    if not len(rows):
        return None
    rows, cols = rows + r0, cols + c0
    xs, ys = (cols + .5) * resolution - half, (rows + .5) * resolution - half
    distances = (xs - point[0]) ** 2 + (ys - point[1]) ** 2
    nearest = int(distances.argmin())
    if distances[nearest] > radius ** 2:
        return None
    return (int(rows[nearest]), int(cols[nearest])), (float(xs[nearest]), float(ys[nearest]))


def astar(start, goal, xy, neighbours, weight):
    """Shared A* kernel for both graph resolutions. Count actual expansions."""
    def heuristic(node):
        return math.hypot(xy[node][0] - xy[goal][0], xy[node][1] - xy[goal][1])

    heap = [(heuristic(start), 0., start)]
    best = {start: 0.}
    parents = {}
    expanded = 0
    while heap:
        _, cost, node = heapq.heappop(heap)
        if cost != best[node]:
            continue
        expanded += 1
        if node == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parents[path[-1]])
            return path[::-1], expanded
        x, y = xy[node]
        for nxt in neighbours(node):
            nxt = int(nxt)
            if not math.isfinite(weight[nxt]):
                continue
            nx, ny = xy[nxt]
            move = math.hypot(x - nx, y - ny) * (weight[node] + weight[nxt]) * .5
            candidate = cost + move
            if candidate < best.get(nxt, math.inf):
                best[nxt] = candidate
                parents[nxt] = node
                heapq.heappush(heap, (candidate + heuristic(nxt), candidate, nxt))
    return [], expanded


def segment_cells(a, b, shape, resolution, half):
    """Yield all raster cells touched by a segment, including corner contacts."""
    start = (np.asarray(a) + half) / resolution
    end = (np.asarray(b) + half) / resolution
    delta = end - start
    events = [np.array([0., 1.])]
    for p, q, d in zip(start, end, delta):
        if d:
            planes = np.arange(math.ceil(min(p, q)), math.floor(max(p, q)) + 1)
            events.append(np.clip((planes - p) / d, 0, 1))
    t = np.sort(np.concatenate(events))
    t = np.r_[t, (t[1:] + t[:-1]) * .5]
    points = start + t[:, None] * delta
    rounded = np.rint(points)
    on_plane = np.abs(points - rounded) < 1e-9
    points = np.where(on_plane, rounded, points)
    upper = np.floor(points).astype(int)
    lower = upper - on_plane
    cells = np.concatenate([np.column_stack((x, y)) for x in (lower[:, 0], upper[:, 0])
                            for y in (lower[:, 1], upper[:, 1])])
    cols, rows = cells.T
    inside = (cols >= 0) & (cols < shape[1]) & (rows >= 0) & (rows < shape[0])
    return rows[inside], cols[inside]


def route_clear(path, blocked, resolution, half):
    """Check endpoints and both sides of every grid crossing for all segments."""
    if not len(path):
        return False
    points = (np.asarray(path, dtype=float) + half) / resolution
    start = points[:-1]
    delta = np.diff(points, axis=0)
    crossings = [points]
    for axis in (0, 1):
        moving = np.abs(delta[:, axis]) > 1e-12
        a = start[moving]
        d = delta[moving]
        if not len(a):
            continue
        p, q = a[:, axis], a[:, axis] + d[:, axis]
        low = np.ceil(np.minimum(p, q))
        high = np.floor(np.maximum(p, q))
        lines = low[:, None] + np.arange(int(max(0, np.max(high - low + 1))))
        segment, plane = np.nonzero(lines <= high[:, None])
        fractions = (lines[segment, plane] - p[segment]) / d[segment, axis]
        crossings.append(a[segment] + fractions[:, None] * d[segment])
    events = np.concatenate(crossings)
    nearest = np.rint(events)
    on_plane = np.abs(events - nearest) < 1e-9
    upper = np.floor(np.where(on_plane, nearest, events)).astype(int)
    lower = upper - on_plane
    for x in (lower[:, 0], upper[:, 0]):
        for y in (lower[:, 1], upper[:, 1]):
            inside = (x >= 0) & (x < blocked.shape[1]) & (y >= 0) & (y < blocked.shape[0])
            if blocked[y[inside], x[inside]].any():
                return False
    return True


def shorten(path, blocked, resolution, half):
    """String-pull, then validate each replacement corner independently."""
    visible = lambda a, b: route_clear([a, b], blocked, resolution, half)
    result = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        candidate = len(path) - 1
        while candidate > anchor + 1 and not visible(path[anchor], path[candidate]):
            candidate -= 1
        result.append(path[candidate])
        anchor = candidate
    if len(result) < 3:
        return result
    smooth = [result[0]]
    for a, b, c in zip(result, result[1:], result[2:]):
        first = tuple(.2 * p + .8 * q for p, q in zip(a, b))
        last = tuple(.8 * p + .2 * q for p, q in zip(b, c))
        if visible(smooth[-1], first) and visible(first, last) and visible(last, c):
            smooth.extend((first, last))
        else:
            smooth.append(b)
    smooth.append(result[-1])
    return smooth if route_clear(smooth, blocked, resolution, half) else result


def fixed_sample(blocked, caution, start, goal, resolution):
    """Measure eight-connected fixed-grid A*, without diagonal corner cutting."""
    start_time = time.perf_counter()
    n = blocked.shape[0]
    free = ~blocked
    weight = np.where(blocked, np.inf, np.where(caution, 4.5, 1.)).ravel().tolist()
    xy = [(col * resolution, row * resolution) for row in range(n) for col in range(n)]

    def adjacent(node):
        row, col = divmod(node, n)
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0),
                       (1, 1), (-1, -1), (1, -1), (-1, 1)):
            r, c = row + dy, col + dx
            if 0 <= r < n and 0 <= c < n and free[r, c]:
                if not (dy and dx) or (free[row, c] and free[r, col]):
                    yield r * n + c

    _, expanded = astar(start[0] * n + start[1], goal[0] * n + goal[1], xy, adjacent, weight)
    return expanded, (time.perf_counter() - start_time) * 1000


class FixedGridComparison:
    """One in-flight job. Never wait for a sample on the frame thread."""
    def __init__(self):
        self.pool = ProcessPoolExecutor(max_workers=1)
        self.pending = None
        self.sample_key = None
        self.sample_time = -math.inf
        self.result = None
        self.last_key = None
        self.last_submit = -math.inf
        self.adaptive_nodes = None
        self.sample_adaptive_nodes = None

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)

    def update(self, now, sample_key, adaptive_nodes, snapshot):
        if self.pending is not None and self.pending.done():
            try:
                self.result = self.pending.result()
            except Exception:
                self.result = None
            self.sample_key, self.sample_time = self.last_key, self.last_submit
            self.sample_adaptive_nodes = self.adaptive_nodes
            self.pending = None
        if (snapshot is not None and self.pending is None
                and (sample_key != self.last_key or now - self.last_submit >= 5)):
            self.last_key, self.last_submit = sample_key, now
            try:
                self.pending = self.pool.submit(fixed_sample, *snapshot)
                self.adaptive_nodes = adaptive_nodes
            except RuntimeError:
                self.pending = None
        if self.result is None or self.sample_key != sample_key:
            return {}
        nodes, elapsed = self.result
        return {"fixed_grid_nodes_expanded": nodes, "fixed_plan_ms": elapsed,
                "fixed_sample_age_s": now - self.sample_time}

class OccupancyPlanner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.comparison = None
        self.generation = 0
        self.reset()

    def reset(self):
        self.generation += 1
        self.request = None
        self.path = []
        self.goal = None
        self.reason = None
        self.last_plan = -math.inf
        self.reroute_until = -math.inf
        self.last_nodes = None
        self.snapshot = None

    def update(self, ids, columns, goal_request, now=None, replan_interval_s=.1, sample_fixed=True):
        clock = time.perf_counter
        start = clock()
        now = start if now is None else now
        cfg = self.cfg
        resolution, half = cfg.finest_res_m, cfg.half_size
        requested = tuple(cfg.planner_default_goal if goal_request is None else goal_request)
        changed = requested != self.request
        flags = columns["flags"]
        blocked, caution = occupancy_masks(ids, flags, cfg)
        previous_valid = bool(self.path and self.reason is None and route_clear(
            self.path, blocked, resolution, half,
        ))
        invalidated = bool(self.path and self.reason is None and not previous_valid)
        search_ms = nodes = None
        if changed or invalidated or now - self.last_plan >= replan_interval_s:
            self.request, self.last_plan = requested, now
            # A leaf is usable only if all of its constituent cells are clear.
            # This avoids disconnecting a partially inflated leaf internally.
            count = len(flags)
            closed = np.bincount(ids[blocked], minlength=count) != 0
            allowed = ~closed[ids]
            origin = (snap((0., 0.), allowed, resolution, half, 3.) or
                      snap((0., 0.), allowed, resolution, half, math.sqrt(2) * half))
            dest = snap(requested, allowed, resolution, half, 3.)
            self.goal = dest[1] if dest else np.clip(requested, -half, half).tolist()
            self.snapshot = None
            if not origin:
                self.reason = "No drivable start"
            elif not dest:
                self.reason = "No drivable goal within 3 m"
            else:
                corners = np.column_stack((columns["x_idx"], columns["y_idx"])).astype(float)
                size = (1 << columns["level"].astype(np.int32))[:, None]
                upper = corners + size
                world_centres = ((corners + size / 2) * resolution - half).tolist()
                weight = np.where(closed, np.inf, np.where(((flags >> 1) & 3) == 1, 4.5, 1.)).tolist()
                csr = leaf_adjacency(ids, count)
                begin = clock()
                route, nodes = astar(int(ids[origin[0]]), int(ids[dest[0]]), world_centres,
                                     lambda node: csr.neighbours[csr.offsets[node]:csr.offsets[node + 1]],
                                     weight)
                search_ms = (clock() - begin) * 1000
                self.snapshot = (blocked, caution, origin[0], dest[0], resolution)
                self.last_nodes = nodes
                self.reason = None
                if route:
                    candidate = [origin[1]]
                    for a, b in zip(route, route[1:]):
                        overlap_low = np.maximum(corners[a], corners[b])
                        overlap_high = np.minimum(upper[a], upper[b])
                        portal = (overlap_low + overlap_high) / 2 * resolution - half
                        candidate.append(tuple(portal))
                    candidate.append(dest[1])
                    if route_clear(candidate, blocked, resolution, half):
                        if changed or invalidated or not previous_valid:
                            replace_path = True
                        else:
                            old_price = self.route_price(self.path, caution, resolution, half)
                            # A straight line is a lower bound on the candidate's
                            # cost, so skip scoring if it cannot beat the ribbon.
                            replace_path = (math.dist(candidate[0], candidate[-1]) < .9 * old_price
                                            and self.route_price(candidate, caution, resolution, half)
                                            < .9 * old_price)
                        if replace_path:
                            if invalidated or (self.path and not changed and not previous_valid):
                                self.reroute_until = now + 1.
                            self.path = shorten(candidate, blocked, resolution, half)
                        self.goal = self.path[-1]
                    elif not (previous_valid and not changed):
                        self.reason = "No clearance-safe path through leaf boundaries"
                elif not (previous_valid and not changed):
                    self.reason = "No connected clearance-safe route"
                if not self.reason and self.path:
                    self.goal = self.path[-1]
        elapsed = (clock() - start) * 1000
        status = "BLOCKED" if self.reason else ("REROUTING" if now < self.reroute_until else "OPTIMAL")
        result = {
            "planned_path": [[float(x), float(y)] for x, y in self.path],
            "goal_xy": list(self.goal) if self.goal is not None else None,
            "path_status": status, "path_reason": self.reason,
            "plan_time_ms": elapsed, "nodes_expanded": nodes,
            "adaptive_plan_ms": search_ms, "adaptive_nodes_expanded": nodes,
        }
        if sample_fixed:
            result.update(self.sample_fixed_grid(now))
        return result

    def sample_fixed_grid(self, now):
        if self.comparison is None:
            return {}
        result = self.comparison.update(now, (self.generation, self.request), self.last_nodes, self.snapshot)
        fixed_nodes = result.get("fixed_grid_nodes_expanded")
        compared = self.comparison.sample_adaptive_nodes
        if fixed_nodes and compared is not None:
            result["nodes_saved_pct"] = 100. * (1. - compared / fixed_nodes)
        return result

    @staticmethod
    def route_price(path, caution, resolution, half):
        total = 0.
        for a, b in zip(path, path[1:]):
            rows, cols = segment_cells(a, b, caution.shape, resolution, half)
            share_caution = float(caution[rows, cols].mean()) if len(rows) else 0.
            total += math.dist(a, b) * (1. + 3.5 * share_caution)
        return total
