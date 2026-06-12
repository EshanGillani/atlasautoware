"""
Grid path planner — inflated-grid Dijkstra + shortcut smoothing (pure, no ROS).
===============================================================================

The planning half of the Navigator (slow indoor "drive to a goal" mode): given
the shared GridMap and a goal, produce a collision-free open path with a gentle
speed profile.  No raceline, no optimization — shortest drivable route.

  1. **Inflated free space.**  A cell is drivable iff the cached distance
     field says the nearest wall is farther than `inflation` (half car width
     + margin), optionally AND-ed with a `drivable` mask of *observed-free*
     cells (`load_drivable`) so SLAM maps whose unknown region is light gray
     (e.g. levine) don't invite plans through unmapped offices.
  2. **8-connected shortest path** with proper diagonal costs (res·√2) and a
     no-corner-cutting rule (a diagonal move needs both orthogonal neighbours
     free).  Solved with scipy's C Dijkstra over a sparse graph that is built
     once per (map, inflation) and reused for every plan — per-plan cost is
     one C shortest-path call, well under the 1 s budget.
  3. **Post-processing:** greedy line-of-sight shortcutting (sampled clearance
     checks on the distance field, margin inflation + res so every point of a
     kept segment provably clears `inflation`), resampling to ~0.2 m spacing,
     a light moving-average corner rounding (reverted pointwise wherever it
     would violate clearance), and a friction-limited speed profile via
     velocity_profiler.velocity_profile (closed=False) with hallway-gentle
     limits, clamped to `v_goal` (~0.5 m/s) at the terminal point.

  `plan()` returns None when start or goal can't be snapped to drivable space
  or no route exists.  `extra_obstacles` (world points, e.g. lidar returns
  that aren't map walls) are inflated and subtracted per-plan for replanning
  around new obstructions.
"""

import math
import os

import numpy as np
import yaml

from grid_map import _read_image
from velocity_profiler import velocity_profile

# half car width (~0.31/2 = 0.16) + safety margin (0.14): the follower's
# residual corner-cutting at low speed is < 0.1 m, so the driven track keeps
# comfortably > 0.15 m to the walls (benchmarked on comp_track + levine)
DEFAULT_INFLATION = 0.30


def load_drivable(yaml_path, max_occ=0.1):
    """Observed-free mask from a ROS map YAML: occupancy prob <= max_occ.

    Distinguishes scanned-free (white) from unknown (gray ~205/216) — the
    standard occupied/free thresholds treat both as free, which would let a
    planner leave the mapped hallway.  Binary maps (pure black/white) yield
    exactly the complement of the occupied mask.
    """
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    img_path = meta['image']
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(yaml_path), img_path)
    img = _read_image(img_path).astype(float) / 255.0
    occ_p = (1.0 - img) if not meta.get('negate', 0) else img
    return occ_p <= float(max_occ)


def _clearance(gm, xs, ys, obstacles=None):
    """Min of distance-to-wall and distance-to-extra-obstacles, per point."""
    d = np.atleast_1d(np.asarray(gm.distance_to_wall(xs, ys), float))
    if obstacles is not None:
        do, _ = obstacles.query(
            np.column_stack([np.atleast_1d(xs), np.atleast_1d(ys)]))
        d = np.minimum(d, do)
    return d


def segment_clear(gm, x0, y0, x1, y1, min_clear, drivable=None,
                  obstacles=None):
    """True iff every sample (spacing res/2) along the segment clears walls
    AND extra obstacles by > min_clear (and lies on `drivable` if given)."""
    n = max(2, int(math.hypot(x1 - x0, y1 - y0) / (gm.res * 0.5)) + 1)
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    if not np.all(_clearance(gm, xs, ys, obstacles) > min_clear):
        return False
    if drivable is not None:
        r, c = gm.world_to_grid(xs, ys)
        if not (np.all(gm.in_bounds(r, c)) and np.all(drivable[r, c])):
            return False
    return True


def shortcut(gm, xs, ys, min_clear, drivable=None, obstacles=None):
    """Greedy line-of-sight pruning: from each kept point, walk forward to
    the last point still reachable with > min_clear everywhere (forward walk
    with early stop keeps this O(path) instead of O(path^2))."""
    keep = [0]
    i = 0
    n = len(xs)
    while i < n - 1:
        j = i + 1
        while j + 1 < n and segment_clear(gm, xs[i], ys[i],
                                          xs[j + 1], ys[j + 1],
                                          min_clear, drivable, obstacles):
            j += 1
        keep.append(j)
        i = j
    return xs[keep], ys[keep]


def elastic_band(gm, xs, ys, min_clear, iters=80, step=0.3, obstacles=None):
    """Clearance-constrained smoothing: iteratively pull interior points
    toward their neighbours' midpoint, accepting each move only where the
    distance field (and any extra obstacles) stays > min_clear.  Rounds
    shortcut corners using all the corridor width the inflation allows — the
    follower's minimum turning radius (~0.76 m) needs gentler corners than
    raw shortcut vertices."""
    xs, ys = xs.copy(), ys.copy()
    if len(xs) < 3:
        return xs, ys
    for _ in range(iters):
        cx = xs[1:-1] + step * (xs[:-2] + xs[2:] - 2.0 * xs[1:-1])
        cy = ys[1:-1] + step * (ys[:-2] + ys[2:] - 2.0 * ys[1:-1])
        ok = _clearance(gm, cx, cy, obstacles) > min_clear
        xs[1:-1] = np.where(ok, cx, xs[1:-1])
        ys[1:-1] = np.where(ok, cy, ys[1:-1])
    return xs, ys


def resample(xs, ys, spacing=0.2):
    """Even arclength resampling (~spacing m); always keeps both endpoints."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    if s[-1] < 1e-9:
        return np.array([xs[0], xs[-1]]), np.array([ys[0], ys[-1]])
    n = max(2, int(round(s[-1] / spacing)) + 1)
    si = np.linspace(0.0, s[-1], n)
    return np.interp(si, s, xs), np.interp(si, s, ys)


def path_curvature(xs, ys):
    """Discrete |unsigned-safe| curvature: heading change per arclength."""
    dx, dy = np.diff(xs), np.diff(ys)
    seg = np.hypot(dx, dy)
    hdg = np.arctan2(dy, dx)
    k = np.zeros(len(xs))
    if len(xs) > 2:
        dth = np.arctan2(np.sin(np.diff(hdg)), np.cos(np.diff(hdg)))
        k[1:-1] = 2.0 * dth / np.maximum(seg[:-1] + seg[1:], 1e-9)
    return k


class GridPlanner:
    """Reusable planner bound to one GridMap + inflation (+ drivable mask).

    Graph construction (the expensive part) happens once in __init__;
    plan() is then a single C Dijkstra + cheap numpy post-processing.
    """

    def __init__(self, grid_map, inflation=DEFAULT_INFLATION, drivable=None):
        self.gm = grid_map
        self.inflation = float(inflation)
        self.drivable = drivable
        free = grid_map.distance_field() > self.inflation
        if drivable is not None:
            free = free & np.asarray(drivable, bool)
        self.free = free
        self._build_graph()

    # ── 8-connected sparse graph (vectorized, no corner cutting) ─────────────
    def _build_graph(self):
        free = self.free
        H, W = free.shape
        res = self.gm.res
        self.node_id = np.full((H, W), -1, np.int32)
        self.n_nodes = int(free.sum())
        self.node_id[free] = np.arange(self.n_nodes, dtype=np.int32)

        def window(arr, dr, dc, sr=0, sc=0):
            """Slice arr over source cells valid for offset (dr,dc),
            itself shifted by (sr,sc)."""
            a0, a1 = max(0, -dr), H - max(0, dr)
            b0, b1 = max(0, -dc), W - max(0, dc)
            return arr[a0 + sr:a1 + sr, b0 + sc:b1 + sc]

        rows, cols, wts = [], [], []
        for dr, dc, w in ((0, 1, res), (1, 0, res),
                          (1, 1, res * math.sqrt(2)),
                          (1, -1, res * math.sqrt(2))):
            ok = window(free, dr, dc) & window(free, dr, dc, dr, dc)
            if dr and dc:        # diagonal: both orthogonal neighbours free
                ok = ok & window(free, dr, dc, dr, 0) \
                        & window(free, dr, dc, 0, dc)
            rows.append(window(self.node_id, dr, dc)[ok])
            cols.append(window(self.node_id, dr, dc, dr, dc)[ok])
            wts.append(np.full(ok.sum(), w))
        self._erow = np.concatenate(rows)
        self._ecol = np.concatenate(cols)
        self._ewt = np.concatenate(wts)

    # ── snapping: nearest drivable cell within max_r ─────────────────────────
    def _snap(self, x, y, free, max_r=0.6):
        gm = self.gm
        r, c = gm.world_to_grid(x, y)
        r, c = int(r), int(c)
        if 0 <= r < gm.h and 0 <= c < gm.w and free[r, c]:
            return r, c
        k = int(math.ceil(max_r / gm.res))
        r0, r1 = max(0, r - k), min(gm.h, r + k + 1)
        c0, c1 = max(0, c - k), min(gm.w, c + k + 1)
        if r1 <= r0 or c1 <= c0:
            return None
        cand = np.argwhere(free[r0:r1, c0:c1])
        if len(cand) == 0:
            return None
        d2 = (cand[:, 0] + r0 - r) ** 2 + (cand[:, 1] + c0 - c) ** 2
        best = cand[int(np.argmin(d2))]
        if math.sqrt(float(d2.min())) * gm.res > max_r:
            return None
        return int(best[0] + r0), int(best[1] + c0)

    # ── raw grid shortest path ───────────────────────────────────────────────
    def grid_path(self, start_xy, goal_xy, extra_obstacles=None):
        """World start/goal -> (xs, ys) cell-centre polyline, or None."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra

        free = self.free
        erow, ecol, ewt = self._erow, self._ecol, self._ewt
        if extra_obstacles is not None and len(extra_obstacles):
            free = free.copy()
            blocked = np.zeros(self.n_nodes, bool)
            k = int(math.ceil(self.inflation / self.gm.res))
            pts = np.atleast_2d(np.asarray(extra_obstacles, float))
            rr, cc = self.gm.world_to_grid(pts[:, 0], pts[:, 1])
            for r, c in zip(np.atleast_1d(rr), np.atleast_1d(cc)):
                r0, r1 = max(0, r - k), min(self.gm.h, r + k + 1)
                c0, c1 = max(0, c - k), min(self.gm.w, c + k + 1)
                if r1 <= r0 or c1 <= c0:
                    continue
                ids = self.node_id[r0:r1, c0:c1]
                blocked[ids[ids >= 0]] = True
                free[r0:r1, c0:c1] = False
            keep = ~(blocked[erow] | blocked[ecol])
            erow, ecol, ewt = erow[keep], ecol[keep], ewt[keep]

        s = self._snap(*start_xy, free)
        g = self._snap(*goal_xy, free)
        if s is None or g is None:
            return None
        s_id = int(self.node_id[s])
        g_id = int(self.node_id[g])
        if s_id == g_id:
            xs, ys = self.gm.grid_to_world(np.array([s[0], g[0]]),
                                           np.array([s[1], g[1]]))
            return xs, ys
        graph = csr_matrix((ewt, (erow, ecol)),
                           shape=(self.n_nodes, self.n_nodes))
        dist, pred = dijkstra(graph, directed=False, indices=s_id,
                              return_predecessors=True)
        if not np.isfinite(dist[g_id]):
            return None
        chain = [g_id]
        while chain[-1] != s_id:
            p = pred[chain[-1]]
            if p < 0:
                return None
            chain.append(int(p))
        chain.reverse()
        flat = np.flatnonzero(self.free.ravel())     # node -> flat cell index
        cells = flat[np.asarray(chain)]
        return self.gm.grid_to_world(cells // self.gm.w, cells % self.gm.w)

    # ── full pipeline: grid path -> smooth -> resample -> speed profile ─────
    def plan(self, start_xy, goal_xy, spacing=0.2,
             v_max=2.0, a_lat_max=3.0, a_accel_max=1.0, a_brake_max=1.5,
             v_goal=0.5, v_floor=0.3, extra_obstacles=None):
        """Returns dict(x, y, v, kappa, length, raw_length) or None."""
        raw = self.grid_path(start_xy, goal_xy, extra_obstacles)
        if raw is None:
            return None
        rx, ry = np.asarray(raw[0], float), np.asarray(raw[1], float)
        raw_length = float(np.hypot(np.diff(rx), np.diff(ry)).sum())
        obstacles = None
        if extra_obstacles is not None and len(extra_obstacles):
            from scipy.spatial import cKDTree
            obstacles = cKDTree(np.atleast_2d(
                np.asarray(extra_obstacles, float)))

        # (a) line-of-sight shortcut; margin inflation+res so *every* point of
        # kept segments clears `inflation` (EDT is 1-Lipschitz, samples res/2)
        margin = self.inflation + self.gm.res
        sx, sy = shortcut(self.gm, rx, ry, margin, self.drivable, obstacles)
        # (b) even spacing, then clearance-constrained corner rounding
        px, py = resample(sx, sy, spacing)
        px, py = elastic_band(self.gm, px, py,
                              self.inflation + 0.5 * self.gm.res,
                              obstacles=obstacles)
        length = float(np.hypot(np.diff(px), np.diff(py)).sum())

        # (c) friction-limited speeds (open path), gentle hallway limits
        kappa = path_curvature(px, py)
        ds = np.hypot(np.diff(px), np.diff(py))
        v = velocity_profile(kappa, ds, a_lat_max=a_lat_max,
                             a_accel_max=a_accel_max, a_brake_max=a_brake_max,
                             v_max=v_max, closed=False)
        v[-1] = min(v[-1], v_goal)               # arrive slow…
        for i in range(len(v) - 2, -1, -1):      # …and brake to it in time
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2
                                       + 2.0 * a_brake_max * ds[i]))
        v = np.maximum(v, v_floor)
        return dict(x=px, y=py, v=v, kappa=kappa,
                    length=length, raw_length=raw_length)
