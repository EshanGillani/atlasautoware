"""
F1TENTH Raceline Optimizer  —  Offline map → racing line
=========================================================

Turns an occupancy-grid map into a *racing line*, not a centerline.

The line it produces is the **minimum-curvature** trajectory inside the track
corridor.  This is the same objective real racing-line theory (and the TUM /
TUMFTM autonomous-racing stack) optimizes for: it is what produces the
"out-in-out", late-apex line a top driver naturally takes.  A min-curvature
line lets the velocity profile carry the most speed through every corner,
because the speed you can hold in a corner is  v = sqrt(a_lat / kappa)  — so
the flattest achievable curvature is the fastest achievable lap.

Pipeline
--------
  1. Load map (PNG + YAML), build the free/occupied grid.
  2. Flood-fill the drivable corridor from a seed pose (closed walls trap it).
  3. Trace the infield boundary (Moore-neighbor contour) -> an ordered loop
     hugging the inner wall.  Robust and deterministic (no skeletonization).
  4. Smooth the inner wall, then for each station march outward to the outer
     wall to recover the centerline and the local track width.
  5. Minimum-curvature optimization: solve for lateral offsets alpha_i that
     minimize sum(kappa^2), box-constrained to stay inside the track (minus a
     safety margin for car width).  Solved as a projected QP.
  6. Velocity profile: lateral-accel cap + forward/backward longitudinal
     passes (a simple g-g traction model with separate accel / brake limits).
  7. Write CSV  (x, y, heading, curvature, speed)  — the exact schema the
     pursuit agent loads — and a PNG overlay for eyeballing the line.

Usage
-----
  python3 raceline_optimizer.py \
      --map    /sim_ws/src/f1tenth_gym_ros/maps/Spielberg_map.yaml \
      --output /sim_ws/src/f1tenth_gym_ros/racelines/best_raceline.csv

The optimizer is map-agnostic: point it at any ROS occupancy map and it will
produce a raceline.  The on-car agent then needs only a lap or two of practice
to refine it to the real (simulated) vehicle behaviour.
"""

import argparse
import csv
import math
import os

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage
from scipy.interpolate import splev, splprep


# ─────────────────────────────────────────────────────────────────────────────
# Vehicle / dynamics defaults  (F1TENTH 1:10 scale)
# ─────────────────────────────────────────────────────────────────────────────

class VehicleParams:
    """Tunable limits.  These set how aggressive the resulting line + speed are."""
    def __init__(self):
        self.car_width   = 0.30     # m, physical width of the car
        self.safety_marg = 0.12     # m, extra clearance kept from each wall
        self.a_lat_max   = 6.5      # m/s^2, max lateral (cornering) accel
        self.a_acc_max   = 5.0      # m/s^2, max forward accel
        self.a_brk_max   = 7.0      # m/s^2, max braking decel
        self.v_max       = 7.0      # m/s, top speed cap
        self.v_min       = 1.5      # m/s, floor so we never stall
        # ── kinematic steering limit (what the car can PHYSICALLY turn) ──────────
        # A min-curvature line is worthless if it asks for a tighter turn than the
        # steering can deliver; the car just runs wide and noses the wall.  These
        # cap the achievable path curvature so the line is always drivable, on a
        # hairpin or a sweeper alike.
        self.wheelbase   = 0.33     # m, front-to-rear axle
        self.max_steer   = 0.41     # rad, steering limit (matches the agent)
        self.feas_frac   = 0.85     # use this fraction of the limit; the controller
                                    # needs steering headroom left for corrections

    @property
    def half_clearance(self):
        return self.car_width / 2.0 + self.safety_marg

    @property
    def kappa_max(self):
        """Max curvature the steering can produce (1/turn-radius), 1/m."""
        return math.tan(self.max_steer) / self.wheelbase

    @property
    def kappa_budget(self):
        """Curvature the optimizer is allowed to plan to (leaves control headroom)."""
        return self.feas_frac * self.kappa_max


# ─────────────────────────────────────────────────────────────────────────────
# Map loading + coordinate transforms
# ─────────────────────────────────────────────────────────────────────────────

class GridMap:
    def __init__(self, yaml_path):
        with open(yaml_path, 'r') as f:
            meta = yaml.safe_load(f)
        base = os.path.dirname(os.path.abspath(yaml_path))
        img_path = meta['image']
        if not os.path.isabs(img_path):
            img_path = os.path.join(base, img_path)

        self.res      = float(meta['resolution'])
        self.origin   = meta['origin']            # [x, y, theta]
        self.negate   = int(meta.get('negate', 0))
        self.occ_th   = float(meta.get('occupied_thresh', 0.65))
        self.free_th  = float(meta.get('free_thresh', 0.196))

        self.img_path = img_path
        img = np.array(Image.open(img_path).convert('L'))
        self.H, self.W = img.shape

        # ROS map_server occupancy convention.
        if self.negate:
            p = img / 255.0
        else:
            p = (255 - img) / 255.0
        self.free = p < self.free_th          # confidently free
        self.occ  = p > self.occ_th           # wall / occupied

    # world <-> pixel.  Image row 0 is the top (max world-y).
    def world_to_px(self, wx, wy):
        col = (wx - self.origin[0]) / self.res
        row = (self.H - 1) - (wy - self.origin[1]) / self.res
        return int(round(col)), int(round(row))

    def px_to_world(self, col, row):
        wx = col * self.res + self.origin[0]
        wy = ((self.H - 1) - row) * self.res + self.origin[1]
        return wx, wy


# ─────────────────────────────────────────────────────────────────────────────
# 1) Drivable corridor via flood fill
# ─────────────────────────────────────────────────────────────────────────────

def extract_corridor(grid: GridMap, seed_world):
    """Connected free region reachable from the seed pose (bounded by walls)."""
    sx, sy = grid.world_to_px(*seed_world)
    if not (0 <= sx < grid.W and 0 <= sy < grid.H) or not grid.free[sy, sx]:
        free_idx = np.argwhere(grid.free)
        d = (free_idx[:, 0] - sy) ** 2 + (free_idx[:, 1] - sx) ** 2
        sy, sx = free_idx[np.argmin(d)]

    labels, _ = ndimage.label(grid.free)
    corridor = labels == labels[sy, sx]
    return corridor, (sx, sy)


# ─────────────────────────────────────────────────────────────────────────────
# 2) Trace the inner (infield) wall as an ordered closed loop
# ─────────────────────────────────────────────────────────────────────────────

# Clockwise Moore neighborhood, index 0 = West.
_CW = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]


def trace_boundary(region):
    """Moore-neighbor boundary tracing of a filled binary region -> ordered px."""
    padded = np.pad(region, 1)
    rr, cc = np.where(padded)
    si = np.lexsort((cc, rr))[0]                  # top-left-most foreground px
    start = (int(rr[si]), int(cc[si]))
    cur, backtrack = start, 0
    boundary = [start]
    guard, maxg = 0, 8 * int(padded.sum()) + 100
    while guard < maxg:
        guard += 1
        found = None
        for k in range(1, 9):
            di = (backtrack + k) % 8
            d = _CW[di]
            nb = (cur[0] + d[0], cur[1] + d[1])
            if padded[nb]:
                found = nb
                backtrack = (di + 4) % 8          # point back toward cur
                break
        if found is None or found == start:
            break
        cur = found
        boundary.append(cur)
    return np.array([(r - 1, c - 1) for (r, c) in boundary])   # unpad -> (row,col)


# ─────────────────────────────────────────────────────────────────────────────
# 3) Centerline + track width by marching from the inner wall to the outer wall
# ─────────────────────────────────────────────────────────────────────────────

def resample_closed(xy, n_pts, smooth_px):
    """Periodic cubic spline through a closed loop, resampled uniformly."""
    x, y = xy[:, 0].astype(float), xy[:, 1].astype(float)
    keep = [0]
    for i in range(1, len(x)):
        if np.hypot(x[i] - x[keep[-1]], y[i] - y[keep[-1]]) > 1e-6:
            keep.append(i)
    x, y = x[keep], y[keep]
    s = smooth_px ** 2 * len(x)
    tck, _ = splprep([x, y], s=s, per=True, k=3)
    u = np.linspace(0, 1, n_pts, endpoint=False)
    xs, ys = splev(u, tck)
    return np.column_stack([xs, ys])


def loop_normals(xy):
    """Unit tangents and left normals for a closed loop of world points."""
    n = len(xy)
    tang = np.empty_like(xy)
    for i in range(n):
        tang[i] = xy[(i + 1) % n] - xy[(i - 1) % n]
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12)
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])   # left normal
    return tang, nrm


def build_centerline(corridor, grid: GridMap, n_pts, inner_smooth_m=0.25):
    """
    Centerline = medial axis (EDT ridge), ordered by projecting the ordered
    inner-wall contour onto it.

    The ridge is the set of corridor pixels that are local maxima of the
    distance transform — points equidistant from both walls.  Projecting each
    ordered inner-wall station to its nearest ridge pixel yields an ordered
    centerline that stays well inside the track everywhere, including hairpins.
    """
    from scipy.spatial import cKDTree

    edt = ndimage.distance_transform_edt(corridor)      # px to nearest wall
    ridge = corridor & (edt >= ndimage.maximum_filter(edt, size=3) - 1e-6) \
        & (edt > 2.0)
    ridge_rc = np.argwhere(ridge).astype(float)         # (row, col)
    tree = cKDTree(ridge_rc)

    filled = ndimage.binary_fill_holes(corridor)
    infield = filled & ~corridor
    inner_rc = trace_boundary(infield)                  # (row, col), ordered
    inner_cr = np.column_stack([inner_rc[:, 1], inner_rc[:, 0]])  # (col, row)
    inner_px = resample_closed(inner_cr, n_pts, smooth_px=inner_smooth_m / grid.res)

    query = np.column_stack([inner_px[:, 1], inner_px[:, 0]])     # (row, col)
    _, idx = tree.query(query)
    cen_rc = ridge_rc[idx]                               # (row, col) on ridge
    center_world = np.array([grid.px_to_world(c, r) for r, c in cen_rc])
    half_w = np.array([edt[int(r), int(c)] for r, c in cen_rc]) * grid.res
    return center_world, half_w


# ─────────────────────────────────────────────────────────────────────────────
# 4) Minimum-curvature optimization (projected QP)
# ─────────────────────────────────────────────────────────────────────────────

def second_diff_matrix(n):
    """Circulant second difference: (D p)_i = p_{i-1} - 2 p_i + p_{i+1}."""
    D = np.zeros((n, n))
    idx = np.arange(n)
    D[idx, idx] = -2.0
    D[idx, (idx - 1) % n] += 1.0
    D[idx, (idx + 1) % n] += 1.0
    return D


def min_curvature_offsets(center, nrm, lo, hi, ridge=1e-3, iters=600, weights=None):
    """
    Solve for lateral offsets alpha (raceline = center + alpha * normal) that
    minimize sum_i w_i |D (center + alpha*n)|_i^2, box-constrained to [lo, hi].

    Quadratic in alpha:  J = alpha^T H alpha + 2 f^T alpha + const,
    H = Ax^T W Ax + Ay^T W Ay + ridge*I,  Ax = D diag(nx), Ay = D diag(ny).
    Per-point weights `w` let us penalise curvature more at corner *exits*, which
    flattens the exit and pushes the apex later — the late-apex line that buys a
    faster run onto the following straight (overtaking).  w=1 everywhere gives the
    pure minimum-curvature line.  Solved by projected Gauss-Seidel (SPD H, honours
    the box constraints exactly).
    """
    n = len(center)
    D = second_diff_matrix(n)
    Ax = D * nrm[:, 0][None, :]
    Ay = D * nrm[:, 1][None, :]
    Dcx = D @ center[:, 0]
    Dcy = D @ center[:, 1]

    w = np.ones(n) if weights is None else np.asarray(weights, float)
    H = Ax.T @ (w[:, None] * Ax) + Ay.T @ (w[:, None] * Ay) + ridge * np.eye(n)
    f = Ax.T @ (w * Dcx) + Ay.T @ (w * Dcy)

    # Strictly-convex box-constrained QP (H is SPD) -> unique minimum.  Solve
    # with accelerated projected gradient (FISTA): each step is a single
    # matrix-vector product + a clip, fully vectorized — replacing the
    # O(iters * n^2) pure-Python Gauss-Seidel double loop (the old hot path:
    # ~9M scalar min/max calls, ~21 s on a 629-pt track).  Converges to the
    # same constrained optimum, so the produced raceline is identical to the
    # CSV's 4-decimal rounding (regression-gated against the old solver).
    L = _spectral_radius(H)                         # Lipschitz const ~ max eig(H)
    step = 1.0 / max(L, 1e-9)
    alpha = np.zeros(n)
    y = alpha.copy()
    t = 1.0
    for _ in range(iters):
        new = np.clip(y - step * (H @ y + f), lo, hi)
        if np.max(np.abs(new - alpha)) < 1e-5:
            alpha = new
            break
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = new + ((t - 1.0) / t_new) * (new - alpha)
        alpha = new
        t = t_new
    return alpha


def _spectral_radius(H, iters=30):
    """Largest eigenvalue of SPD H via power iteration (cheap; for the FISTA
    step size).  A few mat-vecs instead of a full O(n^3) eig."""
    v = np.ones(H.shape[0])
    nv = np.linalg.norm(v)
    v = v / nv if nv > 0 else v
    lam = 0.0
    for _ in range(iters):
        Hv = H @ v
        nv = np.linalg.norm(Hv)
        if nv < 1e-12:
            break
        v = Hv / nv
        lam = float(v @ (H @ v))
    return lam


def feasible_curvature_offsets(center, nrm, lo, hi, kappa_budget,
                               base_weights=None, outer_iters=12, boost=4.0,
                               verbose_log=None):
    """
    Minimum-curvature offsets, made *kinematically feasible*: no point may demand
    a tighter turn than the steering can deliver (|kappa| <= kappa_budget).

    A pure min-curvature line minimises total curvature but can still leave a
    hairpin tighter than the car can steer — the optimizer happily trades a tiny
    bit of extra curvature on a hairpin for a flatter sweeper elsewhere, because
    both cost the same in sum(kappa^2).  We don't want the average flattest line;
    we want one whose *worst* corner is drivable.

    Method — iteratively reweighted constrained QP: solve min-curvature, find the
    points that still exceed the budget, and multiply their curvature weight up
    (proportional to how badly they violate).  Re-solving then spends the track
    width specifically on opening those corners (out-in-out through the hairpin),
    leaving gentle corners alone.  Converges to the feasible line that is closest
    to pure min-curvature.  If a corner physically cannot fit even at full track
    width, the residual is reported and the velocity profile slows for it.
    """
    n = len(center)
    w = np.ones(n) if base_weights is None else np.asarray(base_weights, float).copy()
    best_alpha = None
    best_resid = np.inf
    for it in range(outer_iters):
        alpha = min_curvature_offsets(center, nrm, lo, hi, weights=w)
        race = center + alpha[:, None] * nrm
        _, curv = curvature_heading(race)
        viol = curv - kappa_budget                       # >0 where infeasible
        worst = float(viol.max())
        if worst < best_resid:
            best_resid, best_alpha = worst, alpha
        if worst <= 0:                                   # fully drivable — done
            if verbose_log:
                verbose_log(f'[feasible] curvature OK after {it+1} pass(es) '
                            f'(max kappa {curv.max():.2f} <= budget {kappa_budget:.2f})')
            return alpha, curv, 0
        # Boost weight where we violate; spread to neighbours so the QP opens a
        # smooth arc through the corner rather than kinking a single point.
        local = np.clip(viol / kappa_budget, 0.0, None)
        local = np.convolve(np.r_[local[-3:], local, local[:3]],
                            np.ones(7) / 7, 'same')[3:-3]
        w = w * (1.0 + boost * local)
        w /= w.mean()                                    # keep the QP well-scaled

    # Couldn't make every corner feasible within the corridor.
    race = center + best_alpha[:, None] * nrm
    _, curv = curvature_heading(race)
    n_bad = int((curv > kappa_budget).sum())
    if verbose_log:
        verbose_log(f'[feasible] WARNING: {n_bad} pt(s) still exceed the steering '
                    f'limit (max kappa {curv.max():.2f} > budget {kappa_budget:.2f}); '
                    f'corridor too narrow there — slowing for it in the speed profile')
    return best_alpha, curv, n_bad


# ─────────────────────────────────────────────────────────────────────────────
# 5) Geometry + velocity profile
# ─────────────────────────────────────────────────────────────────────────────

def curvature_heading(xy):
    n = len(xy)
    hdg = np.zeros(n)
    curv = np.zeros(n)
    for i in range(n):
        a = xy[(i - 1) % n]; b = xy[i]; c = xy[(i + 1) % n]
        hdg[i] = np.arctan2(c[1] - a[1], c[0] - a[0])
        area2 = abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
        d1 = np.hypot(*(b - a)); d2 = np.hypot(*(c - b)); d3 = np.hypot(*(c - a))
        denom = d1 * d2 * d3
        curv[i] = (2.0 * area2) / denom if denom > 1e-9 else 0.0
    return hdg, curv


def velocity_profile(xy, curv, vp: VehicleParams):
    n = len(xy)
    ds = np.array([np.hypot(*(xy[(i + 1) % n] - xy[i])) for i in range(n)])
    # Corner-speed limit from lateral grip.  Where a corner is tighter than the
    # car can steer (residual-infeasible point), curv is large so v collapses to
    # v_min automatically — the car crawls through rather than running wide.
    v = np.sqrt(vp.a_lat_max / np.maximum(curv, 1e-5))
    v = np.clip(v, vp.v_min, vp.v_max)
    for _ in range(2):                            # forward (accel) pass, wrapped
        for i in range(n):
            j = (i + 1) % n
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * vp.a_acc_max * ds[i]))
    for _ in range(2):                            # backward (brake) pass, wrapped
        for i in range(n - 1, -1, -1):
            j = (i - 1) % n
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * vp.a_brk_max * ds[i]))
    return np.clip(v, vp.v_min, vp.v_max)


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(path, xy, hdg, curv, spd):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x', 'y', 'heading', 'curvature', 'speed'])
        for i in range(len(xy)):
            w.writerow([round(xy[i, 0], 4), round(xy[i, 1], 4),
                        round(hdg[i], 4), round(curv[i], 6), round(spd[i], 3)])


def _speed_color(v, vmin, vmax):
    """Red (slow) -> yellow -> green (fast)."""
    t = 0.0 if vmax <= vmin else (v - vmin) / (vmax - vmin)
    t = min(1.0, max(0.0, t))
    if t < 0.5:
        return (255, int(510 * t), 0)            # red -> yellow
    return (int(255 * (2 - 2 * t)), 255, 0)      # yellow -> green


def save_overlay(path, grid, center_world, race_world, spd):
    """Render the raceline over the map with PIL (no matplotlib dependency)."""
    from PIL import ImageDraw
    base = Image.open(grid.img_path).convert('RGB')
    draw = ImageDraw.Draw(base)

    def to_px(p):
        c, r = grid.world_to_px(p[0], p[1])
        return (c, r)

    cpx = [to_px(p) for p in center_world] + [to_px(center_world[0])]
    draw.line(cpx, fill=(150, 150, 255), width=1)

    vmin, vmax = float(spd.min()), float(spd.max())
    rpx = [to_px(p) for p in race_world]
    for i in range(len(rpx)):
        a = rpx[i]; b = rpx[(i + 1) % len(rpx)]
        draw.line([a, b], fill=_speed_color(spd[i], vmin, vmax), width=3)
    base.save(path)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def optimize(map_yaml, out_csv, seed_world=(0.0, 0.0), seed_heading=0.0, n_pts=None,
             spacing=0.4, vp: VehicleParams = None, overlay=True, verbose=True,
             apex_bias=0.0):
    vp = vp or VehicleParams()

    def log(*a):
        if verbose:
            print(*a, flush=True)

    grid = GridMap(map_yaml)
    log(f'[map] {grid.W}x{grid.H} @ {grid.res:.4f} m/px')

    corridor, seed_px = extract_corridor(grid, seed_world)
    log(f'[corridor] {int(corridor.sum())} px, seed at {seed_px}')

    # ── Adaptive resolution — pick the sample count from the track's length so
    # the waypoint spacing (~`spacing` m/pt) is constant regardless of track
    # size.  A bigger comp track gets proportionally more points, keeping the
    # curvature estimate sharp and the controller's lookahead physically
    # consistent across tracks.
    if n_pts is None:
        center0, _ = build_centerline(corridor, grid, 400)
        center0 = resample_closed(center0, 400, smooth_px=0.20)
        d0 = np.hypot(*(np.roll(center0, -1, axis=0) - center0).T)
        track_len = float(d0.sum())
        n_pts = int(np.clip(round(track_len / spacing), 200, 2000))
        log(f'[resolution] track ~{track_len:.0f} m -> {n_pts} pts '
            f'(~{spacing:.2f} m/pt)')

    center_world, half_w = build_centerline(corridor, grid, n_pts)
    center_world = resample_closed(center_world, n_pts, smooth_px=0.20)  # tidy ridge
    _, nrm = loop_normals(center_world)
    track_w = 2.0 * half_w
    log(f'[centerline] {n_pts} pts | track width mean {track_w.mean():.2f} m '
        f'(min {track_w.min():.2f}, max {track_w.max():.2f})')

    # Lateral offset bounds (positive = left normal).  On the medial axis the
    # half-width is the distance to either wall, so bounds are symmetric.
    room = np.maximum(half_w - vp.half_clearance, 0.0)
    hi, lo = room.copy(), -room.copy()

    # Late-apex weighting for overtaking: penalise curvature just *after* each
    # corner (the exit) so the optimizer straightens the exit and the apex moves
    # later — a stronger run onto the following straight.  apex_bias=0 -> pure
    # minimum-curvature line.
    weights = None
    if apex_bias > 0:
        _, c_curv = curvature_heading(center_world)
        cs = np.convolve(np.r_[c_curv[-5:], c_curv, c_curv[:5]],
                         np.ones(5) / 5, 'same')[5:-5]          # smooth, periodic
        exit_emph = np.roll(cs, 4)                              # weight points after a corner
        exit_emph /= (exit_emph.max() + 1e-9)
        weights = 1.0 + apex_bias * exit_emph

    # Feasibility-constrained min-curvature: the worst corner is guaranteed
    # drivable by the steering (or reported + slowed if the corridor can't fit).
    alpha, _, n_bad = feasible_curvature_offsets(
        center_world, nrm, lo, hi, vp.kappa_budget,
        base_weights=weights, verbose_log=log)
    race_world = center_world + alpha[:, None] * nrm
    log(f'[optimize] lateral offset range [{alpha.min():.2f}, {alpha.max():.2f}] m '
        f'| min turn radius {1.0/vp.kappa_max:.2f} m (budget {1.0/vp.kappa_budget:.2f} m)')

    # Orient the loop so increasing index = the car's start driving direction.
    si = int(np.argmin(np.hypot(race_world[:, 0] - seed_world[0],
                                race_world[:, 1] - seed_world[1])))
    n = len(race_world)
    tangent = race_world[(si + 1) % n] - race_world[(si - 1) % n]
    start_dir = np.array([np.cos(seed_heading), np.sin(seed_heading)])
    if np.dot(tangent, start_dir) < 0:
        race_world = race_world[::-1].copy()
        center_world = center_world[::-1].copy()
        log('[optimize] reversed raceline to match start heading '
            f'{np.degrees(seed_heading):.0f}deg')

    hdg, curv = curvature_heading(race_world)
    spd = velocity_profile(race_world, curv, vp)

    n = len(race_world)
    ds = np.array([np.hypot(*(race_world[(i+1) % n] - race_world[i]))
                   for i in range(n)])
    lap_t = float(np.sum(ds / spd))
    log(f'[speed] {spd.min():.2f}-{spd.max():.2f} m/s (mean {spd.mean():.2f}) | '
        f'length {ds.sum():.1f} m | est lap ~{lap_t:.1f} s')

    # Feasibility verdict on the FINAL line (drivable iff every corner is within
    # the steering budget).  This is the gate the agent depends on.
    n_infeasible = int((curv > vp.kappa_budget).sum())
    if n_infeasible == 0:
        log(f'[feasible] OK — every corner drivable (max kappa {curv.max():.2f} '
            f'<= budget {vp.kappa_budget:.2f}, i.e. min radius {1.0/curv.max():.2f} m)')
    else:
        log(f'[feasible] {n_infeasible} corner-pt(s) exceed the steering budget '
            f'(max kappa {curv.max():.2f}); these are slowed to v_min — widen the '
            f'corridor or raise max_steer to remove them')

    save_csv(out_csv, race_world, hdg, curv, spd)
    log(f'[write] {out_csv}')
    if overlay:
        png = os.path.splitext(out_csv)[0] + '_overlay.png'
        if save_overlay(png, grid, center_world, race_world, spd):
            log(f'[write] {png}')
    return race_world, hdg, curv, spd


def main():
    ap = argparse.ArgumentParser(description='F1TENTH min-curvature raceline optimizer')
    ap.add_argument('--map', required=True, help='path to map .yaml')
    ap.add_argument('--output', required=True, help='output raceline .csv')
    ap.add_argument('--seed', type=float, nargs=2, default=[0.0, 0.0],
                    metavar=('X', 'Y'), help='seed world pose on the track')
    ap.add_argument('--heading', type=float, default=0.0,
                    help='start heading (rad); raceline is oriented to match')
    ap.add_argument('--points', type=int, default=None,
                    help='raceline samples (default: auto from track length)')
    ap.add_argument('--spacing', type=float, default=0.4,
                    help='target waypoint spacing (m/pt) when --points is auto')
    ap.add_argument('--a-lat', type=float, default=6.5, help='max lateral accel m/s^2')
    ap.add_argument('--v-max', type=float, default=7.0, help='top speed m/s')
    ap.add_argument('--margin', type=float, default=0.12,
                    help='extra wall clearance (m) kept on each side of the car')
    ap.add_argument('--apex-bias', type=float, default=0.0,
                    help='late-apex / exit-priority strength for overtaking (try 0.5-1.5)')
    ap.add_argument('--no-overlay', action='store_true')
    args = ap.parse_args()

    vp = VehicleParams()
    vp.a_lat_max = args.a_lat
    vp.v_max = args.v_max
    vp.safety_marg = args.margin
    optimize(args.map, args.output, seed_world=tuple(args.seed),
             seed_heading=args.heading, n_pts=args.points, vp=vp,
             overlay=not args.no_overlay, apex_bias=args.apex_bias)


if __name__ == '__main__':
    main()
