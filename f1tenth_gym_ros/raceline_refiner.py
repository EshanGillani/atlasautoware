"""
Minimum-curvature raceline refinement — corridor-bounded QP (pure, no ROS).
===========================================================================

The raceline CSVs were fit from driven laps, so they follow where the car
*went*, not the geometrically fastest line.  This module refines such a
closed line the way the TUMFTM global planner does: shift every point
laterally by d_i along its left normal n_i so the squared curvature of the
shifted path is minimal, while no point strays more than `corridor` meters
from where it started — the corridor is the safety knob.  With no map the
corridor is a conservative scalar; with an occupancy map, `map_corridors`
turns the real free space into per-point per-side bounds (corridor=(lo, hi))
so the refiner uses all the room where the track is wide and none where it
is tight, and `verify_wall_clearance` checks the result against the map.

Linearization (Heilmeier): new points p_i' = p_i + d_i * n_i, and curvature
is approximated by the periodic second arc-length difference of positions,

  kappa_i ~ | (D (p + N d))_i |,   D row i:  2/(ds_{i-1}+ds_i) *
            [ 1/ds_{i-1},  -(1/ds_{i-1} + 1/ds_i),  1/ds_i ]  (wrapped),

so  J(d) = ||D (p + N d)||^2 + ridge ||d||^2 + smooth ||L d||^2  is a convex
QP in d (L = periodic first difference of d per arc length, which keeps the
offset profile — and hence the line — smooth).  Solved with OSQP under the
box constraint |d_i| <= corridor (scipy lsq_linear fallback), then the
geometry is re-linearized and re-solved `iterations` times, recomputing the
normals each pass; the box shrinks by the displacement already spent, so the
total excursion from the ORIGINAL line never exceeds `corridor`.

Heading and curvature of the refined line are recomputed with the same
periodic finite differences the raceline CSVs use (central-difference
heading, unsigned Menger curvature — see raceline_optimizer.curvature_heading;
pursuit_agent.fit_raceline uses the same stencil but omits the factor 2).

Reference: Heilmeier et al., "Minimum curvature trajectory planning and
control for an autonomous race car", Vehicle System Dynamics 2020;
github.com/TUMFTM/global_racetrajectory_optimization.
"""

import numpy as np

try:
    import osqp
    import scipy.sparse as _sp
    _HAVE_OSQP = True
except ImportError:                                   # pragma: no cover
    _HAVE_OSQP = False


def segment_lengths(x, y):
    """ds[i] = |p_{i+1} - p_i| with periodic wrap."""
    return np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))


def heading_curvature(x, y):
    """Periodic heading + unsigned curvature, matching the CSV conventions.

    heading[i] = atan2 of the central difference p_{i+1} - p_{i-1};
    curvature[i] = Menger curvature of (p_{i-1}, p_i, p_{i+1}) = 4*Area/(abc).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    xa, ya = np.roll(x, 1), np.roll(y, 1)             # p_{i-1}
    xc, yc = np.roll(x, -1), np.roll(y, -1)           # p_{i+1}
    hdg = np.arctan2(yc - ya, xc - xa)
    area2 = np.abs((x - xa) * (yc - ya) - (xc - xa) * (y - ya))
    d1 = np.hypot(x - xa, y - ya)
    d2 = np.hypot(xc - x, yc - y)
    d3 = np.hypot(xc - xa, yc - ya)
    denom = d1 * d2 * d3
    curv = np.where(denom > 1e-9, 2.0 * area2 / np.maximum(denom, 1e-12), 0.0)
    return hdg, curv


def left_normals(x, y):
    """Unit left normals from central-difference tangents (closed loop)."""
    tx = np.roll(x, -1) - np.roll(x, 1)
    ty = np.roll(y, -1) - np.roll(y, 1)
    norm = np.maximum(np.hypot(tx, ty), 1e-12)
    return -ty / norm, tx / norm


def _second_diff_matrix(ds):
    """Dense periodic second-difference operator scaled by arc length.

    (D p)_i ~ d^2 p / ds^2 at i, so |(D p)_i| approximates curvature.
    """
    n = len(ds)
    dsm = np.roll(ds, 1)                              # ds_{i-1}
    w = 2.0 / (dsm + ds)
    D = np.zeros((n, n))
    idx = np.arange(n)
    D[idx, (idx - 1) % n] = w / dsm
    D[idx, (idx + 1) % n] = w / ds
    D[idx, idx] = -(w / dsm + w / ds)
    return D


def _first_diff_matrix(ds):
    """Dense periodic first difference per arc length: (L d)_i = (d_{i+1}-d_i)/ds_i."""
    n = len(ds)
    L = np.zeros((n, n))
    idx = np.arange(n)
    L[idx, idx] = -1.0 / ds
    L[idx, (idx + 1) % n] = 1.0 / ds
    return L


def _solve_box_qp(H, g, lo, hi):
    """min 0.5 d^T H d + g^T d  s.t.  lo <= d <= hi   (H symmetric PD)."""
    n = len(g)
    if _HAVE_OSQP:
        prob = osqp.OSQP()
        prob.setup(P=_sp.csc_matrix(np.triu(H)), q=g,
                   A=_sp.identity(n, format='csc'), l=lo, u=hi,
                   eps_abs=1e-8, eps_rel=1e-8, max_iter=20000,
                   polish=True, verbose=False)
        d = prob.solve().x
        if d is None or not np.all(np.isfinite(d)):   # pragma: no cover
            d = np.zeros(n)
    else:                                             # pragma: no cover
        from scipy.optimize import lsq_linear
        # H = M^T M up to scaling is not available here; fall back to the
        # generic bounded least-squares on the Cholesky factor of H.
        R = np.linalg.cholesky(H + 1e-10 * np.eye(n)).T
        rhs = -np.linalg.solve(R.T, g)
        d = lsq_linear(R, rhs, bounds=(lo, hi)).x
    return np.clip(d, lo, hi)


def refine_raceline(x, y, corridor=0.25, iterations=3, smooth=0.1, ridge=1e-3):
    """Minimum-curvature refinement of a closed raceline.

    Parameters
    ----------
    x, y       : closed-loop waypoints (no repeated endpoint).
    corridor   : either a scalar — max lateral excursion from the ORIGINAL
                 points (m), kept conservative when wall clearance is
                 unknown — or a pair of arrays ``(lo, hi)`` giving per-point,
                 per-side bounds: the signed offset of point i along its
                 ORIGINAL left normal must stay in [lo_i, hi_i] (lo_i <= 0
                 <= hi_i, so the original point is always feasible).  Use
                 `map_corridors` to derive (lo, hi) from an occupancy map.
    iterations : linearize-solve-shift passes (normals recomputed each pass).
    smooth     : weight on ||(d_{i+1}-d_i)/ds||^2 — smoothness of the offset.
    ridge      : Tikhonov weight on ||d||^2 — keeps the QP well-posed.

    Returns (x_new, y_new, heading, curvature); heading/curvature use the
    same periodic finite differences as the raceline CSVs.
    """
    x0 = np.asarray(x, float)
    y0 = np.asarray(y, float)
    per_side = isinstance(corridor, (tuple, list))
    if per_side:
        lo0 = np.asarray(corridor[0], float)
        hi0 = np.asarray(corridor[1], float)
        if lo0.shape != x0.shape or hi0.shape != x0.shape:
            raise ValueError('corridor (lo, hi) must match x, y in length')
        if np.any(lo0 > 1e-9) or np.any(hi0 < -1e-9):
            raise ValueError('per-side corridor needs lo <= 0 <= hi')
        lo0 = np.minimum(lo0, 0.0)
        hi0 = np.maximum(hi0, 0.0)
        nx0, ny0 = left_normals(x0, y0)               # bounds live on these
    xc, yc = x0.copy(), y0.copy()
    s_total = np.zeros_like(x0)                       # offset along n0
    for _ in range(int(iterations)):
        ds = np.maximum(segment_lengths(xc, yc), 1e-9)
        if per_side:
            # displace along the ORIGINAL normals every pass: the point stays
            # exactly p0 + s*n0, so the per-side wall guarantee is structural
            # (no tangential drift); the QP is still re-linearized around the
            # current geometry each pass.
            nx, ny = nx0, ny0
        else:
            nx, ny = left_normals(xc, yc)
        D = _second_diff_matrix(ds)
        L = _first_diff_matrix(ds)
        Ax = D * nx[None, :]                          # D @ diag(nx)
        Ay = D * ny[None, :]
        H = 2.0 * (Ax.T @ Ax + Ay.T @ Ay + smooth * (L.T @ L)
                   + ridge * np.eye(len(xc)))
        g = 2.0 * (Ax.T @ (D @ xc) + Ay.T @ (D @ yc))
        if per_side:
            blo = lo0 - s_total
            bhi = hi0 - s_total
        else:
            # remaining corridor budget relative to the original line
            # (triangle inequality => total displacement stays <= corridor)
            budget = np.maximum(corridor - np.hypot(xc - x0, yc - y0), 0.0)
            blo, bhi = -budget, budget
        d = _solve_box_qp(H, g, blo, bhi)
        xc = xc + d * nx
        yc = yc + d * ny
        s_total = s_total + d
        if np.abs(d).max() < 1e-4:                    # converged early
            break
    hdg, curv = heading_curvature(xc, yc)
    return xc, yc, hdg, curv


# ── occupancy-aware corridors ─────────────────────────────────────────────────

def _side_profile(grid_map, px, py, vx, vy, margin, cap):
    """Walk the distance field from (px, py) along unit (vx, vy) in half-cell
    steps up to `cap` and return (ts, dist >= margin mask, max dist seen)."""
    step = 0.5 * grid_map.res
    ts = np.arange(step, cap + 1e-9, step)
    dv = np.atleast_1d(grid_map.distance_to_wall(px + ts * vx, py + ts * vy))
    return ts, dv >= margin, float(dv.max())


def _bound_from_zero(grid_map, px, py, vx, vy, margin, cap):
    """Largest offset t <= cap such that EVERY sampled point in (0, t] keeps
    distance_to_wall >= margin (contiguous safe run starting at the point)."""
    ts, ok, _ = _side_profile(grid_map, px, py, vx, vy, margin, cap)
    if len(ts) == 0 or not ok[0]:
        return 0.0
    end = int(np.argmin(ok)) - 1 if not ok.all() else len(ts) - 1
    return float(ts[end])


def _bound_away(grid_map, px, py, vx, vy, margin, cap):
    """For a point already inside `margin` of a wall: offset of the last
    sample in the FIRST contiguous run with distance >= margin (0 if the run
    never starts), plus the best wall distance seen along the way."""
    ts, ok, best = _side_profile(grid_map, px, py, vx, vy, margin, cap)
    if ok.any():
        end = first = int(np.argmax(ok))
        while end + 1 < len(ts) and ok[end + 1]:
            end += 1
        return float(ts[end]), best
    return 0.0, best


def map_corridors(x, y, grid_map, margin=0.35, cap=1.0):
    """Per-point per-side corridor bounds from an occupancy map.

    For each waypoint, ray-cast along the +left-normal and -left-normal
    (GridMap.clearance gives the free distance to the first wall on the ray;
    sampling the distance field along the same ray then guarantees every
    admitted offset keeps distance_to_wall >= margin, which the bare ray
    misses for walls running parallel to the normal near corners), subtract
    `margin` (half car width plus safety), clip to [0, cap].  Points that
    graze a wall (distance_to_wall below half a cell) get a ONE-SIDED
    corridor: the side on which the distance field recovers to `margin`
    opens up (the point may move away from the wall), the wall side stays
    closed at 0.  Returns (lo, hi) with lo <= 0 <= hi, ready to pass as
    ``corridor=(lo, hi)`` to refine_raceline.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    nx, ny = left_normals(x, y)
    n = len(x)
    lo = np.zeros(n)
    hi = np.zeros(n)
    d0 = np.atleast_1d(grid_map.distance_to_wall(x, y))
    max_ray = cap + margin + grid_map.res
    for i in range(n):
        if d0[i] >= margin:
            # clearance ray = cheap upper bound; distance-field scan tightens
            # it so the whole box satisfies the margin, not just the ray hit.
            cl = grid_map.clearance(x[i], y[i], nx[i], ny[i], max_ray)
            cr = grid_map.clearance(x[i], y[i], -nx[i], -ny[i], max_ray)
            hi[i] = min(np.clip(cl - margin, 0.0, cap),
                        _bound_from_zero(grid_map, x[i], y[i],
                                         nx[i], ny[i], margin, cap))
            lo[i] = -min(np.clip(cr - margin, 0.0, cap),
                         _bound_from_zero(grid_map, x[i], y[i],
                                          -nx[i], -ny[i], margin, cap))
        else:                                         # inside margin already
            bp, dp = _bound_away(grid_map, x[i], y[i], nx[i], ny[i],
                                 margin, cap)
            bm, dm = _bound_away(grid_map, x[i], y[i], -nx[i], -ny[i],
                                 margin, cap)
            hi[i] = bp                                # away side(s) open,
            lo[i] = -bm                               # wall side stays 0
            if hi[i] == 0.0 and lo[i] == 0.0:
                # margin unreachable within cap: still allow moving toward
                # the best improvement seen (never toward the wall).
                if dp >= dm and dp > d0[i]:
                    hi[i] = cap
                elif dm > d0[i]:
                    lo[i] = -cap
    return lo, hi


def verify_wall_clearance(x_new, y_new, grid_map, margin,
                          x_orig=None, y_orig=None, tol=None, warn=True):
    """Check a refined line against the map: every point must keep
    distance_to_wall >= margin — except where the ORIGINAL point was already
    inside margin (pixelated wall grazes), which must merely not get worse.
    `tol` absorbs grid quantization (default: half a cell).  Returns
    (ok, d_new); emits a warning listing offending indices when not ok.
    """
    import warnings
    x_new = np.asarray(x_new, float)
    y_new = np.asarray(y_new, float)
    d_new = np.atleast_1d(grid_map.distance_to_wall(x_new, y_new))
    required = np.full_like(d_new, float(margin))
    if x_orig is not None and y_orig is not None:
        d_old = np.atleast_1d(grid_map.distance_to_wall(
            np.asarray(x_orig, float), np.asarray(y_orig, float)))
        required = np.minimum(required, d_old)
    if tol is None:
        tol = 0.5 * grid_map.res
    bad = np.where(d_new < required - tol)[0]
    ok = bad.size == 0
    if not ok and warn:
        worst = bad[np.argmin(d_new[bad] - required[bad])]
        warnings.warn(
            f'refined raceline violates wall margin at {bad.size} point(s) '
            f'{bad.tolist()[:10]}; worst i={worst}: '
            f'{d_new[worst]:.3f} m < required {required[worst]:.3f} m')
    return ok, d_new
