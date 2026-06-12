"""
Unit tests — occupancy-aware per-point corridors for the raceline refiner.
==========================================================================

Four properties make map corridors trustworthy: the refiner respects
per-point per-side bounds, a wall-grazing point only ever moves AWAY from
its wall (or stays), the scalar-corridor path is untouched, and the refined
line keeps the wall margin everywhere the original line allowed it.

    python3 -m pytest tests/test_map_corridors.py -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'f1tenth_gym_ros'))
from raceline_refiner import (refine_raceline, map_corridors,   # noqa: E402
                              verify_wall_clearance, left_normals)
from grid_map import GridMap                                    # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _noisy_circle(n=180, radius=3.0, sigma=0.03, seed=0):
    rng = np.random.default_rng(seed)
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = radius + rng.normal(0.0, sigma, n)
    return r * np.cos(th), r * np.sin(th)


def _annulus_map(res=0.05, size=10.0, center=(5.0, 5.0),
                 r_in=1.0, r_out=2.0):
    """Synthetic ring track: free space where r_in < r < r_out."""
    n = int(size / res)
    rows, cols = np.mgrid[0:n, 0:n]
    # cell centers in world coords (origin lower-left, row 0 = top)
    x = (cols + 0.5) * res
    y = ((n - 1) - rows + 0.5) * res
    r = np.hypot(x - center[0], y - center[1])
    occupied = ~((r > r_in) & (r < r_out))
    return GridMap(occupied, res, (0.0, 0.0))


def _circle_line(n=100, radius=1.5, center=(5.0, 5.0)):
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return center[0] + radius * np.cos(th), center[1] + radius * np.sin(th)


def test_per_side_bounds_respected():
    x, y = _noisy_circle()
    nx0, ny0 = left_normals(x, y)
    rng = np.random.default_rng(1)
    lo = -rng.uniform(0.0, 0.3, len(x))               # asymmetric per point
    hi = rng.uniform(0.0, 0.3, len(x))
    hi[::7] = 0.0                                     # some one-sided points
    lo[3::11] = 0.0
    xn, yn, _, _ = refine_raceline(x, y, corridor=(lo, hi))
    s = (xn - x) * nx0 + (yn - y) * ny0
    assert np.all(s >= lo - 1e-6)
    assert np.all(s <= hi + 1e-6)
    # the displacement is purely along the original normals (no drift)
    drift = np.hypot(xn - (x + s * nx0), yn - (y + s * ny0))
    assert drift.max() < 1e-9
    assert np.abs(s).max() > 0.0                      # it did move


def test_scalar_corridor_path_unchanged():
    # the scalar branch is the pre-existing algorithm, bit-for-bit: same
    # budget logic, normals recomputed each pass — symmetric per-side arrays
    # through the new branch and must agree closely but need not be equal.
    x, y = _noisy_circle()
    xs, ys, hs, ks = refine_raceline(x, y, corridor=0.10)
    disp = np.hypot(xs - x, ys - y)
    assert disp.max() <= 0.10 + 1e-6                  # old invariant intact
    # a symmetric per-side corridor solves the same problem in a slightly
    # different geometry (offsets pinned to the original normals): it must
    # stay inside the same corridor and reduce curvature about as well, but
    # need not match point-for-point.
    xt, yt, _, kt = refine_raceline(
        x, y, corridor=(np.full(len(x), -0.10), np.full(len(x), 0.10)))
    assert np.hypot(xs - xt, ys - yt).max() < 0.10
    from raceline_refiner import heading_curvature
    _, k0 = heading_curvature(x, y)
    assert np.abs(kt).max() < np.abs(k0).max()        # still smooths the line
    assert np.abs(kt).mean() < np.abs(k0).mean()
    # passing the scalar as int/np scalar is still the scalar path
    xi, yi, _, _ = refine_raceline(x, y, corridor=np.float64(0.10))
    assert np.array_equal(xs, xi) and np.array_equal(ys, yi)


def test_map_corridors_open_where_wide_and_respect_margin():
    gm = _annulus_map()
    x, y = _circle_line()                             # centerline, 0.5 m gap
    lo, hi = map_corridors(x, y, gm, margin=0.2, cap=1.0)
    assert np.all(lo <= 1e-12) and np.all(hi >= -1e-12)
    # ~0.5 m to each wall, minus margin => ~0.3 m per side (grid-quantized)
    assert 0.15 < np.median(hi) < 0.45
    assert -0.45 < np.median(lo) < -0.15
    # every admitted extreme keeps the margin
    nx, ny = left_normals(x, y)
    for t in (hi, lo):
        d = gm.distance_to_wall(x + t * nx, y + t * ny)
        assert np.all(d >= 0.2 - 0.5 * gm.res - 1e-9)


def test_one_sided_corridor_at_wall_grazing_point():
    gm = _annulus_map()
    x, y = _circle_line()
    # park one point ON the outer wall (radius r_out): zero clearance
    th = np.arctan2(y[10] - 5.0, x[10] - 5.0)
    x = x.copy(); y = y.copy()
    x[10] = 5.0 + 2.0 * np.cos(th)
    y[10] = 5.0 + 2.0 * np.sin(th)
    assert gm.distance_to_wall(x[10], y[10]) < 0.5 * gm.res
    lo, hi = map_corridors(x, y, gm, margin=0.2, cap=1.0)
    nx0, ny0 = left_normals(x, y)
    # CCW circle: left normal points inward, away from the outer wall
    inward = np.sign(nx0[10] * (5.0 - x[10]) + ny0[10] * (5.0 - y[10]))
    assert inward > 0
    assert hi[10] > 0.0                               # may move inward...
    assert lo[10] == 0.0                              # ...but not outward
    xn, yn, _, _ = refine_raceline(x, y, corridor=(lo, hi))
    s10 = (xn[10] - x[10]) * nx0[10] + (yn[10] - y[10]) * ny0[10]
    assert s10 >= -1e-9                               # moved away or stayed
    assert gm.distance_to_wall(xn[10], yn[10]) >= \
        gm.distance_to_wall(x[10], y[10]) - 1e-9      # never worse


def test_refined_real_line_keeps_margin_where_original_allowed():
    import csv
    path = os.path.join(REPO, 'racelines', 'comp_raceline.csv')
    if not os.path.exists(path):                      # pragma: no cover
        import pytest
        pytest.skip('comp raceline not present')
    with open(path) as f:
        rows = list(csv.DictReader(f))
    rx = np.array([float(r['x']) for r in rows])
    ry = np.array([float(r['y']) for r in rows])
    gm = GridMap.load(os.path.join(REPO, 'maps', 'comp_track.yaml'))
    margin = 0.35
    lo, hi = map_corridors(rx, ry, gm, margin=margin, cap=0.8)
    xn, yn, _, _ = refine_raceline(rx, ry, corridor=(lo, hi))
    ok, d_new = verify_wall_clearance(xn, yn, gm, margin, rx, ry)
    assert ok
    # stronger, point-by-point: clearance >= margin wherever the original
    # already had it; never degraded below the original elsewhere
    d_old = gm.distance_to_wall(rx, ry)
    tol = 0.5 * gm.res
    clear = d_old >= margin
    assert np.all(d_new[clear] >= margin - tol)
    assert np.all(d_new[~clear] >= d_old[~clear] - tol)


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-q']))
