"""
GridMap tests — conventions validated against the real track + raceline.
========================================================================

The coordinate convention is the part that silently breaks everything
downstream (PF likelihoods, corridors, planning), so it is pinned here
empirically: the driven raceline must lie in free space with sensible wall
clearance under the ROS row-flip convention.

    python3 -m pytest tests/test_grid_map.py -q
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'f1tenth_gym_ros'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_map import GridMap, grid_raycast, map_path_for   # noqa: E402
from closed_loop import load_raceline                      # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    gm = GridMap.load(os.path.join(REPO, 'maps', 'comp_track.yaml'))
    rx, ry, *_ = load_raceline(
        os.path.join(REPO, 'racelines', 'comp_raceline.csv'))
    return gm, rx, ry


def test_raceline_lies_in_free_space():
    gm, rx, ry = _load()
    occ = gm.is_occupied(rx, ry)
    d = gm.distance_to_wall(rx, ry)
    # the driven line may graze pixelated walls at a few corners, but the
    # bulk must be clearly free — this pins the row-flip convention
    assert occ.mean() < 0.10, f'{occ.sum()}/300 raceline points on walls'
    assert d.mean() > 0.5, f'mean clearance {d.mean():.2f} m — wrong convention?'


def test_world_grid_roundtrip():
    gm, rx, ry = _load()
    r, c = gm.world_to_grid(rx, ry)
    x2, y2 = gm.grid_to_world(r, c)
    # round-trip lands within one cell
    assert np.max(np.hypot(x2 - rx, y2 - ry)) < gm.res * 0.71 + 1e-9


def test_out_of_bounds_is_occupied():
    gm, _, _ = _load()
    assert gm.is_occupied(-1e3, -1e3)
    assert gm.is_occupied(1e3, 1e3)


def test_clearance_ray_matches_distance_field_floor():
    gm, rx, ry = _load()
    i = int(np.argmax(gm.distance_to_wall(rx, ry)))   # most open point
    d0 = gm.distance_to_wall(rx[i], ry[i])
    for a in np.linspace(0, 2 * math.pi, 8, endpoint=False):
        c = gm.clearance(rx[i], ry[i], math.cos(a), math.sin(a), max_d=12.0)
        assert c >= d0 - gm.res, 'ray shorter than omnidirectional clearance'


def test_raycast_shape_and_caps():
    gm, rx, ry = _load()
    r = grid_raycast(gm, rx[0], ry[0],
                     np.linspace(0, 2 * math.pi, 16, endpoint=False),
                     max_range=3.0)
    assert r.shape == (16,) and np.all(r <= 3.0 + 1e-9) and np.all(r >= 0.0)


def test_map_path_resolution():
    assert map_path_for('comp_raceline.csv') is not None


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-q']))
