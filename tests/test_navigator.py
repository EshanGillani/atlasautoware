"""
Navigator tests — grid planner + open-path follower (pure, no ROS).
===================================================================

Pins the contract of the indoor "drive to a goal" mode: the planner finds
clearance-respecting paths on the real competition map (and refuses politely
when there is none), shortcut smoothing only ever shortens, and the follower
actually arrives and STOPS in a kinematic closed-loop sim — straight-line and
around a corner.

    python3 -m pytest tests/test_navigator.py -q
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'f1tenth_gym_ros'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_map import GridMap                                    # noqa: E402
from grid_planner import (GridPlanner, load_drivable,           # noqa: E402
                          DEFAULT_INFLATION)
from navigator import PathFollower, drive_to_goal               # noqa: E402
from closed_loop import load_raceline                           # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFLATION = DEFAULT_INFLATION

_cache = {}


def comp_planner():
    """Shared (map, planner, raceline) — graph build + EDT happen once."""
    if 'comp' not in _cache:
        gm = GridMap.load(os.path.join(REPO, 'maps', 'comp_track.yaml'))
        drv = load_drivable(os.path.join(REPO, 'maps', 'comp_track.yaml'))
        rx, ry, *_ = load_raceline(
            os.path.join(REPO, 'racelines', 'comp_raceline.csv'))
        _cache['comp'] = (gm, GridPlanner(gm, INFLATION, drivable=drv), rx, ry)
    return _cache['comp']


def box_map(w_m=12.0, h_m=6.0, res=0.05, blocks=()):
    """Synthetic bordered room (+ optional rectangular blocks, in metres)."""
    H, W = int(h_m / res), int(w_m / res)
    occ = np.zeros((H, W), bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    for (x0, y0, x1, y1) in blocks:
        c0, c1 = int(x0 / res), int(x1 / res)
        r1, r0 = (H - 1) - int(y0 / res), (H - 1) - int(y1 / res)
        occ[max(r0, 0):r1 + 1, max(c0, 0):c1 + 1] = True
    return GridMap(occ, res, (0.0, 0.0))


# ── planner ───────────────────────────────────────────────────────────────────
def test_plans_between_free_points_with_clearance():
    gm, pl, rx, ry = comp_planner()
    res = pl.plan((rx[0], ry[0]), (rx[150], ry[150]))
    assert res is not None
    # ends near the requested points (snap tolerance)
    assert math.hypot(res['x'][0] - rx[0], res['y'][0] - ry[0]) < 0.5
    assert math.hypot(res['x'][-1] - rx[150], res['y'][-1] - ry[150]) < 0.5
    # every path point stays beyond the inflation radius from walls
    assert np.all(gm.distance_to_wall(res['x'], res['y']) > INFLATION)
    # ~0.2 m spacing
    ds = np.hypot(np.diff(res['x']), np.diff(res['y']))
    assert ds.max() < 0.45 and abs(np.median(ds) - 0.2) < 0.1


def test_speed_profile_is_gentle_and_slows_at_goal():
    _, pl, rx, ry = comp_planner()
    res = pl.plan((rx[0], ry[0]), (rx[150], ry[150]),
                  v_max=2.0, a_lat_max=3.0, v_goal=0.5)
    assert np.all(res['v'] <= 2.0 + 1e-9)
    assert res['v'][-1] <= 0.5 + 1e-9
    assert res['v'].max() > 1.0          # actually drives on the straights


def test_goal_in_wall_returns_none():
    _, pl, _, _ = comp_planner()
    # (1, 1) is deep inside the occupied margin of comp_track (free space
    # starts around x ~ 14) — snapping must fail and plan return None
    assert pl.plan((30.0, 30.0), (1.0, 1.0)) is None
    assert pl.plan((1.0, 1.0), (30.0, 30.0)) is None


def test_unreachable_goal_returns_none():
    # room split by a full wall: right half unreachable from left half
    gm = box_map(blocks=[(5.9, 0.0, 6.1, 6.0)])
    pl = GridPlanner(gm, INFLATION)
    assert pl.plan((2.0, 3.0), (10.0, 3.0)) is None
    # sanity: same-side goal works
    assert pl.plan((2.0, 3.0), (4.0, 4.0)) is not None


def test_shortcut_smoothing_shortens_without_violating_clearance():
    gm, pl, rx, ry = comp_planner()
    res = pl.plan((rx[50], ry[50]), (rx[220], ry[220]))   # around corners
    assert res is not None
    assert res['length'] < res['raw_length'] - 0.05       # strictly shorter
    assert np.all(gm.distance_to_wall(res['x'], res['y']) > INFLATION)


def test_extra_obstacles_divert_the_plan():
    gm = box_map()
    pl = GridPlanner(gm, INFLATION)
    base = pl.plan((1.0, 3.0), (11.0, 3.0))
    obs = [(6.0, y) for y in np.arange(1.0, 5.01, 0.1)]   # wall of scan hits
    detour = pl.plan((1.0, 3.0), (11.0, 3.0), extra_obstacles=obs)
    assert base is not None and detour is not None
    assert detour['length'] > base['length'] + 0.2        # goes around
    d2 = np.min((detour['x'][:, None] - np.array(obs)[:, 0]) ** 2
                + (detour['y'][:, None] - np.array(obs)[:, 1]) ** 2, axis=1)
    assert np.all(np.sqrt(d2) > INFLATION - 1e-6)


# ── follower (kinematic closed loop) ─────────────────────────────────────────
def _follow(gm, pl, start, goal):
    res = pl.plan(start, goal)
    assert res is not None
    f = PathFollower(goal_tol=0.3)
    f.set_path(res['x'], res['y'], res['v'], kappa=res['kappa'])
    yaw0 = math.atan2(res['y'][1] - res['y'][0], res['x'][1] - res['x'][0])
    out = drive_to_goal(f, res['x'][0], res['y'][0], yaw0)
    return res, f, out


def test_follower_reaches_straight_goal_and_stops():
    gm = box_map()
    pl = GridPlanner(gm, INFLATION)
    res, f, out = _follow(gm, pl, (1.0, 3.0), (11.0, 3.0))
    assert out['reached']
    assert out['final_dist'] < 0.4
    assert out['final_speed'] < 0.05                      # actually stopped
    assert f.done and f.control(*[11.0, 3.0, 0.0, 0.0])[1] == 0.0


def test_follower_reaches_corner_goal_and_stops():
    # L-shaped corridor: a block forces a 90-degree corner
    gm = box_map(w_m=10.0, h_m=10.0, blocks=[(2.0, 2.0, 10.0, 10.0)])
    pl = GridPlanner(gm, INFLATION)
    res, f, out = _follow(gm, pl, (1.0, 9.0), (9.0, 1.0))
    assert res['length'] > math.hypot(8.0, 8.0)           # real corner path
    assert out['reached']
    assert out['final_dist'] < 0.4
    assert out['final_speed'] < 0.05
    # never scrapes the corner
    assert gm.distance_to_wall(out['x'], out['y']).min() > 0.15


def test_follower_done_latches_zero_command():
    f = PathFollower(goal_tol=0.3)
    f.set_path([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
    steer, v, done = f.control(2.05, 0.0, 0.0, 0.3)       # inside tolerance
    assert done and v == 0.0 and steer == 0.0
    steer, v, done = f.control(0.0, 0.0, 0.0, 0.0)        # stays latched
    assert done and v == 0.0


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-q']))
