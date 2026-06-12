"""
Spliner unit tests — Frenet frame + ForzaETH overtake deformation.
==================================================================

Validates the overtaker's geometry without ROS or hardware, all on the real
competition raceline:
  - Frenet round trip (cartesian -> frenet -> cartesian) within 2 cm;
  - the deformed line passes the apex at evasion_dist from the opponent;
  - points outside the spline window are bit-identical to the input;
  - the deformation eases in/out of the window with no discontinuity.

    python3 -m pytest tests/test_spliner.py -q
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'f1tenth_gym_ros'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spliner import (arc_lengths, cartesian_to_frenet, frenet_to_cartesian,  # noqa: E402
                     choose_side, plan_overtake)
from closed_loop import load_raceline                                        # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RX, RY, RH, RC, RV = load_raceline(
    os.path.join(REPO, 'racelines', 'comp_raceline.csv'))
S, TOTAL = arc_lengths(RX, RY)


def _plan(evasion=0.65, ego_speed=0.0):
    oi = int(0.4 * len(RX))
    opp_s, opp_d = cartesian_to_frenet(RX[oi], RY[oi], RX, RY, S, TOTAL)
    side = choose_side(RX, RY, opp_s)
    new = plan_overtake((RX, RY, RV), opp_s, opp_d, side,
                        evasion_dist=evasion, ego_speed=ego_speed)
    return oi, opp_s, side, new


# ─────────────────────────────────────────────────────────────────────────────
# Frenet frame
# ─────────────────────────────────────────────────────────────────────────────

def test_frenet_round_trip_on_comp_raceline():
    # offset points all around the lap must survive cartesian -> frenet ->
    # cartesian within 2 cm (it is exact away from vertex normal cones)
    rng = np.random.default_rng(7)
    for _ in range(400):
        s_q = rng.uniform(0.0, TOTAL)
        d_q = rng.uniform(-0.5, 0.5)
        px, py = frenet_to_cartesian(s_q, d_q, RX, RY, S, TOTAL)
        s2, d2 = cartesian_to_frenet(px, py, RX, RY, S, TOTAL)
        x2, y2 = frenet_to_cartesian(s2, d2, RX, RY, S, TOTAL)
        assert math.hypot(px - x2, py - y2) < 0.02


def test_frenet_d_sign_is_left_positive():
    # a point nudged along the left normal of segment 0 must get d > 0
    tx, ty = RX[1] - RX[0], RY[1] - RY[0]
    seg = math.hypot(tx, ty)
    px = RX[0] + 0.5 * tx - 0.3 * ty / seg
    py = RY[0] + 0.5 * ty + 0.3 * tx / seg
    _, d = cartesian_to_frenet(px, py, RX, RY, S, TOTAL)
    assert abs(d - 0.3) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Overtake deformation
# ─────────────────────────────────────────────────────────────────────────────

def test_apex_clears_opponent_by_evasion_dist():
    oi, opp_s, side, (mx, my, mv) = _plan(evasion=0.65)
    # the opponent sits exactly ON a raceline vertex, so that vertex of the
    # deformed line IS the apex: it must sit evasion_dist away laterally
    apex_gap = math.hypot(mx[oi] - RX[oi], my[oi] - RY[oi])
    assert abs(apex_gap - 0.65) < 0.02
    # and the whole deformed polyline keeps at least that vertex clearance
    gaps = np.hypot(mx - RX[oi], my - RY[oi])
    assert gaps.min() > 0.55


def test_apex_side_matches_choose_side():
    oi, opp_s, side, (mx, my, mv) = _plan()
    _, d_apex = cartesian_to_frenet(mx[oi], my[oi], RX, RY, S, TOTAL)
    assert (d_apex > 0) == (side == 'left')


def test_no_deformation_outside_window():
    oi, opp_s, side, (mx, my, mv) = _plan(ego_speed=8.0)   # widest window
    rel = (S - opp_s + TOTAL / 2) % TOTAL - TOTAL / 2
    outside = np.abs(rel) >= 4.0 * 1.5                     # max scale = 1.5
    assert np.array_equal(mx[outside], RX[outside])
    assert np.array_equal(my[outside], RY[outside])
    assert np.array_equal(mv[outside], RV[outside])
    assert len(mx) == len(RX) and len(mv) == len(RV)


def test_continuous_at_window_edges():
    oi, opp_s, side, (mx, my, mv) = _plan()
    offset = np.hypot(mx - RX, my - RY)                    # lateral deformation
    win = np.flatnonzero(offset > 1e-12)
    # clamped spline (d = 0, d' = 0 at the ends): first/last deformed points
    # have eased barely off the raceline — no step where the window starts
    assert offset[win[0]] < 0.08 and offset[win[-1]] < 0.08
    # and no new kink: per-vertex heading change of the deformed polyline is
    # no worse than what the raw raceline already contains

    def max_turn(x, y):
        ux, uy = x - np.roll(x, 1), y - np.roll(y, 1)
        vx, vy = np.roll(x, -1) - x, np.roll(y, -1) - y
        return np.abs(np.arctan2(ux * vy - uy * vx, ux * vx + uy * vy)).max()

    assert max_turn(mx, my) <= max_turn(RX, RY) + 1e-9


def test_window_scales_with_speed():
    _, opp_s, side, (mx0, my0, _) = _plan(ego_speed=0.0)   # scale 1.0
    _, _, _, (mx1, my1, _) = _plan(ego_speed=8.0)          # scale 1.5 (clipped)
    n0 = (np.hypot(mx0 - RX, my0 - RY) > 1e-12).sum()
    n1 = (np.hypot(mx1 - RX, my1 - RY) > 1e-12).sum()
    assert n1 > n0


def test_inside_pass_slows_down():
    oi = int(0.4 * len(RX))
    opp_s, opp_d = cartesian_to_frenet(RX[oi], RY[oi], RX, RY, S, TOTAL)
    outside = choose_side(RX, RY, opp_s)
    inside = 'left' if outside == 'right' else 'right'
    _, _, v_out = plan_overtake((RX, RY, RV), opp_s, opp_d, outside)
    _, _, v_in = plan_overtake((RX, RY, RV), opp_s, opp_d, inside)
    assert np.allclose(v_out, RV)                          # outside: full speed
    assert v_in[oi] < RV[oi]                               # inside: backed off


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-q']))
