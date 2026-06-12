"""
Curvature-aware lookahead scheduling — backward compatibility + behavior.
=========================================================================

  - k_curv = 0, and curvature=None (any k_curv), must be bit-identical to the
    original speed-only lookahead schedule on real control() calls;
  - with curvature + k_curv > 0, the lookahead shrinks entering a corner and
    stays untouched on a straight.

    python3 -m pytest tests/test_lookahead.py -q
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'f1tenth_gym_ros'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_controller import MAPController, build_lat_accel_lut    # noqa: E402
from closed_loop import load_raceline                            # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LUT = build_lat_accel_lut()


def _comp_raceline():
    return load_raceline(os.path.join(REPO, 'racelines', 'comp_raceline.csv'))


def _legacy_control(ctl, x, y, yaw, v, nearest):
    """Reference oracle: the ORIGINAL control() (speed-only lookahead)."""
    rl = ctl._rl
    v_t = float(rl['v'][nearest % rl['n']])
    L = float(np.clip(ctl.q_la + ctl.m_la * v_t, ctl.la_min, ctl.la_max))
    s_la = (rl['s'][nearest % rl['n']] + L) % rl['total']
    j = int(np.searchsorted(rl['s'], s_la)) % rl['n']
    dx, dy = rl['x'][j] - x, rl['y'][j] - y
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return 0.0, v_t
    eta = math.asin(np.clip(
        (-math.sin(yaw) * dx + math.cos(yaw) * dy) / dist, -1.0, 1.0))
    a_des = 2.0 * v_t ** 2 * math.sin(eta) / L
    return ctl.steer_from_lat_accel(a_des, max(v, v_t * 0.5)), v_t


def _sample_poses(rx, ry, rh):
    n = len(rx)
    poses = []
    for i in range(0, n, n // 12):
        poses.append((rx[i] + 0.21, ry[i] - 0.13, rh[i] + 0.07, 3.4, i))
    return poses


def test_k_curv_zero_bit_identical():
    rx, ry, rh, rc, rv = _comp_raceline()
    ctl = MAPController(lut=LUT, k_curv=0.0)
    ctl.set_raceline(rx, ry, rv, curvature=rc)     # curvature given, gain 0
    for pose in _sample_poses(rx, ry, rh):
        assert ctl.control(*pose) == _legacy_control(ctl, *pose)


def test_curvature_none_bit_identical():
    rx, ry, rh, rc, rv = _comp_raceline()
    ctl = MAPController(lut=LUT, k_curv=2.0)       # gain on, but no curvature
    ctl.set_raceline(rx, ry, rv)
    for pose in _sample_poses(rx, ry, rh):
        assert ctl.control(*pose) == _legacy_control(ctl, *pose)


def _straight_into_corner():
    """Synthetic raceline: 20 m straight then a 2 m-radius half-circle."""
    ds, R = 0.25, 2.0
    xs = np.arange(0.0, 20.0, ds)
    straight_x, straight_y = xs, np.zeros_like(xs)
    ang = np.arange(0.0, math.pi, ds / R)
    arc_x = 20.0 + R * np.sin(ang)
    arc_y = R * (1.0 - np.cos(ang))
    x = np.concatenate([straight_x, arc_x])
    y = np.concatenate([straight_y, arc_y])
    curv = np.concatenate([np.zeros(len(xs)), np.full(len(ang), 1.0 / R)])
    speed = np.full(len(x), 3.0)
    return x, y, curv, speed


def test_lookahead_shrinks_entering_corner_not_on_straight():
    x, y, curv, speed = _straight_into_corner()
    ctl = MAPController(lut=LUT, k_curv=2.0)
    ctl.set_raceline(x, y, speed, curvature=curv)
    L_base = float(np.clip(ctl.q_la + ctl.m_la * 3.0, ctl.la_min, ctl.la_max))

    mid = 30                                     # mid-straight: window all flat
    ctl.control(x[mid], y[mid], 0.0, 3.0, mid)
    assert ctl.last_lookahead == L_base          # untouched on the straight

    entry = len(np.arange(0.0, 20.0, 0.25)) - 1  # last straight point
    ctl.control(x[entry], y[entry], 0.0, 3.0, entry)
    assert ctl.last_lookahead < L_base           # shrinks entering the corner
    expect = L_base / (1.0 + 2.0 * ctl._curvature_ahead(entry, L_base))
    assert math.isclose(ctl.last_lookahead, max(expect, ctl.la_min))

    inside = entry + 4                           # deep in the corner: shorter
    ctl.control(x[inside], y[inside], 1.0, 3.0, inside)
    assert ctl.last_lookahead < ctl.la_min + 0.45  # near full shrink
    assert ctl.last_lookahead >= ctl.la_min        # but never below the floor


def test_k_curv_zero_lookahead_unchanged_everywhere():
    rx, ry, rh, rc, rv = _comp_raceline()
    a = MAPController(lut=LUT, k_curv=0.0)
    a.set_raceline(rx, ry, rv, curvature=rc)
    b = MAPController(lut=LUT, k_curv=2.0)
    b.set_raceline(rx, ry, rv, curvature=rc)
    shrunk = 0
    for i in range(0, len(rx), 10):
        v_t = float(rv[i])
        L0 = float(np.clip(a.q_la + a.m_la * v_t, a.la_min, a.la_max))
        assert a._lookahead(v_t, i) == L0          # gain 0: always base
        if b._lookahead(v_t, i) < L0:
            shrunk += 1
    assert shrunk > 0                              # gain > 0 does act somewhere


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-q']))
