"""
Frenet-frame "spliner" overtaker — ForzaETH evasion lines (pure, no ROS).
=========================================================================

The stack can already SEE an opponent (YOLO + lidar) but the only response is
the AEB stopping the car behind it.  This module adds the missing move: deform
the local raceline around the opponent with a cubic spline in Frenet
coordinates, exactly the recipe ForzaETH race on (their "spliner" planner).

  apex            s_apex = opponent_s,  d_apex = opponent_d +/- evasion_dist
  control points  s offsets {-4, -3, -1.5, 0, +2, +3, +4} m from the apex
                  (optionally scaled by clip(1 + v/v_max, 1, 1.5)), d = 0 at
                  every point except the apex
  spline          CubicSpline(s_offsets, d), clamped (d' = 0) at both ends so
                  the evasion line rejoins the raceline with matching heading
  output          the full raceline with only the points inside the window
                  replaced by frenet_to_cartesian(s, spline(s)) — same length
                  and indexing, so existing controllers track it unchanged

Side selection follows ForzaETH too: pass on the OUTSIDE of the local turn
(sign of the summed signed curvature around the opponent), where a defending
car leaves the most room.  Speeds keep the raceline profile, scaled by 0.9
when forced to pass on the inside.

Frenet frame: s is arc length along the closed raceline polyline, d is the
perpendicular offset, positive to the LEFT of the direction of travel.

Reference: Baumann et al., "ForzaETH Race Stack", Journal of Field Robotics
2024 (arXiv:2403.11784); github.com/ForzaETH/race_stack (planner/spliner).
"""

import math

import numpy as np
from scipy.interpolate import CubicSpline

# Control-point s offsets from the apex (metres), per ForzaETH's spliner.
SPLINE_OFFSETS = np.array([-4.0, -3.0, -1.5, 0.0, 2.0, 3.0, 4.0])


# ─────────────────────────────────────────────────────────────────────────────
# Frenet helpers (closed polyline raceline)
# ─────────────────────────────────────────────────────────────────────────────

def arc_lengths(x, y):
    """Cumulative arc length s[i] at every raceline point + closed-lap total."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    seg = np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))
    return np.concatenate([[0.0], np.cumsum(seg)[:-1]]), float(seg.sum())


def cartesian_to_frenet(px, py, x, y, s=None, total=None):
    """(px, py) -> (s, d) against the closed raceline; d positive left.

    Projects onto the two polyline segments adjacent to the nearest raceline
    point and keeps the closer projection, so the round trip through
    frenet_to_cartesian is exact away from vertex normal-cone gaps.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    if s is None or total is None:
        s, total = arc_lengths(x, y)
    n = len(x)
    j = int(np.argmin((x - px) ** 2 + (y - py) ** 2))
    best = None
    for i in ((j - 1) % n, j):
        k = (i + 1) % n
        tx, ty = x[k] - x[i], y[k] - y[i]
        seg2 = tx * tx + ty * ty
        if seg2 < 1e-12:
            continue
        t = float(np.clip(((px - x[i]) * tx + (py - y[i]) * ty) / seg2, 0.0, 1.0))
        cx, cy = x[i] + t * tx, y[i] + t * ty
        dist2 = (px - cx) ** 2 + (py - cy) ** 2
        if best is None or dist2 < best[0]:
            seg = math.sqrt(seg2)
            d = ((px - cx) * -ty + (py - cy) * tx) / seg     # left normal
            best = (dist2, (s[i] + t * seg) % total, d)
    return best[1], best[2]


def frenet_to_cartesian(s_query, d, x, y, s=None, total=None):
    """(s, d) -> (x, y): walk s along the polyline, step d along the left normal."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if s is None or total is None:
        s, total = arc_lengths(x, y)
    n = len(x)
    s_query = float(s_query) % total
    i = int(np.searchsorted(s, s_query, side='right') - 1)
    k = (i + 1) % n
    tx, ty = x[k] - x[i], y[k] - y[i]
    seg = math.hypot(tx, ty)
    t = (s_query - s[i]) / max(seg, 1e-12)
    nx, ny = -ty / max(seg, 1e-12), tx / max(seg, 1e-12)     # left normal
    return x[i] + t * tx + d * nx, y[i] + t * ty + d * ny


def _signed_curvature(x, y):
    """Discrete signed curvature at every vertex (positive = turning left)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ux, uy = x - np.roll(x, 1), y - np.roll(y, 1)            # incoming
    vx, vy = np.roll(x, -1) - x, np.roll(y, -1) - y          # outgoing
    cross = ux * vy - uy * vx
    dot = ux * vx + uy * vy
    angle = np.arctan2(cross, dot)
    ds = 0.5 * (np.hypot(ux, uy) + np.hypot(vx, vy))
    return angle / np.maximum(ds, 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Overtake planning
# ─────────────────────────────────────────────────────────────────────────────

def choose_side(x, y, opponent_s, window=4.0):
    """'left' or 'right': the OUTSIDE of the local turn around the opponent.

    Sums the signed curvature within +/-window metres of opponent_s; a left
    turn (positive sum) means the outside — and the room — is on the right.
    """
    s, total = arc_lengths(x, y)
    rel = (s - opponent_s + total / 2.0) % total - total / 2.0
    near = np.abs(rel) <= window
    kappa = float(_signed_curvature(x, y)[near].sum())
    return 'right' if kappa > 0.0 else 'left'


def plan_overtake(raceline, opponent_s, opponent_d, side,
                  evasion_dist=0.65, ego_speed=0.0, v_max=8.0,
                  inside_slowdown=0.9, hold_dist=0.0):
    """Deform the raceline around an opponent; returns (x, y, v) full arrays.

    raceline is an (x, y, v) tuple of equal-length closed-lap arrays.  The
    apex sits at the opponent's s, offset evasion_dist to the requested side
    of the opponent's d; control points at SPLINE_OFFSETS (scaled by
    clip(1 + ego_speed/v_max, 1, 1.5)) hold d = 0 except at the apex; a
    clamped CubicSpline over (offset, d) is evaluated at every raceline point
    inside the window and mapped back to Cartesian.  Points outside the
    window are returned untouched, so the result drops into any controller
    that already tracks the raceline.  Speeds keep the raceline profile,
    scaled by inside_slowdown when the pass is on the inside of the turn.

    hold_dist > 0 keeps the full apex offset for that many metres PAST the
    opponent before easing back in.  On a slow pass (small speed difference)
    the ego sits alongside for a while: with the plain spline its lookahead
    controller starts tracking the rejoin leg early and squeezes into the
    opponent — holding the offset keeps the gap until genuinely clear.
    hold_dist = 0 reproduces the original ForzaETH control points exactly.
    """
    rx, ry, rv = (np.asarray(a, float) for a in raceline)
    s, total = arc_lengths(rx, ry)
    scale = float(np.clip(1.0 + ego_speed / v_max, 1.0, 1.5))
    offsets = SPLINE_OFFSETS * scale
    sign = 1.0 if side == 'left' else -1.0
    d_apex = float(opponent_d) + sign * evasion_dist
    if hold_dist > 0.0:
        rear = offsets[offsets < 0.0]
        front = offsets[offsets > 0.0] + float(hold_dist)
        offsets = np.concatenate([rear, [0.0, float(hold_dist)], front])
        d_ctrl = np.zeros(len(offsets))
        d_ctrl[len(rear)] = d_ctrl[len(rear) + 1] = d_apex
    else:
        d_ctrl = np.zeros(len(offsets))
        d_ctrl[np.argmin(np.abs(offsets))] = d_apex
    spline = CubicSpline(offsets, d_ctrl, bc_type='clamped')

    rel = (s - float(opponent_s) + total / 2.0) % total - total / 2.0
    win = (rel > offsets[0]) & (rel < offsets[-1])
    new_x, new_y, new_v = rx.copy(), ry.copy(), rv.copy()
    for i in np.flatnonzero(win):
        new_x[i], new_y[i] = frenet_to_cartesian(
            s[i], float(spline(rel[i])), rx, ry, s, total)

    # inside pass (same side as the turn direction) -> back off the speed
    turn = _signed_curvature(rx, ry)[np.abs(rel) <= offsets[-1]].sum()
    if (turn > 1e-3 and side == 'left') or (turn < -1e-3 and side == 'right'):
        new_v[win] *= inside_slowdown
    return new_x, new_y, new_v
