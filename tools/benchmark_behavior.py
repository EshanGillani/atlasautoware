"""
Benchmark the head-to-head behavior state machine against the status quo.
=========================================================================

Two-car closed-loop laps on the competition raceline (kinematic-bicycle ego,
tests/closed_loop.py plant model, MAP controller — no ROS).  The opponent
drives the raceline at a fixed fraction of its speed profile; the ego runs
one of:

  behavior   RaceBehavior (GB_TRACK / TRAILING / OVERTAKE, race_behavior.py)
  aeb        status quo: raceline + brake behind the opponent (no passing)
  spliner    always-overtake: spliner line whenever the opponent is in the
             engagement horizon, no trailing / feasibility logic
  clean      no opponent: baseline lap time

Scenarios: opponent at 40 / 60 / 75 % of the profile from a rolling gap, plus
a NARROW scenario with a slow opponent placed in a section of comp_track
where the map shows no room on either side at evasion distance — the case
trailing exists for (always-spliner dives into the wall there; the behavior
waits, then passes where the track opens).

Per run: overtake success (clean pass = passed + min car-to-car clearance
>= 0.3 m point-to-point), time lost vs the clean lap, the state-transition
trace, collision count (centre distance < 0.20 m), and `plan_wall_min` — the
minimum map wall clearance over the *planned* evasion-line points (where the
plan deviates > 0.2 m from the raceline), the honest wall metric since the
point-mass plant cannot hit walls.  Cars are points: real vehicles need
evasion_dist > half-widths + margin.  Prints JSON.

    python3 tools/benchmark_behavior.py
"""

import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))
sys.path.insert(0, os.path.join(REPO, 'tests'))
from closed_loop import load_raceline                              # noqa: E402
from map_controller import MAPController, build_lat_accel_lut      # noqa: E402
from grid_map import GridMap                                       # noqa: E402
import spliner as sp                                               # noqa: E402
from race_behavior import RaceBehavior, OVERTAKE                   # noqa: E402

RL = os.path.join(REPO, 'racelines', 'comp_raceline.csv')
MAP_YAML = os.path.join(REPO, 'maps', 'comp_track.yaml')
DT = 0.02                       # 50 Hz, matching the harness
A_ACCEL, A_BRAKE = 4.0, 8.0     # plant accel/brake limits (run_lap defaults)
WHEELBASE, MAX_STEER = 0.33, 0.41
COLLIDE = 0.20                  # m centre-to-centre = collision
CLEAN_CLEARANCE = 0.30          # m required for a "clean pass"

_LUT = None


def _lut():
    global _LUT
    if _LUT is None:
        _LUT = build_lat_accel_lut()
    return _LUT


def _plan_wall_min(line, rx, ry, grid_map, s_arr=None, s_window=None):
    """Min map wall clearance over a plan's deviating points (inf if none).

    s_window=(lo, hi) restricts the check to raceline arc lengths in that
    interval — used to ask "did the planner ever deform INSIDE the blocked
    narrow section?" without the answer being polluted by map noise elsewhere.
    """
    lx, ly, _ = line
    dev = np.hypot(lx - rx, ly - ry) > 0.2
    if s_window is not None:
        dev &= (s_arr >= s_window[0]) & (s_arr <= s_window[1])
    if grid_map is None or not dev.any():
        return float('inf')
    return float(np.min(grid_map.distance_to_wall(lx[dev], ly[dev])))


def race(mode, rl, opp_frac, opp_start_s, *, ego_start_idx=0, grid_map=None,
         max_steps=9000, behavior_kwargs=None, s_window=None):
    """One two-car lap; returns the metrics dict (see module docstring)."""
    rx, ry, rh, rc, rv = rl
    s_arr, total = sp.arc_lengths(rx, ry)
    n = len(rx)

    ctl = MAPController(lut=_lut())
    ctl.set_raceline(rx, ry, rv)
    beh = None
    if mode == 'behavior':
        beh = RaceBehavior((rx, ry, rv), grid_map=grid_map,
                           **(behavior_kwargs or {}))

    i0 = int(ego_start_idx) % n
    px, py = float(rx[i0]) + 0.3, float(ry[i0]) - 0.2
    yaw, v = float(rh[i0]), 2.0
    opp_s = float(opp_start_s) % total
    spl_side = None                                 # always-spliner side lock

    min_clear = float('inf')
    plan_wall_min = float('inf')
    collided = passed = False
    cum, prev_j, t = 0, i0, 0.0

    def wrap(ds):
        return (ds + total / 2.0) % total - total / 2.0

    for _ in range(max_steps):
        # ── opponent: point on the raceline at opp_frac of the profile ──────
        k = int(np.searchsorted(s_arr, opp_s, side='right') - 1)
        v_opp = opp_frac * float(rv[k])
        opp_s = (opp_s + v_opp * DT) % total
        ox, oy = sp.frenet_to_cartesian(opp_s, 0.0, rx, ry, s_arr, total)
        if mode != 'clean':
            cdist = math.hypot(px - ox, py - oy)
            min_clear = min(min_clear, cdist)
            if cdist < COLLIDE:
                collided = True
                break

        # ── ego ──────────────────────────────────────────────────────────────
        j = int(np.argmin((rx - px) ** 2 + (ry - py) ** 2))
        d = j - prev_j
        if d < -n / 2:
            d += n
        if 0 < d < n / 2:
            cum += d
        prev_j = j

        if mode == 'behavior':
            out = beh.update(px, py, v, j, (opp_s, 0.0, v_opp), DT)
            if out['line_changed']:
                ctl.set_raceline(*out['line'])
                plan_wall_min = min(plan_wall_min, _plan_wall_min(
                    out['line'], rx, ry, grid_map, s_arr, s_window))
            steer, v_t = ctl.control(px, py, yaw, v, j)
            v_t = min(float(v_t), out['speed_cap'])
            if out['gap'] is not None and out['gap'] < -1.0:
                passed = True
        elif mode == 'aeb':
            steer, v_t = ctl.control(px, py, yaw, v, j)
            dx, dy = ox - px, oy - py
            if dx * math.cos(yaw) + dy * math.sin(yaw) > 0.0 and \
                    math.hypot(dx, dy) < v * v / (2.0 * A_BRAKE) + 0.6:
                v_t = 0.0
            if wrap(opp_s - sp.cartesian_to_frenet(
                    px, py, rx, ry, s_arr, total)[0]) < -1.0:
                passed = True
        elif mode == 'spliner':
            ego_s, _ = sp.cartesian_to_frenet(px, py, rx, ry, s_arr, total)
            rel = wrap(opp_s - ego_s)
            if -7.0 < rel < 12.0:                   # engagement horizon
                if spl_side is None:
                    spl_side = sp.choose_side(rx, ry, opp_s)
                plan = sp.plan_overtake(
                    (rx, ry, rv), opp_s, 0.0, spl_side, ego_speed=v)
                ctl.set_raceline(*plan)
                plan_wall_min = min(plan_wall_min, _plan_wall_min(
                    plan, rx, ry, grid_map, s_arr, s_window))
            else:
                spl_side = None
                ctl.set_raceline(rx, ry, rv)
            steer, v_t = ctl.control(px, py, yaw, v, j)
            if rel < -1.0:
                passed = True
        else:                                       # clean
            steer, v_t = ctl.control(px, py, yaw, v, j)

        # ── kinematic-bicycle plant (run_lap dynamics) ───────────────────────
        a = float(np.clip((float(v_t) - v) / DT, -A_BRAKE, A_ACCEL))
        px += v * math.cos(yaw) * DT
        py += v * math.sin(yaw) * DT
        yaw += v * math.tan(float(np.clip(steer, -MAX_STEER, MAX_STEER))) \
            / WHEELBASE * DT
        v = max(0.0, v + a * DT)
        t += DT
        if cum >= n:
            break

    res = dict(mode=mode, completed=cum >= n, collided=collided,
               lap_time=round(t, 2), passed=passed,
               min_clearance=(None if min_clear == float('inf')
                              else round(min_clear, 3)),
               plan_wall_min=(None if plan_wall_min == float('inf')
                              else round(plan_wall_min, 3)))
    res['clean_pass'] = bool(passed and not collided
                             and min_clear >= CLEAN_CLEARANCE)
    if beh is not None:
        res['transitions'] = [(round(tt, 2), a, b)
                              for tt, a, b in beh.transitions]
        res['n_transitions'] = len(beh.transitions)
        res['final_state'] = beh.state
        res['overtook'] = any(b == OVERTAKE for _, _, b in beh.transitions)
    return res


def main():
    rl = load_raceline(RL)
    rx, ry = rl[0], rl[1]
    s_arr, total = sp.arc_lengths(rx, ry)
    gm = GridMap.load(MAP_YAML) if os.path.exists(MAP_YAML) else None
    out = {}

    clean = race('clean', rl, 0.0, 0.0)
    out['clean'] = clean

    # ── rolling scenarios: opponent 6 m ahead at 40/60/75 % of the profile ──
    for frac in (0.40, 0.60, 0.75):
        key = f'opp_{int(frac * 100)}pct'
        out[key] = {}
        for mode in ('behavior', 'aeb', 'spliner'):
            r = race(mode, rl, frac, 6.0, grid_map=gm)
            r['time_lost'] = (round(r['lap_time'] - clean['lap_time'], 2)
                              if r['completed'] and not r['collided'] else None)
            if mode == 'aeb' and not r['passed']:
                r['outcome'] = 'stuck behind opponent (DNF as a pass)'
            out[key][mode] = r

    # ── narrow scenario: slow opponent inside a no-room section ─────────────
    # comp_track s ~ 215-221 m has < 0.30 m map clearance at +/-0.65 m on BOTH
    # sides — overtaking there is not feasible.  Ego starts just behind.
    narrow_s = 210.0
    ego_idx = int(np.searchsorted(s_arr, 200.0))
    clean_n = race('clean', rl, 0.0, 0.0, ego_start_idx=ego_idx)
    out['narrow'] = {'clean_lap_time': clean_n['lap_time'],
                     'note': 'plan_wall_min restricted to the blocked '
                             's=213-223 m stretch: inf/None = never planned '
                             'a deformation there'}
    for mode in ('behavior', 'spliner', 'aeb'):
        r = race(mode, rl, 0.35, narrow_s, ego_start_idx=ego_idx, grid_map=gm,
                 s_window=(213.0, 223.0))
        r['time_lost'] = (round(r['lap_time'] - clean_n['lap_time'], 2)
                          if r['completed'] and not r['collided'] else None)
        out['narrow'][mode] = r

    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
