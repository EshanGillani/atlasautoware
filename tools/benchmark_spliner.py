"""
Benchmark the Frenet spliner overtaker against the AEB-only status quo.
=======================================================================

Closed-loop laps on the competition raceline (tests/closed_loop.py harness,
MAP controller), no ROS.  Three scenarios:

  1. clean       — no opponent: baseline lap time;
  2. static      — opponent parked ON the raceline at 40 % of the lap.
                   Status quo (AEB only): the car stops behind it, DNF.
                   With the spliner: track the deformed line, report lap time
                   lost and minimum clearance to the opponent;
  3. moving      — opponent driving the raceline at 40 % of the local raceline
                   speed; the spline is re-planned from its current (s, d)
                   every sim step while it is inside the engagement horizon.

Both cars are treated as POINTS: "clearance" is centre-to-centre distance,
so real vehicles need evasion_dist > (half-widths + margin).  Prints JSON.

    python3 tools/benchmark_spliner.py
"""

import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))
sys.path.insert(0, os.path.join(REPO, 'tests'))
from closed_loop import load_raceline, run_lap                    # noqa: E402
from map_controller import MAPController, build_lat_accel_lut     # noqa: E402
import spliner as sp                                              # noqa: E402

RL = os.path.join(REPO, 'racelines', 'comp_raceline.csv')
DT = 0.02                                                  # harness step
A_BRAKE = 8.0                                              # harness brake limit


def main():
    rx, ry, rh, rc, rv = load_raceline(RL)
    s, total = sp.arc_lengths(rx, ry)
    lut = build_lat_accel_lut()
    oi = int(0.4 * len(rx))                                # opponent @ 40 % lap
    ox, oy = float(rx[oi]), float(ry[oi])
    out = {}

    # ── 1. clean lap ─────────────────────────────────────────────────────────
    ctl = MAPController(lut=lut)
    ctl.set_raceline(rx, ry, rv)
    clean = run_lap(ctl.control, rx, ry, rh)
    out['clean'] = dict(completed=clean['completed'],
                        lap_time=round(clean['lap_time'], 2),
                        xte_mean=round(clean['xte_mean'], 3))

    # ── 2a. status quo: AEB only, static opponent on the line ───────────────
    ctl = MAPController(lut=lut)
    ctl.set_raceline(rx, ry, rv)
    aeb_min = [float('inf')]

    def aeb_control(px, py, yaw, v, j):
        steer, v_t = ctl.control(px, py, yaw, v, j)
        dx, dy = ox - px, oy - py
        dist = math.hypot(dx, dy)
        aeb_min[0] = min(aeb_min[0], dist)
        ahead = dx * math.cos(yaw) + dy * math.sin(yaw) > 0.0
        if ahead and dist < v * v / (2.0 * A_BRAKE) + 0.6:  # brake + margin
            v_t = 0.0
        return steer, v_t

    res = run_lap(aeb_control, rx, ry, rh, max_steps=6000)
    out['static_aeb_baseline'] = dict(
        completed=res['completed'],
        outcome=('collided' if aeb_min[0] < 0.05 else
                 'stopped behind opponent (DNF)'),
        stop_distance=round(aeb_min[0], 3))

    # ── 2b. spliner, static opponent ─────────────────────────────────────────
    opp_s, opp_d = sp.cartesian_to_frenet(ox, oy, rx, ry, s, total)
    side = sp.choose_side(rx, ry, opp_s)
    mx, my, mv = sp.plan_overtake((rx, ry, rv), opp_s, opp_d, side,
                                  ego_speed=float(rv[oi]))
    ctl = MAPController(lut=lut)
    ctl.set_raceline(mx, my, mv)
    clr = [float('inf')]

    def static_control(px, py, yaw, v, j):
        clr[0] = min(clr[0], math.hypot(px - ox, py - oy))
        return ctl.control(px, py, yaw, v, j)

    res = run_lap(static_control, mx, my, rh)
    out['static_spliner'] = dict(
        completed=res['completed'], side=side,
        lap_time=round(res['lap_time'], 2),
        time_lost=round(res['lap_time'] - clean['lap_time'], 2),
        min_clearance=round(clr[0], 3),
        clearance_ok=bool(clr[0] >= 0.3))

    # ── 3. spliner, moving opponent (40 % of local raceline speed) ──────────
    ctl = MAPController(lut=lut)
    ctl.set_raceline(rx, ry, rv)
    state = {'s': float(s[oi]), 'clr': float('inf'), 'side': None}

    def moving_control(px, py, yaw, v, j):
        k = int(np.searchsorted(s, state['s'], side='right') - 1)
        state['s'] = (state['s'] + 0.4 * float(rv[k]) * DT) % total
        o_x, o_y = sp.frenet_to_cartesian(state['s'], 0.0, rx, ry, s, total)
        state['clr'] = min(state['clr'], math.hypot(px - o_x, py - o_y))
        ego_s, _ = sp.cartesian_to_frenet(px, py, rx, ry, s, total)
        rel = (state['s'] - ego_s + total / 2.0) % total - total / 2.0
        if -7.0 < rel < 12.0:                  # engagement horizon: re-plan
            if state['side'] is None:          # lock the side once committed
                state['side'] = sp.choose_side(rx, ry, state['s'])
            ctl.set_raceline(*sp.plan_overtake(
                (rx, ry, rv), state['s'], 0.0, state['side'], ego_speed=v))
        else:
            ctl.set_raceline(rx, ry, rv)
        return ctl.control(px, py, yaw, v, j)

    res = run_lap(moving_control, rx, ry, rh)
    out['moving_spliner'] = dict(
        completed=res['completed'], side=state['side'],
        lap_time=round(res['lap_time'], 2),
        time_lost=round(res['lap_time'] - clean['lap_time'], 2),
        min_clearance=round(state['clr'], 3),
        clearance_ok=bool(state['clr'] >= 0.3))

    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
