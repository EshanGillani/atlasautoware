"""
Race-behavior state machine tests — ForzaETH GB_TRACK/TRAILING/OVERTAKE.
========================================================================

All on the real competition raceline, no ROS:
  - no opponent = GB_TRACK forever, raceline untouched;
  - hysteresis: a noisy gap oscillating across the engage range causes at
    most one committed transition (no chatter);
  - TRAILING holds the speed-dependent desired gap in a 1-D follow sim;
  - a full two-car closed loop overtakes cleanly (GB -> TRAIL -> OVERTAKE ->
    GB, min clearance >= 0.3 m) and rejoins the raceline;
  - behavior wrapped around the controller with no opponent is bit-identical
    to the plain controller over a lap;
  - the map room-check rejects the genuinely narrow sections of comp_track;
  - spliner hold_dist: apex offset held past the opponent, default unchanged.

    python3 -m pytest tests/test_race_behavior.py -q
"""

import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))
sys.path.insert(0, os.path.join(REPO, 'tools'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from closed_loop import load_raceline, run_lap                     # noqa: E402
from map_controller import MAPController, build_lat_accel_lut      # noqa: E402
from grid_map import GridMap                                       # noqa: E402
import spliner as sp                                               # noqa: E402
from race_behavior import (RaceBehavior, GB_TRACK, TRAILING,       # noqa: E402
                           OVERTAKE)

RL = load_raceline(os.path.join(REPO, 'racelines', 'comp_raceline.csv'))
RX, RY, RH, RC, RV = RL
S, TOTAL = sp.arc_lengths(RX, RY)
DT = 0.02
LUT = build_lat_accel_lut()


def _beh(**kw):
    return RaceBehavior((RX, RY, RV), **kw)


# ─────────────────────────────────────────────────────────────────────────────
# State machine + hysteresis
# ─────────────────────────────────────────────────────────────────────────────

def test_no_opponent_stays_gb_track():
    beh = _beh()
    for _ in range(200):
        out = beh.update(RX[0], RY[0], 3.0, 0, None, DT)
        assert out['state'] == GB_TRACK
        assert out['line'][0] is beh.rx                # raceline, untouched
        assert not out['line_changed']
        assert out['speed_cap'] == float('inf')
    assert beh.transitions == []


def test_engage_needs_hysteresis_time():
    beh = _beh(d_max=0.0)                              # overtake impossible
    ego_s = float(S[0])
    opp = (ego_s + 5.0, 0.0, 1.0)
    out = beh.update(RX[0], RY[0], 2.0, 0, opp, DT)
    assert out['state'] == GB_TRACK                    # not after one tick
    for _ in range(int(0.2 / DT)):
        out = beh.update(RX[0], RY[0], 2.0, 0, opp, DT)
    assert out['state'] == TRAILING
    assert len(beh.transitions) == 1
    assert beh.transitions[0][0] >= beh.hyst[TRAILING] - 1e-9


def test_no_chatter_on_noisy_gap():
    # gap oscillates +/-0.8 m around the engage range with measurement noise:
    # the raw condition flips constantly, the machine commits at most twice
    beh = _beh(d_max=0.0)
    rng = np.random.default_rng(3)
    ego_s = float(S[0])
    raw_flips, prev_raw = 0, None
    for i in range(int(12.0 / DT)):
        t = i * DT
        gap = beh.trail_range + 0.6 * math.sin(2.0 * math.pi * 1.5 * t) \
            + rng.normal(0.0, 0.25)
        raw = gap < beh.trail_range
        if prev_raw is not None and raw != prev_raw:
            raw_flips += 1
        prev_raw = raw
        beh.update(RX[0], RY[0], 2.0, 0, (ego_s + gap, 0.0, 1.0), DT)
    assert raw_flips > 20                              # the input IS nasty
    assert len(beh.transitions) <= 2                   # the output is not


# ─────────────────────────────────────────────────────────────────────────────
# TRAILING gap controller (1-D follow sim along the raceline)
# ─────────────────────────────────────────────────────────────────────────────

def test_trailing_holds_desired_gap():
    beh = _beh(d_max=0.0)                              # never overtake
    v_opp = 1.5
    ego_s, opp_s = 0.0, 6.0
    v = 3.0
    gaps = []
    for i in range(int(25.0 / DT)):
        opp_s = (opp_s + v_opp * DT) % TOTAL
        px, py = sp.frenet_to_cartesian(ego_s, 0.0, RX, RY, S, TOTAL)
        j = int(np.searchsorted(S, ego_s, side='right') - 1)
        out = beh.update(px, py, v, j, (opp_s, 0.0, v_opp), DT)
        v_des = min(float(RV[j]), out['speed_cap'])
        v += float(np.clip((v_des - v), -8.0 * DT, 4.0 * DT))
        v = max(0.0, v)
        ego_s = (ego_s + v * DT) % TOTAL
        gap = beh.wrap(opp_s - ego_s)
        if i * DT > 15.0:
            gaps.append((gap, beh.gap_min + beh.t_gap * v))
    assert beh.state == TRAILING
    gap_err = [abs(g - gd) for g, gd in gaps]
    assert max(gap_err) < 0.4                          # holds gap_des +/-0.4 m
    assert min(g for g, _ in gaps) >= gaps[0][1] - 0.5  # never tailgates


# ─────────────────────────────────────────────────────────────────────────────
# Full two-car closed loop: overtake, clear, rejoin
# ─────────────────────────────────────────────────────────────────────────────

def test_overtake_clears_and_rejoins():
    from benchmark_behavior import race
    r = race('behavior', RL, 0.4, 6.0)
    assert r['completed'] and not r['collided']
    assert r['passed'] and r['min_clearance'] >= 0.3   # clean pass
    targets = [b for _, _, b in r['transitions']]
    assert targets[0] == TRAILING                      # GB -> TRAIL first
    assert OVERTAKE in targets                         # then the pass
    assert targets[-1] == GB_TRACK                     # rejoined at the end
    assert r['final_state'] == GB_TRACK
    assert r['n_transitions'] <= 6                     # no chatter


def test_disabled_is_bit_identical():
    ctl = MAPController(lut=LUT)
    ctl.set_raceline(RX, RY, RV)
    base = run_lap(ctl.control, RX, RY, RH)

    ctl2 = MAPController(lut=LUT)
    ctl2.set_raceline(RX, RY, RV)
    beh = _beh()

    def wrapped(px, py, yaw, v, j):                    # node wiring, no opp
        out = beh.update(px, py, v, j, None, DT)
        if out['line_changed']:                        # never fires
            ctl2.set_raceline(*out['line'])
        steer, v_t = ctl2.control(px, py, yaw, v, j)
        return steer, min(float(v_t), out['speed_cap'])

    res = run_lap(wrapped, RX, RY, RH)
    assert res['lap_time'] == base['lap_time']
    assert np.array_equal(res['xte'], base['xte'])
    assert np.array_equal(res['idx'], base['idx'])


# ─────────────────────────────────────────────────────────────────────────────
# Overtake feasibility (map room check + pace check)
# ─────────────────────────────────────────────────────────────────────────────

def test_map_check_blocks_narrow_section():
    gm = GridMap.load(os.path.join(REPO, 'maps', 'comp_track.yaml'))
    beh = _beh(grid_map=gm)
    # comp_track s ~ 215-221 m: < 0.30 m clearance at +/-0.65 m on BOTH sides
    feas, side = beh.overtake_feasible(217.0, 0.0, 0.5)
    assert not feas and side is None
    # and the lap is not all narrow: somewhere a side is open
    assert any(beh.overtake_feasible(float(S[i]), 0.0, 0.5)[0]
               for i in (10, 50, 150, 200))


def test_no_pace_advantage_means_trailing():
    beh = _beh()
    k = 50
    fast_opp = float(RV[k])                            # as fast as our profile
    assert not beh.overtake_feasible(float(S[k]), 0.0, fast_opp)[0]
    assert beh.overtake_feasible(float(S[k]), 0.0, 0.3 * fast_opp)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Spliner hold_dist extension
# ─────────────────────────────────────────────────────────────────────────────

def test_spliner_hold_dist_keeps_offset_past_opponent():
    oi = int(0.4 * len(RX))
    opp_s, opp_d = sp.cartesian_to_frenet(RX[oi], RY[oi], RX, RY, S, TOTAL)
    side = sp.choose_side(RX, RY, opp_s)
    hx, hy, _ = sp.plan_overtake((RX, RY, RV), opp_s, opp_d, side,
                                 hold_dist=3.0)
    rel = (S - opp_s + TOTAL / 2.0) % TOTAL - TOTAL / 2.0
    held = (rel >= 0.0) & (rel <= 3.0)
    off = np.hypot(hx - RX, hy - RY)
    assert off[held].min() > 0.55                      # apex offset held
    # default (hold_dist=0) is bit-identical to the original recipe
    a = sp.plan_overtake((RX, RY, RV), opp_s, opp_d, side)
    b = sp.plan_overtake((RX, RY, RV), opp_s, opp_d, side, hold_dist=0.0)
    for u, v in zip(a, b):
        assert np.array_equal(u, v)


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-q']))
