#!/usr/bin/env python3
"""
Benchmark: safety supervisor against scripted fault scenarios.
==============================================================

Drives the pure HealthMonitor with synthetic message timelines (jittered
rates, +-20% uniform on every period) and injected faults, and reports
detection latency, recovery time and false-alarm counts per scenario:

  (a) lidar dies mid-run            -> EMERGENCY within 1 s
  (b) PF covariance balloons        -> DEGRADED + speed scale; recovers with
      hysteresis; an oscillation around the threshold must NOT flap
  (c) sustained slip                -> DEGRADED (brief blips ignored);
      never-clearing slip escalates -> EMERGENCY
  (d) one hour fully healthy        -> zero false alarms

This candidate is about correctness, not lap time: the numbers that matter
are latency (seconds from fault injection to the matching state) and the
false-alarm count (updates in a non-OK state while the system is healthy).

    python3 tools/benchmark_supervisor.py
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))

from supervisor import HealthMonitor, policy, OK, DEGRADED, EMERGENCY  # noqa: E402

UPDATE_HZ = 50.0                                      # supervisor tick rate
JITTER = 0.20                                         # +-20% on every period
# nominal publish rates (Hz) — comfortably above the configured minimums
NOMINAL = {'scan': 10.0, 'pose': 20.0, 'imu': 100.0, 'drive': 50.0}
RATES = {'scan': (5.0, EMERGENCY), 'pose': (5.0, EMERGENCY),
         'imu': (50.0, DEGRADED), 'drive': (10.0, EMERGENCY)}


def channel_times(hz, t_end, rng, t_start=0.0):
    """Jittered message timestamps: period * U(1-J, 1+J), cumulative."""
    n = int(t_end * hz * 1.5) + 16
    dt = (1.0 / hz) * rng.uniform(1.0 - JITTER, 1.0 + JITTER, n)
    t = t_start + np.cumsum(dt)
    return t[t < t_end]


def make_monitor():
    return HealthMonitor(rates=RATES, cov_threshold=0.5,
                         slip_degraded_after=0.5, slip_emergency_after=3.0,
                         recover_after=2.0, startup_grace=5.0)


def run(hm, t_end, beats, reports=()):
    """Feed time-ordered beats/reports + 50 Hz updates; return state trace.

    beats:   {channel: array of timestamps}
    reports: iterable of (channel, value_fn(t), array of timestamps)
    Returns (tick_times, states) with states as strings per update tick.
    """
    events = [(t, 0, ch, None) for ch, ts in beats.items() for t in ts]
    events += [(t, 0, ch, fn) for ch, fn, ts in reports for t in ts]
    events += [(t, 1, None, None)
               for t in np.arange(0.0, t_end, 1.0 / UPDATE_HZ)]
    events.sort(key=lambda e: (e[0], e[1]))
    tick_t, states = [], []
    for t, kind, ch, fn in events:
        if kind == 0:
            if fn is None:
                hm.note(ch, t)
            else:
                hm.report(ch, fn(t), t)
        else:
            state, _ = hm.update(t)
            tick_t.append(t)
            states.append(state)
    return np.asarray(tick_t), states


def first_at(tick_t, states, target, after=0.0):
    for t, s in zip(tick_t, states):
        if t >= after and s == target:
            return t
    return None


def transitions(states):
    return sum(1 for a, b in zip(states, states[1:]) if a != b)


def false_alarms(tick_t, states, healthy_ranges):
    n = 0
    for t, s in zip(tick_t, states):
        if s != OK and any(a <= t < b for a, b in healthy_ranges):
            n += 1
    return n


def healthy_beats(rng, t_end):
    return {ch: channel_times(hz, t_end, rng) for ch, hz in NOMINAL.items()}


rows = []


def report(name, detect, recover, fa, note=''):
    rows.append((name,
                 '-' if detect is None else f'{detect:.2f} s',
                 '-' if recover is None else f'{recover:.2f} s',
                 str(fa), note))


# ── (a) lidar dies mid-run ───────────────────────────────────────────────────
def scenario_a():
    rng = np.random.default_rng(0)
    t_end, t_fault = 40.0, 30.0
    beats = healthy_beats(rng, t_end)
    beats['scan'] = beats['scan'][beats['scan'] < t_fault]
    tick_t, states = run(make_monitor(), t_end, beats)
    t_det = first_at(tick_t, states, EMERGENCY, after=t_fault)
    lat = None if t_det is None else t_det - t_fault
    fa = false_alarms(tick_t, states, [(0.0, t_fault)])
    ok = lat is not None and lat < 1.0 and fa == 0
    report('(a) lidar dies -> EMERGENCY', lat, None, fa,
           'PASS' if ok else 'FAIL')
    # policy check: EMERGENCY must drop the enable gate
    en, sc = policy(states[-1])
    print(f'  (a) detected {lat:.2f} s after last possible scan; '
          f'final policy enable={en} scale={sc}')


# ── (b) PF covariance balloons + oscillation (hysteresis) ────────────────────
def scenario_b():
    rng = np.random.default_rng(1)
    t_end = 110.0
    beats = healthy_beats(rng, t_end)
    bad0, bad1 = 30.0, 50.0                            # covariance balloon
    osc0, osc1 = 70.0, 90.0                            # rides the threshold

    def cov(t):
        if bad0 <= t < bad1:
            return 1.5
        if osc0 <= t < osc1:
            return 0.5 + 0.08 * np.sin(2.0 * np.pi * 1.0 * t)  # 0.42..0.58
        return 0.05

    tick_t, states = run(make_monitor(), t_end, beats,
                         reports=[('pf_cov', cov, beats['pose'])])
    t_det = first_at(tick_t, states, DEGRADED, after=bad0)
    lat = None if t_det is None else t_det - bad0
    t_rec = first_at(tick_t, states, OK, after=bad1)
    rec = None if t_rec is None else t_rec - bad1
    # flapping check inside the oscillation + its recovery tail
    win = [(t, s) for t, s in zip(tick_t, states) if osc0 <= t < osc1 + 5.0]
    n_tr = transitions([s for _, s in win])
    fa = false_alarms(tick_t, states,
                      [(0.0, bad0), (bad1 + 5.0, osc0)])
    emerg = sum(1 for s in states if s == EMERGENCY)
    ok = (lat is not None and lat < 1.0 and rec is not None
          and n_tr <= 2 and fa == 0 and emerg == 0)
    report('(b) PF cov balloons -> DEGRADED', lat, rec, fa,
           ('PASS' if ok else 'FAIL') + f' ({n_tr} transitions in oscillation)')
    print(f'  (b) degraded {lat:.2f} s after balloon, recovered '
          f'{rec:.2f} s after it cleared (hold 2.0 s); oscillation around '
          f'threshold: {n_tr} state transitions (enter+exit), no flapping; '
          f'policy scale while degraded: {policy(DEGRADED)[1]}')


# ── (c) sustained slip ───────────────────────────────────────────────────────
def scenario_c():
    rng = np.random.default_rng(2)
    t_end = 60.0
    beats = healthy_beats(rng, t_end)
    blip0, blip1 = 10.0, 10.2                          # brief: must be ignored
    sus0, sus1 = 30.0, 32.0                            # sustained: DEGRADED

    def slip(t):
        return (blip0 <= t < blip1) or (sus0 <= t < sus1)

    wheel = channel_times(50.0, t_end, rng)            # slip flag at EKF rate
    tick_t, states = run(make_monitor(), t_end, beats,
                         reports=[('slip', slip, wheel)])
    t_det = first_at(tick_t, states, DEGRADED, after=sus0)
    lat = None if t_det is None else t_det - sus0
    fa = false_alarms(tick_t, states, [(0.0, sus0)])   # incl. the brief blip
    emerg = sum(1 for s in states if s == EMERGENCY)
    ok = lat is not None and 0.5 <= lat < 1.0 and fa == 0 and emerg == 0
    report('(c) sustained slip -> DEGRADED', lat, None, fa,
           'PASS' if ok else 'FAIL')
    print(f'  (c) 2 s slip flagged {lat:.2f} s after onset '
          f'(threshold 0.5 s); 0.2 s blip ignored; no EMERGENCY')

    # escalation: slip that never clears trips the traction emergency
    hm = make_monitor()
    rng = np.random.default_rng(3)
    beats = healthy_beats(rng, 40.0)
    wheel = channel_times(50.0, 40.0, rng)
    tick_t, states = run(hm, 40.0, beats,
                         reports=[('slip', lambda t: t >= 20.0, wheel)])
    t_em = first_at(tick_t, states, EMERGENCY, after=20.0)
    lat = None if t_em is None else t_em - 20.0
    ok = lat is not None and 3.0 <= lat < 3.5
    report('(c2) never-clearing slip -> EMERGENCY', lat, None,
           false_alarms(tick_t, states, [(0.0, 20.0)]),
           'PASS' if ok else 'FAIL')


# ── (d) one healthy hour, jittered rates -> zero false alarms ────────────────
def scenario_d():
    rng = np.random.default_rng(4)
    t_end = 3600.0
    beats = healthy_beats(rng, t_end)

    def cov(t):                                        # realistic noisy cov
        return 0.05 + 0.03 * abs(np.sin(0.1 * t))

    tick_t, states = run(make_monitor(), t_end, beats,
                         reports=[('pf_cov', cov, beats['pose'])])
    fa = sum(1 for s in states if s != OK)
    report('(d) 1 h healthy, +-20% jitter', None, None, fa,
           'PASS' if fa == 0 else 'FAIL')
    print(f'  (d) {len(states)} updates over {t_end / 3600.0:.0f} h '
          f'(scan 10 Hz, pose 20 Hz, imu 100 Hz, drive 50 Hz, '
          f'+-{JITTER * 100:.0f}% jitter): {fa} false alarms')


def main():
    print('supervisor fault-scenario benchmark '
          f'(update {UPDATE_HZ:.0f} Hz, recover_after 2.0 s)\n')
    scenario_a()
    scenario_b()
    scenario_c()
    scenario_d()
    print(f"\n{'scenario':<38} {'detect':>8} {'recover':>8} "
          f"{'false alarms':>13}  result")
    print('-' * 84)
    for name, det, rec, fa, note in rows:
        print(f'{name:<38} {det:>8} {rec:>8} {fa:>13}  {note}')
    if any('FAIL' in r[4] for r in rows):
        sys.exit(1)


if __name__ == '__main__':
    main()
