"""
Supervisor tests — HealthMonitor scenarios + policy mapping.
============================================================

Synthetic-timeline validation of the system-level watchdog:
  - rate/staleness tracking: dead lidar -> EMERGENCY within 1 s, slow-but-
    alive channels caught by the windowed rate check, never-published
    channels only after the startup grace;
  - value monitors: PF covariance threshold (with trip/release hysteresis,
    no flapping on oscillation), sustained-slip detection with blip
    rejection and escalation to EMERGENCY;
  - state machine: severity precedence, recovery only after `recover_after`
    seconds of full health, no false alarms over a long healthy run with
    jittered rates;
  - policy: OK pass-through, DEGRADED speed scale, EMERGENCY disable.

    python3 -m pytest tests/test_supervisor.py -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'f1tenth_gym_ros'))
from supervisor import (HealthMonitor, policy,                   # noqa: E402
                        OK, DEGRADED, EMERGENCY, DEFAULT_RATES)

RATES = dict(DEFAULT_RATES)            # scan/pose 5, imu 50, drive 10 Hz min
NOMINAL = {'scan': 10.0, 'pose': 20.0, 'imu': 100.0, 'drive': 50.0}
TICK = 0.02                            # 50 Hz update


def make(**kw):
    args = dict(rates=RATES, cov_threshold=0.5, slip_degraded_after=0.5,
                slip_emergency_after=3.0, recover_after=2.0,
                startup_grace=5.0)
    args.update(kw)
    return HealthMonitor(**args)


def beat_times(hz, t0, t1, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out, t = [], t0
    while t < t1:
        out.append(t)
        t += (1.0 / hz) * (rng.uniform(1.0 - jitter, 1.0 + jitter)
                           if jitter else 1.0)
    return np.asarray(out)


def drive(hm, t0, t1, dead=(), reports=None, jitter=0.15, seed=0):
    """Run all healthy channels (except `dead`) + updates over [t0, t1).

    reports: optional list of (channel, value_fn).  Reports ride the pose
    beats ('pf_cov') / a 50 Hz wheel clock ('slip').
    Returns (tick_times, states)."""
    events = []
    for i, (ch, hz) in enumerate(NOMINAL.items()):
        if ch in dead:
            continue
        for t in beat_times(hz, t0, t1, jitter, seed + i):
            events.append((t, 0, ch, None))
    for ch, fn in (reports or []):
        hz = NOMINAL['pose'] if ch == 'pf_cov' else 50.0
        for t in beat_times(hz, t0, t1, jitter, seed + 7):
            events.append((t, 0, ch, fn))
    for t in np.arange(t0, t1, TICK):
        events.append((t, 1, None, None))
    events.sort(key=lambda e: (e[0], e[1]))
    ticks, states = [], []
    for t, kind, ch, fn in events:
        if kind == 0:
            if fn is None:
                hm.note(ch, t)
            else:
                hm.report(ch, fn(t), t)
        else:
            s, _ = hm.update(t)
            ticks.append(t)
            states.append(s)
    return np.asarray(ticks), states


def first_at(ticks, states, target, after=0.0):
    for t, s in zip(ticks, states):
        if t >= after and s == target:
            return t
    return None


# ─────────────────────────────────────────────────────────────────────────────
# rate / staleness
# ─────────────────────────────────────────────────────────────────────────────

def test_all_healthy_is_ok():
    hm = make()
    ticks, states = drive(hm, 0.0, 20.0)
    assert all(s == OK for s in states)
    assert hm.reasons == []


def test_dead_lidar_emergency_within_1s():
    hm = make()
    drive(hm, 0.0, 30.0)
    ticks, states = drive(hm, 30.0, 35.0, dead=('scan',))
    t_det = first_at(ticks, states, EMERGENCY)
    assert t_det is not None and t_det - 30.0 < 1.0
    assert any('scan' in r for r in hm.reasons)


def test_dead_pose_and_drive_are_emergency():
    for ch in ('pose', 'drive'):
        hm = make()
        drive(hm, 0.0, 10.0)
        ticks, states = drive(hm, 10.0, 14.0, dead=(ch,))
        assert states[-1] == EMERGENCY, ch
        assert any(ch in r for r in hm.reasons)


def test_stale_imu_is_only_degraded():
    hm = make()
    drive(hm, 0.0, 10.0)
    ticks, states = drive(hm, 10.0, 14.0, dead=('imu',))
    assert states[-1] == DEGRADED
    assert any('imu' in r for r in hm.reasons)
    assert EMERGENCY not in states


def test_slow_but_alive_channel_caught_by_rate_check():
    # 3 Hz scan: every gap is shorter than the staleness cutoff
    # (2 / 5 Hz = 0.4 s > 0.33 s) so only the windowed rate check can see it
    hm = make()
    events = [(t, 0, ch) for ch, hz in NOMINAL.items() if ch != 'scan'
              for t in beat_times(hz, 0.0, 20.0)]
    events += [(t, 0, 'scan') for t in beat_times(3.0, 0.0, 20.0)]
    events += [(t, 1, None) for t in np.arange(0.0, 20.0, TICK)]
    events.sort(key=lambda e: (e[0], e[1]))
    last = None
    for t, kind, ch in events:
        if kind == 0:
            hm.note(ch, t)
        else:
            last, _ = hm.update(t)
    assert last == EMERGENCY
    assert any('rate' in r and 'scan' in r for r in hm.reasons)


def test_exactly_minimum_rate_is_legal():
    # drive at exactly its 10 Hz minimum must not alarm (one-beat tolerance)
    hm = make()
    events = [(t, 0, ch) for ch, hz in NOMINAL.items() if ch != 'drive'
              for t in beat_times(hz, 0.0, 20.0)]
    events += [(t, 0, 'drive') for t in beat_times(10.0, 0.0, 20.0)]
    events += [(t, 1, None) for t in np.arange(0.0, 20.0, TICK)]
    events.sort(key=lambda e: (e[0], e[1]))
    states = []
    for t, kind, ch in events:
        if kind == 0:
            hm.note(ch, t)
        else:
            states.append(hm.update(t)[0])
    assert all(s == OK for s in states)


def test_startup_grace_for_silent_channels():
    hm = make()
    # only updates, no messages at all: quiet through the grace window…
    for t in np.arange(0.0, 4.9, TICK):
        assert hm.update(t)[0] == OK
    # …then every silent channel faults
    state, reasons = hm.update(6.0)
    assert state == EMERGENCY
    assert len(reasons) == len(RATES)


def test_channel_alive_then_dead_needs_no_grace():
    hm = make()
    drive(hm, 0.0, 1.0)                  # alive well inside the grace window
    ticks, states = drive(hm, 1.0, 3.0, dead=('scan',))
    assert states[-1] == EMERGENCY       # death detected during grace


# ─────────────────────────────────────────────────────────────────────────────
# value monitors
# ─────────────────────────────────────────────────────────────────────────────

def test_pf_covariance_degrades_and_recovers_with_hold():
    hm = make()
    cov = lambda t: 1.5 if 10.0 <= t < 15.0 else 0.05      # noqa: E731
    ticks, states = drive(hm, 0.0, 25.0, reports=[('pf_cov', cov)])
    t_deg = first_at(ticks, states, DEGRADED)
    assert t_deg is not None and 10.0 <= t_deg < 10.5
    t_rec = first_at(ticks, states, OK, after=15.0)
    # recovery only after recover_after (2 s) of clean health
    assert t_rec is not None and 16.9 <= t_rec < 17.5
    assert EMERGENCY not in states


def test_pf_covariance_oscillation_does_not_flap():
    hm = make()
    osc = lambda t: (0.5 + 0.08 * np.sin(8.0 * t)          # noqa: E731
                     if 10.0 <= t < 20.0 else 0.05)
    ticks, states = drive(hm, 0.0, 25.0, reports=[('pf_cov', osc)])
    n_tr = sum(1 for a, b in zip(states, states[1:]) if a != b)
    assert n_tr == 2                     # one entry into DEGRADED, one exit
    assert DEGRADED in states and EMERGENCY not in states


def test_pf_covariance_release_threshold_is_hysteretic():
    hm = make()
    t = 0.0
    hm.report('pf_cov', 0.6, t)          # trips (> 0.5)
    assert hm._cov_bad
    hm.report('pf_cov', 0.4, t)          # below trip but above release (0.3)
    assert hm._cov_bad
    hm.report('pf_cov', 0.2, t)          # below release: clears
    assert not hm._cov_bad


def test_brief_slip_blip_ignored():
    hm = make()
    blip = lambda t: 10.0 <= t < 10.2                       # noqa: E731
    ticks, states = drive(hm, 0.0, 15.0, reports=[('slip', blip)])
    assert all(s == OK for s in states)


def test_sustained_slip_degrades():
    hm = make()
    slip = lambda t: 10.0 <= t < 12.0                       # noqa: E731
    ticks, states = drive(hm, 0.0, 20.0, reports=[('slip', slip)])
    t_deg = first_at(ticks, states, DEGRADED)
    assert t_deg is not None and 10.5 <= t_deg < 11.0       # after 0.5 s
    assert EMERGENCY not in states
    assert first_at(ticks, states, OK, after=12.0) is not None


def test_never_clearing_slip_escalates_to_emergency():
    hm = make()
    slip = lambda t: t >= 10.0                              # noqa: E731
    ticks, states = drive(hm, 0.0, 16.0, reports=[('slip', slip)])
    t_deg = first_at(ticks, states, DEGRADED)
    t_em = first_at(ticks, states, EMERGENCY)
    assert t_deg is not None and t_em is not None and t_deg < t_em
    assert 13.0 <= t_em < 13.6           # slip_emergency_after = 3 s
    assert any('traction' in r for r in hm.reasons)


# ─────────────────────────────────────────────────────────────────────────────
# state machine / hysteresis / robustness
# ─────────────────────────────────────────────────────────────────────────────

def test_emergency_takes_precedence_over_degraded():
    hm = make()
    cov = lambda t: 2.0 if t >= 5.0 else 0.05               # noqa: E731
    drive(hm, 0.0, 8.0, reports=[('pf_cov', cov)])
    assert hm.state == DEGRADED
    ticks, states = drive(hm, 8.0, 11.0, dead=('scan',),
                          reports=[('pf_cov', cov)])
    assert states[-1] == EMERGENCY


def test_recovery_requires_sustained_health():
    hm = make(recover_after=2.0)
    drive(hm, 0.0, 10.0)
    drive(hm, 10.0, 12.0, dead=('scan',))                   # EMERGENCY
    ticks, states = drive(hm, 12.0, 16.0)                   # scan returns
    assert states[0] == EMERGENCY                           # still held
    t_ok = first_at(ticks, states, OK)
    assert t_ok is not None and t_ok - 12.0 >= 2.0 - TICK
    assert hm.state == OK and hm.reasons == []


def test_recovering_reason_reported_during_hold():
    hm = make()
    drive(hm, 0.0, 10.0)
    drive(hm, 10.0, 12.0, dead=('scan',))
    # scan back: rate rebuilds by ~12.9, hold runs to ~14.9 — at 14 s the
    # alarm is still held and the reason says so
    drive(hm, 12.0, 14.0)
    assert hm.state == EMERGENCY
    assert any('recovering' in r for r in hm.reasons)


def test_no_false_alarms_long_jittered_run():
    hm = make()
    cov = lambda t: 0.05 + 0.03 * abs(np.sin(0.1 * t))      # noqa: E731
    ticks, states = drive(hm, 0.0, 300.0, jitter=0.2,
                          reports=[('pf_cov', cov)])
    assert sum(1 for s in states if s != OK) == 0


def test_unknown_channel_ignored():
    hm = make()
    hm.note('nonsense', 0.0)             # must not raise or create state
    assert 'nonsense' not in hm._beats


# ─────────────────────────────────────────────────────────────────────────────
# policy
# ─────────────────────────────────────────────────────────────────────────────

def test_policy_mapping():
    assert policy(OK) == (True, 1.0)
    assert policy(DEGRADED) == (True, 0.4)
    assert policy(DEGRADED, degraded_scale=0.25) == (True, 0.25)
    assert policy(EMERGENCY) == (False, 0.0)


def test_module_imports_without_ros():
    # the pure core (and the module itself) must not require rclpy
    import supervisor
    assert not hasattr(supervisor, 'rclpy')
