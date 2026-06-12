#!/usr/bin/env python3
"""
Benchmark: particle-filter localization over a full competition lap.
====================================================================

Simulates the car driving the comp raceline (poses interpolated from the
raceline points + heading at 10 Hz), synthesizes 360-beam lidar scans with
grid_raycast + Gaussian range noise (sigma 0.05 m) + 5% random dropout, and
feeds the PF a motion prior built from the true twist corrupted with a 5%
speed bias + gyro noise/bias — emulating the velocity EKF output.

Reports, over 5 seeds: mean / p95 / max position error, mean / p95 yaw
error, divergence count, and wall-clock per update() (10 Hz budget: the
target is < 10 ms at 1500 particles x 18 beams).  Then a kidnapped-robot
check: global init, time to converge under 0.5 m.

NOTE the circularity: the benchmark scans are synthesized by the very same
distance-field raycaster the likelihood field is built from, so these
numbers are an upper bound on map-consistency, not on real-lidar accuracy.

    python3 tools/benchmark_pf.py
"""

import math
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))
sys.path.insert(0, os.path.join(REPO, 'tests'))

from grid_map import GridMap, grid_raycast            # noqa: E402
from particle_filter import ParticleFilter            # noqa: E402
from closed_loop import load_raceline                 # noqa: E402

DT = 0.1                                              # 10 Hz scans
N_BEAMS = 360
ANGLE_MIN = -math.pi
ANGLE_INC = 2.0 * math.pi / N_BEAMS
MAX_RANGE = 12.0
RANGE_SIGMA = 0.05
DROPOUT = 0.05
SPEED_BIAS = 1.05                                     # 5% fast (EKF-ish)
GYRO_SIGMA = 0.02                                     # rad/s white noise
GYRO_BIAS = 0.005                                     # rad/s
SEEDS = [0, 1, 2, 3, 4]


def lap_trajectory(rx, ry, rh, rv, dt=DT):
    """Resample the raceline at dt by integrating the speed profile.

    Returns t, x, y, yaw, vx (body speed), omega (yaw rate) — one lap."""
    dx = np.diff(np.r_[rx, rx[0]])
    dy = np.diff(np.r_[ry, ry[0]])
    ds = np.hypot(dx, dy)
    s = np.r_[0.0, np.cumsum(ds)]                     # s[-1] = lap length
    h = np.unwrap(np.r_[rh, rh[0]])
    xs, ys, vs = np.r_[rx, rx[0]], np.r_[ry, ry[0]], np.r_[rv, rv[0]]
    out_t, out = [], []
    si, t = 0.0, 0.0
    while si < s[-1]:
        x = np.interp(si, s, xs)
        y = np.interp(si, s, ys)
        yaw = np.interp(si, s, h)
        v = np.interp(si, s, vs)
        out_t.append(t)
        out.append((x, y, yaw, v))
        si += v * dt
        t += dt
    x, y, yaw, v = map(np.array, zip(*out))
    omega = np.gradient(yaw, dt)
    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
    return np.array(out_t), x, y, yaw, v, omega


def synth_scans(gm, x, y, yaw):
    """Clean 360-beam scan per pose (slow sphere tracing — computed once)."""
    body = ANGLE_MIN + np.arange(N_BEAMS) * ANGLE_INC
    scans = np.empty((len(x), N_BEAMS))
    for i in range(len(x)):
        scans[i] = grid_raycast(gm, x[i], y[i], yaw[i] + body, MAX_RANGE)
    return scans


def ang_err(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def run_lap(gm, traj, scans, seed, n_particles=1500, global_init=False):
    t, x, y, yaw, v, omega = traj
    rng = np.random.default_rng(seed + 1000)
    pf = ParticleFilter(gm, n_particles=n_particles, seed=seed)
    if not global_init:
        pf.initialize(x[0], y[0], yaw[0], spread=0.3)
    pos_err, yaw_err, upd_times = [], [], []
    for i in range(len(t)):
        # motion prior: true twist, biased + noisy (emulated EKF)
        vx = v[i] * SPEED_BIAS + rng.normal(0.0, 0.05)
        wz = omega[i] + GYRO_BIAS + rng.normal(0.0, GYRO_SIGMA)
        pf.predict(vx, 0.0, wz, DT)
        # measurement: noisy scan with dropouts (dropped -> max range)
        scan = scans[i] + rng.normal(0.0, RANGE_SIGMA, N_BEAMS)
        scan[rng.random(N_BEAMS) < DROPOUT] = MAX_RANGE
        t0 = time.perf_counter()
        pf.update(scan, ANGLE_MIN, ANGLE_INC, subsample=20)
        upd_times.append(time.perf_counter() - t0)
        ex_, ey_, eyaw, _ = pf.pose()
        pos_err.append(math.hypot(ex_ - x[i], ey_ - y[i]))
        yaw_err.append(ang_err(eyaw, yaw[i]))
    return np.array(pos_err), np.array(yaw_err), np.array(upd_times)


def main():
    gm = GridMap.load(os.path.join(REPO, 'maps', 'comp_track.yaml'))
    rx, ry, rh, _, rv = load_raceline(
        os.path.join(REPO, 'racelines', 'comp_raceline.csv'))
    traj = lap_trajectory(rx, ry, rh, rv)
    t = traj[0]
    print(f'lap: {t[-1]:.1f} s, {len(t)} scan steps at {1/DT:.0f} Hz '
          f'({N_BEAMS} beams, subsample 20 -> 18 used)')
    print('synthesizing clean scans (grid_raycast, once)...')
    scans = synth_scans(gm, traj[1], traj[2], traj[3])

    settle = int(2.0 / DT)                            # ignore first 2 s
    print(f"\n{'seed':>4} {'pos_mean':>9} {'pos_p95':>8} {'pos_max':>8} "
          f"{'yaw_mean':>9} {'yaw_p95':>8} {'ms/upd':>7} {'diverged':>8}")
    print('-' * 66)
    div = 0
    all_pos, all_yaw, all_ms = [], [], []
    for seed in SEEDS:
        pe, ye, ut = run_lap(gm, traj, scans, seed)
        pe_s, ye_s = pe[settle:], ye[settle:]
        d = bool(np.any(pe_s > 1.5))
        div += d
        all_pos.append(pe_s)
        all_yaw.append(ye_s)
        all_ms.append(ut)
        print(f'{seed:>4} {pe_s.mean():>9.3f} {np.percentile(pe_s, 95):>8.3f} '
              f'{pe_s.max():>8.3f} {math.degrees(ye_s.mean()):>8.2f}d '
              f'{math.degrees(np.percentile(ye_s, 95)):>7.2f}d '
              f'{1e3 * ut.mean():>7.2f} {str(d):>8}')
    pe = np.concatenate(all_pos)
    ye = np.concatenate(all_yaw)
    ut = np.concatenate(all_ms)
    print('-' * 66)
    print(f' all {pe.mean():>9.3f} {np.percentile(pe, 95):>8.3f} '
          f'{pe.max():>8.3f} {math.degrees(ye.mean()):>8.2f}d '
          f'{math.degrees(np.percentile(ye, 95)):>7.2f}d '
          f'{1e3 * ut.mean():>7.2f} {div:>7}/5')
    print(f'update() wall-clock: mean {1e3 * ut.mean():.2f} ms, '
          f'p95 {1e3 * np.percentile(ut, 95):.2f} ms, '
          f'max {1e3 * ut.max():.2f} ms '
          f'(budget 100 ms at 10 Hz; target < 10 ms)')

    # ── kidnapped robot: global init, time to converge ───────────────────────
    print('\nkidnapped-robot check (global init, 3000 particles):')
    for seed in SEEDS:
        pe, ye, _ = run_lap(gm, traj, scans, seed,
                            n_particles=3000, global_init=True)
        conv = None
        hold = int(1.0 / DT)
        good = pe < 0.5
        for i in range(len(pe) - hold):
            if good[i:i + hold].all():
                conv = t[i]
                break
        tail = pe[-hold:].mean()
        print(f'  seed {seed}: '
              + (f'converged at t={conv:.1f} s' if conv is not None
                 else 'DID NOT CONVERGE')
              + f' (final-1s mean err {tail:.3f} m)')


if __name__ == '__main__':
    main()
