"""
Shared closed-loop lap harness — the one benchmark every controller runs.
=========================================================================

Kinematic-bicycle plant at 50 Hz lapping a raceline CSV, starting with a
deliberate pose error.  Used by the unit tests and by candidate-feature
benchmarks so "is it actually faster/tighter?" is always answered with the
same metric: perpendicular cross-track error after a settle period, plus lap
time.  No ROS, no hardware.

    from closed_loop import load_raceline, run_lap
    result = run_lap(my_control_fn, rx, ry, rh)
    # control_fn(px, py, yaw, v, nearest_idx) -> (steer, v_target)
"""

import csv
import math

import numpy as np


def load_raceline(path):
    """CSV -> (x, y, heading, curvature, speed) arrays."""
    cols = {k: [] for k in ('x', 'y', 'heading', 'curvature', 'speed')}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k in cols:
                cols[k].append(float(row[k]))
    return tuple(np.array(cols[k]) for k in cols)


def cross_track(px, py, rx, ry, j):
    """Perpendicular distance from (px, py) to the raceline at index j."""
    n = len(rx)
    tx = rx[(j + 1) % n] - rx[(j - 1) % n]
    ty = ry[(j + 1) % n] - ry[(j - 1) % n]
    tn = math.hypot(tx, ty) + 1e-9
    return abs((px - rx[j]) * (-ty / tn) + (py - ry[j]) * (tx / tn))


def run_lap(control_fn, rx, ry, rh, wheelbase=0.33,
            start_offset=(0.3, -0.2, 0.0), v0=2.0, dt=0.02,
            max_steps=12000, settle=100,
            a_accel=4.0, a_brake=8.0, actuator_delay=0.0):
    """One lap under control_fn; returns metrics + traces.

    control_fn(px, py, yaw, v, nearest_idx) -> (steer, v_target).  The plant
    is the kinematic bicycle; speed tracks v_target under accel/brake limits.
    `actuator_delay` (s) holds each command in a FIFO before the plant sees
    it — the sensor-to-actuator latency of a real car (serial links, servo,
    ESC ramping), for testing delay compensation.
    Returns dict with completed, lap_time, xte (full trace), idx (nearest
    raceline index per step), xte_mean / xte_max (post-settle).
    """
    n = len(rx)
    px = float(rx[0]) + start_offset[0]
    py = float(ry[0]) + start_offset[1]
    yaw = float(rh[0]) + start_offset[2]
    v = float(v0)
    prev_j = cum = 0
    t = 0.0
    xte, idx = [], []
    delay_steps = int(round(actuator_delay / dt))
    pipeline = [(0.0, v0)] * delay_steps               # commands in flight
    for _ in range(max_steps):
        j = int(np.argmin((rx - px) ** 2 + (ry - py) ** 2))
        d = j - prev_j
        if d < -n / 2:
            d += n
        if 0 < d < n / 2:
            cum += d
        prev_j = j
        cmd = control_fn(px, py, yaw, v, j)
        if delay_steps:
            pipeline.append(cmd)
            cmd = pipeline.pop(0)
        steer, v_t = cmd
        a = float(np.clip((v_t - v) / dt, -a_brake, a_accel))
        px += v * math.cos(yaw) * dt
        py += v * math.sin(yaw) * dt
        yaw += v * math.tan(float(steer)) / wheelbase * dt
        v = max(0.0, v + a * dt)
        t += dt
        xte.append(cross_track(px, py, rx, ry, j))
        idx.append(j)
        if cum >= n:
            break
    xte = np.array(xte)
    steady = xte[settle:] if len(xte) > settle else xte
    return dict(completed=cum >= n, lap_time=t, xte=xte,
                idx=np.array(idx),
                xte_mean=float(steady.mean()) if len(steady) else float('inf'),
                xte_max=float(steady.max()) if len(steady) else float('inf'))
