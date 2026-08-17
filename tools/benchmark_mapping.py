#!/usr/bin/env python3
"""
Benchmark: the SLAM mapping driver — BEFORE vs AFTER the smoothing/centering fix.
================================================================================

The mapping driver does autonomous exploratory laps so SLAM can build the
occupancy map that the racing-line optimizer needs.  Fewer, cleaner laps that
sweep both walls evenly = a faster, more accurate map.  This script measures
exactly that, deterministically and without ROS, by driving the *pure* control
law (f1tenth_gym_ros/mapping_driver.gap_follow_command) closed-loop in the
deterministic 2D raycast lidar simulator from tests/test_mapping_driver.py:

  * per-scan compute time          (the vectorization win)
  * lap completes without a crash  (min wall clearance > car half-width)
  * lap time + steering smoothness  (steer-rate std / max -> cleaner scans)
  * TRACK COVERAGE after 1 / 2 laps (the efficiency metric: higher = fewer laps)
  * determinism                    (identical metrics on a repeated run)

BEFORE = the original bare disparity-extender argmax follower (center_gain=0,
steer_lp=0).  AFTER = the shipped defaults (center-bias + steering low-pass).

    python3 tools/benchmark_mapping.py
    python3 tools/benchmark_mapping.py --map maps/comp_track.yaml --laps 2
"""

import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tests'))
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))

from mapping_driver import GapParams, gap_follow_command          # noqa: E402
from test_mapping_driver import run_mapping_lap, _sample_scans     # noqa: E402


def scan_compute_time(loops=200):
    """Mean per-scan wall time of the vectorized control law on a 1080-beam scan."""
    amin, inc, scans = _sample_scans()
    ranges = scans[3]                                  # the random-step scan
    p = GapParams()
    # warm up
    for _ in range(5):
        gap_follow_command(ranges, amin, inc, 0.0, p)
    t0 = time.perf_counter()
    prev = 0.0
    for _ in range(loops):
        prev, _ = gap_follow_command(ranges, amin, inc, prev, p)
    return (time.perf_counter() - t0) / loops * 1e3     # ms


def python_loop_compute_time(loops=200):
    """Mean per-scan time of the ORIGINAL per-beam Python loop (the baseline)."""
    from test_mapping_driver import original_scan_command
    amin, inc, scans = _sample_scans()
    ranges = scans[3]
    p = GapParams()
    for _ in range(5):
        original_scan_command(ranges, amin, inc, p)
    t0 = time.perf_counter()
    for _ in range(loops):
        original_scan_command(ranges, amin, inc, p)
    return (time.perf_counter() - t0) / loops * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default=os.path.join(REPO, 'maps', 'comp_track.yaml'))
    ap.add_argument('--laps', type=int, default=2)
    ap.add_argument('--num-beams', type=int, default=1080)
    args = ap.parse_args()

    before = GapParams(center_gain=0.0, steer_lp=0.0)   # original behavior
    after = GapParams()                                  # shipped defaults

    print('=== per-scan compute time (1080-beam scan) ===')
    t_loop = python_loop_compute_time()
    t_vec = scan_compute_time()
    print(f'  original Python for-loop : {t_loop:8.3f} ms / scan')
    print(f'  vectorized               : {t_vec:8.3f} ms / scan')
    print(f'  speedup                  : {t_loop / max(t_vec, 1e-9):8.1f}x')

    print(f'\n=== closed-loop mapping lap on {os.path.basename(args.map)} '
          f'({args.num_beams} beams, {args.laps} laps) ===')
    kw = dict(laps=args.laps, num_beams=args.num_beams)
    rb = run_mapping_lap(args.map, before, **kw)
    ra = run_mapping_lap(args.map, after, **kw)
    # determinism check: rerun AFTER and compare
    ra2 = run_mapping_lap(args.map, after, **kw)
    deterministic = (ra['steps'] == ra2['steps']
                     and ra['coverage_final'] == ra2['coverage_final']
                     and ra['min_clearance'] == ra2['min_clearance']
                     and ra['steer_rate_std'] == ra2['steer_rate_std'])

    hdr = (f"{'metric':<26}{'BEFORE':>14}{'AFTER':>14}")
    print(hdr)
    print('-' * len(hdr))

    def row(name, b, a, fmt='{:.3f}'):
        print(f"{name:<26}{fmt.format(b):>14}{fmt.format(a):>14}")

    row('laps completed', rb['completed_laps'], ra['completed_laps'], '{:d}')
    row('min wall clearance (m)', rb['min_clearance'], ra['min_clearance'])
    row('safe (>car half 0.20m)',
        int(rb['min_clearance'] > before.car_half),
        int(ra['min_clearance'] > after.car_half), '{:d}')
    row('lap time (s, sim)', rb['lap_time'], ra['lap_time'], '{:.1f}')
    row('steer-rate std (rad/s)', rb['steer_rate_std'], ra['steer_rate_std'])
    row('steer-rate max (rad/s)', rb['steer_rate_max'], ra['steer_rate_max'])
    cb = rb['coverage_per_lap']
    ca = ra['coverage_per_lap']
    for i in range(args.laps):
        row(f'coverage after lap {i + 1}',
            cb[i] if i < len(cb) else float('nan'),
            ca[i] if i < len(ca) else float('nan'))
    print('-' * len(hdr))
    print(f"determinism (AFTER == rerun) : {deterministic}")


if __name__ == '__main__':
    main()
