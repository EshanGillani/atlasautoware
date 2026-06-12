#!/usr/bin/env python3
"""
Benchmark: curvature-aware lookahead scheduling for the MAP controller.
========================================================================

Sweeps k_curv (lookahead shrink gain, L = L_base / (1 + k_curv * kappa_ahead))
against the k_curv = 0 baseline on the shared closed-loop harness
(tests/closed_loop.py, kinematic bicycle @ 50 Hz) over the competition
raceline, at nominal speed and with the speed profile scaled x1.15 / x1.30
to stress the controller where lookahead matters most.

    python3 tools/benchmark_lookahead.py
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))
sys.path.insert(0, os.path.join(REPO, 'tests'))

from map_controller import MAPController, build_lat_accel_lut  # noqa: E402
from closed_loop import load_raceline, run_lap                 # noqa: E402

K_CURVS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
SPEED_FACTORS = [1.0, 1.15, 1.30]


def main():
    rx, ry, rh, rc, rv = load_raceline(
        os.path.join(REPO, 'racelines', 'comp_raceline.csv'))
    lut = build_lat_accel_lut()

    print(f"{'v_factor':>8} {'k_curv':>7} {'completed':>9} "
          f"{'lap_time':>9} {'xte_mean':>9} {'xte_max':>8}")
    print('-' * 56)
    for f in SPEED_FACTORS:
        for k in K_CURVS:
            ctl = MAPController(lut=lut, k_curv=k)
            ctl.set_raceline(rx, ry, rv * f,
                             curvature=None if k == 0.0 else rc)
            res = run_lap(ctl.control, rx, ry, rh)
            print(f"{f:>8.2f} {k:>7.2f} {str(res['completed']):>9} "
                  f"{res['lap_time']:>9.2f} {res['xte_mean']:>9.4f} "
                  f"{res['xte_max']:>8.4f}")
        print('-' * 56)


if __name__ == '__main__':
    main()
