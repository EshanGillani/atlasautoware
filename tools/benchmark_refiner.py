"""
Benchmark — minimum-curvature raceline refinement (no ROS, standalone).
=======================================================================

Loads racelines/comp_raceline.csv, refines it with raceline_refiner at
corridors 0.15 / 0.25 / 0.35 m, and answers "is it actually faster?" three
ways, apples-to-apples:

  1. curvature statistics (max / mean |kappa|, same Menger stencil for all);
  2. friction-limited speed profile (velocity_profiler, identical limits on
     the original AND each refined line) -> estimated lap time;
  3. CLOSED-LOOP lap with the MAP controller on the kinematic-bicycle plant
     (tests/closed_loop.run_lap): lap time, cross-track error, completion.

Also verifies every refined point stays within the corridor of the original.

    python3 tools/benchmark_refiner.py
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))
sys.path.insert(0, os.path.join(REPO, 'tests'))

from raceline_refiner import (refine_raceline, heading_curvature,  # noqa: E402
                              segment_lengths)
from velocity_profiler import velocity_profile                     # noqa: E402
from map_controller import MAPController, build_lat_accel_lut      # noqa: E402
from closed_loop import load_raceline, run_lap                     # noqa: E402

PROFILE = dict(a_lat_max=6.0, a_accel_max=4.0, a_brake_max=8.0, v_max=8.0)
CORRIDORS = (0.15, 0.25, 0.35)


def evaluate(label, x, y, hdg, curv, lut, max_disp=None):
    """Profile speeds, estimate lap time, run a closed-loop lap."""
    ds = segment_lengths(x, y)
    v = velocity_profile(curv, ds, **PROFILE)
    v_next = np.roll(v, -1)
    t_est = float(np.sum(ds / np.maximum(0.5 * (v + v_next), 1e-6)))

    ctl = MAPController(lut=lut)
    ctl.set_raceline(x, y, v)
    res = run_lap(ctl.control, x, y, hdg)
    return dict(label=label, length=float(ds.sum()),
                k_max=float(np.abs(curv).max()),
                k_mean=float(np.abs(curv).mean()),
                t_est=t_est, lap=res['lap_time'],
                xte_mean=res['xte_mean'], xte_max=res['xte_max'],
                completed=res['completed'], max_disp=max_disp)


def main():
    path = os.path.join(REPO, 'racelines', 'comp_raceline.csv')
    rx, ry, _, _, _ = load_raceline(path)
    hdg0, curv0 = heading_curvature(rx, ry)        # recompute: same stencil
    lut = build_lat_accel_lut()

    rows = [evaluate('original', rx, ry, hdg0, curv0, lut)]
    for cor in CORRIDORS:
        xn, yn, hn, kn = refine_raceline(rx, ry, corridor=cor)
        disp = float(np.hypot(xn - rx, yn - ry).max())
        assert disp <= cor + 1e-6, f'corridor violated: {disp:.4f} > {cor}'
        rows.append(evaluate(f'refined {cor:.2f} m', xn, yn, hn, kn, lut,
                             max_disp=disp))

    hdr = (f"{'line':<15} {'len(m)':>7} {'max|k|':>7} {'mean|k|':>8} "
           f"{'est lap':>8} {'CL lap':>7} {'xte_mu':>7} {'xte_max':>8} "
           f"{'done':>5} {'maxdisp':>8}")
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        disp = f"{r['max_disp']:.4f}" if r['max_disp'] is not None else '-'
        print(f"{r['label']:<15} {r['length']:>7.2f} {r['k_max']:>7.3f} "
              f"{r['k_mean']:>8.4f} {r['t_est']:>7.2f}s {r['lap']:>6.2f}s "
              f"{r['xte_mean']:>7.3f} {r['xte_max']:>8.3f} "
              f"{str(r['completed']):>5} {disp:>8}")

    base = rows[0]
    print()
    for r in rows[1:]:
        print(f"{r['label']}: est lap {base['t_est'] - r['t_est']:+.2f}s vs "
              f"original, closed-loop {base['lap'] - r['lap']:+.2f}s")


if __name__ == '__main__':
    main()
