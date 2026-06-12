"""
Benchmark — occupancy-aware corridors vs blind scalar corridors (no ROS).
=========================================================================

Loads racelines/comp_raceline.csv + maps/comp_track.yaml and compares, with
identical speed limits and the MAP controller on the kinematic-bicycle plant
(tests/closed_loop.run_lap):

  original line | scalar corridor 0.25 / 0.35 m |
  map corridors margin 0.35 cap 0.8 | map corridors margin 0.25 cap 1.2

For each line: closed-loop lap time, cross-track error, and the safety half
of the story — min / mean distance_to_wall against the real map, plus how
many points sit closer to a wall than the margin (the original line grazes
pixelated walls at ~13 points; map corridors may never make those worse).
Headline: does map-awareness beat scalar 0.35 on lap time while having
better worst-case wall clearance?

    python3 tools/benchmark_map_corridors.py
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))
sys.path.insert(0, os.path.join(REPO, 'tests'))

from raceline_refiner import (refine_raceline, heading_curvature,  # noqa: E402
                              map_corridors, verify_wall_clearance,
                              segment_lengths)
from grid_map import GridMap                                       # noqa: E402
from velocity_profiler import velocity_profile                     # noqa: E402
from map_controller import MAPController, build_lat_accel_lut      # noqa: E402
from closed_loop import load_raceline, run_lap                     # noqa: E402

PROFILE = dict(a_lat_max=6.0, a_accel_max=4.0, a_brake_max=8.0, v_max=8.0)
MARGIN_REF = 0.35            # wall margin used for the "points < margin" count


def evaluate(label, x, y, hdg, curv, lut, gm, d_orig, verified=None):
    """Speed-profile the line, run a closed-loop lap, query wall clearance."""
    ds = segment_lengths(x, y)
    v = velocity_profile(curv, ds, **PROFILE)
    ctl = MAPController(lut=lut)
    ctl.set_raceline(x, y, v)
    res = run_lap(ctl.control, x, y, hdg)
    d = np.atleast_1d(gm.distance_to_wall(x, y))
    clear0 = d_orig >= MARGIN_REF                 # points the original line
    return dict(label=label, lap=res['lap_time'],  # kept clear of walls
                xte_mean=res['xte_mean'], xte_max=res['xte_max'],
                completed=res['completed'],
                d_min=float(d.min()), d_mean=float(d.mean()),
                d_min_clear=float(d[clear0].min()),
                n_below=int(np.sum(d < MARGIN_REF)),
                degraded=int(np.sum(d < np.minimum(d_orig, MARGIN_REF)
                                    - 0.5 * gm.res)),
                verified=verified)


def main():
    rx, ry, _, _, _ = load_raceline(
        os.path.join(REPO, 'racelines', 'comp_raceline.csv'))
    gm = GridMap.load(os.path.join(REPO, 'maps', 'comp_track.yaml'))
    hdg0, curv0 = heading_curvature(rx, ry)       # recompute: same stencil
    lut = build_lat_accel_lut()
    d_orig = np.atleast_1d(gm.distance_to_wall(rx, ry))

    rows = [evaluate('original', rx, ry, hdg0, curv0, lut, gm, d_orig)]

    for cor in (0.25, 0.35):
        xn, yn, hn, kn = refine_raceline(rx, ry, corridor=cor)
        rows.append(evaluate(f'scalar {cor:.2f}', xn, yn, hn, kn,
                             lut, gm, d_orig))

    for margin, cap in ((0.35, 0.8), (0.25, 1.2), (0.15, 1.2)):
        lo, hi = map_corridors(rx, ry, gm, margin=margin, cap=cap)
        xn, yn, hn, kn = refine_raceline(rx, ry, corridor=(lo, hi))
        ok, _ = verify_wall_clearance(xn, yn, gm, margin, rx, ry)
        rows.append(evaluate(f'map m={margin} cap={cap}', xn, yn, hn, kn,
                             lut, gm, d_orig, verified=ok))

    hdr = (f"{'line':<18} {'lap':>7} {'xte_mu':>7} {'xte_max':>8} "
           f"{'done':>5} {'d_min':>6} {'d_mean':>7} {'d_min*':>7} "
           f"{'n<0.35':>6} {'worse':>5} {'verok':>5}")
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        ver = '-' if r['verified'] is None else str(r['verified'])
        print(f"{r['label']:<18} {r['lap']:>6.2f}s {r['xte_mean']:>7.3f} "
              f"{r['xte_max']:>8.3f} {str(r['completed']):>5} "
              f"{r['d_min']:>6.3f} {r['d_mean']:>7.3f} "
              f"{r['d_min_clear']:>7.3f} {r['n_below']:>6} "
              f"{r['degraded']:>5} {ver:>5}")
    print()
    print("d_min*  = min clearance over the points the ORIGINAL line kept")
    print("          >= 0.35 m from walls (the line's controllable worst")
    print("          case; the original grazes pixelated walls elsewhere).")
    print("n<0.35  = points closer than 0.35 m to a wall (original: "
          f"{rows[0]['n_below']}).")
    print("worse   = points pushed closer to a wall than min(original, "
          "0.35 m), beyond half a cell.")

    base = rows[0]
    print()
    for r in rows[1:]:
        print(f"{r['label']}: {base['lap'] - r['lap']:+.2f}s vs original "
              f"({r['lap']:.2f}s), d_min* {r['d_min_clear']:.3f} m, "
              f"{r['degraded']} point(s) made worse")


if __name__ == '__main__':
    main()
