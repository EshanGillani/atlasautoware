"""
Reliability sweep — how consistently does the controller complete a clean lap?
=============================================================================

Runs the MPC over many perturbed start conditions across a range of speed
scales, on the shared kinematic closed-loop harness, and reports the **success
rate**: fraction of laps that finish without leaving the track.

"Hit a wall" proxy: this is the kinematic plant (no wall geometry), so a lap is
counted FAILED if it doesn't complete OR its cross-track error ever exceeds
`--wall` metres (~half the track width minus the car) — i.e. the car has run off
the racing corridor.  Swap in real map-corridor checking later to make it exact.

The point: find the speed at which consistency drops below 95%, so we know how
hard we can push and where to focus tuning.

    python3 tools/reliability_sweep.py                 # default sweep
    python3 tools/reliability_sweep.py --n 40 --vscales 1.0 1.2 1.4 --wall 0.9
"""

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tests'))
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))
import closed_loop as cl
from mpc_controller import KinematicMPC


def make_control_fn(mpc):
    def ctrl(px, py, yaw, v, j):
        out = mpc.solve((px, py, yaw, v), j)
        return out if out is not None else (0.0, v)    # solver miss -> coast
    return ctrl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raceline', default=os.path.join(REPO, 'racelines', 'comp_raceline.csv'))
    ap.add_argument('--n', type=int, default=30, help='trials per speed scale')
    ap.add_argument('--vscales', type=float, nargs='+',
                    default=[1.0, 1.1, 1.2, 1.3, 1.4])
    ap.add_argument('--wall', type=float, default=0.9, help='off-track XTE (m)')
    ap.add_argument('--max-offset', type=float, default=0.5, help='lateral start (m)')
    ap.add_argument('--max-heading', type=float, default=0.30, help='start yaw err (rad)')
    ap.add_argument('--delay', type=float, default=0.0, help='actuation delay (s)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rx, ry, rh, rc, rspeed = cl.load_raceline(args.raceline)
    base_lap_v = rspeed.copy()
    print(f'raceline: {len(rx)} pts, base v {rspeed.min():.1f}-{rspeed.max():.1f} m/s '
          f'| {args.n} trials/scale | wall proxy {args.wall} m '
          f'| delay {args.delay*1000:.0f} ms\n')

    rng = np.random.default_rng(args.seed)
    # same perturbation set across scales (fair comparison)
    starts = [(float(rng.uniform(-args.max_offset, args.max_offset)),
               float(rng.uniform(-0.3, 0.3)),
               float(rng.uniform(-args.max_heading, args.max_heading)))
              for _ in range(args.n)]

    print(f'{"v_scale":>8} {"success":>9} {"lap (mean±sd)":>16} '
          f'{"xte_mean":>9} {"xte_max":>9}  worst-fail')
    print('-' * 72)
    overall = []
    for vs in args.vscales:
        speeds = np.clip(base_lap_v * vs, 0.5, None)
        mpc = KinematicMPC(v_max=float(speeds.max()) + 0.5)
        mpc.set_raceline(rx, ry, rh, rc, speeds)
        ctrl = make_control_fn(mpc)

        ok, lap_times, xte_means, xte_maxes, worst = 0, [], [], [], 0.0
        for (dx, dy, dyaw) in starts:
            r = cl.run_lap(ctrl, rx, ry, rh, start_offset=(dx, dy, dyaw),
                           actuator_delay=args.delay)
            xmax = float(r['xte'].max()) if len(r['xte']) else float('inf')
            success = r['completed'] and xmax < args.wall
            if success:
                ok += 1
                lap_times.append(r['lap_time'])
                xte_means.append(r['xte_mean'])
                xte_maxes.append(r['xte_max'])
            else:
                worst = max(worst, xmax if np.isfinite(xmax) else worst)
        rate = 100.0 * ok / args.n
        lt = (f'{np.mean(lap_times):.1f}±{np.std(lap_times):.1f}s'
              if lap_times else '   —   ')
        xm = f'{np.mean(xte_means):.3f}' if xte_means else '  —  '
        xx = f'{np.mean(xte_maxes):.3f}' if xte_maxes else '  —  '
        flag = '' if rate >= 95 else '  <-- below 95%'
        print(f'{vs:>8.2f} {ok:>3}/{args.n} {rate:>4.0f}% {lt:>16} '
              f'{xm:>9} {xx:>9}  {worst:>5.2f}m{flag}')
        overall.append((vs, rate))

    print('-' * 72)
    best = [vs for vs, r in overall if r >= 95]
    if best:
        print(f'95%+ consistent up to v_scale {max(best):.2f} '
              f'(of those tested). Push speed there; tune to lift the rest.')
    else:
        print('No tested speed hit 95%. Start tuning at the lowest scale '
              '(controller weights / raceline feasibility) before pushing speed.')


if __name__ == '__main__':
    main()
