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

For every failed lap it also records WHERE the car left the line (the raceline
index at peak cross-track), so the sweep names the limiting corner instead of
just saying "it failed".

MPC weights are CLI-overridable so you can A/B-tune without touching the
deployed controller; bake the winner into KinematicMPC defaults afterwards.

    python3 tools/reliability_sweep.py                              # baseline
    # tuned for corner entry (longer preview, faster steer slew, align sooner):
    python3 tools/reliability_sweep.py --horizon 18 --rd-steer 6 --q-yaw 10
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
    ap.add_argument('--flying-start', action='store_true',
                    help='begin at the reference speed (flying lap) not 2 m/s')
    # ── MPC weight overrides (defaults match KinematicMPC) ─────────────────────
    ap.add_argument('--horizon', type=int, default=12, help='MPC preview steps')
    ap.add_argument('--dt', type=float, default=0.08, help='MPC step (s)')
    ap.add_argument('--q-pos', type=float, default=28.0, help='position tracking weight')
    ap.add_argument('--q-yaw', type=float, default=6.0, help='heading tracking weight')
    ap.add_argument('--rd-steer', type=float, default=12.0, help='steer-rate penalty (lower = sharper)')
    args = ap.parse_args()

    rx, ry, rh, rc, rspeed = cl.load_raceline(args.raceline)
    n_pts = len(rx)
    base_lap_v = rspeed.copy()
    print(f'raceline: {n_pts} pts, base v {rspeed.min():.1f}-{rspeed.max():.1f} m/s '
          f'| {args.n} trials/scale | wall {args.wall} m | delay {args.delay*1000:.0f} ms')
    print(f'MPC: horizon {args.horizon} dt {args.dt} | q_pos {args.q_pos} '
          f'q_yaw {args.q_yaw} rd_steer {args.rd_steer} | '
          f'start {"flying" if args.flying_start else "cold 2 m/s"}\n')

    rng = np.random.default_rng(args.seed)
    # same perturbation set across scales (fair comparison)
    starts = [(float(rng.uniform(-args.max_offset, args.max_offset)),
               float(rng.uniform(-0.3, 0.3)),
               float(rng.uniform(-args.max_heading, args.max_heading)))
              for _ in range(args.n)]

    print(f'{"v_scale":>8} {"success":>11} {"lap (mean±sd)":>15} '
          f'{"xte_mean":>9} {"xte_max":>9}  worst-fail')
    print('-' * 74)
    overall = []
    for vs in args.vscales:
        speeds = np.clip(base_lap_v * vs, 0.5, None)
        mpc = KinematicMPC(v_max=float(speeds.max()) + 0.5, horizon=args.horizon,
                           dt=args.dt, q_pos=args.q_pos, q_yaw=args.q_yaw,
                           rd_steer=args.rd_steer)
        mpc.set_raceline(rx, ry, rh, rc, speeds)
        ctrl = make_control_fn(mpc)
        v0 = float(speeds[0]) if args.flying_start else 2.0   # flying vs cold start

        ok, lap_times, xte_means, xte_maxes, worst = 0, [], [], [], 0.0
        fail_idxs = []
        for (dx, dy, dyaw) in starts:
            r = cl.run_lap(ctrl, rx, ry, rh, start_offset=(dx, dy, dyaw),
                           v0=v0, actuator_delay=args.delay)
            xte = r['xte']
            xmax = float(xte.max()) if len(xte) else float('inf')
            if r['completed'] and xmax < args.wall:
                ok += 1
                lap_times.append(r['lap_time'])
                xte_means.append(r['xte_mean'])
                xte_maxes.append(r['xte_max'])
            else:
                worst = max(worst, xmax if np.isfinite(xmax) else worst)
                if len(xte):                            # where did it leave the line?
                    fail_idxs.append(int(r['idx'][int(np.argmax(xte))]))
        rate = 100.0 * ok / args.n
        lt = (f'{np.mean(lap_times):.1f}±{np.std(lap_times):.1f}s'
              if lap_times else '   —   ')
        xm = f'{np.mean(xte_means):.3f}' if xte_means else '  —  '
        xx = f'{np.mean(xte_maxes):.3f}' if xte_maxes else '  —  '
        flag = '' if rate >= 95 else '  <-- below 95%'
        print(f'{vs:>8.2f} {ok:>4}/{args.n} {rate:>4.0f}% {lt:>15} '
              f'{xm:>9} {xx:>9}  {worst:>5.2f}m{flag}')
        # name the limiting corner: densest cluster of failure locations
        if fail_idxs:
            fa = np.array(fail_idxs)
            buckets = (fa // 10) * 10
            vals, counts = np.unique(buckets, return_counts=True)
            b = int(vals[int(np.argmax(counts))])
            k_here = float(abs(rc[b % n_pts]))
            print(f'          └ fails cluster at idx ~{b} '
                  f'(κ {k_here:.2f}, R {1.0/max(k_here,1e-3):.2f} m, '
                  f'pos {rx[b % n_pts]:.1f},{ry[b % n_pts]:.1f}) '
                  f'— {int(counts.max())}/{len(fail_idxs)} fails here')
        overall.append((vs, rate))

    print('-' * 74)
    best = [vs for vs, r in overall if r >= 95]
    if best:
        print(f'95%+ consistent up to v_scale {max(best):.2f} (of those tested).')
    else:
        print('No tested speed hit 95% — tune at the lowest scale first.')


if __name__ == '__main__':
    main()
