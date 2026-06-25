"""
Grip-aware reliability sweep — where does GRIP (not tracking) break the lap?
============================================================================

The kinematic sweep proved the controller can *track* the line at any speed,
because a kinematic bicycle can't slide.  This sweep adds the two physics that
actually limit a real F1TENTH lap, which the kinematic model ignores:

  1. LATERAL GRIP — a corner demands a_lat = v^2 * |kappa|.  If that exceeds
     mu * g, the front tyres can't deliver it and the car UNDERSTEERS wide
     (the real "runs off at the corner" failure).  Modelled as a speed-dependent
     cap on usable steering: |delta| <= atan(mu * g * L / v^2).
  2. STEERING SLEW RATE — the servo can't snap to a new angle instantly; it
     ramps at <= sv_max rad/s, so at speed it can't turn in fast enough.

Everything else mirrors the kinematic harness (perturbed starts, speed scales,
success = completed lap with cross-track under the wall proxy, fail-corner
naming, --flying-start).  Params default to f1tenth's canonical vehicle so the
numbers are representative of the gym sim; sweep `--mu` to bracket the real grip.

This is a faithful *first-order* grip model, not the gym env itself — use it to
find the speed ceiling + a margin, then confirm the final setting in the real
f1tenth_gym dynamic sim.

    python3 tools/dynamic_sweep.py --flying-start
    python3 tools/dynamic_sweep.py --flying-start --mu 0.8 --vscales 1.0 1.1 1.2
"""

import argparse
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tests'))
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))
import closed_loop as cl
from mpc_controller import KinematicMPC

G = 9.81


def run_lap_grip(control_fn, rx, ry, rh, mu, sv_max, max_steer, wheelbase=0.33,
                 start_offset=(0.15, -0.1, 0.0), v0=2.0, dt=0.02, max_steps=12000,
                 settle=100, a_accel=4.0, a_brake=8.0):
    """One lap with steering slew-rate + lateral-grip limits.

    Plant state adds delta (actual steering).  Each tick the MPC's target steer
    is (a) slewed toward at <= sv_max, then (b) clamped to the grip-feasible
    angle for the current speed; the gap between commanded and achievable steer
    is exactly the understeer that pushes the car wide when it's asking for more
    grip than mu*g.  Returns the same dict as closed_loop.run_lap plus
    grip_capped (fraction of steps the grip cap was binding).
    """
    n = len(rx)
    px = float(rx[0]) + start_offset[0]
    py = float(ry[0]) + start_offset[1]
    yaw = float(rh[0]) + start_offset[2]
    v = float(v0)
    delta = 0.0
    prev_j = cum = 0
    t = 0.0
    xte, idx, capped = [], [], 0
    for _ in range(max_steps):
        j = int(np.argmin((rx - px) ** 2 + (ry - py) ** 2))
        d = j - prev_j
        if d < -n / 2:
            d += n
        if 0 < d < n / 2:
            cum += d
        prev_j = j

        target_steer, v_t = control_fn(px, py, yaw, v, j)
        # (a) slew toward the commanded angle (servo rate limit)
        delta += float(np.clip(target_steer - delta, -sv_max * dt, sv_max * dt))
        # (b) grip-feasible steering for this speed: v^2 * tan|delta| / L <= mu*g
        grip_delta = math.atan(mu * G * wheelbase / max(v * v, 1e-3))
        lim = min(max_steer, grip_delta)
        if abs(delta) > lim:
            delta = math.copysign(lim, delta)
            capped += 1
        a = float(np.clip((v_t - v) / dt, -a_brake, a_accel))
        px += v * math.cos(yaw) * dt
        py += v * math.sin(yaw) * dt
        yaw += v * math.tan(delta) / wheelbase * dt
        v = max(0.0, v + a * dt)
        t += dt
        xte.append(cl.cross_track(px, py, rx, ry, j))
        idx.append(j)
        if cum >= n:
            break
    xte = np.array(xte)
    steady = xte[settle:] if len(xte) > settle else xte
    steps = len(xte)
    return dict(completed=cum >= n, lap_time=t, xte=xte, idx=np.array(idx),
                xte_mean=float(steady.mean()) if len(steady) else float('inf'),
                xte_max=float(steady.max()) if len(steady) else float('inf'),
                grip_capped=capped / max(steps, 1))


def make_control_fn(mpc):
    def ctrl(px, py, yaw, v, j):
        out = mpc.solve((px, py, yaw, v), j)
        return out if out is not None else (0.0, v)
    return ctrl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raceline', default=os.path.join(REPO, 'racelines', 'comp_raceline.csv'))
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--vscales', type=float, nargs='+', default=[1.0, 1.1, 1.2, 1.3, 1.4])
    ap.add_argument('--wall', type=float, default=0.9)
    ap.add_argument('--max-offset', type=float, default=0.15)
    ap.add_argument('--max-heading', type=float, default=0.10)
    ap.add_argument('--mu', type=float, default=1.0489, help='tyre friction (f1tenth default)')
    ap.add_argument('--sv-max', type=float, default=3.2, help='steering slew rate (rad/s)')
    ap.add_argument('--max-steer', type=float, default=0.41)
    ap.add_argument('--flying-start', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rx, ry, rh, rc, rspeed = cl.load_raceline(args.raceline)
    n_pts = len(rx)
    print(f'raceline: {n_pts} pts, base v {rspeed.min():.1f}-{rspeed.max():.1f} m/s | '
          f'{args.n} trials/scale | wall {args.wall} m')
    print(f'grip: mu {args.mu} -> a_lat cap {args.mu*G:.1f} m/s^2 | sv_max {args.sv_max} rad/s'
          f' | start {"flying" if args.flying_start else "cold 2 m/s"}\n')

    rng = np.random.default_rng(args.seed)
    starts = [(float(rng.uniform(-args.max_offset, args.max_offset)),
               float(rng.uniform(-0.1, 0.1)),
               float(rng.uniform(-args.max_heading, args.max_heading)))
              for _ in range(args.n)]

    print(f'{"v_scale":>8} {"success":>11} {"lap (mean±sd)":>15} '
          f'{"xte_max":>8} {"grip-cap%":>9}  worst-fail')
    print('-' * 76)
    overall = []
    for vs in args.vscales:
        speeds = np.clip(rspeed * vs, 0.5, None)
        mpc = KinematicMPC(v_max=float(speeds.max()) + 0.5)
        mpc.set_raceline(rx, ry, rh, rc, speeds)
        ctrl = make_control_fn(mpc)
        v0 = float(speeds[0]) if args.flying_start else 2.0

        ok, laps, xmaxes, caps, worst, fail_idxs = 0, [], [], [], 0.0, []
        for (dx, dy, dyaw) in starts:
            r = run_lap_grip(ctrl, rx, ry, rh, mu=args.mu, sv_max=args.sv_max,
                             max_steer=args.max_steer, start_offset=(dx, dy, dyaw),
                             v0=v0)
            caps.append(r['grip_capped'])
            xmax = float(r['xte'].max()) if len(r['xte']) else float('inf')
            if r['completed'] and xmax < args.wall:
                ok += 1
                laps.append(r['lap_time'])
                xmaxes.append(r['xte_max'])
            else:
                worst = max(worst, xmax if np.isfinite(xmax) else worst)
                if len(r['xte']):
                    fail_idxs.append(int(r['idx'][int(np.argmax(r['xte']))]))
        rate = 100.0 * ok / args.n
        lt = f'{np.mean(laps):.1f}±{np.std(laps):.1f}s' if laps else '   —   '
        xx = f'{np.mean(xmaxes):.3f}' if xmaxes else '  —  '
        cap = f'{100.0*np.mean(caps):.0f}%' if caps else '  —  '
        flag = '' if rate >= 95 else '  <-- below 95%'
        print(f'{vs:>8.2f} {ok:>4}/{args.n} {rate:>4.0f}% {lt:>15} '
              f'{xx:>8} {cap:>9}  {worst:>5.2f}m{flag}')
        if fail_idxs:
            fa = np.array(fail_idxs)
            b = int((np.bincount((fa // 10)).argmax()) * 10)
            k_here = float(abs(rc[b % n_pts]))
            print(f'          └ fails cluster at idx ~{b} (κ {k_here:.2f}, '
                  f'R {1.0/max(k_here,1e-3):.2f} m, pos {rx[b%n_pts]:.1f},{ry[b%n_pts]:.1f})')
        overall.append((vs, rate))

    print('-' * 76)
    best = [vs for vs, r in overall if r >= 95]
    if best:
        top = max(best)
        print(f'95%+ grip-feasible up to v_scale {top:.2f}. '
              f'Race ~10-15% under that for margin; the traction governor covers the rest.')
    else:
        print('No tested speed is grip-feasible at 95% — lower the profile speed '
              '(or raise mu only if the real tyres justify it).')
    print('NOTE: representative first-order grip model — confirm the final v_scale '
          'in the real f1tenth_gym dynamic sim before racing it.')


if __name__ == '__main__':
    main()
