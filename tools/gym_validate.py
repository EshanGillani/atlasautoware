"""
Real-dynamics validation — drive the actual f110_gym sim, no ROS needed.
========================================================================

The reliability sweeps used a kinematic / first-order-grip plant.  This drives
the **real f1tenth_gym dynamic single-track model** (the same physics + tyre
behaviour your sim race runs) on comp_track, with the MPC controller, and reports
lap time + whether the car COLLIDED with a wall (the gym detects that for real —
no cross-track proxy).  `--render` opens a window so you can watch it.

f110_gym is a standalone Python package — it does not need ROS.  Install once:
    cd ~ && git clone https://github.com/f1tenth/f1tenth_gym.git
    cd f1tenth_gym && pip3 install -e .

Then, from the atlasautoware repo root:
    python3 tools/gym_validate.py --v-scale 1.1 --render
    python3 tools/gym_validate.py --v-scale 1.2          # headless, faster

Note: f110_gym's API varies a little by version (gym vs gymnasium); this handles
the common cases but if env creation/step errors, paste it and I'll match your
installed version.
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

try:
    import gym
except Exception:
    import gymnasium as gym


def _unpack_reset(res):
    return res[0] if isinstance(res, tuple) else res


def _unpack_step(res):
    obs = res[0]
    done = bool(res[2]) or (len(res) >= 5 and bool(res[3]))   # done | (term|trunc)
    return obs, done


def scalar(v):
    """obs fields come back as arrays (per-agent) or scalars."""
    try:
        return float(v[0])
    except (TypeError, IndexError):
        return float(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default=os.path.join(REPO, 'maps', 'comp_track'),
                    help='map path WITHOUT extension')
    ap.add_argument('--map-ext', default='.png')
    ap.add_argument('--raceline', default=os.path.join(REPO, 'racelines', 'comp_raceline.csv'))
    ap.add_argument('--v-scale', type=float, default=1.1)
    ap.add_argument('--laps', type=int, default=3, help='laps to attempt')
    ap.add_argument('--render', action='store_true')
    args = ap.parse_args()

    rx, ry, rh, rc, rspeed = cl.load_raceline(args.raceline)
    n = len(rx)
    speeds = np.clip(rspeed * args.v_scale, 0.5, None)
    mpc = KinematicMPC(v_max=float(speeds.max()) + 0.5)
    mpc.set_raceline(rx, ry, rh, rc, speeds)
    print(f'raceline {n} pts | v_scale {args.v_scale} '
          f'(v {speeds.min():.1f}-{speeds.max():.1f} m/s) | map {os.path.basename(args.map)}')

    try:
        env = gym.make('f110_gym:f110-v0', map=args.map, map_ext=args.map_ext,
                       num_agents=1)
    except Exception as e:
        raise SystemExit(f'env creation failed ({e}); paste this and I will match '
                         f'your f110_gym version')

    sx, sy, st = float(rx[0]), float(ry[0]), float(rh[0])
    obs = _unpack_reset(env.reset(np.array([[sx, sy, st]])))

    prev_j = cum = 0
    t = 0.0
    dt = 0.01                                   # f110_gym default timestep
    collided = False
    nearest = 0
    while cum < args.laps * n:
        x, y = scalar(obs['poses_x']), scalar(obs['poses_y'])
        yaw = scalar(obs['poses_theta'])
        v = scalar(obs['linear_vels_x'])
        nearest = int(np.argmin((rx - x) ** 2 + (ry - y) ** 2))
        d = nearest - prev_j
        if d < -n / 2:
            d += n
        if 0 < d < n / 2:
            cum += d
        prev_j = nearest

        out = mpc.solve((x, y, yaw, v), nearest)
        steer, v_t = out if out is not None else (0.0, v)
        obs, done = _unpack_step(env.step(np.array([[float(steer), float(v_t)]])))
        t += dt
        if args.render:
            try:
                env.render(mode='human')
            except Exception:
                pass
        if scalar(obs['collisions']) > 0.5:
            collided = True
            break
        if done and not (cum < args.laps * n):
            break

    laps_done = cum / float(n)
    print(f'\nresult: {laps_done:.2f} laps in {t:.1f}s '
          f'({t/max(laps_done,1e-3):.1f}s/lap) | '
          f'{"COLLIDED — left the track" if collided else "clean, no collision"}')
    if collided:
        print(f'  collision near raceline idx {nearest} '
              f'(pos {rx[nearest]:.1f},{ry[nearest]:.1f}) — too fast there; '
              f'lower --v-scale or slow that corner')
    else:
        print(f'  v_scale {args.v_scale} holds in the real dynamic sim. '
              f'Try a higher scale, or race this.')


if __name__ == '__main__':
    main()
