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


def pursuit_steer(x, y, yaw, rx, ry, nearest, look, wheelbase, max_steer):
    """Geometric pure-pursuit steering — works at any speed (incl. ~0), unlike
    the MPC whose steering authority vanishes at v=0.  Used to launch from rest."""
    n = len(rx)
    tj = nearest
    for off in range(1, n):
        j = (nearest + off) % n
        if math.hypot(rx[j] - x, ry[j] - y) >= look:
            tj = j
            break
    dx, dy = rx[tj] - x, ry[tj] - y
    lx = dx * math.cos(yaw) + dy * math.sin(yaw)
    ly = -dx * math.sin(yaw) + dy * math.cos(yaw)
    ld2 = lx * lx + ly * ly
    if ld2 < 1e-6:
        return 0.0
    return float(np.clip(math.atan(wheelbase * 2.0 * ly / ld2),
                         -max_steer, max_steer))


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
    ap.add_argument('--start-idx', type=int, default=0, help='raceline index to spawn at')
    ap.add_argument('--launch-speed', type=float, default=1.5,
                    help='below this speed, launch with pure-pursuit not MPC (m/s)')
    ap.add_argument('--render', action='store_true')
    ap.add_argument('--debug', action='store_true', help='dump first 30 steps')
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

    s0 = args.start_idx % n
    sx, sy, st = float(rx[s0]), float(ry[s0]), float(rh[s0])
    obs = _unpack_reset(env.reset(np.array([[sx, sy, st]])))
    if args.debug:
        print(f'requested spawn ({sx:.2f},{sy:.2f},{st:.2f}) | '
              f'env placed ({scalar(obs["poses_x"]):.2f},{scalar(obs["poses_y"]):.2f},'
              f'{scalar(obs["poses_theta"]):.2f})')
        print(' step    t    x      y     yaw     v   near  steer  v_cmd')

    prev_j = s0
    cum = 0
    step_i = 0
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

        if v < args.launch_speed:
            # launch: MPC steering authority ~ 0 at v~0, so use geometric
            # pursuit to get rolling, commanding a modest launch speed.
            steer = pursuit_steer(x, y, yaw, rx, ry, nearest, 1.0,
                                  mpc.L, mpc.max_steer)
            v_t = args.launch_speed + 1.0
        else:
            out = mpc.solve((x, y, yaw, v), nearest)
            if out is not None:
                steer, v_t = out
            else:
                steer = pursuit_steer(x, y, yaw, rx, ry, nearest, 1.0,
                                      mpc.L, mpc.max_steer)
                v_t = v
        if args.debug and step_i < 30:
            print(f'{step_i:5d} {t:5.2f} {x:6.2f} {y:6.2f} {yaw:6.2f} {v:5.2f} '
                  f'{nearest:5d} {steer:6.3f} {v_t:6.2f}')
        step_i += 1
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
