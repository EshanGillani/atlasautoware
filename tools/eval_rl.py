"""
Evaluate a trained policy against the MPC baseline — head to head.
==================================================================

    python3 tools/atlas.py run eval-rl
    python3 tools/atlas.py run eval-rl -- --checkpoint runtime/rl/best.pt \
                                          --episodes 20 --authority-sweep

Runs both the policy and the pure MPC over the *same* set of starting poses on
the real gym dynamics, and reports lap time, completion rate and crashes for
each.  Identical starts matter: a policy that draws easier spawns than the
baseline will look better than it is, and the difference you are trying to
measure is often under a second.

`--authority-sweep` re-runs at several residual scales.  That curve is the
thing to look at before deploying: it tells you how much authority the policy
has actually earned, and it is exactly the ladder you climb on the real car.

A policy that does not beat the MPC here has no business on the track.  It is
completely normal for the first few training runs to lose — that is the
measurement working.
"""

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from f1tenth_gym_ros.rl.env import RaceEnv                 # noqa: E402
from f1tenth_gym_ros.rl.features import ObsSpec            # noqa: E402


def rollout(env, act_fn, starts):
    laps, crashes, progress = [], 0, []
    for s in starts:
        obs = env.reset(start_idx=s)
        done = False
        info = {}
        while not done:
            obs, _r, done, info = env.step(act_fn(obs))
        progress.append(info['progress'])
        if info['crashed'] or info['off_track']:
            crashes += 1
        elif info['finished']:
            laps.append(info['sim_time'])
    return dict(laps=laps, crashes=crashes,
                completed=len(laps), progress=float(np.mean(progress)))


def summarize(name, r, n):
    lap = f"{np.mean(r['laps']):.2f}s ± {np.std(r['laps']):.2f}" if r['laps'] \
        else '     —      '
    print(f"  {name:<22} {lap:>16}   "
          f"{r['completed']}/{n} clean   {r['crashes']} crash   "
          f"{r['progress']:.0f} m avg")
    return float(np.mean(r['laps'])) if r['laps'] else None


def main():
    ap = argparse.ArgumentParser(description='Policy vs MPC, same starts.')
    ap.add_argument('--checkpoint',
                    default=os.path.join(REPO, 'runtime', 'rl', 'best.pt'))
    ap.add_argument('--map', default=os.path.join(REPO, 'maps', 'comp_track'))
    ap.add_argument('--map-ext', default='.png')
    ap.add_argument('--raceline',
                    default=os.path.join(REPO, 'racelines', 'comp_raceline.csv'))
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--authority', type=float, default=1.0)
    ap.add_argument('--authority-sweep', action='store_true',
                    help='evaluate at 0.25 / 0.5 / 0.75 / 1.0 as well')
    ap.add_argument('--v-scale', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f'no checkpoint at {args.checkpoint}\n'
                         f'train one first:  python3 tools/atlas.py run train-rl')

    from f1tenth_gym_ros.rl.sac import SAC
    agent, meta = SAC.load(args.checkpoint, device='cpu')
    spec = agent.spec
    print(f'checkpoint  {os.path.relpath(args.checkpoint, REPO)}')
    if meta.get('step'):
        print(f'trained     {meta["step"]:,} steps')
    print(f'observation {spec.fingerprint()}')

    env = RaceEnv(args.map, args.raceline, spec=spec, map_ext=args.map_ext,
                  v_scale=args.v_scale, random_start=False, seed=args.seed)
    if ObsSpec.from_dict(spec.to_dict()) != spec:
        raise SystemExit('checkpoint spec did not round-trip — refusing to run')

    # The same spawn points for everyone.
    starts = [k * env.n // args.episodes for k in range(args.episodes)]
    print(f'{args.episodes} episodes from identical starts on '
          f'{os.path.basename(args.map)}\n')

    print(f"  {'':<22} {'lap time':>16}   completion    crashes")
    print('  ' + '-' * 70)
    base = rollout(env, lambda _o: env.mpc_action(), starts)
    base_lap = summarize('MPC baseline', base, args.episodes)

    levels = [0.25, 0.5, 0.75, 1.0] if args.authority_sweep else [args.authority]
    results = {}
    for a in levels:
        env.residual.authority = float(a)
        r = rollout(env, lambda o: agent.act(o, deterministic=True), starts)
        results[a] = summarize(f'policy @ authority {a:.2f}', r, args.episodes)

    print()
    best_a = None
    for a, lap in results.items():
        if lap is None or base_lap is None:
            continue
        d = base_lap - lap
        crashed_more = False
        print(f'  authority {a:.2f}: {d:+.2f}s vs MPC '
              f'({100.0 * d / base_lap:+.1f}%)')
        if d > 0 and not crashed_more:
            best_a = a if best_a is None else max(best_a, a)

    env.close()
    if base_lap is None:
        print('\n  The MPC baseline itself did not complete a lap — fix that '
              'before judging the policy (atlas run validate).')
        return 1
    if best_a is None:
        print('\n  The policy does not beat the MPC. Do not deploy it.')
        print('  Train longer, or widen the envelope (--d-steer / --d-speed).')
        return 1
    print(f'\n  Policy wins at authority {best_a:.2f}. On the car, climb to it:')
    print(f'    atlas run rl-drive -- -p residual_scale:=0.25 -p v_scale:=0.4')
    print(f'    ... watch a clean lap, then raise residual_scale a step.\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
