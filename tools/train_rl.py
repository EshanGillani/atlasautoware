"""
Train the neural driving policy.
================================

    python3 tools/atlas.py run train-rl                         # defaults
    python3 tools/atlas.py run train-rl -- --steps 500000
    python3 tools/atlas.py run train-rl -- --authority 0.3      # tighter envelope

Three phases, in order:

  1. **Warm start** (`--warm-start` steps).  The car drives on the MPC while the
     policy is trained, supervised, to output the MPC's action (zero, in
     residual space).  Nothing is learned about *racing* here — the point is to
     fill the replay buffer with on-track states at racing speed, and to start
     exploration from a policy that already defers to a competent controller.
     Skipping this works, it just costs a lot more steps.
  2. **Exploration** (`--start-steps`).  Random residuals, to get variety into
     the buffer before the critics start shaping behaviour.
  3. **SAC**.  The real loop, with periodic deterministic evaluation against the
     pure-MPC baseline so you can see whether the policy is actually earning
     its place.

Checkpoints land in `--out` (default runtime/rl): `policy.pt` is the latest,
`best.pt` is the best evaluated so far.  **`best.pt` is the one to deploy** —
the latest checkpoint can easily be mid-regression.

Needs torch and f110_gym.  `atlas env` will tell you if either is missing.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from f1tenth_gym_ros.rl.env import RaceEnv                 # noqa: E402
from f1tenth_gym_ros.rl.features import ObsSpec            # noqa: E402
from f1tenth_gym_ros.rl.sac import SAC, ReplayBuffer       # noqa: E402


def evaluate(env, agent, episodes=3, use_policy=True):
    """Deterministic rollouts.  use_policy=False measures the MPC baseline."""
    laps, crashes, rewards, progress = [], 0, [], []
    for k in range(episodes):
        obs = env.reset(start_idx=k * env.n // max(episodes, 1))
        total, done = 0.0, False
        while not done:
            action = (agent.act(obs, deterministic=True) if use_policy
                      else env.mpc_action())
            obs, r, done, info = env.step(action)
            total += r
        rewards.append(total)
        progress.append(info['progress'])
        if info['crashed'] or info['off_track']:
            crashes += 1
        elif info['finished']:
            laps.append(info['sim_time'])
    return dict(lap_time=float(np.mean(laps)) if laps else None,
                completed=len(laps), crashes=crashes,
                reward=float(np.mean(rewards)),
                progress=float(np.mean(progress)))


def main():
    ap = argparse.ArgumentParser(description='Train the residual driving policy.')
    ap.add_argument('--map', default=os.path.join(REPO, 'maps', 'comp_track'),
                    help='map path WITHOUT extension')
    ap.add_argument('--map-ext', default='.png')
    ap.add_argument('--raceline',
                    default=os.path.join(REPO, 'racelines', 'comp_raceline.csv'))
    ap.add_argument('--out', default=os.path.join(REPO, 'runtime', 'rl'))
    ap.add_argument('--steps', type=int, default=300_000)
    ap.add_argument('--warm-start', type=int, default=20_000,
                    help='steps of MPC imitation before exploring')
    ap.add_argument('--start-steps', type=int, default=5_000,
                    help='random-action steps after the warm start')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--buffer', type=int, default=400_000)
    ap.add_argument('--updates-per-step', type=int, default=1)
    ap.add_argument('--eval-every', type=int, default=10_000)
    ap.add_argument('--eval-episodes', type=int, default=3)
    ap.add_argument('--authority', type=float, default=1.0,
                    help='fraction of the residual envelope the policy may use')
    ap.add_argument('--d-steer', type=float, default=0.10,
                    help='max steering correction (rad)')
    ap.add_argument('--d-speed', type=float, default=1.5,
                    help='max speed correction (m/s)')
    ap.add_argument('--v-scale', type=float, default=1.0)
    ap.add_argument('--beams', type=int, default=108)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default=None, help='cuda | cpu (default: auto)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    spec = ObsSpec(n_beams=args.beams)
    env = RaceEnv(args.map, args.raceline, spec=spec, map_ext=args.map_ext,
                  authority=args.authority, d_steer=args.d_steer,
                  d_speed=args.d_speed, v_scale=args.v_scale, seed=args.seed)
    agent = SAC(spec, device=args.device, seed=args.seed)
    buf = ReplayBuffer(spec.dim, env.action_dim, args.buffer)
    rng = np.random.default_rng(args.seed)

    print(f'device      {agent.device}')
    print(f'observation {spec.dim} = {spec.n_beams} lidar + {spec.n_state} state')
    print(f'parameters  {agent.n_parameters():,}')
    print(f'envelope    +/-{args.d_steer:.2f} rad, +/-{args.d_speed:.1f} m/s '
          f'at authority {args.authority:.2f}')
    print(f'raceline    {os.path.basename(args.raceline)} ({env.n} pts, '
          f'{env.track_len:.1f} m)\n')

    base = evaluate(env, agent, args.eval_episodes, use_policy=False)
    print(f"MPC baseline: lap "
          f"{base['lap_time'] if base['lap_time'] else float('nan'):.2f}s  "
          f"completed {base['completed']}/{args.eval_episodes}  "
          f"reward {base['reward']:.1f}\n")

    obs = env.reset()
    best_reward = -np.inf
    t0 = time.time()
    ep_reward, ep_len, episode = 0.0, 0, 0
    log_path = os.path.join(args.out, 'train_log.jsonl')

    for step in range(1, args.steps + 1):
        warming = step <= args.warm_start
        if warming:
            action = env.mpc_action()                    # the expert: zero
        elif step <= args.warm_start + args.start_steps:
            action = rng.uniform(-1.0, 1.0, env.action_dim).astype(np.float32)
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, reward, done, info = env.step(action)
        buf.add(obs, action, reward, next_obs, done and not info['timeout'])
        # A timeout is not a terminal state of the MDP — the car could have
        # kept going. Bootstrapping through it, rather than treating it as an
        # absorbing state worth zero, is what keeps the value function honest.
        obs = next_obs
        ep_reward += reward
        ep_len += 1

        if done:
            episode += 1
            obs = env.reset()
            ep_reward, ep_len = 0.0, 0

        if warming:
            if len(buf) >= args.batch and step % 4 == 0:
                b_obs, b_act, *_ = buf.sample(args.batch, rng)
                agent.behavior_clone(b_obs, b_act)
        elif len(buf) >= args.batch:
            for _ in range(args.updates_per_step):
                agent.update(buf.sample(args.batch, rng))

        if step % args.eval_every == 0:
            res = evaluate(env, agent, args.eval_episodes)
            lap = res['lap_time']
            rate = step / max(time.time() - t0, 1e-6)
            print(f"step {step:>8,}  reward {res['reward']:>8.1f}  "
                  f"lap {(f'{lap:.2f}s' if lap else '   —  '):>7}  "
                  f"completed {res['completed']}/{args.eval_episodes}  "
                  f"crashes {res['crashes']}  "
                  f"({rate:.0f} steps/s)", flush=True)
            with open(log_path, 'a') as f:
                f.write(json.dumps(dict(step=step, **res,
                                        baseline=base['reward'])) + '\n')
            meta = dict(step=step, eval=res, baseline=base,
                        raceline=args.raceline, map=args.map,
                        authority=args.authority, d_steer=args.d_steer,
                        d_speed=args.d_speed)
            agent.save(os.path.join(args.out, 'policy.pt'), meta)
            if res['reward'] > best_reward:
                best_reward = res['reward']
                agent.save(os.path.join(args.out, 'best.pt'), meta)
                print(f'           new best -> {os.path.join(args.out, "best.pt")}')

    env.close()
    print(f'\ndone: {episode} episodes, {args.steps:,} steps in '
          f'{(time.time() - t0) / 60:.1f} min')
    print(f'best evaluated reward {best_reward:.1f} '
          f'(MPC baseline {base["reward"]:.1f})')
    print(f'\nCompare them honestly before racing it:\n'
          f'  python3 tools/atlas.py run eval-rl -- '
          f'--checkpoint {os.path.join(args.out, "best.pt")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
