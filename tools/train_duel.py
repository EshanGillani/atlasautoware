"""
Train the style-conditioned wheel-to-wheel decision policy.
===========================================================

    python3 tools/atlas.py run train-duel
    python3 tools/atlas.py run train-duel -- --steps 200000 --authority 0.5
    python3 tools/atlas.py run train-duel -- --style 1.0     # aggressive only

Same three-phase shape as tools/train_rl.py — warm start on the rule-based
decision, then random exploration, then SAC — for the same reason: with a
residual action space the expert demonstration is a constant, so the warm start
costs almost nothing and fills the buffer with real racing situations before a
single reward gradient is taken.

What differs is that `style` is resampled every episode, so ONE policy learns
the whole conservative-to-aggressive range rather than a network per setting.
Evaluation reports each style separately, because the useful question is not
"is it good" but "does the knob do what it says" — an aggressive setting should
attempt and complete more passes, a conservative one should make contact less
often. If those two columns do not separate, the style conditioning is not
working, whatever the reward says.

Needs torch and f110_gym.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from f1tenth_gym_ros.rl.duel import DuelSpec, FlatObsSpec     # noqa: E402
from f1tenth_gym_ros.rl.duel_env import DuelEnv               # noqa: E402
from f1tenth_gym_ros.rl.sac import SAC, ReplayBuffer          # noqa: E402


def evaluate(env, agent, styles=(0.0, 0.5, 1.0), episodes=4, use_policy=True,
             render=False):
    """Per-style rollouts — the columns that show whether the knob works."""
    out = {}
    for style in styles:
        passes = passed = contacts = 0
        rewards, progress = [], []
        for k in range(episodes):
            obs = env.reset(style=style, start_idx=k * env.n // max(episodes, 1))
            total, done, info = 0.0, False, {}
            while not done:
                action = (agent.act(obs, deterministic=True) if use_policy
                          else env.rule_based_action())
                obs, r, done, info = env.step(action)
                total += r
                if render:
                    env.render()
            rewards.append(total)
            progress.append(info['progress'])
            passes += info['passes']
            passed += info['passed_by']
            contacts += int(info['contact'] or info['off_track'])
        out[style] = dict(reward=float(np.mean(rewards)),
                          progress=float(np.mean(progress)),
                          passes=passes, passed_by=passed, contacts=contacts)
    return out


def show(tag, res):
    print(f'  {tag}')
    print(f"    {'style':>6} {'reward':>9} {'passes':>7} {'passed':>7} "
          f"{'contacts':>9} {'progress':>9}")
    for style, r in res.items():
        print(f"    {style:>6.2f} {r['reward']:>9.1f} {r['passes']:>7} "
              f"{r['passed_by']:>7} {r['contacts']:>9} {r['progress']:>8.0f}m")


def main():
    ap = argparse.ArgumentParser(description='Train the decision policy.')
    ap.add_argument('--map', default=os.path.join(REPO, 'maps', 'comp_track'))
    ap.add_argument('--map-ext', default='.png')
    ap.add_argument('--raceline',
                    default=os.path.join(REPO, 'racelines', 'comp_raceline.csv'))
    ap.add_argument('--out', default=os.path.join(REPO, 'runtime', 'duel'))
    ap.add_argument('--steps', type=int, default=200_000)
    ap.add_argument('--warm-start', type=int, default=15_000)
    ap.add_argument('--start-steps', type=int, default=5_000)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--buffer', type=int, default=300_000)
    ap.add_argument('--eval-every', type=int, default=10_000)
    ap.add_argument('--eval-episodes', type=int, default=4)
    ap.add_argument('--authority', type=float, default=1.0)
    ap.add_argument('--style', type=float, default=None,
                    help='fix the style instead of sampling it each episode')
    ap.add_argument('--v-scale', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default=None)
    ap.add_argument('--render', action='store_true',
                    help='draw the evaluation rollouts (noVNC display in the container)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    spec = DuelSpec()
    env = DuelEnv(args.map, args.raceline, spec=spec, map_ext=args.map_ext,
                  authority=args.authority, style=args.style,
                  v_scale=args.v_scale, seed=args.seed)
    # The decision observation has no lidar block, so the CNN encoder is not
    # wanted here: n_beams=0 makes the SAC torso a plain MLP over the state.
    agent = SAC(FlatObsSpec(spec.dim), action_dim=2, device=args.device,
                seed=args.seed)
    buf = ReplayBuffer(spec.dim, 2, args.buffer)
    rng = np.random.default_rng(args.seed)

    print(f'device      {agent.device}')
    print(f'observation {spec.dim} (decision-layer, no lidar)')
    print(f'envelope    +/-{spec.d_offset:.2f} m offset, '
          f'+/-{spec.d_speed:.2f} speed factor, authority {args.authority:.2f}')
    print(f'style       {"fixed " + str(args.style) if args.style is not None else "sampled per episode"}\n')

    base = evaluate(env, agent, episodes=args.eval_episodes, use_policy=False)
    show('rule-based baseline', base)
    print()

    obs = env.reset()
    best = -np.inf
    t0 = time.time()
    log_path = os.path.join(args.out, 'train_log.jsonl')

    for step in range(1, args.steps + 1):
        if step <= args.warm_start:
            action = env.rule_based_action()
        elif step <= args.warm_start + args.start_steps:
            action = rng.uniform(-1.0, 1.0, 2).astype(np.float32)
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, reward, done, info = env.step(action)
        buf.add(obs, action, reward, next_obs, done and not info['timeout'])
        obs = next_obs
        if done:
            obs = env.reset()

        if step <= args.warm_start:
            if len(buf) >= args.batch and step % 4 == 0:
                b_obs, b_act, *_ = buf.sample(args.batch, rng)
                agent.behavior_clone(b_obs, b_act)
        elif len(buf) >= args.batch:
            agent.update(buf.sample(args.batch, rng))

        if step % args.eval_every == 0:
            res = evaluate(env, agent, episodes=args.eval_episodes,
                           render=args.render)
            mean_r = float(np.mean([r['reward'] for r in res.values()]))
            print(f'step {step:>8,}  mean reward {mean_r:>8.1f}  '
                  f'({step / max(time.time() - t0, 1e-6):.0f} steps/s)')
            show('policy', res)
            with open(log_path, 'a') as f:
                f.write(json.dumps({'step': step,
                                    'eval': {str(k): v for k, v in res.items()},
                                    'baseline': {str(k): v
                                                 for k, v in base.items()}}) + '\n')
            meta = dict(step=step, authority=args.authority,
                        spec=spec.to_dict(), eval={str(k): v for k, v in res.items()})
            agent.save(os.path.join(args.out, 'policy.pt'), meta)
            if mean_r > best:
                best = mean_r
                agent.save(os.path.join(args.out, 'best.pt'), meta)
                print('           new best')
            print()

    env.close()
    print(f'done: {args.steps:,} steps in {(time.time() - t0) / 60:.1f} min')
    print('\nCheck the style knob actually separates before trusting it:')
    print('  aggressive should attempt and complete MORE passes than '
          'conservative,\n  and conservative should make contact LESS often. '
          'If those columns look\n  the same, the conditioning is not working.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
