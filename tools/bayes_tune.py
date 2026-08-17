"""
Bayesian tuning — find the fastest reliable setup in as few runs as possible.
=============================================================================

At a competition you get a fixed, small number of practice runs.  This spends
them the way a good engineer would: model what you have learned about the car so
far, then run the setup that will teach you the most about where the lap time
is.  The search is a Gaussian process with Expected Improvement
(tools/bayes_opt.py); what is new here is the objective and the parameter space.

    python3 tools/atlas.py run bayes-tune                      # 40 evaluations
    python3 tools/atlas.py run bayes-tune -- --iters 80 --trials 24
    python3 tools/atlas.py run bayes-tune -- --backend gym     # real gym physics
    python3 tools/atlas.py run bayes-tune -- --resume          # continue a session

What it tunes
-------------
Speed profile (re-profiled from the line's own curvature each evaluation, so no
map or optimizer re-run is needed):
    a_lat      lateral grip budget the profiler is allowed to spend
    a_accel    engine limit used when profiling
    a_brake    braking limit used when profiling
    v_max      profile ceiling
    v_scale    global multiplier applied on top of the profile
MPC cost weights (what the controller trades off):
    q_pos      position tracking
    q_yaw      heading tracking
    q_v        speed tracking
    rd_steer   steering-rate penalty — smoothness vs response
    horizon    how far ahead it plans

The objective
-------------
    score = mean_lap_time / success_rate          (+ a penalty below the floor)

That is the *expected time to actually complete a lap*, including having to
restart after a crash — which is the quantity a race engineer cares about, and
it is why the search will not hand you a setup that is half a second quicker
and spins one lap in four.  Each candidate is scored over `--trials` laps from
randomly perturbed starting poses, so a config only wins by being repeatable.

Backends (`--backend`)
----------------------
    grip   (default) the first-order grip plant from tools/dynamic_sweep.py —
           lateral-grip saturation plus steering slew rate, which are the two
           physics that actually end an F1TENTH lap.  Pure numpy: it runs on the
           laptop at the track with no ROS and no simulator, in seconds.
    gym    the real f110_gym dynamic single-track model with true collision
           detection.  Slower, needs f110_gym installed; the honest confirmation
           before you field a setup.

Tune on `grip`, confirm the winner on `gym`, then put it on the car with
v_scale backed off ~10%.  Results append to runtime/bayes_log.jsonl and the
best config lands in runtime/bayes_best.json (and, with --apply, in
config/hardware.yaml).
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tools'))
sys.path.insert(0, os.path.join(REPO, 'tests'))
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))

import closed_loop as cl                                   # noqa: E402
from bayes_opt import BayesOpt                             # noqa: E402
from mpc_controller import KinematicMPC                    # noqa: E402
from velocity_profiler import velocity_profile, segment_lengths   # noqa: E402

RUNTIME = os.path.join(REPO, 'runtime')
LOG = os.path.join(RUNTIME, 'bayes_log.jsonl')
BEST = os.path.join(RUNTIME, 'bayes_best.json')

# name -> (low, high).  Ranges are deliberately wide: the GP is cheap, and a
# range that excludes the optimum is the one failure mode you cannot recover
# from by running longer.
SPACE = [
    ('a_lat',    4.5, 9.0),      # m/s^2 lateral grip budget for the profiler
    ('a_accel',  2.5, 6.0),      # m/s^2 engine limit
    ('a_brake',  4.0, 11.0),     # m/s^2 braking limit
    ('v_max',    5.0, 9.0),      # m/s profile ceiling
    ('v_scale',  0.85, 1.35),    # global multiplier on the profile
    ('q_pos',   12.0, 60.0),     # MPC position-tracking weight
    ('q_yaw',    1.0, 18.0),     # MPC heading weight
    ('q_v',      0.5, 8.0),      # MPC speed weight
    ('rd_steer', 3.0, 30.0),     # MPC steer-rate penalty (smoothness)
    ('horizon',  8.0, 22.0),     # MPC prediction steps (rounded)
]

G = 9.81

# Score given to a config that never completes a lap.  Deliberately a bounded
# constant rather than 1e9: the GP standardizes its targets, so a single
# enormous outlier inflates the standard deviation until every *feasible*
# config — the 37s one and the 55s one — collapses into the same sliver of the
# scale and the model can no longer tell them apart.  A few multiples of a real
# lap is unambiguously "worse than anything that works" while leaving the
# interesting range resolvable.  The search additionally models log(score), so
# the failure region compresses further still.
FAIL_SCORE = 300.0


# ── the plant ────────────────────────────────────────────────────────────────
def run_lap_grip(control_fn, rx, ry, rh, mu, sv_max, max_steer, wheelbase=0.33,
                 start_offset=(0.15, -0.1, 0.0), v0=2.0, dt=0.02,
                 max_steps=12000, settle=100, a_accel=4.0, a_brake=8.0):
    """One lap with steering slew-rate and lateral-grip limits.

    Mirrors tools/dynamic_sweep.py: the commanded steer is slewed at the
    servo's rate, then clamped to what the tyres can actually deliver at the
    current speed.  The gap between commanded and achievable steer *is* the
    understeer that runs a car wide, so a config that asks for more grip than
    it has fails here for the same reason it fails on the track.
    """
    n = len(rx)
    px = float(rx[0]) + start_offset[0]
    py = float(ry[0]) + start_offset[1]
    yaw = float(rh[0]) + start_offset[2]
    v, delta = float(v0), 0.0
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
        delta += float(np.clip(target_steer - delta, -sv_max * dt, sv_max * dt))
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
    return dict(completed=cum >= n, lap_time=t, xte=xte, idx=np.array(idx),
                xte_max=float(steady.max()) if len(steady) else float('inf'),
                grip_capped=capped / max(len(xte), 1))


# ── objective ────────────────────────────────────────────────────────────────
class Objective:
    """Turns a config dict into a score.  Lower is better."""

    def __init__(self, raceline, backend='grip', trials=12, wall=0.9,
                 mu=1.0489, sv_max=3.2, max_steer=0.41, wheelbase=0.33,
                 min_success=0.9, seed=0, map_path=None, laps=2, mus=None):
        self.rx, self.ry, self.rh, self.rc, self.base_speed = \
            cl.load_raceline(raceline)
        self.ds = segment_lengths(self.rx, self.ry)
        self.backend = backend
        self.trials = int(trials)
        self.wall = float(wall)
        # Grip values every candidate must survive.  Tuning at a single mu
        # optimizes for one exact surface and silently rewards setups that sit
        # on the edge of the tyres: the fastest config found at mu 1.05 can be
        # 100% reliable there and 0% at 0.90, which is one dusty patch, one set
        # of worn tyres, or one sagging battery away.  Scoring across a range
        # is what makes "consistent" mean "still finishes on a bad day".
        self.mus = [float(m) for m in (mus if mus else [mu])]
        self.mu, self.sv_max = float(mu), float(sv_max)
        self.max_steer, self.wheelbase = float(max_steer), float(wheelbase)
        self.min_success = float(min_success)
        self.map_path = map_path
        self.laps = int(laps)
        # One fixed set of perturbed starts, shared by every candidate: the
        # configs must be compared on identical conditions, otherwise the search
        # is just chasing which candidate drew the easier starts.
        rng = np.random.default_rng(seed)
        self.starts = [(float(rng.uniform(-0.15, 0.15)),
                        float(rng.uniform(-0.10, 0.10)),
                        float(rng.uniform(-0.10, 0.10)))
                       for _ in range(self.trials)]

    def speeds_for(self, cfg):
        """Re-profile the line's speeds from its own curvature under `cfg`.

        This is what makes the search cheap: the geometry stays fixed and only
        the speed column is recomputed, so a candidate costs milliseconds
        instead of a full raceline-optimizer run.
        """
        v = velocity_profile(self.rc, self.ds,
                             a_lat_max=cfg['a_lat'],
                             a_accel_max=cfg['a_accel'],
                             a_brake_max=cfg['a_brake'],
                             v_max=cfg['v_max'])
        return np.clip(v * cfg['v_scale'], 0.5, None)

    def make_mpc(self, cfg, speeds):
        mpc = KinematicMPC(wheelbase=self.wheelbase, max_steer=self.max_steer,
                           horizon=int(round(cfg['horizon'])),
                           v_max=float(speeds.max()) + 0.5,
                           max_accel=cfg['a_accel'], max_brake=cfg['a_brake'],
                           q_pos=cfg['q_pos'], q_yaw=cfg['q_yaw'],
                           q_v=cfg['q_v'], rd_steer=cfg['rd_steer'])
        mpc.set_raceline(self.rx, self.ry, self.rh, self.rc, speeds)
        return mpc

    def __call__(self, cfg):
        speeds = self.speeds_for(cfg)
        mpc = self.make_mpc(cfg, speeds)
        if not mpc.available:
            raise SystemExit('osqp is not installed — the MPC cannot solve.\n'
                             'pip install osqp==0.6.3')

        def ctrl(px, py, yaw, v, j):
            out = mpc.solve((px, py, yaw, v), j)
            return out if out is not None else (0.0, v)

        if self.backend == 'gym':
            return self._score_gym(cfg, speeds, mpc)

        v0 = float(speeds[0])                       # flying start
        per_mu, scores = {}, []
        for mu in self.mus:
            laps, ok, caps = [], 0, []
            for (dx, dy, dyaw) in self.starts:
                mpc.reset()                 # every trial starts identically
                r = run_lap_grip(ctrl, self.rx, self.ry, self.rh, mu=mu,
                                 sv_max=self.sv_max, max_steer=self.max_steer,
                                 wheelbase=self.wheelbase,
                                 start_offset=(dx, dy, dyaw), v0=v0,
                                 a_accel=cfg['a_accel'], a_brake=cfg['a_brake'])
                caps.append(r['grip_capped'])
                wide = float(r['xte'].max()) if len(r['xte']) else float('inf')
                if r['completed'] and wide < self.wall:
                    ok += 1
                    laps.append(r['lap_time'])
            score, info = self._combine(laps, ok,
                                        dict(grip_capped=float(np.mean(caps))))
            scores.append(score)
            per_mu[f'{mu:.2f}'] = dict(success=info['success'],
                                       mean_lap=info.get('mean_lap'))
        if len(self.mus) == 1:
            mu0 = per_mu[f'{self.mus[0]:.2f}']
            return scores[0], dict(success=mu0['success'],
                                   mean_lap=mu0['mean_lap'], per_mu=per_mu)
        # Mean across grip levels. A config that is quick at nominal grip but
        # cannot finish at the low end takes a FAIL_SCORE into the average and
        # loses to a slightly slower one that survives the whole range -- which
        # is the trade the search is supposed to be making.
        nominal = per_mu[f'{self.mus[0]:.2f}']
        return float(np.mean(scores)), dict(
            success=min(m['success'] for m in per_mu.values()),
            mean_lap=nominal['mean_lap'], per_mu=per_mu,
            worst_mu=min(per_mu, key=lambda k: per_mu[k]['success']))

    def _score_gym(self, cfg, speeds, mpc):
        """Real f110_gym dynamics — slower, but with true collision detection."""
        try:
            import gym
        except Exception:
            try:
                import gymnasium as gym
            except Exception:
                raise SystemExit(
                    'the gym backend needs f110_gym:\n'
                    '  git clone https://github.com/f1tenth/f1tenth_gym\n'
                    '  cd f1tenth_gym && pip install -e .')
        env = gym.make('f110_gym:f110-v0', map=self.map_path,
                       map_ext='.png', num_agents=1)
        n = len(self.rx)
        laps, ok = [], 0
        for k, (dx, dy, _dyaw) in enumerate(self.starts):
            s0 = (k * n // max(len(self.starts), 1)) % n     # spread round the lap
            res = env.reset(np.array([[float(self.rx[s0]) + dx,
                                       float(self.ry[s0]) + dy,
                                       float(self.rh[s0])]]))
            obs = res[0] if isinstance(res, tuple) else res
            prev_j, cum, t, crashed = s0, 0, 0.0, False
            while cum < self.laps * n and t < 120.0:
                x, y = _s(obs['poses_x']), _s(obs['poses_y'])
                yaw = math.atan2(math.sin(_s(obs['poses_theta'])),
                                 math.cos(_s(obs['poses_theta'])))
                v = _s(obs['linear_vels_x'])
                j = int(np.argmin((self.rx - x) ** 2 + (self.ry - y) ** 2))
                d = j - prev_j
                if d < -n / 2:
                    d += n
                if 0 < d < n / 2:
                    cum += d
                prev_j = j
                out = mpc.solve((x, y, yaw, v), j)
                steer, v_t = out if out is not None else (0.0, v)
                step = env.step(np.array([[float(steer), float(v_t)]]))
                obs = step[0]
                t += 0.01
                if _s(obs['collisions']) > 0.5:
                    crashed = True
                    break
            if not crashed and cum >= self.laps * n:
                ok += 1
                laps.append(t / self.laps)
        return self._combine(laps, ok, {'backend': 'gym'})

    def _combine(self, laps, ok, extra):
        """Expected time to COMPLETE a lap, including restarts after a crash."""
        rate = ok / float(self.trials)
        if ok == 0:
            return FAIL_SCORE, dict(success=0.0, mean_lap=None, **extra)
        mean_lap = float(np.mean(laps))
        score = mean_lap / max(rate, 1e-3)
        if rate < self.min_success:
            # An explicit cliff below the reliability floor. Without it the
            # search happily trades a big chunk of reliability for a small
            # chunk of lap time, which is exactly the setup that loses races.
            score += 100.0 * (self.min_success - rate)
        return min(score, FAIL_SCORE), dict(success=rate, mean_lap=mean_lap,
                                            lap_sd=float(np.std(laps)), **extra)


def _s(v):
    try:
        return float(v[0])
    except (TypeError, IndexError):
        return float(v)


def hardware_max_steer(default=0.41):
    """The car's actual steering limit, from config/hardware.yaml.

    Read rather than hardcoded because it is a property of the linkage that
    changes with the car: the Traxxas Slash servo tops out at 0.36 rad and 0.41
    drives it into the stops.  A search that assumes more steering than the car
    has will happily hand back a setup that only works in simulation.
    """
    path = os.path.join(REPO, 'config', 'hardware.yaml')
    try:
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        val = cfg.get('drive_node', {}).get('ros__parameters', {}).get('max_steer')
        return float(val) if val else default
    except Exception:
        return default


# ── session ──────────────────────────────────────────────────────────────────
def load_history(path, names):
    """Past evaluations, so --resume continues rather than restarting."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            cfg = rec.get('config') or {}
            if all(n in cfg for n in names) and 'score' in rec:
                out.append((cfg, float(rec['score'])))
    return out


def main():
    ap = argparse.ArgumentParser(
        description='Bayesian optimization of the racing setup.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raceline',
                    default=os.path.join(REPO, 'racelines', 'comp_raceline.csv'))
    ap.add_argument('--map', default=os.path.join(REPO, 'maps', 'comp_track'),
                    help='map WITHOUT extension (gym backend only)')
    ap.add_argument('--backend', choices=['grip', 'gym'], default='grip')
    ap.add_argument('--iters', type=int, default=40,
                    help='total evaluations to spend')
    ap.add_argument('--init', type=int, default=8,
                    help='random designs before the model takes over')
    ap.add_argument('--trials', type=int, default=12,
                    help='perturbed-start laps per candidate')
    ap.add_argument('--laps', type=int, default=2, help='laps per trial (gym)')
    ap.add_argument('--mu', type=float, default=1.0489,
                    help='tyre friction — sweep this to bracket the real surface')
    ap.add_argument('--mu-range', type=float, nargs='+', default=None,
                    metavar='MU',
                    help='score every candidate at these friction values and '
                         'average, so the winner has to survive a worse surface '
                         'than the nominal one (e.g. --mu-range 1.05 0.95 0.85)')
    ap.add_argument('--wall', type=float, default=0.9,
                    help='cross-track (m) counted as leaving the track')
    ap.add_argument('--max-steer', type=float, default=None,
                    help='steering limit (rad); defaults to drive_node.max_steer '
                         'in config/hardware.yaml so the search cannot silently '
                         'assume more steering than the car has')
    ap.add_argument('--min-success', type=float, default=0.9,
                    help='reliability floor; below it a config is penalized hard')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', action='store_true',
                    help='seed the model with runtime/bayes_log.jsonl')
    ap.add_argument('--apply', action='store_true',
                    help='write the winner into config/hardware.yaml')
    args = ap.parse_args()

    os.makedirs(RUNTIME, exist_ok=True)
    max_steer = args.max_steer if args.max_steer else hardware_max_steer()
    obj = Objective(args.raceline, backend=args.backend, trials=args.trials,
                    wall=args.wall, mu=args.mu, min_success=args.min_success,
                    seed=args.seed, map_path=args.map, laps=args.laps,
                    mus=args.mu_range, max_steer=max_steer)
    names = [s[0] for s in SPACE]
    opt = BayesOpt(SPACE, n_init=args.init, seed=args.seed)

    print(f'raceline  {os.path.basename(args.raceline)}  '
          f'({len(obj.rx)} pts, base v {obj.base_speed.min():.1f}-'
          f'{obj.base_speed.max():.1f} m/s)')
    grip = ('mu ' + ', '.join(f'{m:g}' for m in obj.mus)) if len(obj.mus) > 1 \
        else f'mu {args.mu:g}'
    print(f'backend   {args.backend}   {grip}   max_steer {max_steer:.2f} rad   '
          f'{args.trials} perturbed starts per candidate'
          + (f' x {len(obj.mus)} grip levels' if len(obj.mus) > 1 else ''))
    print(f'objective mean lap / success rate, floor {args.min_success:.0%}'
          + (', averaged across grip' if len(obj.mus) > 1 else '') + '\n')

    # The GP models log(score).  Lap time and the failure penalty then live on
    # one multiplicative scale, so a crashed candidate is "much worse" without
    # swamping the differences between the configs that actually work.
    best_cfg, best_score = None, float('inf')

    if args.resume:
        prior = load_history(LOG, names)
        for cfg, score in prior:
            opt.tell(cfg, math.log(max(score, 1e-3)))
            if score < best_score:
                best_cfg, best_score = dict(cfg), score
        if prior:
            print(f'resumed with {len(prior)} past evaluation(s), '
                  f'best {best_score:.2f}\n')

    print(f"{'#':>3} {'score':>8} {'lap':>7} {'ok':>6}  best  config")
    print('-' * 78)
    t0 = time.time()
    for i in range(1, args.iters + 1):
        cfg = opt.ask()
        cfg['horizon'] = float(round(cfg['horizon']))       # integer-valued
        try:
            score, info = obj(cfg)
        except SystemExit:
            raise
        except Exception as e:                              # a bad config must
            score, info = FAIL_SCORE, {'error': str(e)}     # not kill the run
        opt.tell(cfg, math.log(max(score, 1e-3)))

        rec = dict(i=i, t=round(time.time() - t0, 1), config=cfg,
                   score=score, info=info, backend=args.backend)
        with open(LOG, 'a') as f:
            f.write(json.dumps(rec) + '\n')

        improved = score < best_score
        if improved:
            best_cfg, best_score = dict(cfg), score
            with open(BEST, 'w') as f:
                json.dump(dict(config=cfg, score=score, info=info,
                               backend=args.backend, raceline=args.raceline,
                               ts=time.time()), f, indent=2)
        lap = info.get('mean_lap')
        print(f'{i:>3} {score:>8.2f} '
              f"{(f'{lap:.2f}s' if lap else '   —  '):>7} "
              f"{info.get('success', 0.0):>5.0%}  "
              f"{'NEW ' if improved else '    '}  "
              + ' '.join(f'{k}={cfg[k]:.2f}' for k in
                         ('v_scale', 'a_lat', 'q_pos', 'rd_steer')))

    print('-' * 78)
    if best_cfg is None:
        print('\nNo configuration completed a lap. Widen the ranges, lower '
              '--mu, or check the raceline with `atlas run doctor`.\n')
        return 1
    cfg, score = best_cfg, best_score
    _, info = obj(cfg)
    print(f'\nBest after {args.iters} evaluations  (score {score:.2f})')
    for k in names:
        print(f'  {k:<10} {cfg[k]:.3f}')
    print(f"\n  reliability  {info.get('success', 0):.0%} of "
          f"{args.trials} perturbed starts")
    if info.get('mean_lap'):
        print(f"  lap time     {info['mean_lap']:.2f}s "
              f"± {info.get('lap_sd', 0):.2f}")
    imp = opt.importance()
    if imp:
        # importance() is in log-score units; exp(range)-1 turns it back into
        # "how much lap time this knob swings, as a fraction"
        ranked = sorted(imp.items(), key=lambda kv: -kv[1])
        print('\n  what actually mattered (lap-time swing across each range):')
        for k, v in ranked[:5]:
            print(f'    {k:<10} {math.expm1(min(v, 5.0)) * 100:>6.0f}%')
    print(f'\n  written to {os.path.relpath(BEST, REPO)}')

    if args.backend == 'grip':
        print('\n  NEXT: confirm on the real dynamics before you race it —')
        print('    python3 tools/atlas.py run bayes-tune -- --backend gym '
              '--iters 5 --resume')
        print(f"    python3 tools/atlas.py run validate -- --v-scale "
              f"{cfg['v_scale']:.2f}")
    print('  On the car, start ~10% under the winning v_scale.\n')

    if args.apply:
        apply_to_hardware(cfg)
    return 0


def apply_to_hardware(cfg):
    """Write the tuned values into config/hardware.yaml, preserving comments.

    A line-wise edit rather than a YAML round-trip: ruamel is not a dependency
    here, and pyyaml's dump would strip every comment in that file — which is
    where the calibration knowledge lives.
    """
    path = os.path.join(REPO, 'config', 'hardware.yaml')
    mapping = {
        'v_scale': cfg['v_scale'],
        'max_lat_accel': cfg['a_lat'],
        'profile_a_accel': cfg['a_accel'],
        'profile_a_brake': cfg['a_brake'],
        'profile_v_max': cfg['v_max'],
        # The controller half of the result. Without these the search's MPC
        # weights never reach the car and only the speed profile is deployed —
        # which is not the setup that was measured.
        'mpc_q_pos': cfg['q_pos'],
        'mpc_q_yaw': cfg['q_yaw'],
        'mpc_q_v': cfg['q_v'],
        'mpc_rd_steer': cfg['rd_steer'],
        'mpc_horizon': cfg['horizon'],
    }
    try:
        with open(path) as f:
            lines = f.readlines()
    except Exception as e:
        print(f'  could not read {path}: {e}')
        return
    changed = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        for key, val in mapping.items():
            if stripped.startswith(key + ':'):
                indent = line[:len(line) - len(stripped)]
                # keep any trailing comment — that is where the calibration
                # notes live, and they outlive any single tuning run
                comment = '  #' + line.split('#', 1)[1].rstrip() \
                    if '#' in line else ''
                lines[i] = f'{indent}{key}: {val:.3f}{comment}\n'
                changed.append(key)
    with open(path, 'w') as f:
        f.writelines(lines)
    print(f"  applied to config/hardware.yaml: {', '.join(changed)}")


if __name__ == '__main__':
    sys.exit(main())
