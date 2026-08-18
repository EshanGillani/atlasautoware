"""
Racing environment — the f110_gym dynamics wrapped for reinforcement learning.
==============================================================================

Wraps the real f1tenth_gym single-track dynamic model (the same physics the sim
race runs, with genuine collision detection) into a step/reset interface that
produces the observation defined in `features.py` and a reward that means
"go round this track faster without crashing".

The MPC is *inside* the environment
-----------------------------------
Every step, the MPC solves as it normally would, and the policy's action is a
bounded correction to that command (see `features.ResidualAction`).  This is
the single most important design decision here, and it buys three things:

  1. **Exploration starts competent.**  A from-scratch policy on a race track
     spends its first hundred thousand steps discovering that walls are bad.
     Starting from the MPC, the very first episode completes laps, so every
     sample is collected in the part of the state space the car will actually
     race in.
  2. **The learning problem is small.**  "Where does the MPC leave time on the
     table" is a much lower-variance target than "how do I drive".
  3. **The same wrapper is the safety envelope at deployment.**  Training and
     racing share one clamp, so behaviour cannot diverge between them.

Reward
------
    + progress      metres advanced along the raceline this step — the term
                    that actually asks for speed. Distance, not velocity, so
                    the policy gets nothing for pointing fast at a wall.
    - deviation     for drifting off the line, quadratic, so small corrections
                    are nearly free and large ones are not
    - effort        for using the residual at all, which keeps the policy near
                    the MPC unless deviating genuinely pays
    - crash         a large one-off penalty, then the episode ends
    + finish        a bonus for completing the lap, so it prefers a clean lap
                    to a fast fragment

Progress is the dominant term by design: everything else is a shaping term that
stops the policy buying speed with risk it cannot see.

Needs f110_gym (and numpy).  Does not need torch or ROS — you can exercise the
environment and its reward with a scripted policy, which tests/test_rl.py does.
"""

import math
import os

import numpy as np

from .features import (ObsSpec, ResidualAction, arclength, build_observation,
                       pursuit_steer, signed_cross_track, wrap_angle)


def _load_raceline(path):
    """CSV -> (x, y, heading, curvature, speed).  Mirrors closed_loop.load_raceline
    without importing from tests/, so the package stands on its own."""
    import csv
    cols = {k: [] for k in ('x', 'y', 'heading', 'curvature', 'speed')}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k in cols:
                cols[k].append(float(row[k]))
    return tuple(np.asarray(cols[k], dtype=float) for k in cols)


def _scalar(v):
    """f110_gym returns per-agent arrays or bare scalars depending on version."""
    try:
        return float(v[0])
    except (TypeError, IndexError):
        return float(v)


def _unpack_reset(res):
    return res[0] if isinstance(res, tuple) else res


# ── renderer camera ──────────────────────────────────────────────────────────
# f110_gym draws cars at 50x world coordinates and initialises its camera around
# the ORIGIN, with no follow behaviour -- only manual mouse pan and scroll. On a
# map whose coordinates are tens of metres from zero (comp_track's start is near
# (49, 62) m, i.e. (2450, 3100) renderer units) the cars sit far outside the
# default view, so you get a window showing an empty corner of the track and
# nothing else. This registers the documented render callback, which runs after
# update_obs and before on_draw, to keep the ego centred.
RENDER_HALF_WIDTH = 700.0          # renderer units (~14 m at the 50x scale)


def follow_ego(renderer):
    """Centre the camera on the ego car. Registered via add_render_callback."""
    poses = getattr(renderer, 'poses', None)
    if poses is None or not len(poses):
        return
    idx = int(getattr(renderer, 'ego_idx', 0) or 0)
    idx = min(idx, len(poses) - 1)
    cx, cy = float(poses[idx][0]) * 50.0, float(poses[idx][1]) * 50.0
    half_w = RENDER_HALF_WIDTH
    # Preserve the window's aspect ratio so the track is not stretched.
    width = float(getattr(renderer, 'width', 1000) or 1000)
    height = float(getattr(renderer, 'height', 800) or 800)
    half_h = half_w * (height / max(width, 1.0))
    renderer.left, renderer.right = cx - half_w, cx + half_w
    renderer.bottom, renderer.top = cy - half_h, cy + half_h


class RaceEnv:
    """Racing environment over the real gym dynamics.

    Gym-like, but deliberately not a `gym.Env` subclass: f110_gym's own API
    shifts between the gym and gymnasium releases, and keeping our interface
    independent means the training loop does not care which one is installed.

        env = RaceEnv('maps/comp_track', 'racelines/comp_raceline.csv')
        obs = env.reset()
        obs, reward, done, info = env.step(action)   # action in [-1, 1]^2
    """

    def __init__(self, map_path, raceline, spec=None, mpc_factory=None,
                 max_steps=6000, laps=1, crash_penalty=40.0, finish_bonus=25.0,
                 deviation_weight=0.35, effort_weight=0.02, progress_weight=1.0,
                 max_deviation=1.2, d_steer=0.10, d_speed=1.5, authority=1.0,
                 v_scale=1.0, random_start=True, map_ext='.png', seed=0,
                 control_dt=0.01, launch_speed=1.5):
        self.spec = spec or ObsSpec()
        self.rx, self.ry, self.rh, self.rc, self.rspeed = _load_raceline(raceline)
        self.rspeed = self.rspeed * float(v_scale)
        self.n = len(self.rx)
        self.s_cum = arclength(self.rx, self.ry)
        self.track_len = float(self.s_cum[-1])

        self.max_steps = int(max_steps)
        self.laps = int(laps)
        self.crash_penalty = float(crash_penalty)
        self.finish_bonus = float(finish_bonus)
        self.w_dev = float(deviation_weight)
        self.w_eff = float(effort_weight)
        self.w_prog = float(progress_weight)
        self.max_deviation = float(max_deviation)
        self.random_start = bool(random_start)
        self.control_dt = float(control_dt)
        self.launch_speed = float(launch_speed)
        self.rng = np.random.default_rng(seed)

        self.residual = ResidualAction(
            d_steer=d_steer, d_speed=d_speed, max_steer=self.spec.max_steer,
            v_min=0.0, v_max=float(self.rspeed.max()) + d_speed,
            authority=authority)

        self.mpc = (mpc_factory or self._default_mpc)()
        self.obs_dim = self.spec.dim
        self.action_dim = 2

        self._env = self._make_gym(map_path, map_ext)
        self._obs = None
        self.j = 0
        self.steps = 0
        self.progress = 0.0

    # ── construction helpers ────────────────────────────────────────────────
    def _default_mpc(self):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        from mpc_controller import KinematicMPC
        mpc = KinematicMPC(v_max=float(self.rspeed.max()) + 0.5)
        if not mpc.available:
            raise SystemExit(
                'the residual policy needs the MPC baseline, which needs osqp:\n'
                '  pip install osqp==0.6.3')
        mpc.set_raceline(self.rx, self.ry, self.rh, self.rc, self.rspeed)
        return mpc

    @staticmethod
    def _make_gym(map_path, map_ext):
        try:
            import gym
        except Exception:
            try:
                import gymnasium as gym
            except Exception:
                raise SystemExit(
                    'training needs f110_gym:\n'
                    '  git clone https://github.com/f1tenth/f1tenth_gym\n'
                    '  cd f1tenth_gym && pip install -e .')
        return gym.make('f110_gym:f110-v0', map=map_path, map_ext=map_ext,
                        num_agents=1)

    # ── episode ─────────────────────────────────────────────────────────────
    def reset(self, start_idx=None):
        """Start a new episode, by default somewhere random on the line.

        Random starts matter more here than in most RL problems: a fixed start
        lets the policy memorise one lap as a sequence rather than learn a
        control law, and it then falls apart the moment the car is nudged off
        that trajectory — which is exactly what a real start line, or contact
        with another car, does.
        """
        if start_idx is None:
            start_idx = int(self.rng.integers(self.n)) if self.random_start else 0
        j0 = int(start_idx) % self.n
        x, y, yaw = float(self.rx[j0]), float(self.ry[j0]), float(self.rh[j0])
        if self.random_start:
            # perturb within the corridor the car could plausibly be in
            e = float(self.rng.uniform(-0.25, 0.25))
            nx, ny = -math.sin(yaw), math.cos(yaw)          # left normal
            x += e * nx
            y += e * ny
            yaw += float(self.rng.uniform(-0.12, 0.12))

        self.mpc.reset()
        self._obs = _unpack_reset(self._env.reset(np.array([[x, y, yaw]])))
        self.j = j0
        self.steps = 0
        self.progress = 0.0
        self._prev_j = j0
        self._last_action = np.zeros(2)
        return self._observe()[0]

    def _pose(self):
        o = self._obs
        yaw = wrap_angle(_scalar(o['poses_theta']))      # gym gives [0, 2π)
        return (_scalar(o['poses_x']), _scalar(o['poses_y']), yaw,
                _scalar(o['linear_vels_x']), _scalar(o['ang_vels_z'])
                if 'ang_vels_z' in o else 0.0)

    def _observe(self):
        """-> (observation, baseline (steer, speed), pose)."""
        x, y, yaw, v, yaw_rate = self._pose()
        self.j = int(np.argmin((self.rx - x) ** 2 + (self.ry - y) ** 2))
        base = self._baseline(x, y, yaw, v, self.j)
        scan = self._obs.get('scans', [[]])
        scan = scan[0] if len(scan) and np.ndim(scan[0]) else scan
        obs = build_observation(
            self.spec, scan, x, y, yaw, v, yaw_rate,
            self.rx, self.ry, self.rh, self.rc, self.rspeed, self.s_cum, self.j,
            base_steer=base[0], base_speed=base[1])
        return obs, base, (x, y, yaw, v)

    def _baseline(self, x, y, yaw, v, j):
        """The MPC's command, or a pure-pursuit launch if the car is barely moving.

        f110_gym resets with zero velocity, so without the launch branch every
        episode began with the MPC unable to steer and the car leaving the track
        within a few metres -- which made the baseline, and therefore everything
        the policy learned against it, meaningless.
        """
        if v < self.launch_speed:
            steer = pursuit_steer(x, y, yaw, self.rx, self.ry, j, 1.0,
                                  self.mpc.L, self.mpc.max_steer)
            return steer, self.launch_speed + 1.0
        out = self.mpc.solve((x, y, yaw, v), j)
        return out if out is not None else (0.0, float(self.rspeed[j]))

    def step(self, action):
        """action in [-1, 1]^2 — a correction of the MPC, not a raw command."""
        obs, base, (x, y, yaw, v) = self._observe()
        steer, speed = self.residual.apply(action, base[0], base[1])

        prev_s = float(self.s_cum[self.j])
        self._obs, done_env = self._gym_step(steer, speed)
        self.steps += 1

        nx, ny, nyaw, nv, _ = self._pose()
        j_new = int(np.argmin((self.rx - nx) ** 2 + (self.ry - ny) ** 2))
        d_s = self._advance(prev_s, float(self.s_cum[j_new]))
        self.progress += d_s
        self.j = j_new

        crashed = _scalar(self._obs['collisions']) > 0.5
        dev = abs(signed_cross_track(nx, ny, self.rx, self.ry, j_new))
        off = dev > self.max_deviation
        finished = self.progress >= self.laps * self.track_len
        timeout = self.steps >= self.max_steps

        reward = self.w_prog * d_s
        reward -= self.w_dev * (dev / self.max_deviation) ** 2
        reward -= self.w_eff * float(np.sum(np.square(np.clip(action, -1, 1))))
        if crashed or off:
            reward -= self.crash_penalty
        if finished:
            reward += self.finish_bonus

        done = bool(crashed or off or finished or timeout or done_env)
        info = dict(progress=self.progress, deviation=dev, crashed=bool(crashed),
                    off_track=bool(off), finished=bool(finished),
                    timeout=bool(timeout), speed=nv, steer=steer,
                    base_steer=base[0], base_speed=base[1],
                    sim_time=self.steps * self.control_dt)
        next_obs = self._observe()[0] if not done else obs
        return next_obs, float(reward), done, info

    def _advance(self, prev_s, new_s):
        """Arc-length gained, handling the start/finish wrap.

        A raw difference goes hugely negative when the car crosses the line,
        which would hand the policy a giant penalty for the one thing it is
        supposed to do.  Large backward jumps are the wrap; genuine reversing
        stays small and negative, and is penalised as it should be.
        """
        d = new_s - prev_s
        if d < -self.track_len / 2.0:
            d += self.track_len
        elif d > self.track_len / 2.0:
            d -= self.track_len
        return float(d)

    def _gym_step(self, steer, speed):
        res = self._env.step(np.array([[float(steer), float(speed)]]))
        obs = res[0]
        done = bool(res[2]) or (len(res) >= 5 and bool(res[3]))
        return obs, done

    # ── the demonstration policy ────────────────────────────────────────────
    def mpc_action(self):
        """The zero action — i.e. "do exactly what the MPC says".

        This is what warm-starting imitates.  It looks trivial, and that is the
        point: because the action space is defined as a residual, the expert
        demonstration is a constant, and behaviour cloning reduces to teaching
        the policy to output zero everywhere before it starts exploring.  That
        gives the replay buffer a whole distribution of on-track states, at
        racing speed, before a single gradient step is taken on reward.
        """
        return np.zeros(2, dtype=np.float32)

    def render(self, mode='human'):
        """Draw the current frame, if a display is reachable.

        Deliberately best-effort: f110_gym's renderer is pyglet, which needs an
        X display, and a training run must never die because nobody was
        watching.  The first failure disables rendering for the rest of the
        run rather than raising once per step.
        """
        if getattr(self, '_render_broken', False):
            return
        try:
            if not getattr(self, '_camera_hooked', False):
                # Register once. Without it the camera stays at the origin and
                # the cars are off-screen on any map not centred on zero.
                try:
                    self._env.unwrapped.add_render_callback(follow_ego)
                except Exception:
                    self._env.add_render_callback(follow_ego)
                self._camera_hooked = True
            self._env.render(mode=mode)
        except Exception as exc:
            self._render_broken = True
            print(f'[render] disabled ({exc}). Inside the sim container this '
                  f'needs the noVNC display: docker-compose brings one up, then '
                  f'open http://localhost:8080/vnc.html', flush=True)

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass
