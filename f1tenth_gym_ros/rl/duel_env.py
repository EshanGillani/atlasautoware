"""
Two-car racing environment for the decision policy.
===================================================

Wraps f110_gym with two agents.  The ego runs the full deployed stack — race
brain decides, the policy nudges that decision, the MPC tracks the resulting
offset line — and the opponent is scripted.  Training against a scripted rival
first (rather than self-play from scratch) is deliberate: self-play is
non-stationary and unstable, and a policy that cannot beat a predictable
opponent will certainly not beat an unpredictable one.  Self-play belongs on top
of this, not instead of it.

    ego:   RaceStrategist -> Decision(mode, offset, speed_factor)
                          -> + policy residual (bounded)
                          -> MPC.solve(state, nearest, offset) -> /drive
    rival: raceline follower at a configurable pace, holding a lateral bias

The opponent is intentionally simple and *varied* rather than clever.  What
makes an overtaking policy generalize is meeting many different paces and lines,
not one well-played rival — a policy trained against a single scripted
behaviour learns to beat that behaviour, which is exactly the overfitting that
does not survive contact with another team.

Reward (weights from `duel.style_reward_weights`, so style genuinely changes
what is being optimized):

    + progress    metres gained along the racing line
    + overtake    for passing the rival
    - overtaken   for being passed
    - proximity   for sitting inside the contact bubble
    - contact     collision, ends the episode
    - effort      for deviating from the rule-based decision at all

Needs f110_gym and numpy; no torch, no ROS, so the environment and its reward
can be exercised with a scripted policy (tests/test_duel.py does exactly that).
"""

import math
import os
import sys

import numpy as np

from .duel import (DuelSpec, Rival, StrategyResidual, build_duel_observation,
                   style_reward_weights)
from .features import pursuit_steer

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _load_raceline(path):
    import csv
    cols = {k: [] for k in ('x', 'y', 'heading', 'curvature', 'speed')}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k in cols:
                cols[k].append(float(row[k]))
    return tuple(np.asarray(cols[k], dtype=float) for k in cols)


def _scalar(v, i=0):
    try:
        return float(v[i])
    except (TypeError, IndexError):
        return float(v)


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


def load_corridor(map_yaml, rx, ry, car_half=0.15):
    """Per-raceline-point usable half-width, measured from the occupancy map.

    Returns None if the map cannot be read, in which case the caller falls back
    to a constant.

    This exists because assuming a constant corridor is badly wrong on a real
    track.  comp_track's raceline has a median clearance of 0.67 m to the wall
    but a minimum of 0.21 m, so the usable half-width -- what is left after the
    car's own 0.15 m half-width -- ranges from 0.99 m down to 0.06 m.  A
    decision layer told it has 1.0 m everywhere will happily park the car 0.55 m
    off-line to make a pass, which is inside the wall on 54% of this lap.
    """
    try:
        import yaml
        from PIL import Image
        from scipy import ndimage
    except Exception:
        return None
    try:
        with open(map_yaml) as f:
            meta = yaml.safe_load(f)
        img_path = meta['image']
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.path.dirname(map_yaml), img_path)
        res = float(meta['resolution'])
        origin = meta['origin']
        img = np.array(Image.open(img_path).convert('L'))
        H, W = img.shape
        p = img / 255.0 if int(meta.get('negate', 0)) else (255 - img) / 255.0
        occ = p > float(meta.get('occupied_thresh', 0.65))
        edt = ndimage.distance_transform_edt(~occ) * res
        col = (np.asarray(rx, float) - origin[0]) / res
        row = (H - 1) - (np.asarray(ry, float) - origin[1]) / res
        clear = ndimage.map_coordinates(edt, [row, col], order=1)
        return np.maximum(clear - float(car_half), 0.02)
    except Exception:
        return None


class ScriptedRival:
    """A raceline follower with a pace and a lateral bias.

    Pure pursuit rather than MPC: it must be cheap (it runs every tick of every
    training episode) and it must be *beatable but not trivial*. Randomising
    pace and bias per episode is what stops the policy memorising one rival.
    """

    def __init__(self, rx, ry, speed, pace=0.9, bias=0.0, wheelbase=0.33,
                 max_steer=0.36, lookahead=1.2):
        self.rx, self.ry, self.speed = rx, ry, speed
        self.pace = float(pace)
        self.bias = float(bias)
        self.L = float(wheelbase)
        self.max_steer = float(max_steer)
        self.lookahead = float(lookahead)
        self._nx, self._ny = _normals(rx, ry)

    def act(self, x, y, yaw, v):
        n = len(self.rx)
        j = int(np.argmin((self.rx - x) ** 2 + (self.ry - y) ** 2))
        tj = j
        for off in range(1, n):
            k = (j + off) % n
            if math.hypot(self.rx[k] - x, self.ry[k] - y) >= self.lookahead:
                tj = k
                break
        tx = self.rx[tj] + self.bias * self._nx[tj]
        ty = self.ry[tj] + self.bias * self._ny[tj]
        dx, dy = tx - x, ty - y
        lx = dx * math.cos(yaw) + dy * math.sin(yaw)
        ly = -dx * math.sin(yaw) + dy * math.cos(yaw)
        ld2 = lx * lx + ly * ly
        steer = 0.0 if ld2 < 1e-6 else math.atan(self.L * 2.0 * ly / ld2)
        return (float(np.clip(steer, -self.max_steer, self.max_steer)),
                float(self.speed[j] * self.pace))


def _normals(x, y):
    tx = np.roll(x, -1) - np.roll(x, 1)
    ty = np.roll(y, -1) - np.roll(y, 1)
    tn = np.hypot(tx, ty) + 1e-9
    return -ty / tn, tx / tn          # left normal


class DuelEnv:
    """Ego + one scripted rival on the real gym dynamics."""

    def __init__(self, map_path, raceline, spec=None, map_ext='.png',
                 max_steps=6000, laps=1, v_scale=1.0, authority=1.0,
                 style=None, seed=0, contact_bubble=0.9, max_deviation=1.4,
                 max_steer=0.36, rival_pace=(0.80, 0.95), launch_speed=1.5,
                 map_yaml=None, car_half=0.15, off_track_margin=0.75):
        self.spec = spec or DuelSpec()
        self.rx, self.ry, self.rh, self.rc, self.rspeed = _load_raceline(raceline)
        self.rspeed = self.rspeed * float(v_scale)
        self.n = len(self.rx)
        self.nx, self.ny = _normals(self.rx, self.ry)
        dx = np.diff(self.rx, append=self.rx[0])
        dy = np.diff(self.ry, append=self.ry[0])
        self.seg = np.hypot(dx, dy)
        self.s_cum = np.cumsum(self.seg)
        self.track_len = float(self.s_cum[-1])
        self.spacing = float(self.seg.mean())

        self.max_steps = int(max_steps)
        self.laps = int(laps)
        self.contact_bubble = float(contact_bubble)
        self.max_deviation = float(max_deviation)
        self.max_steer = float(max_steer)
        self.rival_pace = tuple(rival_pace)
        self.launch_speed = float(launch_speed)
        self.off_track_margin = float(off_track_margin)
        self.fixed_style = style
        self.rng = np.random.default_rng(seed)

        # Usable corridor around the raceline, MEASURED from the occupancy
        # map where possible.  A constant is badly wrong on a real track:
        # comp_track's usable half-width runs from 0.99 m down to 0.06 m, so a
        # decision layer told it has 1.0 m everywhere parks the car 0.55 m
        # off-line to pass -- inside the wall on 54% of that lap.
        if map_yaml is None:
            guess = str(map_path) + '.yaml'
            map_yaml = guess if os.path.exists(guess) else None
        measured = (load_corridor(map_yaml, self.rx, self.ry, car_half)
                    if map_yaml else None)
        self.corridor_measured = measured is not None
        self.half_width = (measured if self.corridor_measured else
                           np.full(self.n, float(self.spec.track_half)))

        # Where ATTACK parks the car must fit the NARROW part of the track,
        # not the average.  Only derive it when the corridor is real: derived
        # from the fallback constant it would be 0.8 x a made-up width, which
        # is worse than the strategist's own default.
        side = (float(np.percentile(self.half_width, 20)) * 0.8
                if self.corridor_measured else 0.55)

        from race_brain import RaceStrategist
        self.brain = RaceStrategist(attack_range=self.spec.attack_range,
                                    defend_range=self.spec.defend_range,
                                    side_clearance=side,
                                    track_half=float(np.median(self.half_width)))
        self.residual = StrategyResidual(self.spec, authority=authority)
        self.mpc = self._make_mpc()
        self.obs_dim = self.spec.dim
        self.action_dim = 2
        self._env = self._make_gym(map_path, map_ext)
        self.style = 0.5

    def _make_mpc(self):
        from mpc_controller import KinematicMPC
        mpc = KinematicMPC(max_steer=self.max_steer,
                           v_max=float(self.rspeed.max()) + 0.5)
        if not mpc.available:
            raise SystemExit('the decision policy needs the MPC baseline, which '
                             'needs osqp:\n  pip install osqp==0.6.3')
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
                    'the duel environment needs f110_gym:\n'
                    '  git clone https://github.com/f1tenth/f1tenth_gym\n'
                    '  cd f1tenth_gym && pip install -e .')
        return gym.make('f110_gym:f110-v0', map=map_path, map_ext=map_ext,
                        num_agents=2)

    # ── episode ─────────────────────────────────────────────────────────────
    def reset(self, style=None, start_idx=None, rival_gap=None):
        """Start an episode, by default with the rival somewhere plausible.

        The rival is placed either ahead (an overtaking problem) or behind (a
        defending one), because a policy only ever shown one of those learns
        half a race.
        """
        # Precedence: an explicit request wins, then a style fixed for the whole
        # run, then a fresh sample. The explicit case is what per-style
        # evaluation depends on -- ignoring it would silently score random
        # styles under style labels and make the whole comparison meaningless.
        if style is not None:
            self.style = float(np.clip(style, 0.0, 1.0))
        elif self.fixed_style is not None:
            self.style = float(np.clip(self.fixed_style, 0.0, 1.0))
        else:
            self.style = float(self.rng.uniform(0.0, 1.0))
        self.residual.style = self.style
        self.weights = style_reward_weights(self.style)

        j0 = int(self.rng.integers(self.n)) if start_idx is None else int(start_idx)
        gap = rival_gap if rival_gap is not None else \
            float(self.rng.choice([-1, 1]) * self.rng.uniform(1.5, 5.0))
        j_riv = int((j0 + round(gap / self.spacing)) % self.n)

        self.rival = ScriptedRival(
            self.rx, self.ry, self.rspeed,
            pace=float(self.rng.uniform(*self.rival_pace)),
            bias=float(self.rng.uniform(-0.35, 0.35)),
            max_steer=self.max_steer)

        poses = np.array([
            [float(self.rx[j0]), float(self.ry[j0]), float(self.rh[j0])],
            [float(self.rx[j_riv]), float(self.ry[j_riv]), float(self.rh[j_riv])],
        ])
        self.mpc.reset()
        res = self._env.reset(poses)
        self._obs = res[0] if isinstance(res, tuple) else res
        self.steps = 0
        self.progress = 0.0
        self.j = j0
        self._prev_s = float(self.s_cum[j0])
        self._prev_gap = self._rival_gap()
        self._passes = 0
        self._passed_by = 0
        return self._observe()[0]

    # ── geometry helpers ────────────────────────────────────────────────────
    def _pose(self, i):
        o = self._obs
        yaw = _scalar(o['poses_theta'], i)
        return (_scalar(o['poses_x'], i), _scalar(o['poses_y'], i),
                math.atan2(math.sin(yaw), math.cos(yaw)),
                _scalar(o['linear_vels_x'], i))

    def _project(self, x, y):
        j = int(np.argmin((self.rx - x) ** 2 + (self.ry - y) ** 2))
        lateral = (x - self.rx[j]) * self.nx[j] + (y - self.ry[j]) * self.ny[j]
        return j, float(lateral)

    def _rival_gap(self):
        ex, ey, _, _ = self._pose(0)
        rx, ry, _, _ = self._pose(1)
        je, _ = self._project(ex, ey)
        jr, _ = self._project(rx, ry)
        d = (jr - je) % self.n
        gap = d * self.spacing
        if gap > self.track_len / 2.0:
            gap -= self.track_len
        return gap

    def _preview(self, j):
        out = []
        for m in self.spec.preview_m:
            k = int((j + round(m / self.spacing)) % self.n)
            out.append(float(self.rc[k]))
        return out

    def _observe(self):
        ex, ey, eyaw, ev = self._pose(0)
        rx_, ry_, ryaw, rv = self._pose(1)
        self.j, ego_lat = self._project(ex, ey)
        _, riv_lat = self._project(rx_, ry_)
        gap = self._rival_gap()
        closing = ev - rv
        rival = Rival(gap, riv_lat, closing)

        half = float(self.half_width[self.j])
        room_left = max(0.0, half - ego_lat)
        room_right = max(0.0, half + ego_lat)
        heading_err = math.atan2(math.sin(eyaw - self.rh[self.j]),
                                 math.cos(eyaw - self.rh[self.j]))

        decision = self.brain.decide(
            self.j, ev, self.rx, self.ry, self.rspeed, room_left, room_right,
            [_BrainOpponent(rx_, ry_, rv * math.cos(ryaw), rv * math.sin(ryaw))])

        obs = build_duel_observation(
            self.spec, ev, ego_lat, heading_err, self._preview(self.j),
            room_left, room_right, [rival], decision.mode, decision.offset,
            decision.speed_factor, self.style)
        return obs, decision, (ego_lat, room_left, room_right, gap, ev)

    def _baseline(self, x, y, yaw, v, j, offset):
        """The MPC tracking the offset line, or a pure-pursuit launch from rest.

        f110_gym resets with zero velocity and the MPC cannot steer a stationary
        car (yaw_dot = v*tan(delta)/L is zero at v = 0), so without this the
        rule-based baseline left the track within a few metres of every spawn --
        4 contacts in 4 episodes, ~10 m of a 243 m lap -- and the policy was
        being trained against a baseline that could not drive.
        """
        if v < self.launch_speed:
            steer = pursuit_steer(x, y, yaw, self.rx, self.ry, j, 1.0,
                                  self.mpc.L, self.max_steer,
                                  offset=offset, nx=self.nx, ny=self.ny)
            return steer, self.launch_speed + 1.0
        out = self.mpc.solve((x, y, yaw, v), j, offset=offset)
        return out if out is not None else (0.0, float(self.rspeed[j]))

    def step(self, action):
        obs, decision, (ego_lat, room_l, room_r, gap, ev) = self._observe()
        offset, speed_factor = self.residual.apply(
            action, decision.offset, decision.speed_factor, room_l, room_r)

        ex, ey, eyaw, _ = self._pose(0)
        steer, v_cmd = self._baseline(ex, ey, eyaw, ev, self.j, offset)
        v_cmd *= speed_factor

        rx_, ry_, ryaw, rv = self._pose(1)
        r_steer, r_speed = self.rival.act(rx_, ry_, ryaw, rv)

        prev_s = float(self.s_cum[self.j])
        res = self._env.step(np.array([[float(steer), float(v_cmd)],
                                       [float(r_steer), float(r_speed)]]))
        self._obs = res[0]
        done_env = bool(res[2]) or (len(res) >= 5 and bool(res[3]))
        self.steps += 1

        nex, ney, _, nev = self._pose(0)
        j_new, new_lat = self._project(nex, ney)
        d_s = self._advance(prev_s, float(self.s_cum[j_new]))
        self.progress += d_s
        self.j = j_new

        new_gap = self._rival_gap()
        # A pass is a sign change in the along-track gap while the cars are
        # close enough for it to be a real overtake rather than a lap apart.
        overtook = self._prev_gap > 0 >= new_gap and abs(new_gap) < 8.0
        overtaken = self._prev_gap < 0 <= new_gap and abs(new_gap) < 8.0
        self._passes += int(overtook)
        self._passed_by += int(overtaken)
        self._prev_gap = new_gap

        contact = _scalar(self._obs['collisions'], 0) > 0.5
        # f110_gym's own collision detection is the real arbiter of hitting a
        # wall, and `contact` above carries it.  This is only a 'lost the
        # plot' bound for a car that has wandered somewhere the corridor
        # cannot explain.  It deliberately is NOT the measured corridor: the
        # MPC's own cross-track error runs 0.13-0.47 m on this line while the
        # tightest usable half-width is 0.06 m, so a strict corridor test
        # would end almost every episode on normal tracking error.
        limit = min(self.max_deviation,
                    float(self.half_width[j_new]) + self.off_track_margin)
        off_track = abs(new_lat) > limit
        finished = self.progress >= self.laps * self.track_len
        timeout = self.steps >= self.max_steps

        w = self.weights
        reward = w['progress'] * d_s
        reward += w['overtake'] * overtook
        reward += w['overtaken'] * overtaken
        if abs(new_gap) < self.contact_bubble:
            reward += w['proximity']
        reward += w['effort'] * float(np.sum(np.square(np.clip(action, -1, 1))))
        if contact or off_track:
            reward += w['contact']

        done = bool(contact or off_track or finished or timeout or done_env)
        info = dict(style=self.style, mode=decision.mode, gap=new_gap,
                    offset=offset, speed_factor=speed_factor,
                    baseline_offset=decision.offset, progress=self.progress,
                    contact=bool(contact), off_track=bool(off_track),
                    finished=bool(finished), timeout=bool(timeout),
                    passes=self._passes, passed_by=self._passed_by,
                    thought=decision.thought)
        next_obs = self._observe()[0] if not done else obs
        return next_obs, float(reward), done, info

    def _advance(self, prev_s, new_s):
        d = new_s - prev_s
        if d < -self.track_len / 2.0:
            d += self.track_len
        elif d > self.track_len / 2.0:
            d -= self.track_len
        return float(d)

    def rule_based_action(self):
        """Zero — i.e. race exactly as the rule-based brain decides.

        As in the single-agent design, the expert demonstration is a constant
        because the action space is a residual, so warm-starting reduces to
        teaching the policy to defer before it explores.
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


class _BrainOpponent:
    """The shape race_brain.RaceStrategist expects an opponent to have."""

    def __init__(self, x, y, vx, vy):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.s_idx = 0
        self.lateral = 0.0
        self.gap = 0.0
        self.closing = 0.0
