"""
Wheel-to-wheel decision policy — observation, style, and the safety envelope.
=============================================================================

The single-agent policy (`features.py`) learns a correction to the MPC's
*control* command.  This learns a correction to the race brain's *decision*, and
the difference is the whole point.

`race_brain.RaceStrategist` already answers "what is the situation" — CRUISE,
ATTACK, DEFEND, EVADE — and emits a `Decision(mode, offset, speed_factor)`:
where to sit relative to the racing line, and how hard to push.  `spliner` then
turns that offset into a geometrically sane Frenet evasion line, and the MPC
tracks it.  What is *hand-tuned* is the judgement: how much room to take, when
a gap is big enough to commit, how hard to lean on someone defending.

So the policy outputs a bounded correction to `offset` and `speed_factor`, and
nothing else.  That buys the same guarantees the single-agent design has:

  - the **mode stays rule-based and explicit**, so you can still read why the
    car did what it did off `/rl/status` — which matters enormously when a
    steward asks, and when you are debugging at 2 a.m.;
  - **`spliner` still generates the geometry**, so a learned decision cannot
    produce a physically silly line, only a different sane one;
  - **AEB and the traction governor are still downstream**, so the policy can
    never talk the car out of braking;
  - **an untrained policy races exactly as the rule-based brain does**, and
    `authority = 0` is bit-for-bit the existing behaviour.

And the action space is two numbers rather than a control law, which is why this
is trainable in hours on a laptop instead of weeks on a cluster.

Driving styles
--------------
`style` is a single scalar, 0 = conservative through 1 = aggressive.  It enters
in three places, which is what makes it a real behaviour knob rather than a
label:

  1. the **observation**, so one network can produce different behaviour on
     demand instead of needing a policy per style;
  2. the **reward** during training (`style_reward_weights`), so aggressive
     rollouts are actually rewarded for track position and punished less for
     proximity, while conservative ones are punished hard for contact risk;
  3. the **action envelope** at deployment, so turning the knob down at the
     track also physically shrinks how far the policy may deviate.

At a competition that is one number: push it up when you must overtake to
advance, drop it when a points finish is worth more than a place.

Pure numpy — no torch, no ROS, no simulator.  Tested in tests/test_duel.py.
"""

import math

import numpy as np

MODES = ('CRUISE', 'ATTACK', 'DEFEND', 'EVADE')
MODE_INDEX = {m: i for i, m in enumerate(MODES)}


class DuelSpec:
    """Shape and normalization of the wheel-to-wheel observation.

    Carries a fingerprint for the same reason `features.ObsSpec` does: a policy
    is only valid on the layout it was trained against, and a silent mismatch
    is the failure the car cannot detect from the inside.
    """

    def __init__(self, v_ref=8.0, track_half=1.0, attack_range=6.0,
                 defend_range=5.0, preview_m=(2.0, 5.0, 9.0, 14.0),
                 max_curv=1.5, d_offset=0.35, d_speed=0.12):
        self.v_ref = float(v_ref)
        self.track_half = float(track_half)
        self.attack_range = float(attack_range)
        self.defend_range = float(defend_range)
        self.preview_m = tuple(float(p) for p in preview_m)
        self.max_curv = float(max_curv)
        self.d_offset = float(d_offset)      # m of lateral correction allowed
        self.d_speed = float(d_speed)        # fraction of speed_factor allowed

    # ego(4) + preview + room(2) + ahead(4) + behind(4) + zone(1)
    # + baseline mode one-hot(4) + baseline offset/speed(2) + style(1)
    @property
    def dim(self):
        return 4 + len(self.preview_m) + 2 + 4 + 4 + 1 + len(MODES) + 2 + 1

    def fingerprint(self):
        return (f'duel_v{self.v_ref:g}_h{self.track_half:g}'
                f'_a{self.attack_range:g}_d{self.defend_range:g}'
                f'_p{"-".join(f"{p:g}" for p in self.preview_m)}'
                f'_o{self.d_offset:g}_s{self.d_speed:g}')

    def to_dict(self):
        return dict(v_ref=self.v_ref, track_half=self.track_half,
                    attack_range=self.attack_range,
                    defend_range=self.defend_range,
                    preview_m=list(self.preview_m), max_curv=self.max_curv,
                    d_offset=self.d_offset, d_speed=self.d_speed)

    @staticmethod
    def from_dict(d):
        keys = ('v_ref', 'track_half', 'attack_range', 'defend_range',
                'preview_m', 'max_curv', 'd_offset', 'd_speed')
        return DuelSpec(**{k: v for k, v in d.items() if k in keys})

    def __eq__(self, other):
        return isinstance(other, DuelSpec) and \
            self.fingerprint() == other.fingerprint()


class FlatObsSpec:
    """A state-only observation, for policies with no lidar block.

    The decision layer reasons over gaps, closing speeds and the rule-based
    decision — not raw range data — so its network is a plain MLP.  Declaring
    `n_beams = 0` is what tells `networks.Torso` to skip the CNN encoder.
    """

    def __init__(self, dim):
        self.n_beams = 0
        self.n_state = int(dim)
        self.dim = int(dim)

    def fingerprint(self):
        return f'flat{self.dim}'

    def to_dict(self):
        return dict(n_beams=0, n_state=self.n_state, flat=True)

    @staticmethod
    def from_dict(d):
        return FlatObsSpec(int(d.get('n_state', d.get('dim', 0))))

    def __eq__(self, other):
        return isinstance(other, FlatObsSpec) and \
            self.fingerprint() == other.fingerprint()


class Rival:
    """One opponent as the decision layer sees it.

    Deliberately the same quantities `RaceStrategist` computes (gap along the
    track, lateral offset, closing speed) rather than raw poses: the policy
    should reason in the frame the rules already reason in, and those three
    numbers are what actually decide a pass.
    """

    __slots__ = ('gap', 'lateral', 'closing')

    def __init__(self, gap, lateral, closing):
        self.gap = float(gap)              # +ahead / -behind, metres along track
        self.lateral = float(lateral)      # +left of the racing line, metres
        self.closing = float(closing)      # +we are catching them, m/s


def _finite(v, default=0.0):
    v = float(v)
    return v if math.isfinite(v) else default


def build_duel_observation(spec, ego_speed, ego_lateral, ego_heading_err,
                           curvature_preview, room_left, room_right,
                           rivals, baseline_mode, baseline_offset,
                           baseline_speed_factor, style, in_overtake_zone=False):
    """Assemble the decision-layer observation.

    `curvature_preview` is the signed raceline curvature at `spec.preview_m`
    metres ahead — the same information that tells you whether the next corner
    is where a pass sticks or where it ends in the wall.

    `rivals` is any iterable of `Rival`; the nearest ahead and nearest behind
    are selected here, because those are the two that decide the next move and
    a variable-length list cannot be fed to a fixed-width network.
    """
    ahead = [r for r in rivals if r.gap > 0.0]
    behind = [r for r in rivals if r.gap <= 0.0]
    nearest_ahead = min(ahead, key=lambda r: r.gap) if ahead else None
    nearest_behind = max(behind, key=lambda r: r.gap) if behind else None

    obs = [
        np.clip(_finite(ego_speed) / spec.v_ref, 0.0, 2.0),
        np.clip(_finite(ego_lateral) / spec.track_half, -2.0, 2.0),
        np.clip(_finite(ego_heading_err) / math.pi, -1.0, 1.0),
        np.clip(_finite(ego_speed) * abs(_finite(curvature_preview[0]
                                                 if len(curvature_preview) else 0.0))
                / 9.81, 0.0, 2.0),                       # lateral-g demand now
    ]
    for i in range(len(spec.preview_m)):
        k = curvature_preview[i] if i < len(curvature_preview) else 0.0
        obs.append(np.clip(_finite(k) / spec.max_curv, -1.0, 1.0))
    obs.append(np.clip(_finite(room_left) / spec.track_half, 0.0, 2.0))
    obs.append(np.clip(_finite(room_right) / spec.track_half, 0.0, 2.0))

    for rival, rng in ((nearest_ahead, spec.attack_range),
                       (nearest_behind, spec.defend_range)):
        if rival is None:
            # Absent rival: the presence flag is what distinguishes "nobody
            # there" from "somebody exactly alongside", which zeros alone would
            # not — an ambiguity that would teach the policy to treat an empty
            # track like a car on your door.
            obs += [0.0, 1.0, 0.0, 0.0]
        else:
            obs += [1.0,
                    float(np.clip(abs(rival.gap) / rng, 0.0, 2.0)),
                    float(np.clip(rival.lateral / spec.track_half, -2.0, 2.0)),
                    float(np.clip(rival.closing / spec.v_ref, -1.0, 1.0))]

    obs.append(1.0 if in_overtake_zone else 0.0)
    one_hot = [0.0] * len(MODES)
    one_hot[MODE_INDEX.get(str(baseline_mode).upper(), 0)] = 1.0
    obs += one_hot
    obs.append(float(np.clip(_finite(baseline_offset) / spec.track_half, -2, 2)))
    obs.append(float(np.clip(_finite(baseline_speed_factor), 0.0, 2.0)))
    obs.append(float(np.clip(_finite(style), 0.0, 1.0)))

    return np.nan_to_num(np.asarray(obs, dtype=np.float32),
                         nan=0.0, posinf=1.0, neginf=-1.0)


class StrategyResidual:
    """Bounded correction to a `Decision`, scaled by authority and style.

    The safety argument, in one place:

      - the correction is at most `d_offset` metres and `d_speed` of speed
        factor, so a diverged policy nudges the line rather than reinventing it;
      - the corrected offset is clamped into the room actually available on each
        side, so the policy cannot steer the car off the road even if it wants
        to — that clamp is geometric, not learned;
      - non-finite or malformed output falls back to the rule-based decision;
      - `authority = 0` reproduces the rule-based brain exactly.

    `style` additionally scales the envelope, so the trackside knob shrinks the
    physical room the policy has, not just the behaviour it prefers.
    """

    def __init__(self, spec, authority=1.0, style=0.5, min_speed_factor=0.5,
                 max_speed_factor=1.35):
        self.spec = spec
        self.authority = float(np.clip(authority, 0.0, 1.0))
        self.style = float(np.clip(style, 0.0, 1.0))
        self.min_speed_factor = float(min_speed_factor)
        self.max_speed_factor = float(max_speed_factor)

    def scale(self):
        """How much of the envelope is live right now.

        Style floors at 0.25 rather than 0: a fully conservative policy should
        still be able to make small adjustments, because refusing to move at all
        is its own way of causing a collision.
        """
        return self.authority * (0.25 + 0.75 * self.style)

    def apply(self, action, decision_offset, decision_speed_factor,
              room_left, room_right):
        """-> (offset, speed_factor) to hand to spliner / the controller."""
        a = np.asarray(action, dtype=np.float64).ravel()
        if a.size < 2 or not np.all(np.isfinite(a)):
            return float(decision_offset), float(decision_speed_factor)
        a = np.clip(a[:2], -1.0, 1.0) * self.scale()

        offset = float(decision_offset) + a[0] * self.spec.d_offset
        speed = float(decision_speed_factor) + a[1] * self.spec.d_speed
        # Geometric clamp: never outside the room that exists.
        offset = float(np.clip(offset, -abs(float(room_right)),
                               abs(float(room_left))))
        speed = float(np.clip(speed, self.min_speed_factor,
                              self.max_speed_factor))
        return offset, speed


def style_reward_weights(style):
    """Reward shaping as a function of style.

    Interpolating the *weights* rather than training separate policies is what
    lets one network cover the range, and it keeps the styles honest: an
    aggressive setting is not "ignore contact", it is "value track position
    more and proximity less". Contact is never free at either end, because a
    collision ends the race whatever the strategy said.
    """
    s = float(np.clip(style, 0.0, 1.0))
    return dict(
        progress=1.0,
        # aggressive: overtaking is worth much more
        overtake=2.0 + 6.0 * s,
        # aggressive: being passed hurts more (defend harder)
        overtaken=-(1.5 + 4.5 * s),
        # conservative: sitting close to another car is discouraged
        proximity=-(0.9 - 0.75 * s),
        # contact is heavily penalized at BOTH ends, slightly less when
        # aggressive so the policy will still commit to a legitimate move
        contact=-(60.0 - 15.0 * s),
        # conservative: deviating from the rule-based decision costs more
        effort=-(0.30 - 0.25 * s),
    )
