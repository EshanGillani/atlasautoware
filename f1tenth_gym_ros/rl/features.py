"""
Observation construction — the one definition of what the policy sees.
======================================================================

This module exists so that training and deployment cannot drift apart.  A
learned policy is only valid on the exact observation distribution it was
trained on; if the simulator builds its vector one way and the ROS node builds
it another — a different beam count, a different normalization, curvature in
the other sign convention — the policy still *runs*, silently, and drives into
a wall.  So both call `build_observation` here, and the checkpoint records the
`ObsSpec` it was trained with (see `ObsSpec.fingerprint`), which the node
refuses to load against a mismatched spec.

Everything is pure numpy: no torch, no ROS, no gym.  That makes it testable on
any machine (tests/test_rl.py) and cheap enough to run inside a 50 Hz control
loop.

What the policy sees
--------------------
    lidar        `n_beams` downsampled ranges, clipped and scaled to [0, 1]
    kinematics   speed, lateral velocity proxy, yaw rate — all normalized
    line-frame   signed cross-track error and heading error against the raceline
    preview      curvature and target speed at several distances ahead
    baseline     the MPC's own proposed (steer, speed), normalized

The last block is what makes a *residual* policy work: the network is told what
the classical controller intends to do, so its job is the much easier one of
learning a correction ("the MPC understeers through turn 4, carry more angle")
rather than rediscovering how to drive a car from scratch.

The preview block is what makes it race rather than follow walls: a lidar-only
policy cannot know that a hairpin follows the fast right-hander, so it brakes
reactively and late.  Feeding curvature ahead lets it plan.
"""

import math

import numpy as np


class ObsSpec:
    """Shape and normalization of the observation vector.

    Stored in every checkpoint.  Two specs that differ anywhere produce
    incompatible policies, which `fingerprint` makes cheap to detect.
    """

    def __init__(self, n_beams=108, max_range=10.0, v_ref=8.0,
                 preview_m=(0.5, 1.0, 2.0, 3.5, 5.0, 7.0),
                 max_steer=0.41, max_curv=1.5, max_yaw_rate=6.0):
        self.n_beams = int(n_beams)
        self.max_range = float(max_range)
        self.v_ref = float(v_ref)
        self.preview_m = tuple(float(p) for p in preview_m)
        self.max_steer = float(max_steer)
        self.max_curv = float(max_curv)
        self.max_yaw_rate = float(max_yaw_rate)

    # 4 kinematics + 2 line-frame + 2 per preview point + 2 baseline
    @property
    def n_state(self):
        return 4 + 2 + 2 * len(self.preview_m) + 2

    @property
    def dim(self):
        return self.n_beams + self.n_state

    def fingerprint(self):
        """Compact identity of this observation layout."""
        return (f'b{self.n_beams}_r{self.max_range:g}_v{self.v_ref:g}'
                f'_p{"-".join(f"{p:g}" for p in self.preview_m)}'
                f'_s{self.max_steer:g}_k{self.max_curv:g}')

    def to_dict(self):
        return dict(n_beams=self.n_beams, max_range=self.max_range,
                    v_ref=self.v_ref, preview_m=list(self.preview_m),
                    max_steer=self.max_steer, max_curv=self.max_curv,
                    max_yaw_rate=self.max_yaw_rate)

    @staticmethod
    def from_dict(d):
        return ObsSpec(**{k: v for k, v in d.items()
                          if k in ('n_beams', 'max_range', 'v_ref', 'preview_m',
                                   'max_steer', 'max_curv', 'max_yaw_rate')})

    def __eq__(self, other):
        return isinstance(other, ObsSpec) and self.fingerprint() == other.fingerprint()


# ── lidar ────────────────────────────────────────────────────────────────────
def downsample_scan(ranges, n_beams, max_range):
    """Reduce a raw scan to `n_beams` values in [0, 1], nearest-obstacle-wins.

    Min-pooling, not averaging or striding: the value that matters for not
    hitting something is the CLOSEST return in each sector.  Averaging would
    smooth a thin obstacle (a cone, another car's wheel) out of existence, and
    plain striding would miss it whenever it falls between sampled indices.

    Non-finite returns (the lidar's "no echo") become max range, which is what
    they physically mean.
    """
    r = np.asarray(ranges, dtype=np.float32)
    if r.size == 0:
        return np.ones(n_beams, dtype=np.float32)
    r = np.where(np.isfinite(r) & (r > 0.02), r, max_range)
    r = np.clip(r, 0.0, max_range)
    if r.size == n_beams:
        out = r
    elif r.size > n_beams:
        # trim to a whole multiple so the reshape is exact, then min-pool
        k = r.size // n_beams
        out = r[:k * n_beams].reshape(n_beams, k).min(axis=1)
    else:
        idx = np.linspace(0, r.size - 1, n_beams)
        out = np.interp(idx, np.arange(r.size), r).astype(np.float32)
    return (out / max_range).astype(np.float32)


# ── raceline frame ───────────────────────────────────────────────────────────
def signed_cross_track(px, py, rx, ry, j):
    """Signed perpendicular offset from the line at index j.

    Positive = left of the direction of travel.  The sign matters: an unsigned
    error tells the policy it is off-line but not which way to correct, which
    makes the learning problem needlessly ambiguous.
    """
    n = len(rx)
    tx = rx[(j + 1) % n] - rx[(j - 1) % n]
    ty = ry[(j + 1) % n] - ry[(j - 1) % n]
    tn = math.hypot(tx, ty) + 1e-9
    # left normal of the tangent is (-ty, tx)/|t|
    return float((px - rx[j]) * (-ty / tn) + (py - ry[j]) * (tx / tn))


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def preview_indices(j, s_cum, preview_m, n):
    """Raceline indices `preview_m` metres ahead of index j, wrapping the loop."""
    total = float(s_cum[-1]) + 1e-9
    s0 = float(s_cum[j])
    out = []
    for d in preview_m:
        target = (s0 + d) % total
        out.append(int(np.searchsorted(s_cum, target) % n))
    return out


def arclength(rx, ry):
    """Cumulative distance along a closed raceline (same length as the line)."""
    dx = np.diff(rx, append=rx[0])
    dy = np.diff(ry, append=ry[0])
    return np.cumsum(np.hypot(dx, dy))


# ── the observation ──────────────────────────────────────────────────────────
def build_observation(spec, scan, x, y, yaw, v, yaw_rate,
                      rx, ry, rh, rc, rspeed, s_cum, j,
                      base_steer=0.0, base_speed=0.0):
    """Assemble the full observation vector.

    Every element is normalized to roughly [-1, 1] so no single input dominates
    the first layer before the network has learned anything.

    j is the nearest raceline index — the caller already computes it for the
    controller, so it is passed in rather than recomputed.
    """
    n = len(rx)
    lidar = downsample_scan(scan, spec.n_beams, spec.max_range)

    e_lat = signed_cross_track(x, y, rx, ry, j)
    e_yaw = wrap_angle(yaw - float(rh[j]))

    state = [
        # Clipped like every other term: an EKF or odometry glitch reporting a
        # wild speed would otherwise arrive unbounded at the first layer and
        # swamp the inputs that are bounded, which is precisely the silent
        # input corruption this observation contract exists to prevent.
        float(np.clip(v / spec.v_ref, 0.0, 2.0)),        # speed
        float(np.clip(yaw_rate / spec.max_yaw_rate, -1, 1)),
        # a slip proxy: how far the measured yaw rate is from what the current
        # speed and path curvature say it should be.  Large when sliding.
        float(np.clip((yaw_rate - v * float(rc[j])) / spec.max_yaw_rate, -1, 1)),
        float(np.clip(v * abs(float(rc[j])) / 9.81, 0, 2)),   # lateral-g demand
        float(np.clip(e_lat / 1.5, -2, 2)),
        float(np.clip(e_yaw / math.pi, -1, 1)),
    ]
    for k in preview_indices(j, s_cum, spec.preview_m, n):
        state.append(float(np.clip(rc[k] / spec.max_curv, -1, 1)))
        state.append(float(np.clip(rspeed[k] / spec.v_ref, 0, 1.5)))
    state.append(float(np.clip(base_steer / spec.max_steer, -1, 1)))
    state.append(float(np.clip(base_speed / spec.v_ref, 0, 1.5)))

    obs = np.concatenate([lidar, np.asarray(state, dtype=np.float32)])
    # A NaN here would propagate silently through the network and out to the
    # servo; clamp it at the boundary instead.
    return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)


# ── action ───────────────────────────────────────────────────────────────────
class ResidualAction:
    """Maps a policy output in [-1, 1]^2 onto a bounded correction of the MPC.

    The safety argument for the whole approach lives here.  The policy cannot
    command the car directly; it can only bend what the MPC already decided, by
    at most `d_steer` rad and `d_speed` m/s, further scaled by `authority`
    (0 = pure MPC, 1 = the full envelope).  So:

      - an untrained network drives exactly as well as the MPC does,
      - a diverged or NaN-producing network is caught and ignored per-tick,
      - and you raise `authority` a notch at a time on the real car, watching a
        clean lap at each step, instead of handing a black box the servo.

    It also means the AEB and traction governor downstream are untouched — the
    policy never gets to override a brake command.
    """

    def __init__(self, d_steer=0.10, d_speed=1.5, max_steer=0.41,
                 v_min=0.0, v_max=8.0, authority=1.0):
        self.d_steer = float(d_steer)
        self.d_speed = float(d_speed)
        self.max_steer = float(max_steer)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.authority = float(np.clip(authority, 0.0, 1.0))

    def apply(self, action, base_steer, base_speed):
        """(policy_action, MPC steer, MPC speed) -> (steer, speed) to publish."""
        a = np.asarray(action, dtype=np.float64).ravel()
        if a.size < 2 or not np.all(np.isfinite(a)):
            return float(base_steer), float(base_speed)      # fall back cleanly
        a = np.clip(a[:2], -1.0, 1.0) * self.authority
        steer = float(base_steer) + a[0] * self.d_steer
        speed = float(base_speed) + a[1] * self.d_speed
        return (float(np.clip(steer, -self.max_steer, self.max_steer)),
                float(np.clip(speed, self.v_min, self.v_max)))
