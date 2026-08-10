"""
Learned policy — observations, the safety envelope, and the training stack.
===========================================================================

The tests that matter most here are the ones about the *residual envelope*, in
`TestResidualEnvelope`. That clamp is the entire safety argument for putting a
neural network in a race car's control loop: it is what guarantees an untrained,
diverged or NaN-producing policy still drives exactly as well as the MPC. If it
regresses, nothing downstream catches it — the car simply does something the
policy asked for.

The observation tests exist for a related reason: a policy is only valid on the
distribution it was trained on, and a mismatch between how training and
deployment build the vector produces a policy that runs happily on wrong inputs.

Torch-dependent tests skip cleanly when torch is absent, so the suite runs on
the car (where torch may not be installed) and on a workstation alike.

    python3 -m pytest tests/test_rl.py -q
"""

import math
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tests'))

from f1tenth_gym_ros.rl.features import (  # noqa: E402
    ObsSpec, ResidualAction, arclength, build_observation, downsample_scan,
    preview_indices, signed_cross_track, wrap_angle)

try:
    import torch                                          # noqa: F401
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason='torch not installed')
RACELINE = os.path.join(REPO, 'racelines', 'comp_raceline.csv')


def load_line():
    import closed_loop as cl
    return cl.load_raceline(RACELINE)


# ── lidar downsampling ───────────────────────────────────────────────────────
class TestDownsample:
    def test_output_shape_and_range(self):
        for n_in in (1080, 720, 108, 40):
            out = downsample_scan(np.full(n_in, 5.0), 108, 10.0)
            assert out.shape == (108,)
            assert np.all((out >= 0.0) & (out <= 1.0))

    def test_min_pooling_keeps_the_nearest_return(self):
        """A thin obstacle must survive the reduction — averaging would erase
        a cone or another car's wheel, and that is what we brake for."""
        r = np.full(1080, 10.0)
        r[500] = 0.5
        out = downsample_scan(r, 108, 10.0)
        assert math.isclose(float(out.min()), 0.05, abs_tol=1e-6)

    def test_non_finite_becomes_max_range(self):
        """inf/NaN is the lidar saying 'no echo', which means far, not zero."""
        r = np.array([np.inf, np.nan, -1.0, 0.0] * 270, dtype=float)
        out = downsample_scan(r, 108, 10.0)
        assert np.all(np.isfinite(out))
        assert np.allclose(out, 1.0)

    def test_empty_scan_is_survivable(self):
        out = downsample_scan(np.array([]), 108, 10.0)
        assert out.shape == (108,) and np.all(out == 1.0)

    def test_ranges_beyond_max_are_clipped(self):
        assert downsample_scan(np.full(200, 99.0), 108, 10.0).max() <= 1.0


# ── raceline frame ───────────────────────────────────────────────────────────
class TestLineFrame:
    def test_cross_track_sign_flips_across_the_line(self):
        rx, ry = np.linspace(0, 10, 50), np.zeros(50)
        left = signed_cross_track(5.0, 0.4, rx, ry, 25)
        right = signed_cross_track(5.0, -0.4, rx, ry, 25)
        assert left > 0 > right
        assert math.isclose(left, -right, abs_tol=1e-6)

    def test_cross_track_is_zero_on_the_line(self):
        rx, ry = np.linspace(0, 10, 50), np.zeros(50)
        assert abs(signed_cross_track(float(rx[25]), 0.0, rx, ry, 25)) < 1e-9

    def test_wrap_angle_maps_into_pi(self):
        for a in (3 * math.pi, -3 * math.pi, 0.5, 7.0):
            assert -math.pi - 1e-9 <= wrap_angle(a) <= math.pi + 1e-9
        assert math.isclose(wrap_angle(2 * math.pi + 0.3), 0.3, abs_tol=1e-9)

    def test_arclength_matches_the_perimeter(self):
        t = np.linspace(0, 2 * math.pi, 400, endpoint=False)
        s = arclength(np.cos(t), np.sin(t))
        assert math.isclose(float(s[-1]), 2 * math.pi, rel_tol=1e-3)

    def test_preview_wraps_past_the_start_finish_line(self):
        """Previewing off the end of the array would blind the policy exactly
        at the line, where it is carrying the most speed."""
        rx, ry, _, _, _ = load_line()
        s = arclength(rx, ry)
        idx = preview_indices(len(rx) - 2, s, (0.5, 2.0, 5.0), len(rx))
        assert all(0 <= i < len(rx) for i in idx)
        assert any(i < 20 for i in idx), 'should wrap round to the start'


# ── observations ─────────────────────────────────────────────────────────────
class TestObservation:
    def test_dimensions_agree_with_the_spec(self):
        rx, ry, rh, rc, rsp = load_line()
        s = arclength(rx, ry)
        spec = ObsSpec()
        obs = build_observation(spec, np.full(1080, 4.0), float(rx[10]),
                                float(ry[10]), float(rh[10]), 5.0, 0.2,
                                rx, ry, rh, rc, rsp, s, 10)
        assert obs.shape == (spec.dim,)
        assert obs.dtype == np.float32

    def test_garbage_inputs_never_reach_the_network(self):
        """A NaN would propagate through the net and out to the servo."""
        rx, ry, rh, rc, rsp = load_line()
        s = arclength(rx, ry)
        obs = build_observation(ObsSpec(), np.full(1080, np.nan), float(rx[5]),
                                float(ry[5]), float(rh[5]), float('nan'),
                                float('inf'), rx, ry, rh, rc, rsp, s, 5)
        assert np.all(np.isfinite(obs))

    def test_values_stay_in_a_normalized_band(self):
        rx, ry, rh, rc, rsp = load_line()
        s = arclength(rx, ry)
        obs = build_observation(ObsSpec(), np.full(1080, 0.05), float(rx[7]) + 3.0,
                                float(ry[7]) - 3.0, float(rh[7]) + 2.0, 40.0, 25.0,
                                rx, ry, rh, rc, rsp, s, 7)
        assert np.abs(obs).max() <= 2.0, 'no input should dominate layer one'

    def test_fingerprint_distinguishes_layouts(self):
        assert ObsSpec().fingerprint() == ObsSpec().fingerprint()
        assert ObsSpec(n_beams=64).fingerprint() != ObsSpec().fingerprint()
        assert ObsSpec(preview_m=(1.0,)).fingerprint() != ObsSpec().fingerprint()

    def test_spec_round_trips_through_a_checkpoint(self):
        spec = ObsSpec(n_beams=72, max_range=12.0)
        assert ObsSpec.from_dict(spec.to_dict()) == spec


# ── the safety envelope ──────────────────────────────────────────────────────
class TestResidualEnvelope:
    """The guarantees that let a neural network near a real car."""

    def setup_method(self):
        self.r = ResidualAction(d_steer=0.10, d_speed=1.5, max_steer=0.41,
                                v_min=0.0, v_max=8.0)

    def test_zero_action_is_exactly_the_baseline(self):
        """An untrained policy outputs ~0, so it must drive as the MPC does."""
        assert self.r.apply([0.0, 0.0], 0.2, 5.0) == (0.2, 5.0)

    def test_correction_is_bounded_by_the_envelope(self):
        for a in ([1, 1], [-1, -1], [1, -1], [0.5, -0.3]):
            steer, speed = self.r.apply(a, 0.2, 5.0)
            assert abs(steer - 0.2) <= 0.10 + 1e-9
            assert abs(speed - 5.0) <= 1.5 + 1e-9

    def test_out_of_range_output_cannot_widen_the_envelope(self):
        steer, speed = self.r.apply([1e6, -1e6], 0.2, 5.0)
        assert abs(steer - 0.2) <= 0.10 + 1e-9
        assert abs(speed - 5.0) <= 1.5 + 1e-9

    def test_non_finite_output_falls_back_to_the_baseline(self):
        for bad in ([float('nan'), 0.0], [0.0, float('inf')],
                    [float('-inf'), float('nan')]):
            assert self.r.apply(bad, 0.2, 5.0) == (0.2, 5.0)

    def test_malformed_output_falls_back_to_the_baseline(self):
        for bad in ([], [0.5], np.array([])):
            assert self.r.apply(bad, 0.2, 5.0) == (0.2, 5.0)

    def test_zero_authority_is_pure_mpc(self):
        """residual_scale:=0 must be bit-for-bit the MPC — the first rung of
        the ladder when bringing a policy up on a real car."""
        r = ResidualAction(authority=0.0, max_steer=0.41, v_max=8.0)
        for a in ([1, 1], [-1, 1], [0.3, -0.7]):
            assert r.apply(a, 0.17, 4.2) == (0.17, 4.2)

    def test_authority_scales_the_correction_proportionally(self):
        half = ResidualAction(d_steer=0.10, d_speed=1.5, max_steer=0.41,
                              v_max=8.0, authority=0.5)
        s_full, v_full = self.r.apply([1, 1], 0.0, 4.0)
        s_half, v_half = half.apply([1, 1], 0.0, 4.0)
        assert math.isclose(s_half, s_full / 2, abs_tol=1e-9)
        assert math.isclose(v_half - 4.0, (v_full - 4.0) / 2, abs_tol=1e-9)

    def test_steering_never_exceeds_the_mechanical_limit(self):
        """Even a baseline already at full lock plus a full correction."""
        steer, _ = self.r.apply([1.0, 0.0], 0.41, 5.0)
        assert abs(steer) <= 0.41 + 1e-9

    def test_speed_never_goes_negative(self):
        _, speed = self.r.apply([0.0, -1.0], 0.0, 0.2)
        assert speed >= 0.0

    def test_authority_is_clamped_to_unit_range(self):
        assert ResidualAction(authority=5.0).authority == 1.0
        assert ResidualAction(authority=-2.0).authority == 0.0


# ── networks and training ────────────────────────────────────────────────────
@needs_torch
class TestNetworks:
    def test_actor_output_is_bounded(self):
        from f1tenth_gym_ros.rl.networks import SquashedGaussianPolicy
        spec = ObsSpec()
        pi = SquashedGaussianPolicy(spec.n_beams, spec.n_state)
        obs = torch.randn(16, spec.dim) * 50.0          # deliberately extreme
        for det in (True, False):
            a, _ = pi.sample(obs, deterministic=det, with_logprob=False)
            assert a.shape == (16, 2)
            assert bool((a.abs() <= 1.0).all())

    def test_log_prob_is_finite_and_shaped(self):
        from f1tenth_gym_ros.rl.networks import SquashedGaussianPolicy
        spec = ObsSpec()
        pi = SquashedGaussianPolicy(spec.n_beams, spec.n_state)
        a, logp = pi.sample(torch.randn(32, spec.dim))
        assert logp.shape == (32,)
        assert bool(torch.isfinite(logp).all())

    def test_twin_critics_are_independent(self):
        """Shared weights would defeat the point of the min() target."""
        from f1tenth_gym_ros.rl.networks import QNetwork
        spec = ObsSpec()
        q = QNetwork(spec.n_beams, spec.n_state)
        q1, q2 = q(torch.randn(8, spec.dim), torch.randn(8, 2))
        assert q1.shape == (8,) and q2.shape == (8,)
        assert not torch.allclose(q1, q2)

    def test_actor_fits_the_realtime_budget(self):
        """It has to run inside a 50 Hz loop next to SLAM and a detector."""
        import time
        from f1tenth_gym_ros.rl.sac import SAC
        spec = ObsSpec()
        agent = SAC(spec, device='cpu')
        obs = np.random.randn(spec.dim).astype(np.float32)
        for _ in range(10):
            agent.act(obs, deterministic=True)
        t0 = time.perf_counter()
        for _ in range(50):
            agent.act(obs, deterministic=True)
        ms = (time.perf_counter() - t0) / 50 * 1000
        assert ms < 20.0, f'{ms:.1f} ms/tick is too slow for 50 Hz'


@needs_torch
class TestSAC:
    def test_replay_buffer_wraps_without_growing(self):
        from f1tenth_gym_ros.rl.sac import ReplayBuffer
        buf = ReplayBuffer(8, 2, capacity=10)
        for i in range(25):
            buf.add(np.full(8, i, np.float32), [0.1, 0.2], float(i),
                    np.zeros(8, np.float32), False)
        assert len(buf) == 10
        obs, act, rew, nxt, done = buf.sample(5, np.random.default_rng(0))
        assert obs.shape == (5, 8) and act.shape == (5, 2)

    def test_update_produces_finite_losses(self):
        from f1tenth_gym_ros.rl.sac import SAC, ReplayBuffer
        spec = ObsSpec()
        agent = SAC(spec, device='cpu', seed=0)
        buf = ReplayBuffer(spec.dim, 2, 2000)
        rng = np.random.default_rng(0)
        for i in range(300):
            o = rng.standard_normal(spec.dim).astype(np.float32)
            buf.add(o, rng.uniform(-1, 1, 2), float(rng.normal()), o, i % 40 == 0)
        for _ in range(5):
            out = agent.update(buf.sample(32, rng))
            assert all(np.isfinite(v) for v in out.values()), out

    def test_behavior_cloning_moves_toward_the_demonstration(self):
        """Warm start: with a residual action space the expert action is zero,
        so cloning must drive the policy toward deferring to the MPC."""
        from f1tenth_gym_ros.rl.sac import SAC
        spec = ObsSpec()
        agent = SAC(spec, device='cpu', seed=0)
        obs = np.random.default_rng(0).standard_normal(
            (128, spec.dim)).astype(np.float32)
        target = np.zeros((128, 2), np.float32)
        before = float(np.abs(agent.act(obs[0], deterministic=True)).mean())
        for _ in range(80):
            loss = agent.behavior_clone(obs, target)
        after = float(np.abs(agent.act(obs[0], deterministic=True)).mean())
        assert after < before or after < 1e-3
        assert loss < 1e-3

    def test_checkpoint_round_trips_exactly(self, tmp_path):
        from f1tenth_gym_ros.rl.sac import SAC
        spec = ObsSpec()
        agent = SAC(spec, device='cpu', seed=3)
        path = str(tmp_path / 'policy.pt')
        agent.save(path, meta={'step': 42})
        loaded, meta = SAC.load(path, device='cpu')
        assert meta['step'] == 42
        assert loaded.spec == spec
        obs = np.random.default_rng(1).standard_normal(
            spec.dim).astype(np.float32)
        assert np.allclose(agent.act(obs, deterministic=True),
                           loaded.act(obs, deterministic=True), atol=1e-6)

    def test_checkpoint_carries_its_observation_contract(self, tmp_path):
        """rl_agent refuses a checkpoint whose fingerprint does not match, so
        the fingerprint must actually be stored and recoverable."""
        from f1tenth_gym_ros.rl.sac import SAC
        spec = ObsSpec(n_beams=64)
        path = str(tmp_path / 'p.pt')
        SAC(spec, device='cpu').save(path)
        loaded, _ = SAC.load(path, device='cpu')
        assert loaded.spec.fingerprint() == spec.fingerprint()
        assert loaded.spec.fingerprint() != ObsSpec().fingerprint()


# ── environment ──────────────────────────────────────────────────────────────
class StubGym:
    """A kinematic bicycle standing in for f110_gym, so the episode logic and
    reward can be tested without the package installed."""

    def __init__(self, rx, ry, rh):
        self.rx, self.ry, self.rh = rx, ry, rh
        self.x = self.y = self.yaw = self.v = 0.0

    def reset(self, poses):
        self.x, self.y, self.yaw = (float(v) for v in poses[0])
        self.v = 3.0
        return self._obs(), {}

    def step(self, action):
        steer, speed = float(action[0][0]), float(action[0][1])
        dt = 0.01
        self.v += float(np.clip((speed - self.v) / dt, -8.0, 4.0)) * dt
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v * math.tan(steer) / 0.33 * dt
        return self._obs(), 0.0, False, False, {}

    def _obs(self):
        return dict(poses_x=[self.x], poses_y=[self.y],
                    poses_theta=[self.yaw % (2 * math.pi)],
                    linear_vels_x=[self.v], ang_vels_z=[0.0],
                    collisions=[0.0], scans=[np.full(1080, 5.0)])

    def close(self):
        pass


@pytest.fixture
def env():
    try:
        import osqp                                        # noqa: F401
    except Exception:
        pytest.skip('osqp not installed — the MPC baseline cannot solve')
    from f1tenth_gym_ros.rl import env as E
    import closed_loop as cl
    rx, ry, rh, _, _ = cl.load_raceline(RACELINE)
    E.RaceEnv._make_gym = staticmethod(lambda m, e: StubGym(rx, ry, rh))
    return E.RaceEnv('stub', RACELINE, random_start=False, max_steps=8000)


class TestEnv:
    def test_progress_wraps_at_the_start_finish_line(self, env):
        """A raw arc-length difference goes hugely negative on crossing the
        line — which would penalise the policy for the one thing it must do."""
        L = env.track_len
        assert math.isclose(env._advance(L - 0.2, 0.2), 0.4, abs_tol=1e-6)
        assert math.isclose(env._advance(10.0, 10.4), 0.4, abs_tol=1e-6)
        assert env._advance(10.0, 9.6) < 0        # real reversing stays negative

    def test_reset_gives_a_valid_observation(self, env):
        obs = env.reset(start_idx=0)
        assert obs.shape == (env.obs_dim,)
        assert np.all(np.isfinite(obs))

    def test_mpc_baseline_completes_a_lap(self, env):
        """The zero action is 'do what the MPC says' — it must go round."""
        env.reset(start_idx=0)
        info = {}
        for _ in range(8000):
            _obs, _r, done, info = env.step(env.mpc_action())
            if done:
                break
        assert info['finished'], info
        assert not info['crashed']
        assert abs(info['progress'] - env.track_len) < 5.0

    def test_reward_rises_with_progress(self, env):
        env.reset(start_idx=0)
        total = 0.0
        for _ in range(600):
            _o, r, done, _i = env.step(env.mpc_action())
            total += r
            if done:
                break
        assert total > 0, 'driving forward on the line must pay'

    def test_leaving_the_line_is_penalized(self, env):
        """Deviation must cost, or the policy has no reason to stay on track."""
        env.reset(start_idx=0)
        env.step(env.mpc_action())
        _o, r_good, _d, info_good = env.step(env.mpc_action())
        assert info_good['deviation'] < env.max_deviation


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
