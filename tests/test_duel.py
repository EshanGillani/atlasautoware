"""
Wheel-to-wheel decision policy — observation, envelope, styles, reward.
=======================================================================

The load-bearing tests here are `TestDecisionEnvelope`: that clamp is what
lets a learned policy make racing decisions next to another car without being
able to invent a line. In particular it must never place the car outside the
room that physically exists, whatever the network asks for — that bound is
geometric, not learned, so it has to hold for every input including malformed
ones.

`TestStyles` checks the thing that is easy to claim and easy to get wrong: that
the conservative-to-aggressive knob actually changes behaviour monotonically
rather than being a label on an unchanged policy.

    python3 -m pytest tests/test_duel.py -q
"""

import math
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))

from f1tenth_gym_ros.rl.duel import (  # noqa: E402
    MODES, DuelSpec, FlatObsSpec, Rival, StrategyResidual,
    build_duel_observation, style_reward_weights)

try:
    import torch                                          # noqa: F401
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

RACELINE = os.path.join(REPO, 'racelines', 'comp_raceline.csv')


def obs_for(spec, rivals=(), style=0.5, mode='CRUISE', offset=0.0, sf=1.0,
            room_left=1.0, room_right=1.0):
    return build_duel_observation(
        spec, 6.0, 0.0, 0.0, [0.1] * len(spec.preview_m), room_left, room_right,
        list(rivals), mode, offset, sf, style)


# ── observation ──────────────────────────────────────────────────────────────
class TestObservation:
    def test_dimension_matches_the_spec(self):
        spec = DuelSpec()
        assert obs_for(spec).shape == (spec.dim,)

    def test_is_finite_and_normalized(self):
        spec = DuelSpec()
        o = obs_for(spec, [Rival(2.0, 0.3, 1.0), Rival(-1.0, -0.2, -0.4)])
        assert np.all(np.isfinite(o))
        assert np.abs(o).max() <= 2.0

    def test_garbage_never_reaches_the_network(self):
        spec = DuelSpec()
        o = build_duel_observation(
            spec, float('nan'), float('inf'), float('-inf'),
            [float('nan')] * 4, float('nan'), 1.0,
            [Rival(float('nan'), float('inf'), float('nan'))],
            'ATTACK', float('nan'), float('inf'), 0.5)
        assert np.all(np.isfinite(o))

    def test_empty_track_differs_from_a_car_alongside(self):
        """Zeros alone would make 'nobody there' look like 'car on your door' —
        the presence flag is what keeps those apart."""
        spec = DuelSpec()
        empty = obs_for(spec, [])
        alongside = obs_for(spec, [Rival(0.05, 0.0, 0.0)])
        assert not np.allclose(empty, alongside)

    def test_nearest_rival_is_the_one_selected(self):
        spec = DuelSpec()
        near = obs_for(spec, [Rival(1.0, 0.0, 0.0), Rival(5.0, 0.0, 0.0)])
        just_near = obs_for(spec, [Rival(1.0, 0.0, 0.0)])
        assert np.allclose(near, just_near)

    def test_ahead_and_behind_occupy_separate_slots(self):
        spec = DuelSpec()
        ahead = obs_for(spec, [Rival(3.0, 0.0, 1.0)])
        behind = obs_for(spec, [Rival(-3.0, 0.0, 1.0)])
        assert not np.allclose(ahead, behind)

    def test_every_mode_is_encodable(self):
        spec = DuelSpec()
        seen = {tuple(obs_for(spec, mode=m).tolist()) for m in MODES}
        assert len(seen) == len(MODES), 'modes must be distinguishable'

    def test_style_changes_the_observation(self):
        spec = DuelSpec()
        assert not np.allclose(obs_for(spec, style=0.0), obs_for(spec, style=1.0))

    def test_fingerprint_distinguishes_layouts(self):
        assert DuelSpec().fingerprint() == DuelSpec().fingerprint()
        assert DuelSpec(preview_m=(1.0,)).fingerprint() != DuelSpec().fingerprint()
        assert DuelSpec(d_offset=0.5).fingerprint() != DuelSpec().fingerprint()

    def test_spec_round_trips(self):
        spec = DuelSpec(d_offset=0.22, attack_range=7.5)
        assert DuelSpec.from_dict(spec.to_dict()) == spec


# ── the safety envelope ──────────────────────────────────────────────────────
class TestDecisionEnvelope:
    def setup_method(self):
        self.spec = DuelSpec(d_offset=0.35, d_speed=0.12)
        self.r = StrategyResidual(self.spec, authority=1.0, style=1.0)

    def test_zero_action_is_the_rule_based_decision(self):
        assert self.r.apply([0.0, 0.0], 0.4, 1.05, 1.0, 1.0) == (0.4, 1.05)

    def test_correction_is_bounded(self):
        for a in ([1, 1], [-1, -1], [1, -1]):
            off, sf = self.r.apply(a, 0.0, 1.0, 5.0, 5.0)   # room not binding
            assert abs(off) <= self.spec.d_offset + 1e-9
            assert abs(sf - 1.0) <= self.spec.d_speed + 1e-9

    def test_out_of_range_output_cannot_widen_the_envelope(self):
        off, sf = self.r.apply([1e6, -1e6], 0.0, 1.0, 5.0, 5.0)
        assert abs(off) <= self.spec.d_offset + 1e-9
        assert abs(sf - 1.0) <= self.spec.d_speed + 1e-9

    def test_never_steers_into_space_that_does_not_exist(self):
        """The geometric clamp — the policy may pick a line, not invent room."""
        for room_l, room_r in ((0.1, 1.0), (1.0, 0.1), (0.0, 0.0)):
            off, _ = self.r.apply([1.0, 0.0], 0.0, 1.0, room_l, room_r)
            assert -room_r - 1e-9 <= off <= room_l + 1e-9
            off, _ = self.r.apply([-1.0, 0.0], 0.0, 1.0, room_l, room_r)
            assert -room_r - 1e-9 <= off <= room_l + 1e-9

    def test_clamp_holds_even_when_the_baseline_is_already_outside(self):
        """A rule-based decision at the edge plus a correction must not compound
        into something off the road."""
        off, _ = self.r.apply([1.0, 0.0], 0.9, 1.0, 0.5, 0.5)
        assert off <= 0.5 + 1e-9

    def test_non_finite_output_falls_back_to_the_rules(self):
        for bad in ([float('nan'), 0.0], [0.0, float('inf')]):
            assert self.r.apply(bad, 0.3, 1.02, 1.0, 1.0) == (0.3, 1.02)

    def test_malformed_output_falls_back_to_the_rules(self):
        for bad in ([], [0.5], np.array([])):
            assert self.r.apply(bad, 0.3, 1.02, 1.0, 1.0) == (0.3, 1.02)

    def test_zero_authority_reproduces_the_rule_based_brain(self):
        r = StrategyResidual(self.spec, authority=0.0, style=1.0)
        for a in ([1, 1], [-1, 1], [0.4, -0.9]):
            assert r.apply(a, 0.25, 1.03, 1.0, 1.0) == (0.25, 1.03)

    def test_speed_factor_stays_in_sane_bounds(self):
        r = StrategyResidual(self.spec, authority=1.0, style=1.0,
                             min_speed_factor=0.5, max_speed_factor=1.35)
        _, lo = r.apply([0.0, -1.0], 0.0, 0.5, 1.0, 1.0)
        _, hi = r.apply([0.0, 1.0], 0.0, 1.35, 1.0, 1.0)
        assert lo >= 0.5 - 1e-9 and hi <= 1.35 + 1e-9


# ── styles ───────────────────────────────────────────────────────────────────
class TestStyles:
    def test_authority_scales_with_style(self):
        spec = DuelSpec()
        low = StrategyResidual(spec, authority=1.0, style=0.0).scale()
        high = StrategyResidual(spec, authority=1.0, style=1.0).scale()
        assert 0.0 < low < high <= 1.0

    def test_conservative_still_retains_some_room(self):
        """A policy frozen at zero authority cannot avoid anything either —
        refusing to move is its own way of causing a collision."""
        assert StrategyResidual(DuelSpec(), authority=1.0, style=0.0).scale() > 0.2

    def test_aggressive_deviates_further_for_the_same_action(self):
        spec = DuelSpec()
        cons = StrategyResidual(spec, authority=1.0, style=0.0)
        aggr = StrategyResidual(spec, authority=1.0, style=1.0)
        off_c, _ = cons.apply([1.0, 0.0], 0.0, 1.0, 5.0, 5.0)
        off_a, _ = aggr.apply([1.0, 0.0], 0.0, 1.0, 5.0, 5.0)
        assert off_a > off_c > 0.0

    def test_reward_weights_move_monotonically_with_style(self):
        cons = style_reward_weights(0.0)
        mid = style_reward_weights(0.5)
        aggr = style_reward_weights(1.0)
        assert cons['overtake'] < mid['overtake'] < aggr['overtake']
        # aggressive tolerates proximity more (less negative)
        assert cons['proximity'] < aggr['proximity']
        # ... and defends harder against being passed (more negative)
        assert aggr['overtaken'] < cons['overtaken']

    def test_contact_is_never_cheap_at_any_style(self):
        """Aggressive must mean 'values position more', never 'crashing is ok'."""
        for s in (0.0, 0.5, 1.0):
            w = style_reward_weights(s)
            assert w['contact'] < -40.0
            assert abs(w['contact']) > 4 * abs(w['overtake'])

    def test_style_is_clamped_to_unit_range(self):
        assert style_reward_weights(5.0) == style_reward_weights(1.0)
        assert style_reward_weights(-3.0) == style_reward_weights(0.0)
        assert StrategyResidual(DuelSpec(), style=9.0).style == 1.0


# ── flat observation spec ────────────────────────────────────────────────────
class TestFlatSpec:
    def test_declares_no_lidar(self):
        s = FlatObsSpec(26)
        assert s.n_beams == 0 and s.n_state == 26 and s.dim == 26

    def test_round_trips(self):
        s = FlatObsSpec(26)
        assert FlatObsSpec.from_dict(s.to_dict()) == s

    @pytest.mark.skipif(not HAVE_TORCH, reason='torch not installed')
    def test_torso_skips_the_cnn_for_a_flat_observation(self):
        from f1tenth_gym_ros.rl.networks import Torso
        t = Torso(0, 26)
        assert t.encoder is None
        assert t(torch.zeros(4, 26)).shape == (4, 256)

    @pytest.mark.skipif(not HAVE_TORCH, reason='torch not installed')
    def test_a_flat_policy_trains_and_round_trips(self, tmp_path):
        from f1tenth_gym_ros.rl.sac import SAC, ReplayBuffer
        spec = FlatObsSpec(26)
        agent = SAC(spec, action_dim=2, device='cpu', seed=0)
        rng = np.random.default_rng(0)
        buf = ReplayBuffer(26, 2, 500)
        for i in range(200):
            o = rng.standard_normal(26).astype(np.float32)
            buf.add(o, rng.uniform(-1, 1, 2), float(rng.normal()), o, i % 30 == 0)
        out = agent.update(buf.sample(32, rng))
        assert all(np.isfinite(v) for v in out.values())

        path = str(tmp_path / 'duel.pt')
        agent.save(path, meta={'step': 7})
        loaded, meta = SAC.load(path, device='cpu')
        assert meta['step'] == 7
        assert loaded.spec.n_beams == 0, 'must not be rebuilt with a CNN encoder'
        o = rng.standard_normal(26).astype(np.float32)
        assert np.allclose(agent.act(o, deterministic=True),
                           loaded.act(o, deterministic=True), atol=1e-6)


# ── the environment ──────────────────────────────────────────────────────────
class StubTwoCarGym:
    """Two kinematic bicycles, standing in for f110_gym so the reward, the pass
    detection and the episode logic can be tested without the package."""

    def __init__(self, wheelbase=0.33):
        self.L = wheelbase
        self.s = [[0.0] * 4, [0.0] * 4]

    def reset(self, poses):
        self.s = [[float(poses[i][0]), float(poses[i][1]),
                   float(poses[i][2]), 0.0] for i in range(2)]
        return self._obs(), {}

    def step(self, action):
        dt = 0.01
        for i in range(2):
            steer, speed = float(action[i][0]), float(action[i][1])
            x, y, yaw, v = self.s[i]
            v += float(np.clip((speed - v) / dt, -8.0, 4.0)) * dt
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            yaw += v * math.tan(steer) / self.L * dt
            self.s[i] = [x, y, yaw, v]
        return self._obs(), 0.0, False, False, {}

    def _obs(self):
        return dict(
            poses_x=[c[0] for c in self.s], poses_y=[c[1] for c in self.s],
            poses_theta=[c[2] % (2 * math.pi) for c in self.s],
            linear_vels_x=[c[3] for c in self.s],
            ang_vels_z=[0.0, 0.0], collisions=[0.0, 0.0],
            scans=[np.full(1080, 5.0), np.full(1080, 5.0)])

    def close(self):
        pass


@pytest.fixture
def duel_env():
    try:
        import osqp                                        # noqa: F401
    except Exception:
        pytest.skip('osqp not installed — the MPC baseline cannot solve')
    from f1tenth_gym_ros.rl import duel_env as de
    de.DuelEnv._make_gym = staticmethod(lambda m, e: StubTwoCarGym())
    # Pass the real map so the corridor is MEASURED. A fixture that
    # skips it exercises the fallback constant instead of the actual
    # track, which is how the too-wide-corridor bug stayed hidden.
    return de.DuelEnv(os.path.join(REPO, 'maps', 'comp_track'),
                      RACELINE, max_steps=400, seed=0,
                      map_yaml=os.path.join(REPO, 'maps',
                                            'comp_track.yaml'))


class TestDuelEnv:
    def test_reset_returns_a_valid_observation(self, duel_env):
        obs = duel_env.reset(style=0.5, start_idx=0, rival_gap=3.0)
        assert obs.shape == (duel_env.obs_dim,)
        assert np.all(np.isfinite(obs))

    def test_style_is_applied_to_the_envelope_and_reward(self, duel_env):
        duel_env.reset(style=0.0, start_idx=0, rival_gap=3.0)
        cons = duel_env.residual.scale()
        duel_env.reset(style=1.0, start_idx=0, rival_gap=3.0)
        assert duel_env.residual.scale() > cons
        assert duel_env.weights['overtake'] > style_reward_weights(0.0)['overtake']

    def test_rule_based_action_is_zero(self, duel_env):
        assert np.allclose(duel_env.rule_based_action(), 0.0)

    def test_stepping_produces_finite_rewards(self, duel_env):
        obs = duel_env.reset(style=0.5, start_idx=0, rival_gap=4.0)
        for _ in range(60):
            obs, r, done, info = duel_env.step(duel_env.rule_based_action())
            assert np.all(np.isfinite(obs))
            assert math.isfinite(r)
            assert info['mode'] in MODES
            if done:
                break

    def test_progress_wraps_at_the_line(self, duel_env):
        L = duel_env.track_len
        assert math.isclose(duel_env._advance(L - 0.2, 0.2), 0.4, abs_tol=1e-6)
        assert duel_env._advance(10.0, 9.6) < 0

    def test_zero_authority_matches_the_rule_based_offset(self, duel_env):
        """With no authority the commanded offset must be the brain's own."""
        duel_env.residual.authority = 0.0
        duel_env.reset(style=1.0, start_idx=0, rival_gap=2.0)
        for _ in range(20):
            _o, _r, done, info = duel_env.step(np.array([1.0, 1.0]))
            assert math.isclose(info['offset'], info['baseline_offset'],
                                abs_tol=1e-9)
            if done:
                break


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))


# ── launching from rest, and the measured corridor ───────────────────────────
class TestLaunchAndCorridor:
    """The two bugs that made the rule-based baseline undriveable.

    Both were invisible for the same reason: the stub gym used to reset at
    3 m/s and the corridor was assumed rather than measured, so the tests
    agreed with the code instead of with the track.
    """

    def test_stub_resets_from_rest_like_the_real_gym(self):
        """If this regresses, the launch bug becomes invisible again."""
        g = StubTwoCarGym()
        g.reset(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        assert g.s[0][3] == 0.0 and g.s[1][3] == 0.0

    def test_pursuit_steer_works_at_zero_speed(self):
        """The MPC cannot: yaw_dot = v*tan(delta)/L is zero at v = 0."""
        from f1tenth_gym_ros.rl.features import pursuit_steer
        rx = np.linspace(0.0, 10.0, 60)
        ry = 0.3 * rx                      # a path heading up and to the right
        steer = pursuit_steer(0.0, 0.0, 0.0, rx, ry, 0, 1.0, 0.33, 0.36)
        assert steer > 0.01, 'should steer left toward a leftward path'
        assert abs(steer) <= 0.36 + 1e-9

    def test_pursuit_steer_honours_a_lateral_offset(self):
        from f1tenth_gym_ros.rl.features import pursuit_steer
        rx = np.linspace(0.0, 10.0, 60)
        ry = np.zeros(60)
        nx, ny = np.zeros(60), np.ones(60)          # left normal is +y
        straight = pursuit_steer(0.0, 0.0, 0.0, rx, ry, 0, 1.0, 0.33, 0.36)
        offset = pursuit_steer(0.0, 0.0, 0.0, rx, ry, 0, 1.0, 0.33, 0.36,
                               offset=0.5, nx=nx, ny=ny)
        assert offset > straight, 'a left offset should steer further left'

    def test_baseline_launches_and_completes_a_lap(self, duel_env):
        """Without the launch branch this stopped after ~12 m of a 243 m lap."""
        duel_env.max_steps = 12000
        duel_env.reset(style=0.5, start_idx=0, rival_gap=4.0)
        info = {}
        for _ in range(12000):
            _o, _r, done, info = duel_env.step(duel_env.rule_based_action())
            if done:
                break
        assert info['progress'] > 100.0, \
            f'baseline only reached {info["progress"]:.1f} m — is launch broken?'

    def test_corridor_is_measured_from_the_map_when_available(self):
        from f1tenth_gym_ros.rl import duel_env as de
        de.DuelEnv._make_gym = staticmethod(lambda m, e: StubTwoCarGym())
        env = de.DuelEnv(os.path.join(REPO, 'maps', 'comp_track'), RACELINE,
                         map_yaml=os.path.join(REPO, 'maps', 'comp_track.yaml'))
        assert env.corridor_measured
        assert len(env.half_width) == env.n
        # comp_track really is this tight; a constant 1.0 m was fiction
        assert env.half_width.min() < 0.2
        assert float(np.median(env.half_width)) < 0.8

    def test_side_clearance_fits_the_narrow_part_of_the_track(self):
        """0.55 m off-line is inside the wall on 54% of this lap."""
        from f1tenth_gym_ros.rl import duel_env as de
        de.DuelEnv._make_gym = staticmethod(lambda m, e: StubTwoCarGym())
        env = de.DuelEnv(os.path.join(REPO, 'maps', 'comp_track'), RACELINE,
                         map_yaml=os.path.join(REPO, 'maps', 'comp_track.yaml'))
        assert env.brain.side_clearance < 0.35
        assert env.brain.side_clearance <= float(np.percentile(env.half_width, 20))

    def test_falls_back_to_a_constant_without_a_map(self):
        from f1tenth_gym_ros.rl import duel_env as de
        de.DuelEnv._make_gym = staticmethod(lambda m, e: StubTwoCarGym())
        env = de.DuelEnv('nonexistent-map', RACELINE, map_yaml='/no/such.yaml')
        assert not env.corridor_measured
        assert np.allclose(env.half_width, env.spec.track_half)
