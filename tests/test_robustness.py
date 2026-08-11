"""
Grip-robust scoring and the lap-time / safety-margin frontier.
==============================================================

These cover the decision logic, not the physics: whether a setup that cannot
finish on a worse surface is actually punished by the objective, and whether the
frontier that gets put in front of the team omits choices that are strictly
worse on both axes.

Both matter because they shape what a human picks under time pressure. A
frontier listing a dominated setup is an invitation to choose it.

No ROS, no hardware. The lap-running tests need osqp and skip without it.

    python3 -m pytest tests/test_robustness.py -q
"""

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tools'))

from bayes_tune import FAIL_SCORE, Objective                # noqa: E402
from robustness_frontier import pareto                      # noqa: E402

RACELINE = os.path.join(REPO, 'racelines', 'comp_raceline.csv')

needs_osqp = pytest.mark.skipif(
    not __import__('importlib').util.find_spec('osqp'),
    reason='osqp not installed — the MPC cannot solve')


# ── the Pareto frontier ──────────────────────────────────────────────────────
class TestPareto:
    """rows are (lap, grip_floor, ...) — lower is better on both axes."""

    def test_drops_a_setup_beaten_on_both_axes(self):
        quick_and_tough = (39.9, 0.80, 'a')
        slow_and_fragile = (40.2, 0.85, 'b')       # worse lap AND worse floor
        out = pareto([quick_and_tough, slow_and_fragile])
        assert out == [quick_and_tough]

    def test_keeps_a_genuine_trade(self):
        """Quicker but more fragile vs slower but tougher: both are choices."""
        rows = [(35.3, 0.95, 'fast'), (39.9, 0.80, 'tough')]
        assert set(pareto(rows)) == set(rows)

    def test_keeps_everything_on_a_clean_curve(self):
        rows = [(35.0, 0.95, 'a'), (38.0, 0.90, 'b'), (41.0, 0.85, 'c')]
        assert len(pareto(rows)) == 3

    def test_single_row_survives(self):
        assert pareto([(35.0, 0.9, 'a')]) == [(35.0, 0.9, 'a')]

    def test_empty_input(self):
        assert pareto([]) == []

    def test_exact_duplicates_do_not_annihilate_each_other(self):
        """Two identical rows must not both be dropped for dominating each
        other — that would silently empty the frontier."""
        rows = [(35.0, 0.9, 'a'), (35.0, 0.9, 'b')]
        assert len(pareto(rows)) == 2

    def test_dominated_middle_is_removed(self):
        rows = [(35.0, 0.95, 'a'), (40.0, 0.90, 'b'), (39.0, 0.85, 'c')]
        out = pareto(rows)
        assert (40.0, 0.90, 'b') not in out       # 'c' is quicker AND tougher
        assert (35.0, 0.95, 'a') in out and (39.0, 0.85, 'c') in out


# ── the robust objective ─────────────────────────────────────────────────────
class TestRobustObjective:
    def test_defaults_to_a_single_grip_level(self):
        obj = Objective(RACELINE, trials=1, mu=1.0)
        assert obj.mus == [1.0]

    def test_accepts_a_grip_range(self):
        obj = Objective(RACELINE, trials=1, mus=[1.05, 0.95, 0.85])
        assert obj.mus == [1.05, 0.95, 0.85]

    def test_total_failure_scores_the_bounded_penalty(self):
        """Bounded, not 1e9: a huge outlier destroys the GP's target scaling."""
        obj = Objective(RACELINE, trials=4)
        score, info = obj._combine([], 0, {})
        assert score == FAIL_SCORE
        assert info['success'] == 0.0

    def test_unreliable_setups_are_penalized_below_the_floor(self):
        """A quick lap that only finishes sometimes must lose to a slower one
        that always does — this is the trade that decides races."""
        obj = Objective(RACELINE, trials=10, min_success=0.9)
        flaky, _ = obj._combine([30.0] * 3, 3, {})     # 30 % success, quick
        solid, _ = obj._combine([40.0] * 10, 10, {})   # 100 % success, slower
        assert solid < flaky

    def test_score_is_expected_time_including_restarts(self):
        obj = Objective(RACELINE, trials=10, min_success=0.0)
        score, _ = obj._combine([40.0] * 5, 5, {})     # half the runs finish
        assert score == pytest.approx(80.0)            # 40 / 0.5

    def test_scores_are_capped_at_the_failure_penalty(self):
        obj = Objective(RACELINE, trials=10, min_success=1.0)
        score, _ = obj._combine([200.0], 1, {})
        assert score <= FAIL_SCORE

    @needs_osqp
    def test_a_grip_range_reports_every_level(self):
        obj = Objective(RACELINE, trials=2, mus=[1.05, 0.85], min_success=0.0)
        cfg = dict(a_lat=6.0, a_accel=4.0, a_brake=8.0, v_max=6.0, v_scale=1.0,
                   q_pos=28.0, q_yaw=6.0, q_v=2.5, rd_steer=12.0, horizon=12.0)
        _score, info = obj(cfg)
        assert set(info['per_mu']) == {'1.05', '0.85'}
        assert 'worst_mu' in info

    @needs_osqp
    def test_a_setup_that_cannot_take_low_grip_scores_far_worse(self):
        """The whole point of --mu-range.

        This is the actual winner a fixed-mu search produced on comp_track: a
        35.3 s lap that is 100 % reliable at nominal grip and 0 % once friction
        drops to 0.90. Tuned at one mu it looks like the best setup available;
        scored across a range it must lose badly, or the search will keep
        handing the team a car that only works on a perfect surface.
        """
        fragile = dict(a_lat=5.4703, a_accel=5.1008, a_brake=4.0, v_max=7.8386,
                       v_scale=1.2223, q_pos=16.0544, q_yaw=1.0, q_v=2.2610,
                       rd_steer=11.1546, horizon=16.0)
        across = Objective(RACELINE, trials=3, mus=[1.05, 0.85], min_success=0.0)
        range_score, info = across(dict(fragile))

        assert info['per_mu']['1.05']['success'] == 1.0, \
            'precondition: this config is meant to be perfect at nominal grip'
        assert info['per_mu']['0.85']['success'] == 0.0, \
            'precondition: this config is meant to fail at low grip'

        nominal_lap = info['per_mu']['1.05']['mean_lap']
        assert info['worst_mu'] == '0.85'

        # The property that decides races: this setup must lose to a robust one
        # even when the robust one is far slower. A setup lapping 55 s at every
        # grip level scores 55; the fragile 35.3 s setup must score worse than
        # that, or the search would still prefer the car that only works on a
        # perfect surface.
        robust_reference, _ = across._combine([55.0] * 3, 3, {})
        assert range_score > robust_reference, (
            f'fragile setup scored {range_score:.1f} but a reliable 55 s setup '
            f'scores {robust_reference:.1f} — the fragile one must lose')
        assert range_score > 4 * nominal_lap, (
            f'{range_score:.1f} against a {nominal_lap:.1f}s nominal lap is too '
            f'mild a penalty for never finishing at 0.85')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
