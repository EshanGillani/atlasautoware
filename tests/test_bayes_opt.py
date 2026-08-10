"""
Bayesian optimizer — GP inference, the acquisition, and the search itself.
==========================================================================

The optimizer decides how scarce practice runs get spent, so a silent bug here
is expensive in a way that is hard to notice: a broken GP still returns
*something*, and the search just quietly degrades to random sampling. These
tests pin down the properties that would fail if it did.

No ROS, no hardware, no torch — numpy only.

    python3 -m pytest tests/test_bayes_opt.py -q
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'tools'))
from bayes_opt import (BayesOpt, GP, expected_improvement, fit_gp,  # noqa: E402
                       matern52)


# ── kernel ───────────────────────────────────────────────────────────────────
def test_kernel_is_one_on_the_diagonal():
    X = np.random.default_rng(0).random((6, 3))
    K = matern52(X, X, 0.4)
    assert np.allclose(np.diag(K), 1.0)


def test_kernel_decays_with_distance():
    a = np.array([[0.0]])
    near, far = matern52(a, np.array([[0.1]]), 0.5), matern52(a, np.array([[2.0]]), 0.5)
    assert near[0, 0] > far[0, 0]
    assert far[0, 0] >= 0.0


def test_kernel_is_symmetric_positive_definite():
    X = np.random.default_rng(1).random((10, 2))
    K = matern52(X, X, 0.3) + np.eye(10) * 1e-8
    assert np.allclose(K, K.T)
    np.linalg.cholesky(K)          # raises if not positive definite


# ── GP ───────────────────────────────────────────────────────────────────────
def test_gp_interpolates_training_points():
    """With low noise the posterior mean must pass through the data."""
    X = np.linspace(0, 1, 8).reshape(-1, 1)
    y = np.sin(5 * X.ravel())
    gp = GP(lengthscale=0.25, noise=1e-6).fit(X, y)
    mean, _ = gp.predict(X)
    assert np.abs(mean - y).max() < 1e-3


def test_gp_is_more_uncertain_away_from_data():
    X = np.array([[0.0], [1.0]])
    gp = GP(lengthscale=0.2, noise=1e-6).fit(X, np.array([0.0, 1.0]))
    _, s_at = gp.predict(X)
    _, s_gap = gp.predict(np.array([[0.5]]))
    assert s_gap[0] > s_at.max() * 5


def test_gp_handles_constant_targets():
    """std()==0 would divide by zero in the standardization."""
    X = np.linspace(0, 1, 5).reshape(-1, 1)
    gp = fit_gp(X, np.full(5, 7.0))
    mean, std = gp.predict(X)
    assert np.allclose(mean, 7.0, atol=1e-6)
    assert np.all(np.isfinite(std))


def test_log_marginal_likelihood_prefers_the_right_lengthscale():
    """A smooth function should score better under a long lengthscale."""
    X = np.linspace(0, 1, 12).reshape(-1, 1)
    y = 2.0 * X.ravel()                                  # perfectly smooth
    smooth = GP(lengthscale=1.0, noise=1e-4).fit(X, y).log_marginal_likelihood()
    wiggly = GP(lengthscale=0.03, noise=1e-4).fit(X, y).log_marginal_likelihood()
    assert smooth > wiggly


def test_fit_gp_selects_hyperparameters():
    X = np.linspace(0, 1, 15).reshape(-1, 1)
    y = np.sin(3 * X.ravel())
    gp = fit_gp(X, y)
    assert gp.lengthscale > 0 and gp.noise > 0
    assert np.isfinite(gp.lml)


# ── acquisition ──────────────────────────────────────────────────────────────
def test_ei_is_zero_when_improvement_is_impossible():
    """Certain to be worse than the incumbent -> nothing to gain."""
    assert expected_improvement([5.0], [1e-12], best=0.0)[0] == 0.0


def test_ei_rewards_a_better_predicted_mean():
    better = expected_improvement([-1.0], [0.3], best=0.0)[0]
    worse = expected_improvement([0.5], [0.3], best=0.0)[0]
    assert better > worse


def test_ei_rewards_uncertainty_at_equal_mean():
    """The explore half of the trade-off: same mean, more unknown, more value."""
    certain = expected_improvement([0.0], [0.01], best=0.0)[0]
    unsure = expected_improvement([0.0], [1.0], best=0.0)[0]
    assert unsure > certain


def test_ei_is_never_negative():
    rng = np.random.default_rng(0)
    ei = expected_improvement(rng.normal(size=500), np.abs(rng.normal(size=500)),
                              best=0.0)
    assert np.all(ei >= 0.0)


# ── the optimizer ────────────────────────────────────────────────────────────
def test_ask_returns_values_inside_the_space():
    opt = BayesOpt([('a', -2.0, 3.0), ('b', 10.0, 20.0)], n_init=3, seed=0)
    for _ in range(15):
        cfg = opt.ask()
        assert -2.0 <= cfg['a'] <= 3.0
        assert 10.0 <= cfg['b'] <= 20.0
        opt.tell(cfg, float(cfg['a'] ** 2 + cfg['b']))


def test_rejects_a_degenerate_space():
    for bad in ([], [('a', 1.0, 1.0)], [('a', 5.0, 2.0)]):
        try:
            BayesOpt(bad)
        except ValueError:
            continue
        raise AssertionError(f'should have rejected {bad}')


def test_finds_the_optimum_of_a_smooth_function():
    """The core claim: it beats random search on the same budget."""
    def f(c):
        return (c['a'] - 0.3) ** 2 + (c['b'] + 0.4) ** 2

    space = [('a', -1.0, 1.0), ('b', -1.0, 1.0)]
    opt = BayesOpt(space, n_init=6, seed=0)
    for _ in range(30):
        cfg = opt.ask()
        opt.tell(cfg, f(cfg))
    best_cfg, best_score = opt.best()
    assert best_score < 0.01, best_score
    assert abs(best_cfg['a'] - 0.3) < 0.15
    assert abs(best_cfg['b'] + 0.4) < 0.15


def test_beats_random_search_on_the_same_budget():
    def f(c):
        x = np.array([c['a'], c['b'], c['c']])
        return float(np.sum((x - np.array([0.4, -0.6, 0.2])) ** 2))

    space = [('a', -1.0, 1.0), ('b', -1.0, 1.0), ('c', -1.0, 1.0)]
    n = 35
    wins = 0
    for seed in range(3):
        opt = BayesOpt(space, n_init=7, seed=seed)
        for _ in range(n):
            cfg = opt.ask()
            opt.tell(cfg, f(cfg))
        _, bo = opt.best()
        rng = np.random.default_rng(seed)
        rand = min(f(dict(zip('abc', rng.uniform(-1, 1, 3)))) for _ in range(n))
        wins += bo < rand
    assert wins == 3, 'Bayesian search should beat random search every time here'


# A cliff objective: below v=0.7 faster is better, above it the car crashes.
# This is the shape of real lap time, and the reason bayes_tune models
# log(score) rather than the score itself.
def _cliff(c):
    return 300.0 if c['v'] > 0.7 else (1.0 - c['v'])


def test_never_converges_into_the_infeasible_region():
    """Whatever the scaling, the search must not settle past the cliff."""
    for seed in range(4):
        opt = BayesOpt([('v', 0.0, 1.0)], n_init=5, seed=seed)
        for _ in range(30):
            cfg = opt.ask()
            opt.tell(cfg, _cliff(cfg))
        best_cfg, best_score = opt.best()
        assert best_cfg['v'] <= 0.7
        assert best_score < 1.0            # a feasible point, not a crash


def test_log_scores_let_it_resolve_right_up_to_the_cliff():
    """Why tools/bayes_tune.py takes the log of the score before telling.

    The failure penalty is ~1000x a good score here. Fed in raw, it dominates
    the GP's target standardization: every genuinely good config collapses into
    a sliver of the standardized range and the model cannot tell them apart, so
    the search stalls well short of the optimum (~0.60 instead of 0.70). The
    log puts the failure region a bounded distance above the feasible one and
    the fine structure survives.
    """
    def best_found(transform):
        found = []
        for seed in range(4):
            opt = BayesOpt([('v', 0.0, 1.0)], n_init=5, seed=seed)
            best = 1e9
            for _ in range(30):
                cfg = opt.ask()
                score = _cliff(cfg)
                opt.tell(cfg, transform(score))
                best = min(best, score)
            found.append(best)
        return float(np.mean(found))

    raw = best_found(lambda s: s)
    logged = best_found(math.log)
    assert logged < raw, (logged, raw)
    assert logged < 0.40, logged           # optimum is 0.30


def test_non_finite_scores_are_absorbed():
    opt = BayesOpt([('a', 0.0, 1.0)], n_init=3, seed=0)
    opt.tell({'a': 0.5}, float('nan'))
    opt.tell({'a': 0.2}, float('inf'))
    opt.tell({'a': 0.8}, 1.0)
    cfg = opt.ask()                       # must not raise
    assert 0.0 <= cfg['a'] <= 1.0
    assert opt.best()[1] == 1.0


def test_can_be_seeded_with_past_results():
    """--resume: telling it configs it never asked for must work."""
    opt = BayesOpt([('a', 0.0, 10.0)], n_init=4, seed=0)
    for a, s in [(1.0, 9.0), (5.0, 1.0), (9.0, 8.0)]:
        opt.tell({'a': a}, s)
    cfg, score = opt.best()
    assert score == 1.0 and abs(cfg['a'] - 5.0) < 1e-6


def test_importance_ranks_the_influential_parameter_first():
    def f(c):
        return 10.0 * c['strong'] + 0.01 * c['weak']

    opt = BayesOpt([('strong', 0.0, 1.0), ('weak', 0.0, 1.0)], n_init=8, seed=0)
    for _ in range(30):
        cfg = opt.ask()
        opt.tell(cfg, f(cfg))
    imp = opt.importance()
    assert imp['strong'] > imp['weak'] * 3


def test_best_is_empty_before_any_result():
    cfg, score = BayesOpt([('a', 0.0, 1.0)]).best()
    assert cfg is None and math.isinf(score)


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-q']))
