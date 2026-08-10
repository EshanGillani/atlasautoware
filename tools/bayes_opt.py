"""
Bayesian optimization core — Gaussian process + Expected Improvement.
=====================================================================

A small, dependency-free (numpy only) implementation of the loop that matters
when every evaluation is expensive:

    1. fit a Gaussian process to the (config, lap time) pairs seen so far,
    2. ask the GP where the *expected improvement* over the best-so-far is
       largest — which balances "try near the current best" against "try where
       the model is most uncertain",
    3. evaluate exactly there, refit, repeat.

Why this and not the coordinate-descent loop in tools/auto_tune.py: coordinate
descent nudges one parameter at a time and throws away everything it learns
about the shape of the space.  When one evaluation costs a minute of practice
time — and at a competition you might get forty of them, total — you cannot
afford to spend runs re-deriving what the previous runs already implied.  The GP
uses every past run to choose the next one, and it handles the interactions
between parameters (higher grip budget only pays off if the tracking weights
follow) that a one-at-a-time search cannot see at all.

Design notes
------------
- **Matérn 5/2 kernel.**  The squared-exponential assumes the objective is
  infinitely smooth; lap time is not (there is a cliff exactly where the car
  starts running wide).  Matérn 5/2 is the standard choice for this reason.
- **Inputs normalized to [0,1]^d, outputs standardized.**  One isotropic
  lengthscale is then meaningful across parameters with wildly different units
  (metres, m/s^2, cost weights).
- **Hyperparameters by grid search on the log marginal likelihood.**  A handful
  of candidate (lengthscale, noise) pairs, scored exactly.  Gradient-based
  hyperparameter fitting is better in theory but fragile with the ~10-50 points
  a tuning session produces, and this keeps the file dependency-free.
- **Observation noise is a first-class parameter**, because repeat runs of the
  same config genuinely differ (perturbed starts, solver timing).

Used by tools/bayes_tune.py; tested in tests/test_bayes_opt.py.

    space = [('v_scale', 0.8, 1.4), ('a_lat', 5.0, 8.0)]
    opt = BayesOpt(space, seed=0)
    for _ in range(30):
        cfg = opt.ask()
        opt.tell(cfg, evaluate(cfg))       # lower is better
    print(opt.best())
"""

import math

import numpy as np

SQRT5 = math.sqrt(5.0)


# ── kernel ───────────────────────────────────────────────────────────────────
def matern52(a, b, lengthscale):
    """Matérn 5/2 covariance between two sets of points (rows), unit variance."""
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    # pairwise euclidean distance, guarded against tiny negatives from rounding
    d2 = (np.sum(a ** 2, 1)[:, None] + np.sum(b ** 2, 1)[None, :]
          - 2.0 * a @ b.T)
    r = np.sqrt(np.maximum(d2, 0.0)) / float(lengthscale)
    return (1.0 + SQRT5 * r + (5.0 / 3.0) * r ** 2) * np.exp(-SQRT5 * r)


# ── Gaussian process ─────────────────────────────────────────────────────────
class GP:
    """Zero-mean GP on standardized targets, exact inference via Cholesky."""

    def __init__(self, lengthscale=0.3, noise=1e-2):
        self.lengthscale = float(lengthscale)
        self.noise = float(noise)
        self._X = self._L = self._alpha = self._z = None
        self._mu = 0.0
        self._sd = 1.0

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, float))
        y = np.asarray(y, float).ravel()
        # standardize targets so one kernel variance (1.0) is the right scale
        self._mu = float(y.mean())
        self._sd = float(y.std()) or 1.0
        z = (y - self._mu) / self._sd

        K = matern52(X, X, self.lengthscale)
        K[np.diag_indices_from(K)] += self.noise + 1e-8      # noise + jitter
        self._L = np.linalg.cholesky(K)
        self._alpha = _cho_solve(self._L, z)
        self._X = X
        self._z = z                                          # kept for the LML
        return self

    def log_marginal_likelihood(self):
        """log p(y | X, theta) — the score the hyperparameter search maximizes.

            -0.5 z'K⁻¹z  -  sum log diag(L)  -  n/2 log 2π
        """
        if self._L is None:
            raise RuntimeError('fit() first')
        n = len(self._X)
        return (-0.5 * float(self._z @ self._alpha)
                - float(np.log(np.diag(self._L)).sum())
                - 0.5 * n * math.log(2.0 * math.pi))

    def predict(self, X):
        """-> (mean, std) in the ORIGINAL target units."""
        X = np.atleast_2d(np.asarray(X, float))
        Ks = matern52(self._X, X, self.lengthscale)          # (n_train, n_test)
        mean = Ks.T @ self._alpha
        v = np.linalg.solve(self._L, Ks)
        var = 1.0 + self.noise - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-12)
        return mean * self._sd + self._mu, np.sqrt(var) * self._sd


def _cho_solve(L, rhs):
    """Solve K x = rhs given the Cholesky factor K = L L'."""
    return np.linalg.solve(L.T, np.linalg.solve(L, rhs))


def fit_gp(X, y, lengthscales=None, noises=None):
    """Fit a GP, choosing (lengthscale, noise) by exact log marginal likelihood.

    A grid rather than gradient descent: with the few dozen points a tuning
    session produces, the likelihood surface is lumpy enough that a local
    optimizer regularly lands somewhere useless, and scoring 30 candidates
    exactly costs microseconds.
    """
    X = np.atleast_2d(np.asarray(X, float))
    y = np.asarray(y, float).ravel()
    lengthscales = lengthscales if lengthscales is not None else \
        [0.08, 0.12, 0.2, 0.3, 0.45, 0.7, 1.0, 1.5]
    noises = noises if noises is not None else [1e-4, 1e-3, 1e-2, 5e-2, 0.15]

    mu, sd = float(y.mean()), float(y.std()) or 1.0
    z = (y - mu) / sd
    n = len(X)
    best, best_lml = None, -np.inf
    for ls in lengthscales:
        K0 = matern52(X, X, ls)
        for nz in noises:
            K = K0.copy()
            K[np.diag_indices_from(K)] += nz + 1e-8
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                continue
            alpha = _cho_solve(L, z)
            lml = (-0.5 * float(z @ alpha)
                   - float(np.log(np.diag(L)).sum())
                   - 0.5 * n * math.log(2.0 * math.pi))
            if lml > best_lml:
                best_lml, best = lml, (ls, nz)
    ls, nz = best if best else (0.3, 1e-2)
    gp = GP(ls, nz).fit(X, y)
    gp.lml = best_lml
    return gp


# ── acquisition ──────────────────────────────────────────────────────────────
def _norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _norm_pdf(x):
    return np.exp(-0.5 * x ** 2) / math.sqrt(2.0 * math.pi)


def expected_improvement(mean, std, best, xi=0.01):
    """EI for MINIMIZATION: how much better than `best` we expect to do.

    xi adds a small constant to the improvement threshold, which stops the
    search collapsing onto the incumbent once the model gets confident.
    """
    mean = np.asarray(mean, float)
    std = np.maximum(np.asarray(std, float), 1e-12)
    imp = best - mean - xi
    z = imp / std
    ei = imp * _norm_cdf(z) + std * _norm_pdf(z)
    return np.maximum(ei, 0.0)


# ── the optimizer ────────────────────────────────────────────────────────────
class BayesOpt:
    """Ask/tell Bayesian optimizer over a box space.  Lower scores are better.

    space   [(name, low, high), ...]
    n_init  random points evaluated before the model takes over.  Below ~5 the
            GP has nothing to generalize from and picks nonsense.
    """

    def __init__(self, space, n_init=8, xi=0.01, seed=0, candidates=4000):
        if not space:
            raise ValueError('empty search space')
        self.names = [s[0] for s in space]
        self.lo = np.array([float(s[1]) for s in space])
        self.hi = np.array([float(s[2]) for s in space])
        if np.any(self.hi <= self.lo):
            bad = [n for n, l, h in zip(self.names, self.lo, self.hi) if h <= l]
            raise ValueError(f'high must exceed low for: {", ".join(bad)}')
        self.d = len(space)
        self.n_init = max(2, int(n_init))
        self.xi = float(xi)
        self.candidates = int(candidates)
        self.rng = np.random.default_rng(seed)
        self.X = []                     # normalized rows, [0,1]^d
        self.y = []                     # raw scores
        self.gp = None
        self._pending = None

    # -- unit-cube <-> real units ------------------------------------------
    def _to_real(self, u):
        return self.lo + np.asarray(u, float) * (self.hi - self.lo)

    def _to_unit(self, x):
        return (np.asarray(x, float) - self.lo) / (self.hi - self.lo)

    def _as_dict(self, u):
        return {n: float(v) for n, v in zip(self.names, self._to_real(u))}

    # -- ask / tell ---------------------------------------------------------
    def ask(self):
        """Next config to evaluate, as {name: value}."""
        if len(self.X) < self.n_init:
            u = self._latin_point(len(self.X))
        else:
            u = self._propose()
        self._pending = u
        return self._as_dict(u)

    def tell(self, config, score):
        """Record a result.  `config` may be the dict from ask(), or any dict
        of real-unit values (so you can seed the optimizer with past runs)."""
        if not np.isfinite(score):
            score = 1e6                          # a crash is a very bad score
        if isinstance(config, dict):
            u = self._to_unit([config[n] for n in self.names])
        else:
            u = self._to_unit(config)
        self.X.append(np.clip(u, 0.0, 1.0))
        self.y.append(float(score))
        self.gp = None                           # invalidate; refit lazily
        self._pending = None

    def _model(self):
        if self.gp is None and len(self.X) >= 2:
            self.gp = fit_gp(np.array(self.X), np.array(self.y))
        return self.gp

    def _latin_point(self, i):
        """Stratified random start: spread the initial designs out instead of
        letting uniform sampling clump them (which wastes scarce evaluations)."""
        u = (i + self.rng.random(self.d)) / float(self.n_init)
        # independent random permutation per dimension keeps it space-filling
        return np.clip(u[self.rng.permutation(self.d)], 0.0, 1.0)

    def _propose(self):
        gp = self._model()
        if gp is None:
            return self.rng.random(self.d)
        best = float(np.min(self.y))

        # Candidate set: mostly uniform (exploration) plus a cloud around the
        # incumbent (exploitation / local refinement), which is what lets the
        # search actually converge in the last handful of evaluations.
        n_uni = int(self.candidates * 0.7)
        cand = self.rng.random((n_uni, self.d))
        x_best = self.X[int(np.argmin(self.y))]
        for scale in (0.02, 0.05, 0.12):
            local = x_best + self.rng.normal(0.0, scale,
                                             (self.candidates // 10, self.d))
            cand = np.vstack([cand, np.clip(local, 0.0, 1.0)])

        mean, std = gp.predict(cand)
        ei = expected_improvement(mean, std, best, self.xi)
        if not np.any(ei > 0):
            return self.rng.random(self.d)       # model is flat — explore
        return cand[int(np.argmax(ei))]

    # -- results ------------------------------------------------------------
    def best(self):
        """-> (config_dict, score) of the best evaluation so far."""
        if not self.y:
            return None, float('inf')
        i = int(np.argmin(self.y))
        return self._as_dict(self.X[i]), float(self.y[i])

    def predict(self, config):
        """Model's (mean, std) for a config — useful for reporting confidence."""
        gp = self._model()
        if gp is None:
            return float('nan'), float('nan')
        u = self._to_unit([config[n] for n in self.names])
        m, s = gp.predict(np.atleast_2d(u))
        return float(m[0]), float(s[0])

    def importance(self):
        """Crude per-parameter sensitivity: how much the GP mean moves when one
        parameter is swept across its range from the incumbent.  Not a Sobol
        index — just enough to tell you which knobs are worth arguing about."""
        gp = self._model()
        if gp is None:
            return {}
        x_best = np.array(self.X[int(np.argmin(self.y))])
        out = {}
        sweep = np.linspace(0.0, 1.0, 25)
        for k, name in enumerate(self.names):
            pts = np.tile(x_best, (len(sweep), 1))
            pts[:, k] = sweep
            m, _ = gp.predict(pts)
            out[name] = float(m.max() - m.min())
        return out
