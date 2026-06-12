"""
Particle-filter tests — convergence, tracking, resampling, vectorization.
=========================================================================

Synthetic-truth validation of the MCL localizer on the real competition
map: scans are synthesized with grid_raycast at ground-truth poses (note:
the same raycaster family the likelihood field is built from — these tests
pin map-consistency, not real-lidar accuracy), then the filter must

  - converge from a deliberate 0.5 m / 20 deg initial offset,
  - track through the sharpest corner sequence of the raceline under a
    biased + noisy motion prior (emulated EKF twist),
  - preserve weight mass and particle support under resampling,
  - stay vectorized: a full update at 1500 particles must run far below
    any per-particle-Python-loop time.

    python3 -m pytest tests/test_particle_filter.py -q
"""

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'f1tenth_gym_ros'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_map import GridMap, grid_raycast                  # noqa: E402
from particle_filter import ParticleFilter                  # noqa: E402
from closed_loop import load_raceline                       # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_BEAMS = 360
ANGLE_MIN = -math.pi
ANGLE_INC = 2.0 * math.pi / N_BEAMS
BODY = ANGLE_MIN + np.arange(N_BEAMS) * ANGLE_INC


def _load():
    gm = GridMap.load(os.path.join(REPO, 'maps', 'comp_track.yaml'))
    rl = load_raceline(os.path.join(REPO, 'racelines', 'comp_raceline.csv'))
    return gm, rl


def _scan(gm, x, y, yaw, rng=None, sigma=0.03):
    s = grid_raycast(gm, x, y, yaw + BODY, 12.0)
    if rng is not None:
        s = s + rng.normal(0.0, sigma, N_BEAMS)
    return s


def _ang(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def _converge(gm, x, y, yaw, n_updates=15, seed=0):
    rng = np.random.default_rng(seed)
    pf = ParticleFilter(gm, n_particles=1500, seed=seed + 1)
    pf.initialize(x + 0.5, y, yaw + math.radians(20.0), spread=0.4)
    for _ in range(n_updates):
        pf.predict(0.0, 0.0, 0.0, 0.05)               # standing still
        pf.update(_scan(gm, x, y, yaw, rng), ANGLE_MIN, ANGLE_INC, 20)
    ex, ey, eyaw, conf = pf.pose()
    return math.hypot(ex - x, ey - y), _ang(eyaw, yaw), conf


def test_converges_from_offset():
    """0.5 m / 20 deg initial error pulled in within 15 updates (corner pose,
    where the scan geometry constrains all three states)."""
    gm, (rx, ry, rh, rc, _) = _load()
    i = int(np.argmax(np.abs(rc)))                    # sharpest corner
    pos_err, yaw_err, conf = _converge(gm, rx[i], ry[i], rh[i])
    assert pos_err < 0.20, f'position error {pos_err:.3f} m after 15 updates'
    assert yaw_err < math.radians(5.0), \
        f'yaw error {math.degrees(yaw_err):.1f} deg after 15 updates'
    assert conf > 0.05                                # weights not degenerate


def test_converges_on_straight_with_known_ambiguity():
    """On a straight, parallel walls leave longitudinal position weakly
    observable to the likelihood field — convergence is to ~0.35 m, not
    0.15 m.  Pinned here honestly; lateral + yaw still lock in (driving
    through corners resolves the rest, see the corner/lap tests)."""
    gm, (rx, ry, rh, _, _) = _load()
    pos_err, yaw_err, _ = _converge(gm, rx[10], ry[10], rh[10])
    assert pos_err < 0.45, f'position error {pos_err:.3f} m after 15 updates'
    assert yaw_err < math.radians(5.0)


def test_tracks_through_corner():
    """Follow the sharpest corner of the raceline under a biased twist."""
    gm, (rx, ry, rh, rc, rv) = _load()
    n = len(rx)
    j0 = int(np.argmax(np.abs(rc)))                   # sharpest corner
    j0 = (j0 - 8) % n                                 # start a bit before it
    rng = np.random.default_rng(2)
    pf = ParticleFilter(gm, n_particles=1500, seed=3)
    pf.initialize(rx[j0], ry[j0], rh[j0], spread=0.1)
    h = np.unwrap(rh)
    err = []
    for k in range(25):                               # ~3.5 m of arc
        a, b = (j0 + k) % n, (j0 + k + 1) % n
        dt_seg = math.hypot(rx[b] - rx[a], ry[b] - ry[a]) / rv[a]
        vx = rv[a] * 1.05 + rng.normal(0.0, 0.05)     # emulated EKF twist
        dyaw = math.atan2(math.sin(h[b] - h[a]), math.cos(h[b] - h[a]))
        wz = dyaw / dt_seg + rng.normal(0.0, 0.02)
        pf.predict(vx, 0.0, wz, dt_seg)
        pf.update(_scan(gm, rx[b], ry[b], rh[b], rng), ANGLE_MIN, ANGLE_INC, 20)
        ex, ey, eyaw, _ = pf.pose()
        err.append(math.hypot(ex - rx[b], ey - ry[b]))
    err = np.array(err)
    assert err[-1] < 0.30, f'final corner error {err[-1]:.3f} m'
    assert err.mean() < 0.30, f'mean corner error {err.mean():.3f} m'


def test_resampling_preserves_weight_mass_and_support():
    gm, (rx, ry, rh, _, _) = _load()
    pf = ParticleFilter(gm, n_particles=500, seed=4)
    pf.initialize(rx[0], ry[0], rh[0], spread=0.5)
    # skew the weights hard, remember the support
    w = np.exp(np.linspace(-8.0, 0.0, pf.n))
    pf.w = w / w.sum()
    before = pf.p.copy()
    pf.resample()
    assert abs(pf.w.sum() - 1.0) < 1e-12              # mass preserved
    assert np.allclose(pf.w, 1.0 / pf.n)              # uniform after resample
    # every resampled particle existed in the prior support
    match = (pf.p[:, None, :] == before[None, :, :]).all(axis=2).any(axis=1)
    assert match.all(), 'resampled particle not drawn from the prior set'
    # heavy tail dominates: the low-weight half should be mostly culled
    from_low_half = np.isin(pf.p[:, 0], before[:pf.n // 2, 0])
    assert from_low_half.mean() < 0.30


def test_global_init_lies_in_free_space():
    gm, _ = _load()
    pf = ParticleFilter(gm, n_particles=800, seed=5)  # global init by default
    assert not np.any(gm.is_occupied(pf.p[:, 0], pf.p[:, 1]))
    assert pf.lost                                    # starts in recovery mode


def test_update_is_vectorized_fast():
    """1500 particles x 18 beams: far under any Python-loop regression.

    A per-particle Python loop costs >100 ms here; the vectorized path runs
    in ~1-2 ms.  The 50 ms bound is generous for slow CI machines while
    still catching any loop regression by an order of magnitude."""
    gm, (rx, ry, rh, _, _) = _load()
    pf = ParticleFilter(gm, n_particles=1500, seed=6)
    pf.initialize(rx[0], ry[0], rh[0], spread=0.3)
    scan = _scan(gm, rx[0], ry[0], rh[0])
    pf.update(scan, ANGLE_MIN, ANGLE_INC, 20)         # warm-up (JIT caches)
    t0 = time.perf_counter()
    n_runs = 5
    for _ in range(n_runs):
        pf.predict(2.0, 0.0, 0.1, 0.05)
        pf.update(scan, ANGLE_MIN, ANGLE_INC, 20)
    per_update = (time.perf_counter() - t0) / n_runs
    assert per_update < 0.050, f'update took {1e3 * per_update:.1f} ms'


def test_pose_circular_mean_handles_pi_wrap():
    gm, _ = _load()
    pf = ParticleFilter(gm, n_particles=100, seed=7)
    pf.p[:, 2] = math.pi - 0.05
    pf.p[1::2, 2] = -math.pi + 0.05                   # straddle the wrap
    pf.w = np.full(pf.n, 1.0 / pf.n)
    _, _, yaw, _ = pf.pose()
    assert _ang(yaw, math.pi) < 0.06                  # NOT the naive mean ~0


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-q']))
