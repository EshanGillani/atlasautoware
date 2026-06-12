"""
Unit tests — minimum-curvature raceline refiner.
================================================

Checks the three properties that make the refiner safe to trust:
curvature actually goes down, no point ever leaves the corridor, and the
closed loop stays seamless across the wrap at index 0.

    python3 -m pytest tests/test_refiner.py -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'f1tenth_gym_ros'))
from raceline_refiner import refine_raceline, heading_curvature  # noqa: E402


def _noisy_circle(n=180, radius=3.0, sigma=0.03, seed=0):
    rng = np.random.default_rng(seed)
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = radius + rng.normal(0.0, sigma, n)
    return r * np.cos(th), r * np.sin(th)


def test_curvature_strictly_reduced_on_noisy_circle():
    x, y = _noisy_circle()
    _, k0 = heading_curvature(x, y)
    _, _, _, k1 = refine_raceline(x, y, corridor=0.10)
    assert k1.max() < k0.max()
    assert np.abs(k1).mean() < np.abs(k0).mean()
    # and it lands near the clean circle's curvature (1/R), not just "less"
    assert abs(k1.mean() - 1.0 / 3.0) < 0.05


def test_displacement_respects_corridor():
    x, y = _noisy_circle()
    for corridor in (0.05, 0.10, 0.25):
        xn, yn, _, _ = refine_raceline(x, y, corridor=corridor)
        disp = np.hypot(xn - x, yn - y)
        assert disp.max() <= corridor + 1e-6
        assert disp.max() > 0.0                      # it did move


def test_wrap_continuity_no_kink_at_index_zero():
    x, y = _noisy_circle()
    xn, yn, hdg, k = refine_raceline(x, y, corridor=0.10)
    n = len(xn)
    # curvature at the seam is no outlier vs the rest of the loop
    for i in (-1, 0, 1):
        assert abs(k[i] - np.median(k)) < 3.0 * k.std() + 1e-3
    # heading steps (wrapped) are uniform across the seam, ~2*pi/n each
    step = np.abs(np.angle(np.exp(1j * np.diff(hdg, append=hdg[0]))))
    assert step.max() < 2.5 * (2.0 * np.pi / n)


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-q']))
