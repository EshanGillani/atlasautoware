"""
System-identification tests — synthetic logs with known ground truth.
=====================================================================

Each estimator must recover a planted constant from a noisy synthetic
drive: 90 ms actuation delay from a PRBS steering trace, an erpm-gain
ratio, and a 1.5-degree steering bias.

    python3 -m pytest tests/test_sysid.py -q
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'f1tenth_gym_ros'))
from sysid import (estimate_delay, estimate_erpm_gain,   # noqa: E402
                   estimate_steering_bias)


def test_delay_recovered_from_prbs_steering():
    rng = np.random.default_rng(0)
    dt, T, true_delay = 0.005, 25.0, 0.09
    t = np.arange(0.0, T, dt)
    # pseudo-random binary steering, switching every ~0.4 s
    switches = rng.choice([-0.15, 0.15], size=int(T / 0.4) + 1)
    steer = switches[(t / 0.4).astype(int)]
    v, L = 2.5, 0.33
    lag = int(round(true_delay / dt))
    delayed = np.concatenate([np.zeros(lag), steer[:len(t) - lag]])
    yaw_rate = v / L * np.tan(delayed) + rng.normal(0, 0.05, len(t))
    # commands logged at 50 Hz, gyro at 200 Hz (mimic real topic rates)
    delay, corr = estimate_delay(t[::4], steer[::4], t, yaw_rate)
    assert abs(delay - true_delay) <= 0.015, f'{delay:.3f} vs {true_delay}'
    assert corr > 0.6


def test_delay_needs_overlap():
    try:
        estimate_delay([0, 1], [0, 0.1], [10, 11], [0, 0.1])
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_erpm_gain_ratio():
    rng = np.random.default_rng(1)
    v = np.concatenate([np.zeros(20), np.linspace(0.6, 6.0, 200)])
    erpm = 4614.0 * v + rng.normal(0, 50.0, len(v))
    gain = estimate_erpm_gain(erpm, v)
    assert abs(gain - 4614.0) / 4614.0 < 0.02


def test_steering_bias_recovered():
    rng = np.random.default_rng(2)
    bias = math.radians(1.5)
    n = 2000
    v = np.full(n, 3.0)
    steer_cmd = np.zeros(n)                      # commanding straight
    steer_cmd[::7] = 0.2                         # occasional turns: excluded
    actual = steer_cmd + bias
    yaw = v / 0.33 * np.tan(actual) + rng.normal(0, 0.05, n)
    est, used = estimate_steering_bias(v, yaw, steer_cmd, wheelbase=0.33)
    assert abs(est - bias) < math.radians(0.4)
    assert used > 1000


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-q']))
