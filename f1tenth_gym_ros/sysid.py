"""
On-track system identification — measure the car instead of guessing.
=====================================================================

Pure estimators for the calibration constants the stack depends on, fitted
from a short logged drive (see data_logger.py + tools/sysid_report.py):

  actuation delay   cross-correlate commanded steering against the IMU yaw
                    rate: at near-constant speed omega ~ (v/L_wb) tan(delta),
                    so the yaw-rate trace is a scaled, DELAYED copy of the
                    steering command.  The lag maximizing their correlation
                    is the sensor-to-actuator delay `actuation_delay` that
                    the latency compensator needs.  Drive ~20 s of gentle
                    S-turns (vary the steering, hold speed) to excite it.
  erpm gain         least-squares ratio of VESC erpm to a reference speed
                    (PF speed, measured distance/time, or tape-measure runs).
  steering bias     mean yaw rate while commanding straight at speed implies
                    the actual steering angle: delta = atan(omega L / v);
                    feeds `steer_trim_us` (convert via your servo scale).

All numpy, all unit-tested on synthetic logs with known ground truth
(tests/test_sysid.py).  ForzaETH's on-track sysid (arXiv:2411.17508) goes
further (tire curves); these three constants are the ones this stack's
parameters consume directly.
"""

import numpy as np


def _resample(t, v, t_grid):
    return np.interp(t_grid, np.asarray(t, float), np.asarray(v, float))


def estimate_delay(t_cmd, steer_cmd, t_gyro, yaw_rate,
                   max_delay=0.4, dt=0.005):
    """Sensor->actuator delay (s) by normalized cross-correlation.

    Returns (delay_s, peak_correlation).  Correlation < ~0.5 means the run
    didn't excite steering enough — drive S-turns and retry.
    """
    t0 = max(float(np.min(t_cmd)), float(np.min(t_gyro)))
    t1 = min(float(np.max(t_cmd)), float(np.max(t_gyro)))
    if t1 - t0 < 2.0:
        raise ValueError('need >= 2 s of overlapping log')
    grid = np.arange(t0, t1, dt)
    c = _resample(t_cmd, steer_cmd, grid)
    g = _resample(t_gyro, yaw_rate, grid)
    c = c - c.mean()
    g = g - g.mean()
    denom = np.sqrt((c * c).sum() * (g * g).sum()) + 1e-12
    lags = np.arange(0, int(round(max_delay / dt)) + 1)
    corr = np.array([(c[:len(c) - k or None] * g[k:]).sum() / denom
                     for k in lags])
    best = int(np.argmax(np.abs(corr)))
    return float(lags[best] * dt), float(abs(corr[best]))


def estimate_erpm_gain(erpm, v_ref, v_min=0.5):
    """erpm-per-(m/s): least-squares ratio over samples moving > v_min."""
    erpm = np.asarray(erpm, float)
    v = np.asarray(v_ref, float)
    keep = v > float(v_min)
    if keep.sum() < 10:
        raise ValueError('need >= 10 samples above v_min')
    return float((erpm[keep] * v[keep]).sum() / (v[keep] * v[keep]).sum())


def estimate_steering_bias(speed, yaw_rate, steer_cmd, wheelbase=0.33,
                           v_min=1.0, cmd_tol=0.02):
    """Steering bias (rad) from straight-commanded segments at speed.

    Positive result = the car curves left when told to go straight, so trim
    right by this much (steer_trim_us = -bias / max_steer * half_range_us).
    Returns (bias_rad, n_samples_used).
    """
    v = np.asarray(speed, float)
    w = np.asarray(yaw_rate, float)
    c = np.asarray(steer_cmd, float)
    keep = (v > float(v_min)) & (np.abs(c) < float(cmd_tol))
    if keep.sum() < 20:
        raise ValueError('need >= 20 straight-commanded samples at speed')
    # each sample implies delta = atan(omega * L / v); average robustly
    delta = np.arctan(w[keep] * float(wheelbase) / v[keep])
    return float(np.median(delta)), int(keep.sum())
