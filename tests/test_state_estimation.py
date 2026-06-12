"""
State-estimation tests — velocity EKF (slip gating) and lidar de-skew.
======================================================================

Synthetic-truth validation of the two estimation-layer additions:
  - VelocityEKF: convergence on clean data, slip-pulse rejection (the gated
    filter must beat both the raw wheel signal and an ungated filter while
    the wheel is spinning 25% fast), slip-flag correctness;
  - scan de-skew: a scan synthesized from a moving, turning sensor must be
    restored to the static-pose ground truth (cloud RMS), identity at rest.

    python3 -m pytest tests/test_state_estimation.py -q
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'f1tenth_gym_ros'))
from velocity_ekf import VelocityEKF                             # noqa: E402
from scan_deskew import deskew_points, deskew_ranges             # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic drive: accel/brake cycles + coordinated corners, IMU @200 Hz,
# wheel speed @50 Hz with slip pulses during hard accel/brake
# ─────────────────────────────────────────────────────────────────────────────

def synth_drive(t_end=30.0, imu_hz=200, wheel_hz=50, slip_ratio=0.25,
                seed=0):
    rng = np.random.default_rng(seed)
    dt = 1.0 / imu_hz
    t = np.arange(0.0, t_end, dt)
    # longitudinal: 3 m/s^2 accel to 6 m/s, cruise, -5 m/s^2 brake to 2, repeat
    ax = np.zeros_like(t)
    vx = np.zeros_like(t)
    v = 2.0
    for i, ti in enumerate(t):
        phase = ti % 10.0
        a = 3.0 if phase < 2.0 else (0.0 if phase < 6.0 else -5.0)
        if (v >= 6.0 and a > 0) or (v <= 2.0 and a < 0):
            a = 0.0
        ax[i] = a
        vx[i] = v
        v += a * dt
    # corners: yaw rate on for 2 s out of every 5 s.  Kinematically consistent
    # truth: the CG slips sideways at vy = l_r * omega (rear axle is the
    # nonholonomic point), and the accelerometer measures specific force
    # ax_meas = vdot_x - omega*vy, ay_meas = vdot_y + omega*vx.
    l_r = 0.17145
    omega = np.where((t % 5.0) > 3.0, 1.8, 0.0)
    ramp = int(0.3 / dt)                             # ~0.3 s steering ramp
    omega = np.convolve(omega, np.ones(ramp) / ramp, mode='same')
    vy = l_r * omega
    vy_dot = np.gradient(vy, dt)
    # wheel slips when pushed hard; first seconds are slip-free so the filter
    # anchors on honest data (matches reality: filters start at standstill)
    slip = (np.abs(ax) > 2.5) & (t > 3.0)
    imu_ax = (ax - omega * vy) + rng.normal(0.0, 0.2, len(t)) + 0.05
    imu_ay = (vy_dot + omega * vx) + rng.normal(0.0, 0.2, len(t))
    imu_w = omega + rng.normal(0.0, 0.01, len(t))
    every = imu_hz // wheel_hz
    wheel_t = t[::every]
    wheel_v = (vx * (1.0 + slip_ratio * slip))[::every] \
        + rng.normal(0.0, 0.05, len(wheel_t))
    return t, dt, vx, slip, imu_ax, imu_ay, imu_w, every, wheel_v


def run_filter(gated=True):
    t, dt, vx_true, slip, ax, ay, w, every, wheel_v = synth_drive()
    ekf = VelocityEKF()
    ekf.accel_bias = np.array([0.05, 0.0])           # standstill calibration
    ekf.x[0] = 2.0
    if not gated:
        ekf.slip_thresh = float('inf')
    est, flags = np.zeros(len(t)), np.zeros(len(t), bool)
    k = 0
    for i in range(len(t)):
        ekf.predict(ax[i], ay[i], dt)
        ekf.update_gyro(w[i])
        ekf.update_nonholonomic()
        if i % every == 0:
            ekf.update_wheel_speed(wheel_v[k])
            k += 1
        est[i] = ekf.x[0]
        flags[i] = ekf.slip
    return t, vx_true, slip, est, flags, every, wheel_v


def test_ekf_converges_on_clean_segments():
    t, vx_true, slip, est, _, _, _ = run_filter(gated=True)
    clean = ~slip
    rmse = math.sqrt(float(np.mean((est[clean] - vx_true[clean]) ** 2)))
    assert rmse < 0.08, f'clean-segment RMSE too high ({rmse:.3f} m/s)'


def test_gating_beats_raw_wheel_and_ungated_during_slip():
    t, vx_true, slip, est_g, _, every, wheel_v = run_filter(gated=True)
    _, _, _, est_u, _, _, _ = run_filter(gated=False)
    wheel_full = np.repeat(wheel_v, every)[:len(t)]
    e_gated = math.sqrt(float(np.mean((est_g[slip] - vx_true[slip]) ** 2)))
    e_ungated = math.sqrt(float(np.mean((est_u[slip] - vx_true[slip]) ** 2)))
    e_raw = math.sqrt(float(np.mean((wheel_full[slip] - vx_true[slip]) ** 2)))
    # raw wheel reads ~25% fast during slip; the gate must reject most of it
    assert e_gated < 0.5 * e_raw, f'gated {e_gated:.3f} vs raw {e_raw:.3f}'
    assert e_gated < 0.7 * e_ungated, \
        f'gated {e_gated:.3f} vs ungated {e_ungated:.3f}'


def test_slip_flag_fires_during_pulses():
    t, _, slip, _, flags, _, _ = run_filter(gated=True)
    # the flag is sampled at wheel updates; require decent hit rate in-pulse
    # and a low false-positive rate on clean cruise segments
    assert flags[slip].mean() > 0.5
    assert flags[~slip].mean() < 0.1


def test_cmd_speed_fallback_bounds_imu_drift():
    # PCA9685-only mode (no VESC -> no wheel odometry): an uncalibrated IMU
    # bias makes pure inertial vx drift without bound; the weakly-trusted
    # commanded-speed update must keep the estimate bounded
    dt, t_end, v_true = 1.0 / 200, 10.0, 3.0
    rng = np.random.default_rng(1)

    def run(with_cmd):
        ekf = VelocityEKF()
        ekf.x[0] = v_true                            # starts converged
        for i in range(int(t_end / dt)):
            ekf.predict(0.3 + rng.normal(0, 0.2), rng.normal(0, 0.2), dt)
            ekf.update_gyro(rng.normal(0, 0.01))
            ekf.update_nonholonomic()
            if with_cmd and i % 4 == 0:              # 50 Hz /drive commands
                ekf.update_cmd_speed(v_true)
        return abs(ekf.x[0] - v_true)

    drift_free = run(with_cmd=False)
    drift_cmd = run(with_cmd=True)
    assert drift_free > 1.5                          # bias really does run away
    assert drift_cmd < 0.5, f'cmd fallback too loose ({drift_cmd:.2f} m/s)'


# ─────────────────────────────────────────────────────────────────────────────
# De-skew: synthesize a skewed scan of a circular wall from a moving sensor
# ─────────────────────────────────────────────────────────────────────────────

def _circle_range(px, py, dx, dy, R):
    """Range from (px,py) along (dx,dy) to the circle of radius R at origin."""
    b = px * dx + py * dy
    disc = b * b + R * R - (px * px + py * py)
    return -b + math.sqrt(disc)


def synth_skewed_scan(n=360, scan_time=0.1, vx=7.0, vy=0.0, omega=2.3,
                      R=5.0):
    """Per-beam honest raycast from the sensor's true pose at each beam time.

    World frame = sensor pose at scan END (so the truth cloud is simply the
    static scan from the origin).  Beam i fires at s_i = (i/n - 1)*scan_time.
    """
    angle_min, inc = -math.pi, 2.0 * math.pi / n
    ranges = np.zeros(n)
    for i in range(n):
        s = (i / n - 1.0) * scan_time
        th = omega * s                              # sensor yaw at fire time
        px, py = vx * s, vy * s                     # sensor position
        phi = angle_min + i * inc
        d_world = (math.cos(th + phi), math.sin(th + phi))
        ranges[i] = _circle_range(px, py, d_world[0], d_world[1], R)
    truth = np.full(n, R)                           # static scan from origin
    return ranges, truth, angle_min, inc


def test_deskew_restores_static_geometry():
    # all wall returns must lie back ON the wall after correction (a corrected
    # beam lands at a different bearing, so compare distance-to-wall, not
    # per-index points)
    n, R = 360, 5.0
    ranges, _, amin, inc = synth_skewed_scan(n=n, R=R)
    phi = amin + np.arange(n) * inc
    xs, ys = np.cos(phi) * ranges, np.sin(phi) * ranges
    err_before = float(np.sqrt(np.mean((np.hypot(xs, ys) - R) ** 2)))
    xc, yc = deskew_points(ranges, amin, inc, 0.1, 7.0, 0.0, 2.3)
    err_after = float(np.sqrt(np.mean((np.hypot(xc, yc) - R) ** 2)))
    assert err_before > 0.15                        # the skew is real
    assert err_after < 0.05 * err_before, \
        f'deskew {err_after:.4f} m vs skewed {err_before:.4f} m'


def test_deskew_identity_at_rest():
    n = 180
    r = np.full(n, 4.0)
    out = deskew_ranges(r, -math.pi, 2 * math.pi / n, 0.1, 0.0, 0.0, 0.0)
    assert np.allclose(out, 4.0, atol=1e-6)


def test_deskew_handles_invalid_beams():
    r = np.array([np.inf, 2.0, np.nan, 3.0])
    xc, yc = deskew_points(r, -math.pi, math.pi / 2, 0.1, 5.0, 0.0, 1.0)
    assert np.isnan(xc[0]) and np.isnan(xc[2])
    assert np.isfinite(xc[1]) and np.isfinite(xc[3])


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-q']))
