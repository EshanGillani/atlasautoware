"""
Racing-tech unit + closed-loop tests — MAP controller and velocity profiler.
============================================================================

Validates the two research-stack upgrades without ROS or hardware:
  - velocity_profiler: lateral cap, braking feasibility, friction-ellipse
    coupling, closed-track wraparound;
  - map_controller: LUT physics (kinematic agreement at low g, understeer
    compensation at high g, grip-limit clamp), and a full closed-loop lap on
    the real competition raceline with a kinematic plant.

    python3 -m pytest tests/test_racing_tech.py -q
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'f1tenth_gym_ros'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from velocity_profiler import velocity_profile, segment_lengths  # noqa: E402
from map_controller import MAPController, build_lat_accel_lut    # noqa: E402
from mpc_controller import predict_state                         # noqa: E402
from closed_loop import load_raceline, run_lap                   # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────────────────────────────────────
# Velocity profiler
# ─────────────────────────────────────────────────────────────────────────────

def test_profile_constant_radius_circle():
    # steady circle: v = sqrt(a_lat * R) everywhere, capped by v_max
    kappa = np.full(100, 1.0 / 2.0)                  # R = 2 m
    ds = np.full(100, 0.2)
    v = velocity_profile(kappa, ds, a_lat_max=6.0, v_max=10.0)
    assert np.allclose(v, math.sqrt(6.0 * 2.0), atol=1e-6)
    v = velocity_profile(kappa, ds, a_lat_max=6.0, v_max=3.0)
    assert np.allclose(v, 3.0)                       # ceiling wins


def test_profile_straight_hits_vmax():
    v = velocity_profile(np.zeros(50), np.full(50, 1.0), v_max=8.0,
                         closed=False)
    assert v.max() == 8.0


def test_profile_brakes_for_hairpin():
    # long straight into a tight corner: approach speeds must satisfy
    # v[i]^2 <= v_corner^2 + 2*a_brake*dist  (braking feasibility)
    kappa = np.zeros(200)
    kappa[100:110] = 1.0 / 0.5                       # R = 0.5 m hairpin
    ds = np.full(200, 0.25)
    a_brake = 8.0
    v = velocity_profile(kappa, ds, a_lat_max=6.0, a_brake_max=a_brake,
                         v_max=8.0)
    v_corner = math.sqrt(6.0 * 0.5)
    assert v[100] <= v_corner + 1e-9
    for i in range(60, 100):                         # entire approach feasible
        dist = (100 - i) * 0.25
        assert v[i] ** 2 <= v_corner ** 2 + 2.0 * a_brake * dist + 1e-6
    # and the deceleration between consecutive points never exceeds the limit
    dec = (v[:-1] ** 2 - v[1:] ** 2) / (2.0 * ds[:-1])
    assert dec.max() <= a_brake + 1e-6


def test_profile_friction_ellipse_coupling():
    # at the lateral limit there is NO accel budget left: speed cannot rise
    # while still in the corner
    kappa = np.concatenate([np.full(50, 1.0 / 2.0), np.zeros(50)])
    ds = np.full(100, 0.2)
    v = velocity_profile(kappa, ds, a_lat_max=6.0, a_accel_max=4.0, v_max=9.0)
    v_corner = math.sqrt(6.0 * 2.0)
    assert np.all(v[10:49] <= v_corner + 1e-6)
    assert v[60] > v_corner                          # accelerates on the straight


def test_profile_closed_track_wraparound():
    # corner right AFTER the start line must brake the END of the lap
    kappa = np.zeros(100)
    kappa[2:8] = 1.0 / 0.5
    ds = np.full(100, 0.3)
    v = velocity_profile(kappa, ds, a_lat_max=6.0, a_brake_max=8.0, v_max=8.0)
    v_corner = math.sqrt(6.0 * 0.5)
    # last points of the lap are already braking for the corner past the line
    assert v[-1] ** 2 <= v_corner ** 2 + 2.0 * 8.0 * (0.3 * 4) + 1e-6
    open_v = velocity_profile(kappa, ds, a_lat_max=6.0, a_brake_max=8.0,
                              v_max=8.0, closed=False)
    assert v[-1] < open_v[-1]                        # wraparound did constrain it


# ─────────────────────────────────────────────────────────────────────────────
# MAP controller
# ─────────────────────────────────────────────────────────────────────────────

LUT = build_lat_accel_lut()                          # f1tenth_gym defaults


def _ctl():
    return MAPController(lut=LUT)


def test_lut_matches_kinematics_at_low_lateral_g():
    # well below the grip limit, tire slip is tiny: LUT^-1 ~ atan(L*a/v^2)
    ctl = _ctl()
    v, a = 3.0, 1.5
    kin = math.atan(0.33 * a / v ** 2)
    assert abs(ctl.steer_from_lat_accel(a, v) - kin) < 0.015


def test_lut_understeer_compensation_at_high_g():
    # near the limit the dynamic model needs MORE steer than kinematics —
    # exactly the systematic error pure pursuit leaves uncorrected
    ctl = _ctl()
    v, a = 5.0, 8.0
    kin = math.atan(0.33 * a / v ** 2)
    assert ctl.steer_from_lat_accel(a, v) > kin


def test_lut_inversion_sign_and_clamp():
    ctl = _ctl()
    s = ctl.steer_from_lat_accel(4.0, 4.0)
    assert ctl.steer_from_lat_accel(-4.0, 4.0) == -s  # symmetric
    big = ctl.steer_from_lat_accel(50.0, 4.0)         # beyond grip: clamps,
    assert 0.0 < big <= 0.41                          # never explodes


def test_map_closed_loop_lap():
    # full lap on the real competition raceline, kinematic plant @50 Hz —
    # the MAP fallback must lap cleanly with bounded cross-track error
    rx, ry, rh, rc, rv = load_raceline(
        os.path.join(REPO, 'racelines', 'comp_raceline.csv'))
    ctl = _ctl()
    ctl.set_raceline(rx, ry, rv)
    res = run_lap(ctl.control, rx, ry, rh)
    assert res['completed'], 'did not complete a lap'
    assert res['xte_mean'] < 0.15, f"loose tracking ({res['xte_mean']:.2f} m)"
    assert res['xte_max'] < 0.6, f"ran wide ({res['xte_max']:.2f} m)"


# ─────────────────────────────────────────────────────────────────────────────
# Actuation-delay compensation
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_state_straight_line():
    # constant speed, zero steer: the car just travels v*delay forward
    x, y, yaw, v = predict_state(0.0, 0.0, 0.0, 4.0, 0.0, 4.0, 0.1, 0.33)
    assert abs(x - 0.4) < 1e-9 and y == 0.0 and yaw == 0.0 and v == 4.0
    assert predict_state(1.0, 2.0, 0.3, 4.0, 0.1, 4.0, 0.0, 0.33) == \
        (1.0, 2.0, 0.3, 4.0)                            # zero delay: identity


def test_delay_compensation_recovers_tracking():
    # 100 ms actuator latency wrecks the uncompensated lap; predicting the
    # state by the delay before each control step must restore it
    rx, ry, rh, rc, rv = load_raceline(
        os.path.join(REPO, 'racelines', 'comp_raceline.csv'))
    ctl = _ctl()
    ctl.set_raceline(rx, ry, rv)
    delay = 0.10
    plain = run_lap(ctl.control, rx, ry, rh, actuator_delay=delay)

    last = {'steer': 0.0, 'v': 2.0}

    def compensated(px, py, yaw, v, j):
        px, py, yaw, v = predict_state(px, py, yaw, v, last['steer'],
                                       last['v'], delay, 0.33)
        j = int(np.argmin((rx - px) ** 2 + (ry - py) ** 2))
        steer, v_t = ctl.control(px, py, yaw, v, j)
        last['steer'], last['v'] = float(steer), float(v_t)
        return steer, v_t

    comp = run_lap(compensated, rx, ry, rh, actuator_delay=delay)
    assert comp['completed'] and plain['completed']
    assert comp['xte_max'] < plain['xte_max'] * 0.75    # clearly tighter
    assert comp['lap_time'] <= plain['lap_time'] + 0.1  # and no slower


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-q']))
