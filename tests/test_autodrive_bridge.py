"""
Unit tests for the AutoDRIVE bridge's pure conversions (no ROS needed).

    python3 -m pytest tests/test_autodrive_bridge.py -q
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'f1tenth_gym_ros'))
from autodrive_bridge import (throttle_from_speed, steering_from_angle,   # noqa
                              mask_scan_ranges)


def test_throttle_normalizes_and_clips():
    assert throttle_from_speed(0.0, 22.8) == 0.0
    assert throttle_from_speed(22.8, 22.8) == 1.0
    assert throttle_from_speed(-22.8, 22.8) == -1.0
    assert throttle_from_speed(99.0, 22.8) == 1.0          # clipped
    assert throttle_from_speed(-99.0, 22.8) == -1.0        # clipped
    assert abs(throttle_from_speed(11.4, 22.8) - 0.5) < 1e-9
    assert throttle_from_speed(5.0, 0.0) == 0.0            # guard div-by-zero


def test_steering_normalizes_clips_and_signs():
    assert steering_from_angle(0.0, 0.5236) == 0.0
    assert steering_from_angle(0.5236, 0.5236) == 1.0
    assert steering_from_angle(-0.5236, 0.5236) == -1.0
    assert steering_from_angle(1.0, 0.5236) == 1.0         # clipped to lock
    # sign flip (for sims where +command turns the other way)
    assert steering_from_angle(0.5236, 0.5236, steer_sign=-1.0) == -1.0
    assert steering_from_angle(5.0, 0.0) == 0.0            # guard


def test_scan_masks_clamped_maxima_to_inf():
    # AutoDRIVE clamps no-return beams to range_max; we convert to inf
    out = mask_scan_ranges([0.5, 9.9995, 10.0, 3.0, float('nan')],
                           range_max=10.0)
    assert out[0] == 0.5
    assert math.isinf(out[1])           # ~range_max -> inf
    assert math.isinf(out[2])           # range_max -> inf
    assert out[3] == 3.0
    assert math.isinf(out[4])           # nan -> inf (no-return)


def test_scan_keeps_in_range_values():
    out = mask_scan_ranges([0.06, 1.0, 5.0], range_max=10.0)
    assert out == [0.06, 1.0, 5.0]


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-q']))
