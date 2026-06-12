"""
Autonomy-mode tests — RC kill-switch logic and sidewalk-following core.
=======================================================================

The safety-relevant pure logic: the RC pulse classifier (hysteresis,
boot-disarmed, stale-equals-off) and the drivable-mask steering core
(centroid following, surface-lost stop, speed governance).

    python3 -m pytest tests/test_autonomy_modes.py -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'f1tenth_gym_ros'))
from rc_monitor import PulseClassifier                       # noqa: E402
from sidewalk_follow import (mask_to_steering,               # noqa: E402
                             speed_from_confidence)


# ─────────────────────────────────────────────────────────────────────────────
# RC kill switch
# ─────────────────────────────────────────────────────────────────────────────

def test_rc_boots_disarmed_and_arms_on_high_pulse():
    clf = PulseClassifier()
    assert clf.state(0.0) is False                  # never saw a pulse
    clf.feed(1900, t=1.0)
    assert clf.state(1.0) is True


def test_rc_hysteresis_band_keeps_state():
    clf = PulseClassifier(manual_below=1300, auto_above=1700)
    clf.feed(1900, t=0.0)
    clf.feed(1500, t=0.1)                           # in the dead band
    assert clf.state(0.1) is True                   # keeps armed
    clf.feed(1200, t=0.2)
    assert clf.state(0.2) is False
    clf.feed(1500, t=0.3)
    assert clf.state(0.3) is False                  # keeps disarmed


def test_rc_stale_means_disarmed():
    # transmitter off / out of range: no pulses -> off, regardless of state
    clf = PulseClassifier(stale_after=0.5)
    clf.feed(1900, t=0.0)
    assert clf.state(0.4) is True
    assert clf.state(0.6) is False


# ─────────────────────────────────────────────────────────────────────────────
# Sidewalk-following core
# ─────────────────────────────────────────────────────────────────────────────

def _mask(w_center, width=40, h=100, w=200):
    m = np.zeros((h, w), bool)
    lo = max(0, int(w_center - width / 2))
    m[:, lo:int(w_center + width / 2)] = True
    return m


def test_centered_sidewalk_steers_straight():
    steer, frac, ok = mask_to_steering(_mask(100))
    assert ok and abs(steer) < 0.02


def test_offset_sidewalk_steers_toward_it():
    left, _, ok_l = mask_to_steering(_mask(50))     # sidewalk left of centre
    right, _, ok_r = mask_to_steering(_mask(150))
    assert ok_l and ok_r
    assert left > 0.05                              # REP 103: + steer = left
    assert right < -0.05
    assert abs(left + right) < 0.02                 # symmetric


def test_surface_lost_stops():
    steer, frac, ok = mask_to_steering(np.zeros((100, 200), bool))
    assert not ok and steer == 0.0
    # a sliver below min_fraction also stops
    sliver = np.zeros((100, 200), bool)
    sliver[60:62, 95:105] = True
    assert mask_to_steering(sliver)[2] is False


def test_speed_governs_with_confidence():
    assert speed_from_confidence(0.5) == 1.2        # plenty of sidewalk
    assert speed_from_confidence(0.15) < 1.2        # scaled down
    assert speed_from_confidence(0.01) == 0.4       # floor


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-q']))
