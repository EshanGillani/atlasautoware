#!/usr/bin/env python3
"""
Fit the car's calibration constants from a data_logger CSV.
===========================================================

    python3 tools/sysid_report.py /tmp/run1.csv [--wheelbase 0.33] \
        [--steer-half-range-us 400] [--max-steer 0.41]

Prints the hardware.yaml values to set: actuation_delay (from steering<->
yaw-rate cross-correlation), erpm_gain (wheel speed vs PF speed), and
steer_trim_us (straight-commanded yaw drift).  Each estimate reports its
own quality signal; rerun the suggested maneuver if one is weak.
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'f1tenth_gym_ros'))
from sysid import (estimate_delay, estimate_erpm_gain,   # noqa: E402
                   estimate_steering_bias)


def load(path):
    cols = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v) if v else math.nan)
    return {k: np.array(v) for k, v in cols.items()}


def series(d, field):
    ok = np.isfinite(d[field])
    return d['t'][ok], d[field][ok]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('csv')
    ap.add_argument('--wheelbase', type=float, default=0.33)
    ap.add_argument('--max-steer', type=float, default=0.41)
    ap.add_argument('--steer-half-range-us', type=float, default=400.0)
    a = ap.parse_args()
    d = load(a.csv)

    print(f'{os.path.basename(a.csv)}: {len(d["t"])} rows, '
          f'{d["t"][-1] - d["t"][0]:.1f} s\n')

    # actuation delay
    try:
        tc, sc = series(d, 'steer_cmd')
        tg, gz = series(d, 'gyro_z')
        delay, corr = estimate_delay(tc, sc, tg, gz)
        q = 'good' if corr > 0.6 else ('weak — drive S-turns' if corr > 0.3
                                       else 'unusable — drive S-turns')
        print(f'actuation_delay: {delay:.3f}   # correlation {corr:.2f} ({q})')
    except Exception as e:
        print(f'actuation_delay: n/a ({e})')

    # erpm gain (vesc wheel speed is erpm/erpm_gain at the CURRENT setting —
    # we fit the correction factor against PF ground-truth speed)
    try:
        tw, vw = series(d, 'wheel_speed')
        tp, vp = series(d, 'pf_speed')
        vp_i = np.interp(tw, tp, vp)
        ratio = estimate_erpm_gain(vw, vp_i)   # current-gain wheel vs truth
        print(f'erpm_gain: multiply current value by {ratio:.3f}   '
              f'# wheel/PF speed ratio')
    except Exception as e:
        print(f'erpm_gain: n/a ({e})')

    # steering trim
    try:
        tg, gz = series(d, 'gyro_z')
        tc, sc = series(d, 'steer_cmd')
        tw, vw = series(d, 'wheel_speed')
        sc_i = np.interp(tg, tc, sc)
        vw_i = np.interp(tg, tw, vw)
        bias, n = estimate_steering_bias(vw_i, gz, sc_i, a.wheelbase)
        trim_us = -bias / a.max_steer * a.steer_half_range_us
        print(f'steer_trim_us: {trim_us:+.1f}   '
              f'# bias {math.degrees(bias):+.2f} deg over {n} samples')
    except Exception as e:
        print(f'steer_trim_us: n/a ({e})')


if __name__ == '__main__':
    main()
