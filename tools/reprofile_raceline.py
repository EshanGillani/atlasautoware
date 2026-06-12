#!/usr/bin/env python3
"""
Recompute a raceline CSV's speed column with the friction-limited profiler.
===========================================================================

Reads a raceline CSV (x, y, heading, curvature, speed), replaces the speed
column with the TUMFTM forward-backward friction-limited profile, and writes
the result — so the speeds the car races on provably fit the grip budget
instead of whatever the optimizer/hand-tuning left there.

    python3 tools/reprofile_raceline.py racelines/best_raceline.csv \
        --a-lat 6.0 --a-accel 4.0 --a-brake 8.0 --v-max 8.0 [-o out.csv]

(The same profile can be applied at runtime with raceline_mpc's
`reprofile_speeds` parameter; this tool bakes it into the file and prints the
estimated lap-time change.)
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'f1tenth_gym_ros'))
from velocity_profiler import velocity_profile, segment_lengths  # noqa: E402


def load_raceline(path):                  # local: pursuit_agent needs rclpy
    cols = {k: [] for k in ('x', 'y', 'heading', 'curvature', 'speed')}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k in cols:
                cols[k].append(float(row[k]))
    return tuple(np.array(cols[k]) for k in cols)


def save_raceline(path, x, y, hdg, curv, spd):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x', 'y', 'heading', 'curvature', 'speed'])
        for i in range(len(x)):
            w.writerow([round(x[i], 4), round(y[i], 4), round(hdg[i], 4),
                        round(curv[i], 6), round(spd[i], 3)])


def lap_time(speeds, ds):
    v_seg = np.maximum((speeds + np.roll(speeds, -1)) / 2.0, 0.1)
    return float(np.sum(ds / v_seg))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('csv', help='raceline CSV (x,y,heading,curvature,speed)')
    ap.add_argument('-o', '--out', default=None,
                    help='output CSV (default: overwrite input)')
    ap.add_argument('--a-lat', type=float, default=6.0, help='m/s^2 grip budget')
    ap.add_argument('--a-accel', type=float, default=4.0, help='m/s^2 engine limit')
    ap.add_argument('--a-brake', type=float, default=8.0, help='m/s^2 brake limit')
    ap.add_argument('--v-max', type=float, default=8.0, help='m/s ceiling')
    args = ap.parse_args()

    x, y, hdg, curv, old = load_raceline(args.csv)
    ds = segment_lengths(x, y)
    new = velocity_profile(curv, ds, a_lat_max=args.a_lat,
                           a_accel_max=args.a_accel, a_brake_max=args.a_brake,
                           v_max=args.v_max)
    out = args.out or args.csv
    save_raceline(out, x, y, hdg, curv, new)
    print(f'{os.path.basename(args.csv)}: {len(x)} pts')
    print(f'  speeds: {old.min():.1f}-{old.max():.1f} -> '
          f'{new.min():.1f}-{new.max():.1f} m/s')
    print(f'  est. lap: {lap_time(old, ds):.2f} s -> {lap_time(new, ds):.2f} s')
    print(f'  wrote {out}')


if __name__ == '__main__':
    main()
