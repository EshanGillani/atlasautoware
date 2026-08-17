#!/usr/bin/env python3
"""
One-command racing line for ANY track map — unseen-track ready (RoboRacer).
===========================================================================

The competition qualification races a *previously unseen* track, so the
"model" you bring is: a fast, feasible racing line + the MPC that tracks it.
This tool turns any occupancy-grid map into that line in one command:

    python3 tools/build_raceline.py maps/<track>.yaml

It auto-picks a drivable seed (the widest point on the track, which is always
on the racing surface), runs the minimum-curvature optimizer, then VALIDATES
the line in the deterministic closed-loop MPC benchmark and prints a
feasibility report (completes? lap time, cross-track error, and the planned
lateral-acceleration peak vs the grip budget). Output is a raceline CSV that
`raceline_mpc` (and the AutoDRIVE bridge) load directly.

No manual seed, no per-track hand-tuning — point it at a new map and go.
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tests'))
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))


def load_map(yaml_path):
    meta = yaml.safe_load(open(yaml_path))
    res = float(meta['resolution'])
    origin = meta['origin']
    img_path = meta['image']
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), img_path)
    img = np.array(Image.open(img_path).convert('L'))
    H, W = img.shape
    occ = (255 - img) / 255.0 > float(meta.get('occupied_thresh', 0.65))
    return res, origin, img, occ, H, W


def auto_seed(occ, res, origin, H):
    """Widest free point = max distance to any wall → guaranteed on-track."""
    edt = ndimage.distance_transform_edt(~occ)
    py, px = np.unravel_index(int(np.argmax(edt)), edt.shape)
    x = origin[0] + (px + 0.5) * res
    y = origin[1] + (H - 1 - py + 0.5) * res
    return x, y, float(edt[py, px] * res)


def clearance_fn(occ, res, origin, H):
    edt = ndimage.distance_transform_edt(~occ) * res

    def clr(xs, ys):
        px = (np.asarray(xs) - origin[0]) / res
        py = (H - 1) - (np.asarray(ys) - origin[1]) / res
        return ndimage.map_coordinates(edt, [py, px], order=1)
    return clr


def main():
    ap = argparse.ArgumentParser(description='Build + validate a racing line for any map.')
    ap.add_argument('map', help='track map .yaml')
    ap.add_argument('--out', default=None, help='output raceline csv (default: racelines/<track>_auto.csv)')
    ap.add_argument('--seed', type=float, nargs=2, default=None,
                    help='world x y drivable seed (default: auto = widest point)')
    ap.add_argument('--a-lat', type=float, default=6.5, help='grip budget m/s^2')
    ap.add_argument('--v-max', type=float, default=7.0, help='top speed m/s')
    ap.add_argument('--margin', type=float, default=0.20, help='wall safety margin m')
    ap.add_argument('--no-validate', action='store_true', help='skip closed-loop check')
    args = ap.parse_args()

    track = os.path.splitext(os.path.basename(args.map))[0]
    out = args.out or os.path.join(REPO, 'racelines', f'{track}_auto.csv')

    res, origin, img, occ, H, W = load_map(args.map)
    print(f'[map] {track}: {W}x{H} @ {res:.4f} m/px')

    if args.seed:
        sx, sy = args.seed
        print(f'[seed] user seed ({sx:.2f}, {sy:.2f})')
    else:
        sx, sy, clr = auto_seed(occ, res, origin, H)
        print(f'[seed] auto (widest point): ({sx:.2f}, {sy:.2f}), clearance {clr:.2f} m')

    # ── minimum-curvature optimizer (the tested CLI) ────────────────────────
    cmd = [sys.executable, os.path.join(REPO, 'f1tenth_gym_ros', 'raceline_optimizer.py'),
           '--map', os.path.abspath(args.map), '--output', out,
           '--seed', str(sx), str(sy),
           '--a-lat', str(args.a_lat), '--v-max', str(args.v_max),
           '--margin', str(args.margin)]
    print(f'[optimize] {" ".join(cmd[1:])}')
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0 or not os.path.exists(out):
        sys.stderr.write(r.stderr)
        print(f'[FAIL] optimizer did not produce {out} '
              f'(seed may be off-track — pass --seed x y on a drivable point)')
        sys.exit(1)
    print(f'[write] {out}')

    # ── feasibility validation in the closed-loop MPC benchmark ─────────────
    if args.no_validate:
        return
    from closed_loop import load_raceline, run_lap
    try:
        from mpc_controller import KinematicMPC
    except Exception:
        print('[validate] osqp/scipy unavailable — skipping closed-loop check')
        return

    rx, ry, rh, rc, rv = load_raceline(out)
    n = len(rx)
    length = float(np.sum(np.hypot(np.diff(rx, append=rx[0]),
                                   np.diff(ry, append=ry[0]))))
    planned_alat = float(np.max(np.abs(rc) * rv ** 2))      # curvature * v^2
    clr = clearance_fn(occ, res, origin, H)
    cmin = float(clr(rx, ry).min())

    mpc = KinematicMPC(wheelbase=0.33, horizon=12, dt=0.08, v_max=float(rv.max()) + 0.5)
    mpc.set_raceline(rx, ry, rh, rc, rv)

    def ctrl(px, py, yaw, v, j):
        o = mpc.solve((px, py, yaw, v), j)
        return o if o is not None else (0.0, max(v, 1.0))

    res_lap = run_lap(ctrl, rx, ry, rh)
    ok = (res_lap['completed'] and res_lap['xte_max'] < 0.5
          and planned_alat <= args.a_lat + 1e-6 and cmin >= args.margin - 0.05)
    print('\n========== FEASIBILITY REPORT ==========')
    print(f'  track          {track}  ({n} pts, {length:.1f} m)')
    print(f'  wall clearance min {cmin:.3f} m  (margin {args.margin} m)')
    print(f'  planned a_lat  {planned_alat:.2f} / {args.a_lat} m/s^2 budget')
    print(f'  closed-loop    completed={res_lap["completed"]}  '
          f'lap {res_lap["lap_time"]:.1f} s  '
          f'xte {res_lap["xte_mean"]:.3f}/{res_lap["xte_max"]:.3f} m')
    print(f'  VERDICT        {"FEASIBLE ✓" if ok else "REVIEW ✗ (see flags above)"}')
    print('========================================')
    print(f'\nUse it:  ros2 run f1tenth_gym_ros raceline_mpc --ros-args '
          f'-p raceline:={out} -p v_scale:=0.4   (raise v_scale per clean lap)')


if __name__ == '__main__':
    main()
