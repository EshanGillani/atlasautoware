"""
Benchmark the Navigator (grid planner + open-path follower), no ROS.
====================================================================

On BOTH real maps (comp_track, levine), pick 5 challenging start/goal pairs
programmatically (fixed seed): far apart in straight-line distance AND with no
line of sight (so every run goes around corners), sampled from the largest
connected component of the planner's inflated drivable space.

Each pair: plan (timed), then drive the path closed-loop with the kinematic
bicycle at 50 Hz (navigator.drive_to_goal — the open-path twin of
tests/closed_loop.run_lap).  Reported per run:

  straight   straight-line start->goal distance (m)
  path       planned path length (m)  [ratio = path/straight]
  plan_ms    planning time (budget: < 1000 ms)
  t_goal     sim time to reach + stop (s)
  min_clr    min distance_to_wall along the DRIVEN trace (budget: > 0.15 m)
  ok         reached within 0.4 m of the goal and stopped

    python3 tools/benchmark_navigator.py
"""

import json
import math
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))
from grid_map import GridMap                                    # noqa: E402
from grid_planner import (GridPlanner, load_drivable,           # noqa: E402
                          segment_clear, DEFAULT_INFLATION)
from navigator import PathFollower, drive_to_goal               # noqa: E402

SEED = 42
N_PAIRS = 5
SUCCESS_RADIUS = 0.4
MIN_CLEAR_BUDGET = 0.15
PLAN_BUDGET_MS = 1000.0


def pick_pairs(gm, planner, rng, n_pairs=N_PAIRS, n_candidates=400):
    """Challenging start/goal pairs: drawn from the largest connected
    component of drivable space, >= 0.5 m wall clearance, ranked by
    straight-line distance, keeping only pairs WITHOUT line of sight
    (i.e. the route must turn corners), with endpoints spread apart."""
    from scipy.ndimage import label
    lab, _ = label(planner.free, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    comp = lab == int(np.argmax(sizes))
    open_enough = comp & (gm.distance_field() > 0.5)
    cells = np.argwhere(open_enough)
    cells = cells[rng.choice(len(cells), min(n_candidates, len(cells)),
                             replace=False)]
    xs, ys = gm.grid_to_world(cells[:, 0], cells[:, 1])
    pts = np.column_stack([xs, ys])
    d = np.hypot(pts[:, 0, None] - pts[None, :, 0],
                 pts[:, 1, None] - pts[None, :, 1])
    order = np.argsort(d, axis=None)[::-1]
    pairs, used = [], []
    for flat in order:
        i, j = np.unravel_index(flat, d.shape)
        if i >= j or d[i, j] < 5.0:
            continue
        a, b = pts[i], pts[j]
        if any(min(np.hypot(*(a - u)), np.hypot(*(b - u))) < 3.0
               for u in used):
            continue                       # spread the endpoints around
        if segment_clear(gm, a[0], a[1], b[0], b[1], planner.inflation,
                         planner.drivable):
            continue                       # line of sight — too easy
        pairs.append((tuple(a), tuple(b)))
        used += [a, b]
        if len(pairs) == n_pairs:
            break
    return pairs


def run_map(yaml_path, rng):
    name = os.path.splitext(os.path.basename(yaml_path))[0]
    gm = GridMap.load(yaml_path)
    t0 = time.perf_counter()
    planner = GridPlanner(gm, DEFAULT_INFLATION,
                          drivable=load_drivable(yaml_path))
    init_s = time.perf_counter() - t0
    pairs = pick_pairs(gm, planner, rng)
    print(f'\n== {name}  ({gm.h}x{gm.w} @ {gm.res:.3f} m, '
          f'{planner.n_nodes} drivable cells, planner init {init_s:.2f} s, '
          f'inflation {planner.inflation:.2f} m) ==')
    hdr = (f'{"pair":<4}{"start":>16}{"goal":>16}{"straight":>9}'
           f'{"path":>7}{"ratio":>6}{"plan_ms":>8}{"t_goal":>7}'
           f'{"min_clr":>8}{"ok":>4}')
    print(hdr)
    rows = []
    for k, (a, b) in enumerate(pairs):
        t0 = time.perf_counter()
        res = planner.plan(a, b)
        plan_ms = (time.perf_counter() - t0) * 1e3
        straight = math.hypot(b[0] - a[0], b[1] - a[1])
        if res is None:
            print(f'{k:<4}{str(np.round(a,1)):>16}{str(np.round(b,1)):>16}'
                  f'{straight:>9.1f}{"-":>7}{"-":>6}{plan_ms:>8.0f}'
                  f'{"-":>7}{"-":>8}{"NO":>4}')
            rows.append(dict(ok=False, plan_ms=plan_ms))
            continue
        fol = PathFollower(goal_tol=0.3)
        fol.set_path(res['x'], res['y'], res['v'], kappa=res['kappa'])
        yaw0 = math.atan2(res['y'][1] - res['y'][0],
                          res['x'][1] - res['x'][0])
        out = drive_to_goal(fol, res['x'][0], res['y'][0], yaw0,
                            t_max=30.0 + res['length'] / 0.4)
        min_clr = float(gm.distance_to_wall(out['x'], out['y']).min())
        ok = (out['reached'] and out['final_dist'] <= SUCCESS_RADIUS
              and out['final_speed'] < 0.05 and min_clr > MIN_CLEAR_BUDGET
              and plan_ms < PLAN_BUDGET_MS)
        rows.append(dict(straight=straight, path=res['length'],
                         ratio=res['length'] / straight, plan_ms=plan_ms,
                         t_goal=out['t'], min_clear=min_clr, ok=bool(ok)))
        print(f'{k:<4}{str(np.round(a,1)):>16}{str(np.round(b,1)):>16}'
              f'{straight:>9.1f}{res["length"]:>7.1f}'
              f'{res["length"]/straight:>6.2f}{plan_ms:>8.0f}'
              f'{out["t"]:>7.1f}{min_clr:>8.3f}'
              f'{"yes" if ok else "NO":>4}')
    done = [r for r in rows if 'min_clear' in r]
    summary = dict(
        map=name, pairs=len(rows),
        success_rate=sum(r['ok'] for r in rows) / max(len(rows), 1),
        plan_ms_max=max(r['plan_ms'] for r in rows),
        min_clear=min((r['min_clear'] for r in done), default=None),
        ratio_mean=float(np.mean([r['ratio'] for r in done]))
        if done else None)
    print('summary:', json.dumps(summary))
    return summary


def main():
    rng = np.random.default_rng(SEED)
    out = [run_map(os.path.join(REPO, 'maps', m), rng)
           for m in ('comp_track.yaml', 'levine.yaml')]
    ok = all(s['success_rate'] == 1.0 for s in out)
    print('\nOVERALL:', 'PASS' if ok else 'FAIL',
          json.dumps(out))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
