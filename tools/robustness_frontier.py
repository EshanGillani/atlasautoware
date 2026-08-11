"""
Robustness frontier — how much lap time does each margin of safety cost?
========================================================================

"Faster without sacrificing consistency" is a trade, not a single answer, and
the honest output is the curve rather than one config.  This takes the
candidates a tuning session produced, re-scores each across a fine grid of tyre
friction values, and reports for every one:

    nominal lap time   at the surface you expect
    grip floor         the lowest friction it still completes a lap at

Then it prints the frontier: for each grip floor, the fastest configuration
that survives it.  That is the table to make the call from — if the track is
clean and you trust it, take the quick fragile setup; if it is dusty, or the
tyres have a session on them, or the battery is halfway down, buy the margin
and know exactly what it cost.

Why friction and not starting position: perturbing the start pose barely moves
this stack (the controller converges back to the same line within a corner, and
lap times across perturbed starts vary by ~0.01 s).  Grip is the axis that
actually ends races, and it is invisible to a search that pins mu at one value.

    python3 tools/robustness_frontier.py
    python3 tools/robustness_frontier.py --top 12 --trials 8
    python3 tools/robustness_frontier.py --log runtime/bayes_log_fixedmu.jsonl
"""

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tools'))

from bayes_tune import SPACE, Objective                    # noqa: E402

RUNTIME = os.path.join(REPO, 'runtime')
NAMES = [s[0] for s in SPACE]


def load_candidates(paths):
    """Distinct configs from one or more tuning logs / best-files."""
    out, seen = [], set()
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            if path.endswith('.json'):
                recs = [json.load(f)]
            else:
                recs = []
                for line in f:
                    try:
                        recs.append(json.loads(line))
                    except Exception:
                        pass
        for rec in recs:
            cfg = rec.get('config')
            if not cfg or not all(n in cfg for n in NAMES):
                continue
            key = tuple(round(float(cfg[n]), 4) for n in NAMES)
            if key in seen:
                continue
            seen.add(key)
            out.append((cfg, rec.get('score'), os.path.basename(path)))
    return out


def pareto(rows):
    """Keep only setups nothing else beats on BOTH axes.

    rows are (lap, grip_floor, ...); lower lap is better and lower grip floor
    is better (it survives a worse surface).  A setup that survives only to
    0.85 while another survives to 0.80 *and* laps quicker is never the right
    choice, and listing it invites someone to pick it under pressure.
    """
    out = []
    for r in rows:
        dominated = any(
            o is not r and o[0] <= r[0] and o[1] <= r[1]
            and (o[0] < r[0] or o[1] < r[1])
            for o in rows)
        if not dominated:
            out.append(r)
    return out


def grip_floor(cfg, raceline, mus, trials, threshold, seed):
    """-> (nominal lap, lowest mu completed at >= threshold, per-mu successes).

    Walks DOWN the grid and stops at the first failure: below the point where a
    setup starts losing the car, whether it would have coped further down is not
    a question worth spending simulation on.
    """
    per, floor, nominal = {}, None, None
    for mu in mus:
        obj = Objective(raceline, trials=trials, seed=seed, mu=mu,
                        min_success=0.0)
        _score, info = obj(dict(cfg))
        per[mu] = info['success']
        if nominal is None:
            nominal = info.get('mean_lap')
        if info['success'] >= threshold:
            floor = mu
        else:
            break
    return nominal, floor, per


def main():
    ap = argparse.ArgumentParser(
        description='Lap time vs how bad a surface the setup survives.')
    ap.add_argument('--log', nargs='+',
                    default=[os.path.join(RUNTIME, 'bayes_log.jsonl'),
                             os.path.join(RUNTIME, 'bayes_best_fixedmu.json')])
    ap.add_argument('--raceline',
                    default=os.path.join(REPO, 'racelines', 'comp_raceline.csv'))
    ap.add_argument('--top', type=int, default=10,
                    help='how many of the quickest candidates to examine')
    ap.add_argument('--trials', type=int, default=8,
                    help='perturbed starts per grip level')
    ap.add_argument('--threshold', type=float, default=1.0,
                    help='success rate that counts as "survives"')
    ap.add_argument('--mus', type=float, nargs='+',
                    default=[1.05, 1.00, 0.95, 0.90, 0.85, 0.80, 0.75])
    ap.add_argument('--seed', type=int, default=11,
                    help='starts unseen by the tuner, so this is not a re-read '
                         'of what the search already optimized against')
    args = ap.parse_args()

    cands = load_candidates(args.log)
    if not cands:
        raise SystemExit('no candidates found — run `atlas run bayes-tune` first')

    # Rank by the score the search recorded, then examine the quickest few.
    cands.sort(key=lambda c: (c[1] if c[1] is not None else 1e9))
    cands = cands[:args.top]
    print(f'{len(cands)} candidates, {args.trials} starts at each of '
          f'{len(args.mus)} grip levels (seed {args.seed}, unseen by the tuner)\n')

    mus = sorted(args.mus, reverse=True)
    header = ' '.join(f'{m:>5.2f}' for m in mus)
    print(f"{'#':>3} {'nominal':>8} {'floor':>6}  {header}   source")
    print('-' * (30 + 6 * len(mus)))

    rows = []
    for i, (cfg, _score, src) in enumerate(cands, 1):
        nominal, floor, per = grip_floor(cfg, args.raceline, mus, args.trials,
                                         args.threshold, args.seed)
        cells = ' '.join(
            (f'{per[m]*100:>4.0f}%' if m in per else '    -') for m in mus)
        print(f"{i:>3} {(f'{nominal:.2f}s' if nominal else '   --'):>8} "
              f"{(f'{floor:.2f}' if floor else ' none'):>6}  {cells}   {src}")
        if nominal and floor:
            rows.append((nominal, floor, cfg, i))

    if not rows:
        print('\nNothing completed a lap at the top grip level — the raceline or '
              'the ranges are wrong for this track.')
        return 1

    print('\n' + '=' * 62)
    print('FRONTIER — the quickest setup for each margin of safety\n')
    print(f"  {'survives down to':>17}  {'lap':>8}  {'cost vs quickest':>17}  cfg")
    best_overall = min(r[0] for r in rows)

    frontier = pareto(rows)
    for nominal, floor, cfg, idx in sorted(frontier, key=lambda r: -r[1]):
        print(f'  {f"mu {floor:.2f}":>17}  {nominal:>7.2f}s  '
              f'{nominal - best_overall:>+16.2f}s  #{idx}')
    dominated = len(rows) - len(frontier)
    if dominated:
        print(f'\n  ({dominated} candidate(s) omitted: another setup is both '
              f'more robust and quicker)')

    print('\nReading this: pick the lowest grip you actually want to survive, '
          'then\ntake the fastest row at or below it. The cost column is what '
          'the margin\nis buying you — decide with the track in front of you, '
          'not in advance.')

    best_robust = max(rows, key=lambda r: (r[1], -r[0]))
    out = os.path.join(RUNTIME, 'frontier.json')
    with open(out, 'w') as f:
        json.dump({'frontier': [{'lap': n, 'grip_floor': fl, 'config': c}
                                for n, fl, c, _ in sorted(frontier)],
                   'most_robust': {'lap': best_robust[0],
                                   'grip_floor': best_robust[1],
                                   'config': best_robust[2]}}, f, indent=2)
    print(f'\nwritten to {os.path.relpath(out, REPO)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
