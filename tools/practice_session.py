"""
Practice-session manager — turn track time into a faster raceline.
==================================================================

A competition practice slot is short, noisy and stressful, and the useful work
in it is all bookkeeping: record what the car did, extract the line it actually
drove, re-optimize against the map you just built, check it is feasible, and
write down what changed so you can compare it to the last run.  Doing that by
hand between sessions is how teams lose a whole slot to a mistyped path.

    # at the track, with the car running
    python3 tools/atlas.py run practice -- --name friday-am --stage record
    python3 tools/atlas.py run practice -- --name friday-am --stage build
    # or both at once
    python3 tools/atlas.py run practice -- --name friday-am

    python3 tools/atlas.py run practice -- --list          # what have we got
    python3 tools/atlas.py run practice -- --compare       # every session, ranked

Stages
------
record  Subscribe to /scan, the pose topic and /drive, and log everything to
        practice/<name>/run.jsonl at 20 Hz.  Needs ROS; on a machine without
        it, `--replay` lets you build from a log recorded elsewhere.
build   Offline, no ROS needed.  From the recorded run it:
          - extracts the *driven* line and resamples it evenly,
          - measures the speeds actually achieved and the lateral acceleration
            actually sustained — the real grip of this surface, today,
          - re-profiles the reference raceline against that measured grip,
          - validates the result and writes practice/<name>/report.md.

Why measured grip matters
-------------------------
Every speed in the raceline comes from an assumed `a_lat`. Guess it too high
and the car understeers off at the first fast corner; too low and you leave
seconds on the table. The car has been telling you the real number all along —
v²·κ at every point it drove cleanly. This reads that number off the run and
re-profiles against it, which is the single highest-value thing you can do with
a practice session.

The recorded runs are also the demonstration data for the RL policy: same
format, and `tools/train_rl.py --replay` can seed a replay buffer from them.
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))
sys.path.insert(0, os.path.join(REPO, 'tests'))

from velocity_profiler import velocity_profile, segment_lengths   # noqa: E402

PRACTICE = os.path.join(REPO, 'practice')
G = 9.81

for _s in (sys.stdout, sys.stderr):     # m/s² and friends on a cp1252 console
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


# ── recording ────────────────────────────────────────────────────────────────
def record(name, duration, hz, scan_topic, odom_topic, drive_topic):
    """Log the car's state to practice/<name>/run.jsonl.  Needs ROS."""
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import LaserScan
        from nav_msgs.msg import Odometry
        from ackermann_msgs.msg import AckermannDriveStamped
        from transforms3d.euler import quat2euler
    except Exception as e:
        raise SystemExit(
            f'recording needs ROS 2 ({e}).\n'
            f'Run this on the car (or in the sim container):\n'
            f'  python3 tools/atlas.py run practice -- --name {name} '
            f'--stage record')

    out_dir = os.path.join(PRACTICE, name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'run.jsonl')

    class Recorder(Node):
        def __init__(self):
            super().__init__('practice_recorder')
            self.state = dict(x=0.0, y=0.0, yaw=0.0, v=0.0, yaw_rate=0.0,
                              steer=0.0, v_cmd=0.0)
            self.scan = None
            self.have_odom = False
            self.n = 0
            self.f = open(path, 'w')
            self.create_subscription(LaserScan, scan_topic, self._scan, 10)
            self.create_subscription(Odometry, odom_topic, self._odom, 10)
            self.create_subscription(AckermannDriveStamped, drive_topic,
                                     self._drive, 10)
            self.create_timer(1.0 / hz, self._tick)
            self.t0 = time.time()
            self.get_logger().info(f'recording -> {path} ({duration:.0f}s)')

        def _scan(self, m):
            self.scan = m

        def _odom(self, m):
            s = self.state
            s['x'] = m.pose.pose.position.x
            s['y'] = m.pose.pose.position.y
            s['v'] = float(math.hypot(m.twist.twist.linear.x,
                                      m.twist.twist.linear.y))
            s['yaw_rate'] = float(m.twist.twist.angular.z)
            q = m.pose.pose.orientation
            _, _, s['yaw'] = quat2euler([q.w, q.x, q.y, q.z])
            self.have_odom = True

        def _drive(self, m):
            self.state['steer'] = float(m.drive.steering_angle)
            self.state['v_cmd'] = float(m.drive.speed)

        def _tick(self):
            if not self.have_odom:
                return
            rec = dict(t=round(time.time() - self.t0, 3), **self.state)
            if self.scan is not None:
                # Subsample the scan: a full 1080-beam scan at 20 Hz is ~50 MB
                # a minute, and 108 beams is what the policy consumes anyway.
                r = np.asarray(self.scan.ranges, dtype=np.float32)
                r = np.where(np.isfinite(r), r, 30.0)
                k = max(1, len(r) // 108)
                rec['scan'] = [round(float(x), 3)
                               for x in r[:k * 108].reshape(108, k).min(axis=1)]
            self.f.write(json.dumps(rec) + '\n')
            self.n += 1
            if self.n % (int(hz) * 10) == 0:
                self.get_logger().info(
                    f'{self.n} samples, {time.time() - self.t0:.0f}s')

    rclpy.init()
    node = Recorder()
    try:
        end = time.time() + duration
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.f.close()
        print(f'\nrecorded {node.n} samples -> {path}')
        node.destroy_node()
        rclpy.shutdown()
    return path


# ── analysis ─────────────────────────────────────────────────────────────────
def load_run(path):
    """practice/<name>/run.jsonl -> dict of arrays."""
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        raise SystemExit(f'{path} has no usable samples')
    keys = ('t', 'x', 'y', 'yaw', 'v', 'yaw_rate', 'steer', 'v_cmd')
    return {k: np.array([float(r.get(k, 0.0)) for r in rows]) for k in keys}


def driven_line(run, spacing=0.4, min_speed=0.3):
    """Resample the recorded path to evenly spaced points with curvature.

    Raw samples are unevenly spaced (fast on straights, dense in slow corners),
    and every downstream calculation — curvature, arc length, profiling —
    assumes even spacing. Standing-still samples are dropped first, or they
    collapse into a single point that produces an infinite curvature.
    """
    moving = run['v'] > min_speed
    x, y, v = run['x'][moving], run['y'][moving], run['v'][moving]
    if len(x) < 20:
        raise SystemExit('not enough moving samples — was the car driving?')

    d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    total = float(d[-1])
    if total < 1.0:
        raise SystemExit(f'the car covered only {total:.2f} m')
    n = max(30, int(total / spacing))
    s = np.linspace(0.0, total, n)
    rx = np.interp(s, d, x)
    ry = np.interp(s, d, y)
    rv = np.interp(s, d, v)

    # curvature from finite differences of the resampled path
    dx = np.gradient(rx, s)
    dy = np.gradient(ry, s)
    ddx = np.gradient(dx, s)
    ddy = np.gradient(dy, s)
    denom = np.power(dx * dx + dy * dy, 1.5) + 1e-9
    curv = (dx * ddy - dy * ddx) / denom
    hdg = np.arctan2(dy, dx)
    return dict(x=rx, y=ry, heading=hdg, curvature=curv, speed=rv,
                s=s, length=total)


def measured_grip(line, percentile=95.0):
    """The lateral acceleration this car actually sustained: a_lat = v^2 * |k|.

    Reported at a high percentile rather than the maximum: a single noisy
    curvature spike would otherwise set the grip estimate, and profiling the
    whole lap against one sample is how you generate a raceline that only works
    if nothing goes slightly wrong.
    """
    a_lat = line['speed'] ** 2 * np.abs(line['curvature'])
    a_lat = a_lat[np.isfinite(a_lat)]
    if not len(a_lat):
        return 0.0, 0.0
    return float(np.percentile(a_lat, percentile)), float(a_lat.max())


def write_csv(path, line, speeds=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sp = speeds if speeds is not None else line['speed']
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x', 'y', 'heading', 'curvature', 'speed'])
        for i in range(len(line['x'])):
            w.writerow([f"{line['x'][i]:.4f}", f"{line['y'][i]:.4f}",
                        f"{line['heading'][i]:.4f}", f"{line['curvature'][i]:.5f}",
                        f'{sp[i]:.3f}'])


def build(name, reference, a_lat_override=None, a_accel=4.0, a_brake=8.0,
          v_max=None, margin=0.9, spacing=0.4):
    """Offline stage: driven line -> measured grip -> re-profiled raceline."""
    out_dir = os.path.join(PRACTICE, name)
    run_path = os.path.join(out_dir, 'run.jsonl')
    if not os.path.exists(run_path):
        raise SystemExit(f'no recording at {run_path}\n'
                         f'record one first: atlas run practice -- '
                         f'--name {name} --stage record')

    run = load_run(run_path)
    line = driven_line(run, spacing=spacing)
    p95, peak = measured_grip(line)

    duration = float(run['t'][-1] - run['t'][0])
    v_top = float(run['v'].max())
    v_mean = float(run['v'][run['v'] > 0.3].mean()) if (run['v'] > 0.3).any() else 0.0

    # Grip to profile against: what we measured, backed off by `margin`, since
    # the peak we saw is the edge and racing at the edge everywhere is how you
    # lose the car in the one corner that is dustier than the rest.
    a_lat_measured = p95 * margin
    a_lat = a_lat_override if a_lat_override else a_lat_measured
    v_ceiling = v_max if v_max else max(v_top * 1.1, 2.0)

    write_csv(os.path.join(out_dir, 'driven_line.csv'), line)

    # Re-profile the REFERENCE raceline (the geometry we race) against the grip
    # we just measured — not the driven line, whose geometry is whatever the
    # driver managed on the day.
    result = {}
    if reference and os.path.exists(reference):
        ref = load_reference(reference)
        ds = segment_lengths(ref['x'], ref['y'])
        speeds = velocity_profile(ref['curvature'], ds, a_lat_max=a_lat,
                                  a_accel_max=a_accel, a_brake_max=a_brake,
                                  v_max=v_ceiling)
        out_csv = os.path.join(out_dir, 'reprofiled_raceline.csv')
        write_csv(out_csv, ref, speeds)
        est = float(np.sum(ds / np.maximum(speeds, 0.1)))
        old_est = float(np.sum(ds / np.maximum(ref['speed'], 0.1)))
        # Which limit actually shaped the profile?  If the car never got near
        # its top speed, the ceiling binds and the new profile is slower for a
        # reason that has nothing to do with grip — worth saying out loud, or
        # the team reads a slower estimate as "the tyres got worse".
        lat_limited = np.sqrt(a_lat / np.maximum(np.abs(ref['curvature']), 1e-6))
        cap_binding = float(np.mean(lat_limited > v_ceiling))
        result = dict(reference=reference, out_csv=out_csv,
                      v_min=float(speeds.min()), v_max=float(speeds.max()),
                      est_lap=est, old_est_lap=old_est,
                      old_v_max=float(ref['speed'].max()),
                      cap_binding=cap_binding)

    report = dict(
        name=name, ts=time.time(), duration_s=duration,
        samples=len(run['t']), distance_m=line['length'],
        v_top=v_top, v_mean=v_mean,
        a_lat_p95=p95, a_lat_peak=peak, a_lat_used=a_lat,
        margin=margin, a_accel=a_accel, a_brake=a_brake, v_ceiling=v_ceiling,
        **result)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(report, f, indent=2)
    write_report(out_dir, report)
    return report


def load_reference(path):
    cols = {k: [] for k in ('x', 'y', 'heading', 'curvature', 'speed')}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k in cols:
                cols[k].append(float(row[k]))
    return {k: np.array(v) for k, v in cols.items()}


def write_report(out_dir, r):
    lines = [
        f"# Practice session — {r['name']}",
        '',
        time.strftime('%Y-%m-%d %H:%M', time.localtime(r['ts'])),
        '',
        '## What the car did',
        '',
        f"- ran for **{r['duration_s']:.0f} s**, covering "
        f"**{r['distance_m']:.0f} m** ({r['samples']} samples)",
        f"- top speed **{r['v_top']:.2f} m/s**, average while moving "
        f"**{r['v_mean']:.2f} m/s**",
        '',
        '## Measured grip',
        '',
        f"- sustained lateral acceleration (95th pct): "
        f"**{r['a_lat_p95']:.2f} m/s²**  (~{r['a_lat_p95'] / G:.2f} g)",
        f"- peak seen: {r['a_lat_peak']:.2f} m/s²",
        f"- profiled against **{r['a_lat_used']:.2f} m/s²** "
        f"({r['margin']:.0%} of measured — margin for a dusty corner)",
        '',
        'This is the real grip of this surface today. If it is well below what '
        'the raceline assumed, that gap is why the car was running wide.',
        '',
    ]
    if r.get('out_csv'):
        delta = r['old_est_lap'] - r['est_lap']
        verdict = (f'**{abs(delta):.2f} s faster**' if delta > 0
                   else f'**{abs(delta):.2f} s slower**')
        lines += [
            '## Re-profiled raceline',
            '',
            f"- reference: `{os.path.basename(r['reference'])}`",
            f"- new speeds: {r['v_min']:.1f} – {r['v_max']:.1f} m/s "
            f"(was up to {r['old_v_max']:.1f})",
            f"- estimated lap: **{r['est_lap']:.2f} s** vs "
            f"{r['old_est_lap']:.2f} s before — {verdict}",
            f"- written to `{os.path.relpath(r['out_csv'], REPO)}`",
            '',
        ]
        if r.get('cap_binding', 0) > 0.25:
            lines += [
                f"> **The speed ceiling is binding, not grip.** On "
                f"{r['cap_binding']:.0%} of the lap the tyres would allow more "
                f"speed than the {r['v_ceiling']:.1f} m/s ceiling permits, and "
                f"that ceiling came from the {r['v_top']:.1f} m/s you actually "
                f"reached. If the car had more to give, do a run with a higher "
                f"`v_scale` and rebuild — or set `--v-max` explicitly.",
                '',
            ]
        lines += [
            '## Next',
            '',
            '```bash',
            f"python3 tools/atlas.py run validate -- --raceline "
            f"{os.path.relpath(r['out_csv'], REPO)}",
            '```',
            '',
            'If that is clean, install it as the raceline and raise `v_scale` '
            'one step at a time.',
            '',
        ]
    else:
        lines += ['## Re-profiled raceline', '',
                  'No reference raceline was given, so only the driven line was '
                  'extracted. Pass `--reference racelines/comp_raceline.csv` to '
                  're-profile against the measured grip.', '']
    path = os.path.join(out_dir, 'report.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    return path


# ── session listing ──────────────────────────────────────────────────────────
def list_sessions():
    if not os.path.isdir(PRACTICE):
        print('no practice sessions yet — run one with:\n'
              '  python3 tools/atlas.py run practice -- --name friday-am')
        return []
    out = []
    for name in sorted(os.listdir(PRACTICE)):
        s = os.path.join(PRACTICE, name, 'summary.json')
        if os.path.exists(s):
            try:
                with open(s) as f:
                    out.append(json.load(f))
            except Exception:
                pass
    return out


def compare(sessions):
    if not sessions:
        print('no completed sessions to compare '
              '(record one, then run --stage build)')
        return
    print(f"\n{'session':<20} {'grip p95':>9} {'top v':>7} {'est lap':>9}  "
          f"{'distance':>9}")
    print('-' * 60)
    for s in sorted(sessions, key=lambda r: r.get('est_lap', 1e9)):
        est = s.get('est_lap')
        print(f"{s['name']:<20} {s['a_lat_p95']:>8.2f}  {s['v_top']:>6.2f} "
              f"{(f'{est:.2f}s' if est else '    —'):>9}  "
              f"{s['distance_m']:>8.0f}m")
    best = min((s for s in sessions if s.get('est_lap')),
               key=lambda r: r['est_lap'], default=None)
    if best:
        print(f"\nfastest setup came from '{best['name']}' "
              f"(grip {best['a_lat_p95']:.2f} m/s²) -> "
              f"{os.path.relpath(best['out_csv'], REPO)}")
    print()


def main():
    ap = argparse.ArgumentParser(
        description='Record a practice run and turn it into a faster raceline.')
    ap.add_argument('--name', default=time.strftime('session-%m%d-%H%M'))
    ap.add_argument('--stage', choices=['record', 'build', 'all'], default='all')
    ap.add_argument('--duration', type=float, default=120.0,
                    help='seconds to record')
    ap.add_argument('--hz', type=float, default=20.0, help='recording rate')
    ap.add_argument('--scan-topic', default='/scan')
    ap.add_argument('--odom-topic', default='/pf/pose/odom')
    ap.add_argument('--drive-topic', default='/drive')
    ap.add_argument('--reference',
                    default=os.path.join(REPO, 'racelines', 'comp_raceline.csv'),
                    help='raceline to re-profile against the measured grip')
    ap.add_argument('--a-lat', type=float, default=None,
                    help='override the measured grip (m/s^2)')
    ap.add_argument('--a-accel', type=float, default=4.0)
    ap.add_argument('--a-brake', type=float, default=8.0)
    ap.add_argument('--v-max', type=float, default=None)
    ap.add_argument('--margin', type=float, default=0.9,
                    help='fraction of measured grip to actually profile against')
    ap.add_argument('--spacing', type=float, default=0.4)
    ap.add_argument('--list', action='store_true', help='list sessions')
    ap.add_argument('--compare', action='store_true', help='rank all sessions')
    args = ap.parse_args()

    if args.list or args.compare:
        sessions = list_sessions()
        if args.compare:
            compare(sessions)
        else:
            for s in sessions:
                print(f"  {s['name']:<20} {s['distance_m']:>6.0f} m  "
                      f"grip {s['a_lat_p95']:.2f} m/s²")
            if not sessions:
                print('  (none yet)')
        return 0

    if args.stage in ('record', 'all'):
        record(args.name, args.duration, args.hz, args.scan_topic,
               args.odom_topic, args.drive_topic)
    if args.stage in ('build', 'all'):
        r = build(args.name, args.reference, a_lat_override=args.a_lat,
                  a_accel=args.a_accel, a_brake=args.a_brake, v_max=args.v_max,
                  margin=args.margin, spacing=args.spacing)
        print(f"\nsession '{r['name']}'")
        print(f"  drove          {r['distance_m']:.0f} m in {r['duration_s']:.0f} s")
        print(f"  measured grip  {r['a_lat_p95']:.2f} m/s² "
              f"({r['a_lat_p95'] / G:.2f} g), profiling at {r['a_lat_used']:.2f}")
        if r.get('est_lap'):
            d = r['old_est_lap'] - r['est_lap']
            print(f"  estimated lap  {r['est_lap']:.2f} s vs "
                  f"{r['old_est_lap']:.2f} s before "
                  f"({abs(d):.2f} s {'faster' if d > 0 else 'slower'})")
            if r.get('cap_binding', 0) > 0.25:
                print(f"                 note: the {r['v_ceiling']:.1f} m/s "
                      f"ceiling binds on {r['cap_binding']:.0%} of the lap, "
                      f"not grip — see the report")
            print(f"  raceline       {os.path.relpath(r['out_csv'], REPO)}")
        print(f"  report         "
              f"{os.path.relpath(os.path.join(PRACTICE, r['name'], 'report.md'), REPO)}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
