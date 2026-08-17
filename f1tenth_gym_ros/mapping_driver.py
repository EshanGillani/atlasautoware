"""
Mapping driver — safe, smooth autonomous laps for building a map with SLAM.
===========================================================================

On a new track you don't have a map for, you get practice time to make one:
run `slam_toolbox` (see launch/slam_mapping_launch.py) and drive a few clean
laps while it builds the occupancy grid.  This node drives those laps for you,
conservatively, so you don't have to hand-pilot.

It uses the **disparity extender** follow-the-gap algorithm (robust on closed
circuits — far more reliable than naive gap-finding): at every big range step it
"extends" the nearer obstacle by the car's half-width so the car never clips an
edge, then aims at the deepest remaining gap.  On top of that it adds a mild
**center-biasing** term (balance left/right clearance so the lidar sweeps both
walls evenly -> full coverage in fewer laps) and a **steering low-pass** filter
(a smooth path -> cleaner SLAM scans).  Speed is held low and scaled by forward
clearance — the goal is a clean, complete map, not a fast lap.

    python3 f1tenth_gym_ros/mapping_driver.py --speed 1.5

The control law lives in the pure, ROS-free function ``gap_follow_command`` so
it can be unit-tested and benchmarked in a raycast simulator (see
tests/test_mapping_driver.py).  The node is a thin wrapper that keeps the
``prev_steer`` state and forwards parameters.

Topics, steering limit and speed are ROS parameters so the same node drives the
sim and the real car.  Hardware (f1tenth_system) uses /scan and /drive too, so
the defaults work as-is; override if your stack remaps them, e.g.
    ros2 run f1tenth_gym_ros mapping_driver --ros-args -p drive_topic:=/nav/drive
"""

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass
class GapParams:
    """Tuning for the pure follow-the-gap control law.

    The first four mirror the original node's attributes exactly; the last
    three are the new map-quality terms and default to a no-op-safe behaviour
    that still gap-follows through tight sections.
    """

    max_steer: float = 0.41      # rad, steering limit
    speed: float = 1.5           # m/s, nominal mapping-lap speed
    car_half: float = 0.20       # m, inflate obstacles by this (disparity span)
    disparity: float = 0.30      # m, range step that counts as an edge
    max_range: float = 6.0       # m, clip / fill for inf beams
    center_gain: float = 0.60    # rad per unit (L-R)/(L+R) wall-clearance imbalance
    steer_lp: float = 0.55       # steering low-pass factor in [0, 1)


def gap_follow_command(ranges, angle_min, angle_inc, prev_steer, params):
    """Disparity-extender follow-the-gap, vectorized + smoothed + center-biased.

    Pure (no ROS, no I/O, deterministic).  Given one lidar scan it returns the
    ``(steer, speed)`` command and keeps no state — the caller passes the
    previous steering angle in for the low-pass filter.

    Parameters
    ----------
    ranges : array_like
        Raw beam ranges (metres); non-finite values are filled with max_range.
    angle_min, angle_inc : float
        Scan angle of beam 0 and the per-beam angular increment (rad).
    prev_steer : float
        Last commanded steering angle (rad), for the low-pass filter.
    params : GapParams
        Tuning (see above).

    Returns
    -------
    (steer, speed) : (float, float)
    """
    p = params
    r = np.asarray(ranges, np.float32)
    r = np.where(np.isfinite(r), r, p.max_range)
    r = np.clip(r, 0.0, p.max_range)
    n = len(r)
    if n == 0:
        return 0.0, max(p.speed * 0.3, 0.5)
    ang = angle_min + np.arange(n) * angle_inc

    # ── Vectorized disparity extender ───────────────────────────────────────
    # At each big range step, extend the nearer side by the number of beams
    # spanning the car's half-width at that range.  The original code mutated
    # `proc` in place with np.minimum over (possibly overlapping) windows; min
    # is commutative+associative, so the result is order-independent and equals
    # the per-index minimum over the original range and every covering window.
    #
    # We turn every step into one "window" (start index, fill value, length):
    #   - forward extension  (r[i] < r[i-1]): proc[i : i+span] = min(proc, near)
    #   - backward extension (r[i-1] < r[i]): proc[i-span : i] = min(proc, near)
    # and apply them all at once in _apply_windows (no per-beam Python loop).
    proc = r.copy()
    diff = r[1:] - r[:-1]                         # r[i] - r[i-1] for i=1..n-1
    big = np.abs(diff) > p.disparity
    idx = np.nonzero(big)[0] + 1                  # the i's with a disparity
    if idx.size:
        near = np.minimum(r[idx], r[idx - 1]).astype(np.float64)
        span = (np.arctan2(p.car_half, np.maximum(near, 0.1)) / angle_inc
                ).astype(np.int64)
        forward = r[idx] < r[idx - 1]             # near side is at/after i

        # Two seed arrays: at the *start* index of each window we drop the
        # window's value and length; overlapping windows take the elementwise
        # min/max.  A bounded shift-and-min over offsets then realises every
        # window exactly (min is associative, so order is irrelevant).
        #
        #   forward window:  proc[i : i+span]      start = i,        len = span
        #   backward window: proc[i-span : i]      start = i-span,   len = span
        fwd_start = idx[forward]
        fwd_val = near[forward]
        fwd_len = np.minimum(span[forward], n - fwd_start)   # clamp to n, like min(n, i+span)
        bwd_start = np.maximum(idx[~forward] - span[~forward], 0)
        bwd_val = near[~forward]
        # length clamped so the clamped start still ends exactly at i.
        bwd_len = idx[~forward] - bwd_start

        starts = np.concatenate([fwd_start, bwd_start])
        vals = np.concatenate([fwd_val, bwd_val])
        lens = np.concatenate([fwd_len, bwd_len])
        keep = lens > 0
        starts, vals, lens = starts[keep], vals[keep], lens[keep]
        if starts.size:
            proc = _apply_windows(proc, starts, vals, lens, n)

    # Consider only the forward 180° (ignore beams pointing backwards).
    fwd = np.abs(ang) < (np.pi / 2)
    proc_fwd = np.where(fwd, proc, 0.0)
    target = int(np.argmax(proc_fwd))             # deepest forward gap
    gap_steer = float(ang[target])

    # ── Center bias: balance left vs right wall clearance so the car tracks
    # the track centerline and the lidar sweeps both walls evenly -> full map
    # coverage in fewer laps.  The signal is the *nearest wall* on each side in
    # the lateral band (|angle| in [30°, 90°]); using the nearest beam (not the
    # mean) makes it the true "how close is the wall on my left vs my right"
    # measure, immune to a deep gap straight ahead swamping it.  It also reads
    # the raw ranges r (real walls), not the disparity-inflated proc, so it
    # never reacts to the gap-following inflation. ───────────────────────────
    side = (np.abs(ang) >= np.pi / 6.0) & (np.abs(ang) <= np.pi / 2.0)
    left_band = side & (ang > 0.0)
    right_band = side & (ang < 0.0)
    if left_band.any() and right_band.any():
        lc = float(r[left_band].min())            # nearest left wall
        rc = float(r[right_band].min())           # nearest right wall
        denom = lc + rc
        # closer to the right wall (rc < lc) -> imbalance > 0 -> steer left.
        imbalance = (lc - rc) / denom if denom > 1e-6 else 0.0
        center_steer = p.center_gain * imbalance
    else:
        center_steer = 0.0

    raw_steer = gap_steer + center_steer
    raw_steer = float(np.clip(raw_steer, -p.max_steer, p.max_steer))

    # ── Steering low-pass: smooth the command toward a clean path.  Bounds the
    # steering rate, which makes SLAM scans cleaner. ─────────────────────────
    a = float(np.clip(p.steer_lp, 0.0, 0.999))
    steer = a * float(prev_steer) + (1.0 - a) * raw_steer
    steer = float(np.clip(steer, -p.max_steer, p.max_steer))

    # ── Speed: low, scaled by forward clearance. ─────────────────────────────
    front = proc[np.abs(ang) < 0.26]
    clear = float(front.min()) if front.size else p.max_range
    spd = p.speed * float(np.clip((clear - 0.5) / 2.0, 0.3, 1.0))
    return steer, max(spd, 0.5)


def _apply_windows(proc, starts, vals, lens, n):
    """proc[s : s+L] = min(proc, v) for every (start s, value v, length L).

    Fully vectorized (no per-beam Python loop): for each window we drop its
    value at its start index and propagate it forward only as far as its length
    via a bounded shift-and-min over offsets ``d = 0 .. max_len-1`` — a window
    at ``s`` with length ``L`` covers index ``s + d`` iff ``d < L``.  Because
    ``min`` is associative and commutative, accumulating all windows this way
    reproduces the original sequential ``np.minimum`` writes exactly, regardless
    of overlap or processing order.  The offset loop runs ``max_len`` times
    (bounded by the car-half-width beam span, ~hundreds at most), each iteration
    a single O(n) numpy op — independent of the number of beams or disparities.
    """
    out = proc.astype(np.float64, copy=True)
    max_len = int(lens.max())
    starts = starts.astype(np.int64)
    lens = lens.astype(np.int64)
    for d in range(max_len):
        # every window still covering offset d (i.e. lens > d) writes its value
        # to index start+d, taking the min — scatter handles duplicate targets.
        active = lens > d
        if not active.any():
            continue
        tgt = starts[active] + d                   # always < n by construction
        np.minimum.at(out, tgt, vals[active])
    return out.astype(np.float32)


# ── ROS node — thin wrapper around the pure control law ─────────────────────
def _build_node():
    """Import ROS lazily so the pure function is usable without a ROS install."""
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
    from ackermann_msgs.msg import AckermannDriveStamped

    class MappingDriver(Node):
        def __init__(self, speed=1.5, car_half=0.20, disparity=0.30, max_range=6.0):
            super().__init__('mapping_driver')
            # ROS params (CLI defaults below are overridden by any -p value).
            self.declare_parameter('scan_topic', '/scan')
            self.declare_parameter('drive_topic', '/drive')
            self.declare_parameter('max_steer', 0.41)        # rad, steering limit
            self.declare_parameter('speed', speed)           # m/s, mapping-lap speed
            scan_topic  = self.get_parameter('scan_topic').value
            drive_topic = self.get_parameter('drive_topic').value
            self.params = GapParams(
                max_steer=float(self.get_parameter('max_steer').value),
                speed=float(self.get_parameter('speed').value),
                car_half=car_half,
                disparity=disparity,
                max_range=max_range,
            )
            self.prev_steer = 0.0                            # low-pass state
            self.create_subscription(LaserScan, scan_topic, self._scan, 10)
            self.pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
            self.get_logger().info(
                f'mapping driver @ {self.params.speed} m/s on {scan_topic} '
                f'-> {drive_topic} — drive clean laps for SLAM')

        def _scan(self, msg):
            steer, spd = gap_follow_command(
                msg.ranges, msg.angle_min, msg.angle_increment,
                self.prev_steer, self.params)
            self.prev_steer = steer
            m = AckermannDriveStamped()
            m.drive.steering_angle = steer
            m.drive.speed = spd
            self.pub.publish(m)

    return rclpy, MappingDriver


def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed', type=float, default=1.5)
    a, _ = ap.parse_known_args()
    rclpy, MappingDriver = _build_node()
    rclpy.init(args=args)
    try:
        rclpy.spin(MappingDriver(speed=a.speed))
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
