"""
Lidar scan de-skew — per-beam motion correction from the EKF twist.
===================================================================

A 10 Hz mechanically-spinning 2D lidar sweeps for ~100 ms; at 7 m/s the car
translates 0.7 m and, in a tight corner, rotates >10 degrees *within one
scan*.  The resulting warp biases scan matching and shifts opponent-cluster
centroids by decimetres — far above the sensor's range noise.  The fix is
microseconds of numpy: rotate/translate every beam endpoint by the body
twist times that beam's age, so the whole cloud corresponds to one instant
(the scan-end timestamp, matching the message stamp convention).

  beam i at bearing phi_i, range r_i, age s_i = (i/N - 1) * scan_time
  theta_i = omega * s_i ;  p_i = (vx, vy) * s_i           (constant twist)
  [xc, yc]_i = R(theta_i) @ [r_i cos phi_i, r_i sin phi_i] + p_i

Effect is near-zero on straights and dominant in fast corners; apply it
between the driver and anything consuming geometry (particle filter,
opponent clustering).  Pure function; the rplidar node wires it to the
velocity EKF's twist when `deskew_odom_topic` is set.

References: piecewise-linear lidar de-skewing (arXiv:2108.06078,
arXiv:2303.07312); ForzaETH race stack perception (arXiv:2403.11784).
"""

import numpy as np


def deskew_points(ranges, angle_min, angle_increment, scan_time,
                  vx, vy, omega, ages=None):
    """Skewed LaserScan -> (x, y) cloud at scan-END time, sensor frame.

    `ages` (s, <= 0) gives each beam's measurement time relative to the
    scan-end stamp; default assumes array order = firing order,
    s_i = (i/N - 1) * scan_time.  Pass explicit ages when the publisher
    re-binned beams (e.g. rplidar_node's clockwise-to-CCW grid).  Constant
    body twist (vx, vy, omega) over the sweep; invalid ranges
    (inf/nan/<=0) come back as nan points.
    """
    r = np.asarray(ranges, float)
    n = len(r)
    phi = angle_min + np.arange(n) * angle_increment
    if ages is None:
        s = (np.arange(n) / float(n) - 1.0) * float(scan_time)   # age <= 0
    else:
        s = np.asarray(ages, float)
    theta = omega * s
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    xc = ct * x - st * y + vx * s
    yc = st * x + ct * y + vy * s
    bad = ~np.isfinite(r) | (r <= 0.0)
    xc[bad] = np.nan
    yc[bad] = np.nan
    return xc, yc


def deskew_ranges(ranges, angle_min, angle_increment, scan_time,
                  vx, vy, omega, ages=None):
    """De-skewed scan re-binned onto the original angular grid.

    Drop-in for consumers that want a LaserScan rather than a cloud: corrects
    every beam, then re-bins by corrected bearing (nearest return wins, empty
    bins inf — same convention as rplidar_node.bin_scan).
    """
    n = len(ranges)
    xc, yc = deskew_points(ranges, angle_min, angle_increment, scan_time,
                           vx, vy, omega, ages=ages)
    ok = np.isfinite(xc)
    out = np.full(n, np.inf, np.float32)
    if not ok.any():
        return out
    r_new = np.hypot(xc[ok], yc[ok])
    phi_new = np.arctan2(yc[ok], xc[ok])
    idx = np.round((phi_new - angle_min) / angle_increment).astype(np.intp) % n
    np.minimum.at(out, idx, r_new.astype(np.float32))
    return out
