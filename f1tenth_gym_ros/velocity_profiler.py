"""
Friction-limited velocity profiler — forward-backward pass (pure, no ROS).
==========================================================================

Replaces hand-tuned raceline speed columns with the profile every serious
racing stack uses (TUMFTM / ForzaETH): the fastest speed at each point that
(a) stays inside the lateral grip budget, (b) is reachable under the engine's
acceleration limit, and (c) leaves enough distance to brake for everything
ahead — with longitudinal and lateral demands coupled through the friction
ellipse, so the car doesn't ask for full brakes mid-corner.

  pass 0   v = min( sqrt(a_lat_max / |kappa|), v_max )          lateral limit
  pass 1   forward:  v[i+1] <= sqrt(v[i]^2 + 2*ax_avail*ds)     acceleration
  pass 2   backward: v[i-1] <= sqrt(v[i]^2 + 2*ax_avail*ds)     braking
  where    ax_avail = ax_max * (1 - (a_lat_used/a_lat_max)^p)^(1/p)

Closed tracks run each pass twice around so the start/finish constraint
propagates across the line (TUM's "two-lap" trick).  O(N); cheap enough to
re-run live when the traction governor revises the grip estimate.

Reference: Heilmeier et al., "Minimum curvature trajectory planning and
control for an autonomous race car", Vehicle System Dynamics 2020;
github.com/TUMFTM/trajectory_planning_helpers (calc_vel_profile).
"""

import math

import numpy as np


def velocity_profile(kappa, ds, a_lat_max=6.0, a_accel_max=4.0,
                     a_brake_max=8.0, v_max=8.0, v_min=0.0, p=2.0,
                     closed=True):
    """Curvature kappa[i] + segment lengths ds[i] (i -> i+1) -> speeds (m/s)."""
    kappa = np.asarray(kappa, float)
    ds = np.asarray(ds, float)
    n = len(kappa)
    radii = 1.0 / np.maximum(np.abs(kappa), 1e-6)

    # pass 0 — quasi-steady-state lateral limit
    v = np.minimum(np.sqrt(a_lat_max * radii), v_max)

    def ax_avail(v_i, r_i, ax_max):
        """Friction-ellipse share left for longitudinal accel at this speed."""
        a_lat_used = min(v_i * v_i / r_i, a_lat_max)
        radicand = max(0.0, 1.0 - (a_lat_used / a_lat_max) ** p)
        return ax_max * radicand ** (1.0 / p)

    laps = 2 if closed else 1
    # pass 1 — forward, acceleration-limited
    for _ in range(laps):
        for i in range(n if closed else n - 1):
            j = (i + 1) % n
            reachable = math.sqrt(
                v[i] ** 2 + 2.0 * ax_avail(v[i], radii[i], a_accel_max) * ds[i])
            if reachable < v[j]:
                v[j] = reachable
    # pass 2 — backward, braking-limited
    for _ in range(laps):
        for i in range(n - 1, -1 if closed else 0, -1):
            j = (i - 1) % n
            reachable = math.sqrt(
                v[i] ** 2 + 2.0 * ax_avail(v[i], radii[i], a_brake_max) * ds[j])
            if reachable < v[j]:
                v[j] = reachable
    return np.maximum(v, v_min)


def segment_lengths(x, y, closed=True):
    """ds[i] = distance from point i to i+1 (wrapping if closed)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if closed:
        return np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))
    return np.hypot(np.diff(x), np.diff(y))
