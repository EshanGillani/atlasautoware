"""
MAP controller — Model- and Acceleration-based Pursuit (pure, no ROS).
======================================================================

The geometric-fallback upgrade from the winning F1TENTH stacks: keep pure
pursuit's lookahead-point construction, but turn the lookahead angle into a
*desired lateral acceleration* (L1 guidance) and invert a tire-aware
steady-state map a_lat(steer, v) -> steer, instead of pure pursuit's purely
kinematic triangle.  This compensates the tire slip that makes plain pure
pursuit run wide exactly when it matters (high lateral g), at microsecond
cost — near-MPC tracking with none of the solver risk.

    L_base = clip(q_la + m_la * v_target, la_min, la_max)     # lookahead
    L      = clip(L_base / (1 + k_curv * kappa_ahead), la_min, la_max)
             # kappa_ahead = mean |curvature| over [s_near, s_near + L_base];
             # only when set_raceline() got a curvature array and k_curv > 0,
             # otherwise L = L_base (the original speed-only schedule).
    eta    = asin( dot([-sin(yaw), cos(yaw)], p_la - p_car) / |p_la - p_car| )
    a_des  = 2 * v_target^2 * sin(eta) / L
    steer  = LUT^-1(a_des, v)        (kinematic atan(L_wb*a/v^2) at low speed)

The LUT is generated once at startup by integrating a dynamic single-track
model (linear tire with friction saturation — the f1tenth_gym formulation,
with the gym's default parameters) to steady state over a steer x speed grid,
recording a_lat_ss = yaw_rate * v.  Cells that never converge are beyond the
grip limit; the inversion clamps there, which IS the grip-limit steering.

Reference: Becker et al., "Model- and Acceleration-based Pursuit Controller
for High-Performance Autonomous Racing", ICRA 2023 (arXiv:2209.04346);
github.com/ETH-PBL/MAP-Controller (ForzaETH race stack).
"""

import math

import numpy as np


def build_lat_accel_lut(wheelbase=0.33, max_steer=0.41,
                        mass=3.74, inertia_z=0.04712,
                        l_f=0.15875, l_r=0.17145,
                        mu=1.0489, c_sf=4.718, c_sr=5.4562,
                        n_steer=40, n_speed=35,
                        v_min=0.5, v_max=8.0,
                        dt=2e-3, t_end=3.0, conv_tol=0.05):
    """Steady-state lateral acceleration map over a (steer, speed) grid.

    Vectorized over the whole grid: forward-Euler integration of the dynamic
    single-track (states v_y, yaw rate r; v_x held), linear tire saturated at
    mu*F_z.  Defaults are the f1tenth_gym single-track parameters, so sim and
    LUT share one vehicle model.  Returns (steers, speeds, a_lat) with
    a_lat[i, j] for steers[i] x speeds[j], NaN where no steady state exists.
    """
    g = 9.81
    steers = np.linspace(0.0, max_steer, n_steer)
    speeds = np.linspace(v_min, v_max, n_speed)
    delta, vx = np.meshgrid(steers, speeds, indexing='ij')
    f_zf = mass * g * l_r / (l_f + l_r)
    f_zr = mass * g * l_f / (l_f + l_r)
    vy = np.zeros_like(vx)
    r = np.zeros_like(vx)
    r_prev = np.zeros_like(vx)
    steps = int(round(t_end / dt))
    check_at = steps - int(round(0.25 / dt))      # compare r over the last 250 ms
    for k in range(steps):
        alpha_f = delta - np.arctan2(vy + l_f * r, vx)
        alpha_r = -np.arctan2(vy - l_r * r, vx)
        f_yf = np.clip(mu * f_zf * c_sf * alpha_f, -mu * f_zf, mu * f_zf)
        f_yr = np.clip(mu * f_zr * c_sr * alpha_r, -mu * f_zr, mu * f_zr)
        vy += dt * ((f_yf * np.cos(delta) + f_yr) / mass - vx * r)
        r += dt * (l_f * f_yf * np.cos(delta) - l_r * f_yr) / inertia_z
        np.clip(vy, -30.0, 30.0, out=vy)          # keep diverging cells finite
        np.clip(r, -30.0, 30.0, out=r)
        if k == check_at:
            r_prev = r.copy()
    a_lat = r * vx                                # steady-state: a_y = r * v_x
    a_lat[np.abs(r - r_prev) > conv_tol] = np.nan
    return steers, speeds, a_lat


class MAPController:
    def __init__(self, wheelbase=0.33, max_steer=0.41,
                 m_la=0.3, q_la=0.15, la_min=0.3, la_max=5.0,
                 k_curv=2.0, lut=None, **lut_kwargs):
        self.L_wb = float(wheelbase)
        self.max_steer = float(max_steer)
        self.m_la, self.q_la = float(m_la), float(q_la)
        self.la_min, self.la_max = float(la_min), float(la_max)
        # curvature-aware lookahead: L = L_base / (1 + k_curv * kappa_ahead).
        # k_curv = 0 (or no curvature passed to set_raceline) reproduces the
        # plain speed-scheduled lookahead exactly.
        self.k_curv = float(k_curv)
        self.last_lookahead = None
        if lut is None:
            lut = build_lat_accel_lut(wheelbase=wheelbase,
                                      max_steer=max_steer, **lut_kwargs)
        self.lut_steer, self.lut_speed, self.lut_alat = lut
        self._rl = None

    # ── raceline ────────────────────────────────────────────────────────────
    def set_raceline(self, x, y, speed, curvature=None):
        x = np.asarray(x, float); y = np.asarray(y, float)
        dx = np.diff(x, append=x[0]); dy = np.diff(y, append=y[0])
        seg = np.hypot(dx, dy)
        kabs = kcum = None
        if curvature is not None:
            # prefix integral of |kappa| ds: mean curvature over the arc
            # [s_i, s_j] is then an O(1) lookup per control step.
            kabs = np.abs(np.asarray(curvature, float))
            kcum = np.concatenate([[0.0], np.cumsum(kabs * seg)])
        self._rl = dict(x=x, y=y, v=np.asarray(speed, float),
                        s=np.concatenate([[0.0], np.cumsum(seg)[:-1]]),
                        total=float(seg.sum()), n=len(x),
                        kabs=kabs, kcum=kcum)

    def _curvature_ahead(self, i, L):
        """Mean |curvature| over the raceline arc [s_i, s_i + L] (wraps)."""
        rl = self._rl
        s1 = rl['s'][i] + L
        if s1 < rl['total']:
            j = int(np.searchsorted(rl['s'], s1))
            if j <= i:                            # window inside one segment
                return float(rl['kabs'][i])
            integral = rl['kcum'][j] - rl['kcum'][i]
        else:                                     # wraps past the start line
            j = int(np.searchsorted(rl['s'], s1 - rl['total'])) % rl['n']
            integral = (rl['kcum'][rl['n']] - rl['kcum'][i]) + rl['kcum'][j]
        return float(integral / L)

    def _lookahead(self, v_t, i):
        L = float(np.clip(self.q_la + self.m_la * v_t, self.la_min, self.la_max))
        if self.k_curv > 0.0 and self._rl['kabs'] is not None:
            kappa = self._curvature_ahead(i, L)
            L = float(np.clip(L / (1.0 + self.k_curv * kappa),
                              self.la_min, self.la_max))
        return L

    # ── tire-aware inversion: a_lat desired -> steering ─────────────────────
    def steer_from_lat_accel(self, a_lat, v):
        a = abs(float(a_lat))
        if v < 1.0:                               # LUT and kinematics coincide
            steer = math.atan(self.L_wb * a / max(v, 0.5) ** 2)
            return math.copysign(min(steer, self.max_steer), a_lat)
        j = int(np.argmin(np.abs(self.lut_speed - v)))
        col = self.lut_alat[:, j]
        ok = np.isfinite(col)
        # keep the strictly-increasing (stable) branch of a_lat(steer)
        idx = np.flatnonzero(ok)
        if len(idx) < 2:
            steer = math.atan(self.L_wb * a / v ** 2)
            return math.copysign(min(steer, self.max_steer), a_lat)
        rising = np.maximum.accumulate(col[idx])
        last = int(np.argmax(rising)) + 1
        xs, ys = col[idx[:last]], self.lut_steer[idx[:last]]
        if a >= xs[-1]:
            steer = ys[-1]                        # grip limit — max useful steer
        else:
            steer = float(np.interp(a, xs, ys))
        return math.copysign(min(steer, self.max_steer), a_lat)

    # ── one control step ────────────────────────────────────────────────────
    def control(self, x, y, yaw, v, nearest):
        """Pose + speed + nearest raceline index -> (steer, v_target)."""
        rl = self._rl
        nearest = nearest % rl['n']
        v_t = float(rl['v'][nearest])
        L = self._lookahead(v_t, nearest)
        self.last_lookahead = L
        s_la = (rl['s'][nearest] + L) % rl['total']
        j = int(np.searchsorted(rl['s'], s_la)) % rl['n']
        dx, dy = rl['x'][j] - x, rl['y'][j] - y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return 0.0, v_t
        eta = math.asin(np.clip(
            (-math.sin(yaw) * dx + math.cos(yaw) * dy) / dist, -1.0, 1.0))
        a_des = 2.0 * v_t ** 2 * math.sin(eta) / L      # L1 guidance
        return self.steer_from_lat_accel(a_des, max(v, v_t * 0.5)), v_t
