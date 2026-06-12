"""
Monte-Carlo localization on the frozen occupancy map (SynPF pattern).
=====================================================================

Race-time localization for every map-based feature in the stack: a particle
filter on the GridMap with a *likelihood-field* measurement model — exactly
the ForzaETH/SynPF recipe (arXiv:2401.07658) that wins on the rubber-marked,
glass-walled tracks where scan matching gets brittle.  The motion prior is
the body twist from the velocity EKF (velocity_ekf.py), so wheel-slip gating
happens upstream and the PF only needs to absorb the residual twist error.

  predict   per-particle noisy body twist (vx, vy, omega) integrated over dt
            (mid-yaw integration), plus a small additive diffusion floor so
            the cloud never collapses while standing still
  update    likelihood field: every subsampled beam endpoint of every
            particle is transformed into the map (vectorized N x B outer
            product), the cached distance field gives metres-to-wall at the
            endpoint, scored with a Gaussian(0, sigma_hit) + uniform mixture;
            log-weights are accumulated and normalized stably
  resample  low-variance (systematic), only when ESS < n/2 — keeps diversity
            and preserves weight mass by construction
  pose      weighted mean with a circular mean for yaw, plus ESS/n as a
            weight-spread confidence proxy

Pure numpy core (no rclpy import needed) + a thin ROS node publishing
/pf/pose/odom — the topic raceline_mpc already expects on the real car —
with the EKF twist passed through so downstream consumers get pose + twist
in one message.

    ros2 run f1tenth_gym_ros particle_filter --ros-args \
        -p map_yaml:=maps/comp_track.yaml -p initial_pose:='[49.8, 62.2, -2.64]'

References: SynPF (arXiv:2401.07658); ForzaETH race stack (arXiv:2403.11784);
Thrun, Burgard, Fox, "Probabilistic Robotics", ch. 6.4 + 8.3.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_map import GridMap, map_path_for                       # noqa: E402,F401


class ParticleFilter:
    """MCL on a GridMap.  States (x, y, yaw) per particle, fully vectorized.

    Process-noise knobs (per-particle twist corruption, motion-scaled):
      alpha_v / alpha_w   fractional noise on linear / angular speed
      sigma_v / sigma_w   absolute twist noise floors (m/s, rad/s)
      sigma_xy / sigma_yaw  additive diffusion per sqrt(s) — keeps the cloud
                            alive at standstill and absorbs unmodelled slip
    Measurement-model knobs (likelihood field):
      z_hit, sigma_hit    Gaussian hit component (weight, std in metres)
      z_rand              uniform floor — robustness to dropouts/dynamic crud
    """

    def __init__(self, grid_map, n_particles=1500,
                 alpha_v=0.15, alpha_w=0.15,
                 sigma_v=0.08, sigma_w=0.06,
                 sigma_xy=0.01, sigma_yaw=0.01,
                 z_hit=0.85, sigma_hit=0.2, z_rand=0.10,
                 q_lost=0.85, inject_frac=0.15,
                 max_range=12.0, seed=None):
        self.gm = grid_map
        self.df = grid_map.distance_field()           # cached once, O(1) lookups
        self.n = int(n_particles)
        self.alpha_v, self.alpha_w = float(alpha_v), float(alpha_w)
        self.sigma_v, self.sigma_w = float(sigma_v), float(sigma_w)
        self.sigma_xy, self.sigma_yaw = float(sigma_xy), float(sigma_yaw)
        self.z_hit, self.sigma_hit = float(z_hit), float(sigma_hit)
        self.z_rand = float(z_rand)
        self.q_lost, self.inject_frac = float(q_lost), float(inject_frac)
        self.lost_patience = 3                        # consecutive bad updates
        self.spread_lost = 1.0                        # m — cloud not unimodal
        self.quality = 0.0                            # best-particle beam fit
        self.lost = True                              # recovery mode flag
        self._lost_run = 0
        self._est = None                              # cached (x, y, yaw, conf)
        self.max_range = float(max_range)
        self.rng = np.random.default_rng(seed)
        self._free_rc = np.nonzero(self.df > 0.15)    # sampleable free cells
        # corridor direction per free cell (wall-tangent from the distance-
        # field gradient): injected/global particles get a yaw near the local
        # track direction (either way), not uniform — far better hit rate
        ddr, ddc = np.gradient(self.df)               # d/d(row), d/d(col)
        gx, gy = ddc[self._free_rc], -ddr[self._free_rc]   # world frame
        self._free_tangent = np.arctan2(gy, gx) + 0.5 * math.pi
        self.p = np.zeros((self.n, 3))                # columns: x, y, yaw
        self.w = np.full(self.n, 1.0 / self.n)
        self.initialize_global()

    def _sample_free(self, k):
        """k poses uniform over free space; yaw ~ local corridor direction
        (random sign) + noise, so injected hypotheses point along the track."""
        rows, cols = self._free_rc
        pick = self.rng.integers(0, len(rows), k)
        x, y = self.gm.grid_to_world(rows[pick], cols[pick])
        jitter = self.gm.res * 0.5
        yaw = self._free_tangent[pick] \
            + self.rng.choice([0.0, math.pi], k) \
            + self.rng.normal(0.0, 0.3, k)
        out = np.empty((k, 3))
        out[:, 0] = x + self.rng.uniform(-jitter, jitter, k)
        out[:, 1] = y + self.rng.uniform(-jitter, jitter, k)
        out[:, 2] = np.arctan2(np.sin(yaw), np.cos(yaw))
        return out

    # ── initialization ───────────────────────────────────────────────────────
    def initialize(self, x, y, yaw, spread=0.5, yaw_spread=None):
        """Gaussian cloud around a known pose (e.g. the grid start box)."""
        if yaw_spread is None:
            yaw_spread = spread                       # radians, same knob
        self.p[:, 0] = x + self.rng.normal(0.0, spread, self.n)
        self.p[:, 1] = y + self.rng.normal(0.0, spread, self.n)
        self.p[:, 2] = yaw + self.rng.normal(0.0, yaw_spread, self.n)
        self.w[:] = 1.0 / self.n
        self.lost, self._lost_run, self._est = False, 0, None

    def initialize_global(self):
        """Kidnapped-robot init: uniform over free space, uniform yaw."""
        self.p = self._sample_free(self.n)
        self.w = np.full(self.n, 1.0 / self.n)
        self.lost, self._lost_run, self._est = True, self.lost_patience, None

    # ── motion model: body twist + tuned process noise ───────────────────────
    def predict(self, vx, vy, omega, dt):
        """Propagate every particle by the (noisily perturbed) body twist."""
        n, rng = self.n, self.rng
        dt = float(dt)
        if dt <= 0.0:
            return
        sv = self.alpha_v * abs(vx) + self.sigma_v
        sw = self.alpha_w * abs(omega) + self.sigma_w
        vxs = vx + rng.normal(0.0, sv, n)
        vys = vy + rng.normal(0.0, 0.5 * sv, n)       # lateral: smaller, real
        ws = omega + rng.normal(0.0, sw, n)
        yaw_mid = self.p[:, 2] + 0.5 * ws * dt        # midpoint integration
        c, s = np.cos(yaw_mid), np.sin(yaw_mid)
        rt = math.sqrt(dt)
        self.p[:, 0] += (vxs * c - vys * s) * dt + rng.normal(0, self.sigma_xy * rt, n)
        self.p[:, 1] += (vxs * s + vys * c) * dt + rng.normal(0, self.sigma_xy * rt, n)
        self.p[:, 2] += ws * dt + rng.normal(0.0, self.sigma_yaw * rt, n)
        self.p[:, 2] = np.arctan2(np.sin(self.p[:, 2]), np.cos(self.p[:, 2]))
        if self._est is not None:                     # dead-reckon the cached
            x, y, yw, conf = self._est                # estimate between scans
            ym = yw + 0.5 * omega * dt
            self._est = (x + (vx * math.cos(ym) - vy * math.sin(ym)) * dt,
                         y + (vx * math.sin(ym) + vy * math.cos(ym)) * dt,
                         yw + omega * dt, conf)

    # ── likelihood-field measurement model ───────────────────────────────────
    def update(self, scan_ranges, angle_min, angle_increment, subsample=20):
        """Score all particles against a subsampled scan; resample if needed.

        Vectorized outer product: endpoints (n_particles x n_beams) -> grid
        indices -> cached distance field -> Gaussian + uniform mixture.
        Returns the number of beams actually used.
        """
        r = np.asarray(scan_ranges, float)
        # recovery mode densifies the scan 4x: 18 beams alias badly on a
        # corridor track (wrong poses can explain them perfectly), 70+ beams
        # discriminate the true section from look-alikes
        step = max(1, int(subsample) // 4 if self.lost else int(subsample))
        idx = np.arange(0, len(r), step)
        rb = r[idx]
        ok = np.isfinite(rb) & (rb > 0.05) & (rb < self.max_range * 0.99)
        if ok.sum() < 3:                              # nothing usable: skip
            return 0
        rb = rb[ok]
        ab = angle_min + idx[ok] * angle_increment    # (B,) beam angles, body
        x, y, yaw = self.p[:, 0], self.p[:, 1], self.p[:, 2]
        a = yaw[:, None] + ab[None, :]                # (N, B) world angles
        ex = x[:, None] + rb[None, :] * np.cos(a)     # (N, B) endpoints
        ey = y[:, None] + rb[None, :] * np.sin(a)
        rr, cc = self.gm.world_to_grid(ex, ey)
        inb = (rr >= 0) & (rr < self.gm.h) & (cc >= 0) & (cc < self.gm.w)
        d = np.full(ex.shape, 10.0 * self.sigma_hit)  # off-map: miss, floor only
        np.clip(rr, 0, self.gm.h - 1, out=rr)
        np.clip(cc, 0, self.gm.w - 1, out=cc)
        dv = self.df[rr, cc]
        d[inb] = dv[inb]
        p_beam = self.z_hit * np.exp(-0.5 * (d / self.sigma_hit) ** 2) \
            + self.z_rand
        per = np.log(p_beam).mean(axis=1)             # per-particle beam fit
        # quality: best particle's normalized geometric-mean beam likelihood
        # — ~0.95+ when localized, lower when even the best pose explains the
        # scan poorly.  Interpretable because the mixture maximum is known.
        self.quality = float(np.exp(per.max())) / (self.z_hit + self.z_rand)
        if self.lost:
            # recovery: widen the basin (2x sigma) and temper to an effective
            # ~9 beams so several hypotheses survive resampling while the
            # dense quality signal arbitrates
            p_beam = self.z_hit * np.exp(-0.125 * (d / self.sigma_hit) ** 2) \
                + self.z_rand
            per = np.log(p_beam).mean(axis=1)
            n_eff_beams = 9.0
        else:
            n_eff_beams = float(len(rb))
        logw = np.log(np.maximum(self.w, 1e-300)) + per * n_eff_beams
        # particles inside walls / off the map are not poses
        pr, pc = self.gm.world_to_grid(x, y)
        pin = (pr >= 0) & (pr < self.gm.h) & (pc >= 0) & (pc < self.gm.w)
        bad = ~pin
        bad[pin] = self.gm.occupied[pr[pin], pc[pin]]
        logw[bad] -= 30.0
        logw -= logw.max()                            # stable normalization
        self.w = np.exp(logw)
        self.w /= self.w.sum()
        self._est = self._estimate()                  # snapshot BEFORE any
        # lost detection, two signals: poor best-particle fit (kidnapped /
        # diverged) OR a non-unimodal cloud (estimate is a meaningless mean
        # of competing clusters).  `lost_patience` consecutive bad updates
        # are required so transient dips (occlusion, dropouts) don't trigger
        # recovery; one clean unimodal update ends it.
        bad_now = self.quality < self.q_lost or self.spread() > self.spread_lost
        self._lost_run = self._lost_run + 1 if bad_now else 0
        self.lost = self._lost_run >= self.lost_patience
        if self.ess() < 0.5 * self.n:                 # resample/injection can
            self.resample(self.inject_frac if self.lost else 0.0)  # pollute it
        return int(ok.sum())

    # ── resampling ───────────────────────────────────────────────────────────
    def ess(self):
        """Effective sample size 1 / sum(w^2) — in (1, n]."""
        return 1.0 / float(np.sum(self.w ** 2))

    def resample(self, inject_frac=0.0):
        """Low-variance (systematic) resampling; resets weights uniform.

        `inject_frac` of the new set is drawn uniformly from free space —
        the kidnapped-robot recovery path, driven by lost detection."""
        edges = np.cumsum(self.w)
        edges[-1] = 1.0                               # guard float drift
        u = (self.rng.random() + np.arange(self.n)) / self.n
        self.p = self.p[np.searchsorted(edges, u)].copy()
        self.w[:] = 1.0 / self.n
        n_inj = int(round(inject_frac * self.n))
        if n_inj:
            slot = self.rng.choice(self.n, n_inj, replace=False)
            self.p[slot] = self._sample_free(n_inj)

    # ── estimate ─────────────────────────────────────────────────────────────
    def _estimate(self):
        x = float(self.w @ self.p[:, 0])
        y = float(self.w @ self.p[:, 1])
        yaw = math.atan2(float(self.w @ np.sin(self.p[:, 2])),
                         float(self.w @ np.cos(self.p[:, 2])))
        return x, y, yaw, self.ess() / self.n

    def pose(self):
        """(x, y, yaw, confidence): weighted mean, circular mean for yaw,
        confidence = ESS / n of the last measurement update (1.0 = uniform
        weights, ~0 = degenerate).  Snapshotted pre-resample so injected
        recovery particles never pollute the published estimate, and
        dead-reckoned through predict() between scans."""
        return self._est if self._est is not None else self._estimate()

    def spread(self):
        """Weighted position std (m) — a second dispersion diagnostic."""
        x, y, _, _ = self._estimate()
        return float(np.sqrt(self.w @ ((self.p[:, 0] - x) ** 2
                                       + (self.p[:, 1] - y) ** 2)))


# ─────────────────────────────────────────────────────────────────────────────
# ROS node (only imported/used on the car)
# ─────────────────────────────────────────────────────────────────────────────

def _make_node():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseWithCovarianceStamped

    class ParticleFilterNode(Node):
        def __init__(self):
            super().__init__('particle_filter')
            self.declare_parameter('map_yaml', '')        # '' = auto comp_track
            self.declare_parameter('scan_topic', '/scan')
            self.declare_parameter('ekf_odom_topic', '/ekf/odom')
            self.declare_parameter('out_topic', '/pf/pose/odom')
            self.declare_parameter('map_frame', 'map')
            self.declare_parameter('base_frame', 'base_link')
            self.declare_parameter('n_particles', 1500)
            self.declare_parameter('subsample', 20)
            self.declare_parameter('sigma_hit', 0.2)
            self.declare_parameter('z_hit', 0.85)
            self.declare_parameter('z_rand', 0.10)
            self.declare_parameter('max_range', 12.0)
            # [x, y, yaw] start pose; empty list = global (kidnapped) init
            self.declare_parameter('initial_pose', [0.0] * 0)
            self.declare_parameter('initial_spread', 0.5)
            p = lambda n: self.get_parameter(n).value   # noqa: E731

            yaml_path = p('map_yaml') or map_path_for('comp_track')
            if not yaml_path or not os.path.exists(yaml_path):
                raise RuntimeError(f'map not found: {yaml_path!r} '
                                   '(set the map_yaml parameter)')
            self.pf = ParticleFilter(
                GridMap.load(yaml_path), n_particles=int(p('n_particles')),
                z_hit=float(p('z_hit')), sigma_hit=float(p('sigma_hit')),
                z_rand=float(p('z_rand')), max_range=float(p('max_range')))
            init = list(p('initial_pose') or [])
            if len(init) == 3:
                self.pf.initialize(*init, spread=float(p('initial_spread')))
                mode = f'init at ({init[0]:.2f}, {init[1]:.2f}, {init[2]:.2f})'
            else:
                mode = 'global init (kidnapped)'
            self.subsample = int(p('subsample'))
            self.map_frame, self.base_frame = p('map_frame'), p('base_frame')
            self.twist = (0.0, 0.0, 0.0)              # latest EKF body twist
            self._last_odom_t = None
            self.pub = self.create_publisher(Odometry, p('out_topic'), 10)
            self.create_subscription(Odometry, p('ekf_odom_topic'),
                                     self._odom_cb, 50)
            self.create_subscription(LaserScan, p('scan_topic'),
                                     self._scan_cb, 10)
            self.create_subscription(PoseWithCovarianceStamped, '/initialpose',
                                     self._initialpose_cb, 1)
            self.get_logger().info(
                f"particle_filter ready — map={os.path.basename(yaml_path)} "
                f"n={self.pf.n} {mode}; scan={p('scan_topic')} "
                f"twist={p('ekf_odom_topic')} -> {p('out_topic')}")

        def _odom_cb(self, m):
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            tw = m.twist.twist
            self.twist = (tw.linear.x, tw.linear.y, tw.angular.z)
            if self._last_odom_t is not None:
                dt = t - self._last_odom_t
                if 0.0 < dt < 0.5:
                    self.pf.predict(*self.twist, dt)
            self._last_odom_t = t

        def _scan_cb(self, m):
            self.pf.update(m.ranges, m.angle_min, m.angle_increment,
                           subsample=self.subsample)
            x, y, yaw, conf = self.pf.pose()
            out = Odometry()
            out.header.stamp = m.header.stamp
            out.header.frame_id = self.map_frame
            out.child_frame_id = self.base_frame
            out.pose.pose.position.x = x
            out.pose.pose.position.y = y
            out.pose.pose.orientation.z = math.sin(0.5 * yaw)
            out.pose.pose.orientation.w = math.cos(0.5 * yaw)
            sp = self.pf.spread()
            out.pose.covariance[0] = out.pose.covariance[7] = sp * sp
            out.pose.covariance[35] = max(1e-6, 1.0 - conf)
            vx, vy, w = self.twist                   # EKF twist passed through
            out.twist.twist.linear.x = vx
            out.twist.twist.linear.y = vy
            out.twist.twist.angular.z = w
            self.pub.publish(out)

        def _initialpose_cb(self, m):
            q = m.pose.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self.pf.initialize(m.pose.pose.position.x,
                               m.pose.pose.position.y, yaw)
            self.get_logger().info('re-initialized from /initialpose')

    return rclpy, ParticleFilterNode


def main(args=None):
    rclpy, NodeCls = _make_node()
    rclpy.init(args=args)
    try:
        rclpy.spin(NodeCls())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
