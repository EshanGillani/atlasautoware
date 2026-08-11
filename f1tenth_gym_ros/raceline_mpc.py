"""
Raceline-MPC racing agent — the clean competition car.
=======================================================

The minimal, bulletproof time-trial deployment: load the optimized raceline,
track it with the kinematic MPC, brake for anything genuinely in the way, drive.
No opponent strategy, no progressive learning, no camera — just the racing line
+ MPC + a lidar emergency brake.  This is the single path you make race-ready and
field, with the heavier agents reserved for their own jobs (mapping, opponents).

  best_raceline.csv ──► MPC (track) ──► /drive
        localization ──┘        lidar AEB ──┘

Design choices that matter for a race:
  - **MPC with an automatic fallback.**  If osqp is missing or a solve fails,
    the loop transparently falls back to the MAP controller (model- and
    acceleration-based pursuit, Becker et al. ICRA 2023 — pure pursuit's
    geometry with tire-aware steering inversion) that tick, so the car never
    stalls on a solver hiccup.
  - **Friction-limited speeds on demand.**  `reprofile_speeds` replaces the
    raceline CSV's speed column at load with the TUMFTM forward-backward
    profile (lateral budget + friction-ellipse-coupled accel/brake limits).
  - **Hardware-portable.**  Every sim-only topic/frame is a ROS parameter; the
    only thing that differs on the real car is `odom_topic` (the pose source from
    your localization — a particle filter publishing map-relative odometry, NOT
    raw drifting VESC odom).
  - **Light control loop.**  No disk I/O, no heavy perception — just solve + brake
    + publish, so it holds 50 Hz and never trips a watchdog.

Run:
    # sim
    ros2 run f1tenth_gym_ros raceline_mpc
    # real car (pose from your localization, e.g. the particle filter)
    ros2 run f1tenth_gym_ros raceline_mpc --ros-args -p odom_topic:=/pf/pose/odom
"""

import math
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from transforms3d.euler import quat2euler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pursuit_agent import find_best_raceline, load_raceline, find_nearest
from mpc_controller import KinematicMPC, TractionGovernor, predict_state
from map_controller import MAPController
from velocity_profiler import velocity_profile, segment_lengths
from raceline_refiner import refine_raceline


class RacelineMPC(Node):
    def __init__(self):
        super().__init__('raceline_mpc')

        # ── parameters (sim defaults; override on hardware) ────────────────────
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('raceline', '')          # explicit CSV; '' = auto-find
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('max_steer', 0.41)
        self.declare_parameter('control_hz', 50.0)
        self.declare_parameter('v_scale', 1.0)          # global speed cap (start low!)
        self.declare_parameter('aeb_dist', 0.45)        # m, hard stop if wall/obstacle closer
        self.declare_parameter('aeb_cone', 0.20)        # rad (~11deg) forward cone
        self.declare_parameter('aeb_decel', 6.0)        # m/s^2 — extends aeb_dist with speed
        self.declare_parameter('min_speed', 0.6)        # m/s creep floor when rolling
        self.declare_parameter('imu_topic', '')         # e.g. /oakd/imu; '' = off (sim)
        self.declare_parameter('max_lat_accel', 6.0)    # m/s^2 traction-governor limit
        self.declare_parameter('actuation_delay', 0.0)  # s sensor->actuator latency
        self.declare_parameter('refine_corridor', 0.0)  # m min-curvature refinement
        self.declare_parameter('reprofile_speeds', False)  # recompute CSV speeds
        self.declare_parameter('profile_a_accel', 4.0)  # m/s^2 engine limit
        self.declare_parameter('profile_a_brake', 8.0)  # m/s^2 braking limit
        self.declare_parameter('profile_v_max', 8.0)    # m/s profile ceiling
        # MPC cost weights.  These are what tools/bayes_tune.py searches over,
        # so without them on the node the tuned controller has no way onto the
        # car and only the speed-profile half of a tuning result is deployable.
        self.declare_parameter('mpc_horizon', 12)       # prediction steps
        self.declare_parameter('mpc_q_pos', 28.0)       # position tracking
        self.declare_parameter('mpc_q_yaw', 6.0)        # heading tracking
        self.declare_parameter('mpc_q_v', 2.5)          # speed tracking
        self.declare_parameter('mpc_rd_steer', 12.0)    # steer-rate penalty
        scan_topic  = self.get_parameter('scan_topic').value
        odom_topic  = self.get_parameter('odom_topic').value
        drive_topic = self.get_parameter('drive_topic').value
        self.L          = float(self.get_parameter('wheelbase').value)
        self.max_steer  = float(self.get_parameter('max_steer').value)
        self.v_scale    = float(self.get_parameter('v_scale').value)
        self.aeb_dist   = float(self.get_parameter('aeb_dist').value)
        self.aeb_cone   = float(self.get_parameter('aeb_cone').value)
        self.aeb_decel  = float(self.get_parameter('aeb_decel').value)
        self.min_speed  = float(self.get_parameter('min_speed').value)
        self.delay      = float(self.get_parameter('actuation_delay').value)
        self._last_cmd  = (0.0, 0.0)                    # published (steer, speed)
        # commands still in flight toward the actuator, oldest first — one per
        # control tick over the delay window.  predict_state must integrate
        # through THESE (the actuator executes the previously issued commands
        # during the latency window, not the newest one).
        hz = float(self.get_parameter('control_hz').value)
        self._delay_ticks = int(round(self.delay * hz))
        self._cmd_buf = [(0.0, 0.0)] * self._delay_ticks

        # ── raceline ───────────────────────────────────────────────────────────
        rl = self.get_parameter('raceline').value or self._find_raceline()
        if not rl or not os.path.exists(rl):
            self.get_logger().error('No raceline CSV found — run the optimizer first.')
            raise FileNotFoundError('no raceline')
        self.rl_x, self.rl_y, self.rl_hdg, self.rl_curv, self.rl_speed = load_raceline(rl)
        self.n = len(self.rl_x)
        corridor = float(self.get_parameter('refine_corridor').value)
        if corridor > 0.0:
            # minimum-curvature refinement (TUMFTM) within +/- corridor of the
            # loaded line — validate wall clearance before enabling on a car
            self.rl_x, self.rl_y, self.rl_hdg, self.rl_curv = refine_raceline(
                self.rl_x, self.rl_y, corridor=corridor)
            self.get_logger().info(
                f'raceline refined (min-curvature, corridor {corridor:.2f} m)')
        if self.get_parameter('reprofile_speeds').value:
            # friction-limited forward-backward profile (TUMFTM) — replaces the
            # CSV speed column with one that provably fits the grip budget
            self.rl_speed = velocity_profile(
                self.rl_curv, segment_lengths(self.rl_x, self.rl_y),
                a_lat_max=float(self.get_parameter('max_lat_accel').value),
                a_accel_max=float(self.get_parameter('profile_a_accel').value),
                a_brake_max=float(self.get_parameter('profile_a_brake').value),
                v_max=float(self.get_parameter('profile_v_max').value))
            self.get_logger().info(
                f'speeds reprofiled (friction-limited): '
                f'{self.rl_speed.min():.1f}-{self.rl_speed.max():.1f} m/s')
        self.v_max = float(self.rl_speed.max())
        self.get_logger().info(
            f'raceline: {self.n} pts, v {self.rl_speed.min():.1f}-{self.v_max:.1f} m/s '
            f'(x{self.v_scale:.2f}) from {os.path.basename(rl)}')

        # ── MPC (with pure-pursuit fallback) ───────────────────────────────────
        self.mpc = KinematicMPC(
            wheelbase=self.L, max_steer=self.max_steer, v_max=self.v_max + 0.5,
            horizon=int(self.get_parameter('mpc_horizon').value),
            max_accel=float(self.get_parameter('profile_a_accel').value),
            max_brake=float(self.get_parameter('profile_a_brake').value),
            q_pos=float(self.get_parameter('mpc_q_pos').value),
            q_yaw=float(self.get_parameter('mpc_q_yaw').value),
            q_v=float(self.get_parameter('mpc_q_v').value),
            rd_steer=float(self.get_parameter('mpc_rd_steer').value))
        if self.mpc.available:
            self.mpc.set_raceline(self.rl_x, self.rl_y, self.rl_hdg,
                                  self.rl_curv, self.rl_speed)
            self.get_logger().info(
                f'controller: MPC (kinematic LTV, osqp) N={self.mpc.N} '
                f'q_pos={self.get_parameter("mpc_q_pos").value} '
                f'rd_steer={self.get_parameter("mpc_rd_steer").value}')
        else:
            self.get_logger().warning(
                'osqp not available — using MAP fallback full-time '
                '(pip install osqp==0.6.3 on the car to enable MPC)')

        # ── MAP fallback (Becker et al., ICRA 2023) — replaces pure pursuit ───
        self.map_ctl = MAPController(wheelbase=self.L, max_steer=self.max_steer)
        self.map_ctl.set_raceline(self.rl_x, self.rl_y, self.rl_speed,
                                  curvature=self.rl_curv)
        self.get_logger().info('fallback: MAP (model- and acceleration-based pursuit)')

        # ── traction governor (IMU; inert until imu_topic is set) ─────────────
        self.governor = TractionGovernor(
            max_lat_accel=float(self.get_parameter('max_lat_accel').value))
        self.yaw_rate = 0.0
        imu_topic = self.get_parameter('imu_topic').value
        if imu_topic:
            self.create_subscription(Imu, imu_topic, self._imu_cb, 10)
            self.get_logger().info(f'traction governor on (imu={imu_topic})')

        # ── state + ROS wiring ─────────────────────────────────────────────────
        self.x = self.y = self.yaw = self.speed = 0.0
        self.scan = None
        self.have_odom = False
        self.have_imu = False
        self.nearest = 0
        self.lap = 0
        self._prev_near = 0
        self._log = 0
        self._cone_key = None                           # cached AEB cone mask
        self._cone_mask = None
        self.create_subscription(LaserScan, scan_topic, self._scan_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.create_timer(1.0 / float(self.get_parameter('control_hz').value), self._loop)
        self.get_logger().info(
            f'raceline_mpc ready — scan={scan_topic} odom={odom_topic} drive={drive_topic}')

    def _find_raceline(self):
        rl = find_best_raceline()                       # sim path / F1_RACELINE / best_*
        if rl and os.path.exists(rl):
            return rl
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local = os.path.join(repo, 'racelines', 'best_raceline.csv')
        return local if os.path.exists(local) else None

    # ── callbacks ──────────────────────────────────────────────────────────────
    def _scan_cb(self, m):
        self.scan = m

    def _odom_cb(self, m):
        self.x = m.pose.pose.position.x
        self.y = m.pose.pose.position.y
        self.speed = float(np.hypot(m.twist.twist.linear.x, m.twist.twist.linear.y))
        q = m.pose.pose.orientation
        _, _, self.yaw = quat2euler([q.w, q.x, q.y, q.z])
        self.have_odom = True

    def _imu_cb(self, m):
        self.yaw_rate = m.angular_velocity.z            # |.| used; sign-agnostic
        self.have_imu = True

    # ── emergency brake: min range in a narrow forward cone ────────────────────
    def _forward_clear(self):
        s = self.scan
        r = np.asarray(s.ranges, np.float32)
        key = (len(r), s.angle_min, s.angle_increment)
        if key != self._cone_key:                       # scan geometry is static —
            ang = s.angle_min + np.arange(len(r)) * s.angle_increment
            self._cone_mask = np.abs(ang) < self.aeb_cone
            self._cone_key = key                        # build the mask once
        r = np.where(np.isfinite(r) & (r > 0.03), r, 30.0)
        cone = self._cone_mask
        return float(r[cone].min()) if cone.any() else 30.0

    # ── control loop ───────────────────────────────────────────────────────────
    def _loop(self):
        if self.scan is None or not self.have_odom:
            return
        # delay compensation: solve from where the car will be when the
        # command actually reaches the wheels, not where it was last measured.
        # Integrate through the in-flight command pipeline (oldest first) —
        # holding only the last command over-rotates the prediction whenever
        # the steer is changing and destabilizes the loop at larger delays.
        px, py, pyaw, pv = self.x, self.y, self.yaw, self.speed
        if self.delay > 0.0:
            px, py, pyaw, pv = predict_state(
                px, py, pyaw, pv,
                [c[0] for c in self._cmd_buf] or self._last_cmd[0],
                [c[1] for c in self._cmd_buf] or self._last_cmd[1],
                self.delay, self.L)
        self.nearest = find_nearest(px, py, self.rl_x, self.rl_y, self.nearest)

        steer = v_cmd = None
        if self.mpc.available:
            out = self.mpc.solve((px, py, pyaw, pv), self.nearest)
            if out is not None:
                steer, v_cmd = out
        if steer is None:                               # MPC off or solve failed
            steer, v_cmd = self.map_ctl.control(px, py, pyaw, pv, self.nearest)

        v_cmd = float(v_cmd) * self.v_scale

        # Traction governor — scale down when the IMU says we're past the
        # lateral-grip budget (no-op until an IMU is publishing).
        if self.have_imu:
            v_cmd *= self.governor.update(self.yaw_rate, self.speed)

        # Emergency brake — something genuinely in the path (wall on a missed
        # corner, or an obstacle).  Hard stop; the only thing that overrides
        # MPC.  Trigger distance grows with speed so the stop fits within
        # `aeb_decel` of real braking, not just the standstill margin.
        stop_dist = self.aeb_dist + self.speed ** 2 / (2.0 * self.aeb_decel)
        if self._forward_clear() < stop_dist:
            steer, v_cmd = 0.0, 0.0
            if self._log % 10 == 0:
                self.get_logger().warning('AEB — obstacle ahead, stopping')
        else:
            v_cmd = max(v_cmd, self.min_speed)

        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(np.clip(steer, -self.max_steer, self.max_steer))
        msg.drive.speed = float(v_cmd)
        self.drive_pub.publish(msg)
        self._last_cmd = (msg.drive.steering_angle, msg.drive.speed)
        if self._delay_ticks:                           # advance the pipeline
            self._cmd_buf.append(self._last_cmd)
            self._cmd_buf.pop(0)

        # lap counter (index wraps past start/finish)
        if self._prev_near > self.n - 12 and self.nearest < 12:
            self.lap += 1
            self.get_logger().info(f'lap {self.lap}')
        self._prev_near = self.nearest
        self._log += 1
        if self._log % 50 == 0:
            self.get_logger().info(
                f'wp {self.nearest}/{self.n} v={v_cmd:.1f} steer={math.degrees(steer):.0f}deg')


def main(args=None):
    rclpy.init(args=args)
    try:
        node = RacelineMPC()
        rclpy.spin(node)
    except (FileNotFoundError, KeyboardInterrupt):
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
