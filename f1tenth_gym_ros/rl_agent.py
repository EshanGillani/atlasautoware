"""
RL racing agent — deploy the learned policy, with the MPC underneath it.
========================================================================

The same control loop as `raceline_mpc`, with one addition: after the MPC has
produced its command, the trained policy gets to bend it — by a bounded amount,
and only if everything about the inference looked right.

    scan + odom ──► MPC ──► + policy residual ──► governor ──► AEB ──► /drive
                     └──────── fallback ────────────┘

Every safety layer that protects the MPC still protects this.  The policy sits
*inside* the envelope, not around it:

  - it can move the command by at most `d_steer` rad and `d_speed` m/s, scaled
    by `residual_scale`, so `residual_scale:=0` is exactly the MPC;
  - a missing, mismatched or slow checkpoint means the policy is simply not
    consulted, and the car races on the MPC — never stops, never stalls;
  - a NaN or out-of-range network output is discarded for that tick;
  - the traction governor and the lidar AEB are applied *after* the policy, so
    it cannot talk the car out of braking.

Bringing it up on a real car
----------------------------
Raise `residual_scale` one step at a time — 0.0, 0.25, 0.5, 1.0 — and watch a
full clean lap at each before going further.  `/rl/status` publishes what the
policy is actually doing so you can see it earning (or losing) its place.

    ros2 run f1tenth_gym_ros rl_agent --ros-args \
        -p checkpoint:=runtime/rl/best.pt -p residual_scale:=0.25 -p v_scale:=0.4
"""

import math
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from ackermann_msgs.msg import AckermannDriveStamped
from transforms3d.euler import quat2euler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pursuit_agent import find_best_raceline, load_raceline, find_nearest
from mpc_controller import KinematicMPC, TractionGovernor
from map_controller import MAPController
from f1tenth_gym_ros.rl.features import (ObsSpec, ResidualAction, arclength,
                                         build_observation)


class RLAgent(Node):
    def __init__(self):
        super().__init__('rl_agent')

        # ── parameters ────────────────────────────────────────────────────────
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('status_topic', '/rl/status')
        self.declare_parameter('imu_topic', '')
        self.declare_parameter('raceline', '')
        self.declare_parameter('checkpoint', 'runtime/rl/best.pt')
        self.declare_parameter('residual_scale', 0.0)   # START AT 0
        self.declare_parameter('d_steer', 0.10)
        self.declare_parameter('d_speed', 1.5)
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('max_steer', 0.41)
        self.declare_parameter('control_hz', 50.0)
        self.declare_parameter('v_scale', 1.0)
        self.declare_parameter('aeb_dist', 0.45)
        self.declare_parameter('aeb_cone', 0.20)
        self.declare_parameter('aeb_decel', 6.0)
        self.declare_parameter('min_speed', 0.6)
        self.declare_parameter('max_lat_accel', 6.0)
        self.declare_parameter('max_infer_ms', 8.0)     # policy time budget

        g = self.get_parameter
        scan_topic = g('scan_topic').value
        odom_topic = g('odom_topic').value
        drive_topic = g('drive_topic').value
        self.L = float(g('wheelbase').value)
        self.max_steer = float(g('max_steer').value)
        self.v_scale = float(g('v_scale').value)
        self.aeb_dist = float(g('aeb_dist').value)
        self.aeb_cone = float(g('aeb_cone').value)
        self.aeb_decel = float(g('aeb_decel').value)
        self.min_speed = float(g('min_speed').value)
        self.max_infer_s = float(g('max_infer_ms').value) / 1000.0

        # ── raceline ──────────────────────────────────────────────────────────
        rl = g('raceline').value or self._find_raceline()
        if not rl or not os.path.exists(rl):
            self.get_logger().error('No raceline CSV — run the optimizer first.')
            raise FileNotFoundError('no raceline')
        self.rl_x, self.rl_y, self.rl_hdg, self.rl_curv, self.rl_speed = \
            load_raceline(rl)
        self.n = len(self.rl_x)
        self.s_cum = arclength(self.rl_x, self.rl_y)
        self.v_max = float(self.rl_speed.max())
        self.get_logger().info(
            f'raceline: {self.n} pts, v {self.rl_speed.min():.1f}-{self.v_max:.1f} '
            f'm/s from {os.path.basename(rl)}')

        # ── controllers ───────────────────────────────────────────────────────
        self.mpc = KinematicMPC(wheelbase=self.L, max_steer=self.max_steer,
                                v_max=self.v_max + 0.5)
        if self.mpc.available:
            self.mpc.set_raceline(self.rl_x, self.rl_y, self.rl_hdg,
                                  self.rl_curv, self.rl_speed)
        self.map_ctl = MAPController(wheelbase=self.L, max_steer=self.max_steer)
        self.map_ctl.set_raceline(self.rl_x, self.rl_y, self.rl_speed,
                                  curvature=self.rl_curv)
        self.governor = TractionGovernor(
            max_lat_accel=float(g('max_lat_accel').value))

        # ── policy ────────────────────────────────────────────────────────────
        self.authority = float(np.clip(g('residual_scale').value, 0.0, 1.0))
        self.spec = ObsSpec(max_steer=self.max_steer, v_ref=max(self.v_max, 1.0))
        self.residual = ResidualAction(
            d_steer=float(g('d_steer').value), d_speed=float(g('d_speed').value),
            max_steer=self.max_steer, v_min=0.0, v_max=self.v_max + 2.0,
            authority=self.authority)
        self.agent = None
        self.policy_state = 'disabled'
        self._load_policy(g('checkpoint').value)

        # ── state + wiring ────────────────────────────────────────────────────
        self.x = self.y = self.yaw = self.speed = 0.0
        self.yaw_rate = 0.0
        self.scan = None
        self.have_odom = self.have_imu = False
        self.nearest = 0
        self.lap = 0
        self._prev_near = 0
        self._log = 0
        self._cone_key = self._cone_mask = None
        self._infer_overruns = 0
        self._policy_used = 0
        self._ticks = 0

        imu_topic = g('imu_topic').value
        if imu_topic:
            self.create_subscription(Imu, imu_topic, self._imu_cb, 10)
        self.create_subscription(LaserScan, scan_topic, self._scan_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               drive_topic, 10)
        self.status_pub = self.create_publisher(String, g('status_topic').value, 10)
        self.create_timer(1.0 / float(g('control_hz').value), self._loop)
        self.get_logger().info(
            f'rl_agent ready — policy {self.policy_state}, '
            f'authority {self.authority:.2f}')

    # ── setup helpers ─────────────────────────────────────────────────────────
    def _find_raceline(self):
        rl = find_best_raceline()
        if rl and os.path.exists(rl):
            return rl
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local = os.path.join(repo, 'racelines', 'best_raceline.csv')
        return local if os.path.exists(local) else None

    def _load_policy(self, path):
        """Load the checkpoint, or explain clearly why we are racing on the MPC.

        Every failure here is non-fatal by design: a car that will not start
        because a checkpoint is missing is worse than a car that races on the
        controller that was already good enough to qualify.
        """
        if self.authority <= 0.0:
            self.policy_state = 'disabled (residual_scale=0)'
            self.get_logger().info('policy disabled — pure MPC')
            return
        if not path:
            self.policy_state = 'no checkpoint given'
            return
        if not os.path.isabs(path):
            path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
        if not os.path.exists(path):
            self.policy_state = f'missing {os.path.basename(path)}'
            self.get_logger().warning(
                f'checkpoint {path} not found — racing on the MPC. '
                f'Train one with: atlas run train-rl')
            return
        try:
            from f1tenth_gym_ros.rl.sac import SAC
        except Exception as e:
            self.policy_state = 'torch unavailable'
            self.get_logger().warning(f'cannot import the policy stack ({e}) — '
                                      f'racing on the MPC. pip3 install torch')
            return
        try:
            agent, meta = SAC.load(path, device='cpu')
        except Exception as e:
            self.policy_state = 'checkpoint unreadable'
            self.get_logger().error(f'could not load {path}: {e} — racing on the MPC')
            return

        # The observation contract: a policy trained on a different layout would
        # run happily on wrong inputs, which is the one failure the car cannot
        # detect from the outside.
        if agent.spec.fingerprint() != self.spec.fingerprint():
            self.policy_state = 'observation mismatch'
            self.get_logger().error(
                f'checkpoint expects observations "{agent.spec.fingerprint()}" '
                f'but this node builds "{self.spec.fingerprint()}" — refusing to '
                f'use it. Retrain against the current raceline/parameters.')
            return

        self.agent = agent
        self.policy_state = 'active'
        trained = meta.get('step', '?')
        self.get_logger().info(
            f'policy loaded from {os.path.basename(path)} '
            f'(trained {trained} steps), authority {self.authority:.2f}')

    # ── callbacks ─────────────────────────────────────────────────────────────
    def _scan_cb(self, m):
        self.scan = m

    def _odom_cb(self, m):
        self.x = m.pose.pose.position.x
        self.y = m.pose.pose.position.y
        self.speed = float(np.hypot(m.twist.twist.linear.x,
                                    m.twist.twist.linear.y))
        self.yaw_rate = float(m.twist.twist.angular.z)
        q = m.pose.pose.orientation
        _, _, self.yaw = quat2euler([q.w, q.x, q.y, q.z])
        self.have_odom = True

    def _imu_cb(self, m):
        self.yaw_rate = m.angular_velocity.z
        self.have_imu = True

    def _forward_clear(self):
        s = self.scan
        r = np.asarray(s.ranges, np.float32)
        key = (len(r), s.angle_min, s.angle_increment)
        if key != self._cone_key:
            ang = s.angle_min + np.arange(len(r)) * s.angle_increment
            self._cone_mask = np.abs(ang) < self.aeb_cone
            self._cone_key = key
        r = np.where(np.isfinite(r) & (r > 0.03), r, 30.0)
        return float(r[self._cone_mask].min()) if self._cone_mask.any() else 30.0

    # ── control loop ──────────────────────────────────────────────────────────
    def _loop(self):
        if self.scan is None or not self.have_odom:
            return
        self._ticks += 1
        self.nearest = find_nearest(self.x, self.y, self.rl_x, self.rl_y,
                                    self.nearest)

        # 1. baseline command
        steer = v_cmd = None
        if self.mpc.available:
            out = self.mpc.solve((self.x, self.y, self.yaw, self.speed),
                                 self.nearest)
            if out is not None:
                steer, v_cmd = out
        if steer is None:
            steer, v_cmd = self.map_ctl.control(self.x, self.y, self.yaw,
                                                self.speed, self.nearest)
        base_steer, base_speed = float(steer), float(v_cmd)

        # 2. policy residual — bounded, time-budgeted, and entirely optional
        used_policy = False
        if self.agent is not None:
            try:
                t0 = time.perf_counter()
                obs = build_observation(
                    self.spec, self.scan.ranges, self.x, self.y, self.yaw,
                    self.speed, self.yaw_rate, self.rl_x, self.rl_y, self.rl_hdg,
                    self.rl_curv, self.rl_speed, self.s_cum, self.nearest,
                    base_steer=base_steer, base_speed=base_speed)
                action = self.agent.act(obs, deterministic=True)
                dt = time.perf_counter() - t0
                if dt > self.max_infer_s:
                    # Late is the same as wrong in a 50 Hz loop: a command built
                    # from a stale state is worse than the MPC's fresh one.
                    self._infer_overruns += 1
                    if self._infer_overruns % 25 == 1:
                        self.get_logger().warning(
                            f'policy inference {dt*1000:.1f} ms > budget — '
                            f'skipped (x{self._infer_overruns})')
                else:
                    steer, v_cmd = self.residual.apply(action, base_steer,
                                                       base_speed)
                    used_policy = True
                    self._policy_used += 1
            except Exception as e:
                self.agent = None                    # one bad tick is a hiccup;
                self.policy_state = 'runtime error'  # a broken net is permanent
                self.get_logger().error(
                    f'policy failed ({e}) — reverting to the MPC for the '
                    f'rest of this run')

        v_cmd = float(v_cmd) * self.v_scale

        # 3. safety layers, AFTER the policy so it cannot override them
        if self.have_imu:
            v_cmd *= self.governor.update(self.yaw_rate, self.speed)
        stop_dist = self.aeb_dist + self.speed ** 2 / (2.0 * self.aeb_decel)
        braking = self._forward_clear() < stop_dist
        if braking:
            steer, v_cmd = 0.0, 0.0
            if self._log % 10 == 0:
                self.get_logger().warning('AEB — obstacle ahead, stopping')
        else:
            v_cmd = max(v_cmd, self.min_speed)

        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(np.clip(steer, -self.max_steer,
                                                 self.max_steer))
        msg.drive.speed = float(v_cmd)
        self.drive_pub.publish(msg)

        # 4. tell the operator what the policy is actually doing
        if self._log % 5 == 0:
            self.status_pub.publish(String(data=(
                f'{{"policy":"{self.policy_state}",'
                f'"authority":{self.authority:.2f},'
                f'"using_policy":{str(used_policy).lower()},'
                f'"d_steer":{msg.drive.steering_angle - base_steer:.4f},'
                f'"d_speed":{v_cmd - base_speed * self.v_scale:.3f},'
                f'"speed":{self.speed:.2f},"wp":{self.nearest},'
                f'"lap":{self.lap},"aeb":{str(braking).lower()},'
                f'"usage":{self._policy_used / max(self._ticks, 1):.2f}}}')))

        if self._prev_near > self.n - 12 and self.nearest < 12:
            self.lap += 1
            self.get_logger().info(f'lap {self.lap}')
        self._prev_near = self.nearest
        self._log += 1
        if self._log % 50 == 0:
            self.get_logger().info(
                f'wp {self.nearest}/{self.n} v={v_cmd:.1f} '
                f'steer={math.degrees(msg.drive.steering_angle):.0f}deg '
                f'policy={"on" if used_policy else "off"} '
                f'(delta {msg.drive.steering_angle - base_steer:+.3f} rad, '
                f'{v_cmd - base_speed * self.v_scale:+.2f} m/s)')


def main(args=None):
    rclpy.init(args=args)
    try:
        node = RLAgent()
        rclpy.spin(node)
    except (FileNotFoundError, KeyboardInterrupt):
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
