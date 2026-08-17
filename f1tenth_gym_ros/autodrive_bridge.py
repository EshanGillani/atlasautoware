"""
AutoDRIVE <-> AtlasAutoware bridge (RoboRacer Sim Racing League).
================================================================

Translates between the AutoDRIVE Simulator's ROS 2 interface and the topics
the AtlasAutoware racing stack already speaks, so the *same* nodes
(`raceline_mpc`, `race_agent`, ...) that drive the f1tenth_gym sim and the
real car can race in AutoDRIVE without modification.

AutoDRIVE side (RoboRacer generation; legacy "f1tenth_1" supported via the
`vehicle` param).  Source: AutoDRIVE-Ecosystem devkit config.py/bridge.py.

  in  (from sim):  /autodrive/<veh>/lidar     sensor_msgs/LaserScan  (270 deg, 1080 rays, 0.06-10 m)
                   /autodrive/<veh>/imu       sensor_msgs/Imu
                   /autodrive/<veh>/odom      nav_msgs/Odometry      (world -> <veh>, ground truth)
  out (to sim):    /autodrive/<veh>/throttle_command  std_msgs/Float32  normalized [-1, 1]
                   /autodrive/<veh>/steering_command  std_msgs/Float32  normalized [-1, 1]

AtlasAutoware side (what raceline_mpc / race_agent consume / produce):
  out:  /scan (LaserScan), <odom_topic> (Odometry), /oakd/imu (Imu)
  in :  /drive (AckermannDriveStamped)

Conversions (calibrate to your sim build — see NEEDS-CONFIRMATION below):
  throttle_command = clip(speed / max_speed, -1, 1)      max_speed ~ 22.8 m/s
  steering_command = clip(steer_sign * angle / max_steer, -1, 1)
                                                          max_steer ~ 0.5236 rad (30 deg)

NEEDS CONFIRMATION in the running sim (flagged here, not guessed):
  * steer_sign  - which sign of steering_command turns LEFT (param `steer_sign`)
  * max_speed / max_steer calibration constants (params)
  * the AutoDRIVE frame tree is rooted at `world` with no `base_link`; this
    bridge re-stamps odom as odom_frame->base_frame and (optionally) emits the
    map->odom->base_link + base_link->laser TF chain the stack/SLAM expect.

The pure conversion helpers below are unit-tested (tests/test_autodrive_bridge.py);
the ROS wiring is built lazily in _make_node() so the maths is importable
without rclpy (matching drive_node.py's pattern).
"""

import math


# ── pure conversions (no ROS; unit-tested) ──────────────────────────────────

def throttle_from_speed(speed, max_speed):
    """m/s -> AutoDRIVE normalized throttle command in [-1, 1]."""
    if max_speed <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, float(speed) / float(max_speed)))


def steering_from_angle(angle, max_steer, steer_sign=1.0):
    """rad -> AutoDRIVE normalized steering command in [-1, 1].

    `steer_sign` flips the convention if +command turns the car the wrong way
    (NEEDS CONFIRMATION in the running sim).
    """
    if max_steer <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, float(steer_sign) * float(angle) / float(max_steer)))


def mask_scan_ranges(ranges, range_max, eps=1e-3):
    """AutoDRIVE clamps no-return beams to range_max with no inf/NaN masking;
    f1tenth-style consumers expect inf for "no return".  Convert clamped
    maxima back to inf so AEB / gap-following behave as on the real lidar.
    """
    rmax = float(range_max)
    out = []
    for r in ranges:
        r = float(r)
        out.append(float('inf') if (not math.isfinite(r) or r >= rmax - eps)
                   else r)
    return out


# ── ROS node (lazy import; not exercised in the test sandbox) ────────────────

def _make_node():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import LaserScan, Imu
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Float32
    from ackermann_msgs.msg import AckermannDriveStamped
    from geometry_msgs.msg import TransformStamped
    from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

    class AutoDriveBridge(Node):
        def __init__(self):
            super().__init__('autodrive_bridge')
            # AutoDRIVE vehicle namespace: 'roboracer_1' (current) | 'f1tenth_1' (legacy)
            self.declare_parameter('vehicle', 'roboracer_1')
            # AtlasAutoware-side topics (point raceline_mpc/race_agent at these)
            self.declare_parameter('scan_out', '/scan')
            self.declare_parameter('odom_out', '/ego_racecar/odom')
            self.declare_parameter('imu_out', '/oakd/imu')
            self.declare_parameter('drive_in', '/drive')
            # actuation calibration (NEEDS CONFIRMATION per sim build)
            self.declare_parameter('max_speed', 22.8)     # m/s at throttle=1.0
            self.declare_parameter('max_steer', 0.5236)   # rad at steering=1.0 (~30 deg)
            self.declare_parameter('steer_sign', 1.0)     # +1 or -1; confirm in sim
            # frames / TF
            self.declare_parameter('map_frame', 'map')
            self.declare_parameter('odom_frame', 'odom')
            self.declare_parameter('base_frame', 'base_link')
            self.declare_parameter('laser_frame', 'laser')
            self.declare_parameter('mask_max_range', True)   # clamp->inf
            self.declare_parameter('broadcast_tf', True)
            self.declare_parameter('laser_x', 0.27)          # base->laser offset (m)
            self.declare_parameter('laser_z', 0.11)

            g = lambda n: self.get_parameter(n).value
            veh = g('vehicle')
            self.pfx = f'/autodrive/{veh}'
            self.max_speed = float(g('max_speed'))
            self.max_steer = float(g('max_steer'))
            self.steer_sign = float(g('steer_sign'))
            self.map_frame = g('map_frame')
            self.odom_frame = g('odom_frame')
            self.base_frame = g('base_frame')
            self.laser_frame = g('laser_frame')
            self.mask = bool(g('mask_max_range'))
            self.do_tf = bool(g('broadcast_tf'))

            # AutoDRIVE devkit publishes RELIABLE / KEEP_LAST / depth=1
            qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=1)

            # publishers (AtlasAutoware side)
            self.scan_pub = self.create_publisher(LaserScan, g('scan_out'), 10)
            self.odom_pub = self.create_publisher(Odometry, g('odom_out'), 10)
            self.imu_pub = self.create_publisher(Imu, g('imu_out'), 10)
            # publishers (AutoDRIVE actuation)
            self.thr_pub = self.create_publisher(
                Float32, f'{self.pfx}/throttle_command', qos)
            self.str_pub = self.create_publisher(
                Float32, f'{self.pfx}/steering_command', qos)

            # subscribers (AutoDRIVE sensors)
            self.create_subscription(LaserScan, f'{self.pfx}/lidar',
                                     self._scan_cb, qos)
            self.create_subscription(Imu, f'{self.pfx}/imu', self._imu_cb, qos)
            self.create_subscription(Odometry, f'{self.pfx}/odom',
                                     self._odom_cb, qos)
            # subscriber (AtlasAutoware /drive -> AutoDRIVE commands)
            self.create_subscription(AckermannDriveStamped, g('drive_in'),
                                     self._drive_cb, 10)

            if self.do_tf:
                self.tf_bc = TransformBroadcaster(self)
                self.static_bc = StaticTransformBroadcaster(self)
                self._publish_static_tf(TransformStamped)
                self._TransformStamped = TransformStamped

            self.get_logger().info(
                f'autodrive_bridge: {self.pfx}/* <-> [{g("scan_out")}, '
                f'{g("odom_out")}, {g("imu_out")}, {g("drive_in")}]  '
                f'max_speed={self.max_speed} max_steer={self.max_steer} '
                f'steer_sign={self.steer_sign:+.0f}')

        # ── sim -> stack ────────────────────────────────────────────────────
        def _scan_cb(self, msg):
            msg.header.frame_id = self.laser_frame
            if self.mask:
                msg.ranges = [float('inf') if (r >= msg.range_max - 1e-3)
                              else r for r in msg.ranges]
            self.scan_pub.publish(msg)

        def _imu_cb(self, msg):
            # keep AutoDRIVE imu frame under base; restamp to base for the
            # traction governor (it only uses angular_velocity.z magnitude)
            self.imu_pub.publish(msg)

        def _odom_cb(self, msg):
            # AutoDRIVE odom is world->vehicle (ground truth).  Re-stamp as the
            # REP-105 odom->base_link the stack expects; the pose is global, so
            # raceline_mpc tracks the raceline directly (no particle filter
            # needed in sim).
            msg.header.frame_id = self.odom_frame
            msg.child_frame_id = self.base_frame
            self.odom_pub.publish(msg)
            if self.do_tf:
                self._broadcast_odom_tf(msg)

        # ── stack -> sim ────────────────────────────────────────────────────
        def _drive_cb(self, msg):
            thr = Float32()
            thr.data = float(throttle_from_speed(msg.drive.speed, self.max_speed))
            steer = Float32()
            steer.data = float(steering_from_angle(
                msg.drive.steering_angle, self.max_steer, self.steer_sign))
            self.thr_pub.publish(thr)
            self.str_pub.publish(steer)

        # ── TF helpers ──────────────────────────────────────────────────────
        def _publish_static_tf(self, TransformStamped):
            now = self.get_clock().now().to_msg()
            # map -> odom identity (sim odom is already global / ground truth)
            t_mo = TransformStamped()
            t_mo.header.stamp = now
            t_mo.header.frame_id = self.map_frame
            t_mo.child_frame_id = self.odom_frame
            t_mo.transform.rotation.w = 1.0
            # base_link -> laser mounting offset
            t_bl = TransformStamped()
            t_bl.header.stamp = now
            t_bl.header.frame_id = self.base_frame
            t_bl.child_frame_id = self.laser_frame
            t_bl.transform.translation.x = float(
                self.get_parameter('laser_x').value)
            t_bl.transform.translation.z = float(
                self.get_parameter('laser_z').value)
            t_bl.transform.rotation.w = 1.0
            self.static_bc.sendTransform([t_mo, t_bl])

        def _broadcast_odom_tf(self, odom):
            t = self._TransformStamped()
            t.header.stamp = odom.header.stamp
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = odom.pose.pose.position.x
            t.transform.translation.y = odom.pose.pose.position.y
            t.transform.translation.z = odom.pose.pose.position.z
            t.transform.rotation = odom.pose.pose.orientation
            self.tf_bc.sendTransform(t)

    return rclpy, AutoDriveBridge


def main(args=None):
    rclpy, NodeCls = _make_node()
    rclpy.init(args=args)
    node = NodeCls()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
