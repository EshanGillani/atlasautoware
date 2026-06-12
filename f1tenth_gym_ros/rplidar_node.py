"""
RPLidar driver — Slamtec RPLidar -> sensor_msgs/LaserScan.
==========================================================

Reads a Slamtec RPLidar (A1/A2/A3/S-series) over USB serial via the `rplidar`
pip package and publishes a fixed-grid 360 deg LaserScan on /scan — the same
message the sim bridge produces, so every racing node (raceline_mpc AEB,
race_brain gap detection, slam_toolbox mapping) runs unmodified on hardware.

Why a fixed grid: RPLidar measurements arrive at irregular angles that drift
scan-to-scan; the racing code indexes beams by `angle_min + i*increment`, so
each revolution is binned into `num_bins` even slots (nearest-return wins,
empty slots = inf).  RPLidar angles increase clockwise with 0 deg at the
connector; ROS wants CCW, handled in `bin_scan` plus a mounting-offset param.

Reliability: the serial read runs in a daemon thread with automatic
reconnect — a USB hiccup mid-race costs a scan or two, not the node.

Run:
    ros2 run f1tenth_gym_ros rplidar_node --ros-args --params-file config/hardware.yaml
    # A1/A2: baud 115200      A3/S1: baud 256000
"""

import math
import threading
import time

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Pure scan geometry (unit-tested without hardware)
# ─────────────────────────────────────────────────────────────────────────────

def bin_scan(measurements, num_bins, range_min=0.10, range_max=25.0,
             angle_offset=0.0):
    """One revolution of (quality, angle_deg, dist_mm) -> fixed-grid ranges.

    Grid covers [-pi, pi) CCW (ROS convention); RPLidar's clockwise angles are
    negated.  `angle_offset` (rad, CCW) corrects how the unit is mounted.
    Returns float32 ranges, inf where no valid return landed in a bin.
    """
    num_bins = int(num_bins)
    ranges = np.full(num_bins, np.inf, np.float32)
    if not len(measurements):
        return ranges
    m = np.asarray(measurements, np.float64)            # (M, 3) q/angle/dist
    d = m[:, 2] / 1000.0
    keep = (m[:, 0] > 0) & (d >= range_min) & (d <= range_max)
    theta = -np.radians(m[keep, 1]) + angle_offset
    theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
    inc = 2.0 * math.pi / num_bins
    idx = ((theta + math.pi) / inc).astype(np.intp) % num_bins
    np.minimum.at(ranges, idx, d[keep].astype(np.float32))  # nearest return wins
    return ranges


def _make_node():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry
    from scan_deskew import deskew_ranges

    class RPLidarNode(Node):
        def __init__(self):
            super().__init__('rplidar_node')
            self.declare_parameter('port', '/dev/ttyUSB0')
            self.declare_parameter('baudrate', 115200)
            self.declare_parameter('scan_topic', '/scan')
            self.declare_parameter('frame_id', 'laser')
            self.declare_parameter('num_bins', 720)
            self.declare_parameter('range_min', 0.10)
            self.declare_parameter('range_max', 25.0)
            self.declare_parameter('angle_offset', 0.0)   # rad, mounting yaw
            self.declare_parameter('deskew_odom_topic', '')  # e.g. /ekf/odom
            p = lambda n: self.get_parameter(n).value     # noqa: E731
            self.port = p('port')
            self.baud = int(p('baudrate'))
            self.frame = p('frame_id')
            self.num_bins = int(p('num_bins'))
            self.range_min = float(p('range_min'))
            self.range_max = float(p('range_max'))
            self.angle_offset = float(p('angle_offset'))

            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.pub = self.create_publisher(LaserScan, p('scan_topic'), qos)
            self._twist = None
            if p('deskew_odom_topic'):
                self.create_subscription(Odometry, p('deskew_odom_topic'),
                                         self._odom_cb, 10)
                # beam age per ROS-grid bin: the unit sweeps its OWN angle
                # 0->2pi clockwise over scan_time; ROS bin theta corresponds
                # to rplidar angle -theta, hence this time fraction
                theta = -math.pi + np.arange(self.num_bins) \
                    * (2.0 * math.pi / self.num_bins)
                frac = ((-theta - self.angle_offset) % (2.0 * math.pi)) \
                    / (2.0 * math.pi)
                self._ages_frac = frac - 1.0          # x scan_time at publish
                self.get_logger().info(
                    f"de-skew on (twist from {p('deskew_odom_topic')})")
            self._stop = False
            self._last_pub = time.monotonic()
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            self.get_logger().info(
                f'rplidar_node ready — {self.port}@{self.baud} '
                f'{self.num_bins} bins -> {p("scan_topic")}')

        def _publish(self, measurements):
            now = time.monotonic()
            scan = LaserScan()
            scan.header.stamp = self.get_clock().now().to_msg()
            scan.header.frame_id = self.frame
            scan.angle_min = -math.pi
            scan.angle_max = math.pi - 2.0 * math.pi / self.num_bins
            scan.angle_increment = 2.0 * math.pi / self.num_bins
            scan.scan_time = max(now - self._last_pub, 1e-3)
            scan.time_increment = scan.scan_time / self.num_bins
            scan.range_min = self.range_min
            scan.range_max = self.range_max
            ranges = bin_scan(measurements, self.num_bins, self.range_min,
                              self.range_max, self.angle_offset)
            if self._twist is not None:
                vx, vy, w = self._twist
                ranges = deskew_ranges(
                    ranges, scan.angle_min, scan.angle_increment,
                    scan.scan_time, vx, vy, w,
                    ages=self._ages_frac * scan.scan_time)
            scan.ranges = ranges.tolist()
            self.pub.publish(scan)
            self._last_pub = now

        def _odom_cb(self, m):
            self._twist = (m.twist.twist.linear.x, m.twist.twist.linear.y,
                           m.twist.twist.angular.z)

        def _read_loop(self):
            try:
                from rplidar import RPLidar
            except ImportError:
                self.get_logger().error(
                    'rplidar package missing — pip3 install rplidar-roboticia')
                return
            while not self._stop:
                lidar = None
                try:
                    lidar = RPLidar(self.port, baudrate=self.baud)
                    self.get_logger().info(f'connected: {lidar.get_info()}')
                    for measurements in lidar.iter_scans(max_buf_meas=5000):
                        if self._stop:
                            break
                        self._publish(measurements)
                except Exception as e:
                    if not self._stop:
                        self.get_logger().warning(f'lidar error ({e}) — reconnecting')
                        time.sleep(2.0)
                finally:
                    if lidar is not None:
                        try:
                            lidar.stop()
                            lidar.stop_motor()
                            lidar.disconnect()
                        except Exception:
                            pass

        def shutdown(self):
            self._stop = True
            self.thread.join(timeout=3.0)

    return rclpy, RPLidarNode


def main(args=None):
    rclpy, NodeCls = _make_node()
    rclpy.init(args=args)
    node = None
    try:
        node = NodeCls()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
