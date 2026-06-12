"""
Data logger — record the signals tools/sysid_report.py fits.
============================================================

Writes a flat CSV (one row per message, NaN for absent fields) of the
commanded drive, IMU, and wheel telemetry while you drive the calibration
patterns (S-turns for delay, straights for trim, steady runs for erpm
gain).  Keep runs short (~30-60 s); analysis is offline:

    ros2 run f1tenth_gym_ros data_logger --ros-args -p out:=/tmp/run1.csv
    python3 tools/sysid_report.py /tmp/run1.csv

Columns: t, steer_cmd, speed_cmd, gyro_z, accel_x, wheel_speed, erpm_speed
(wheel_speed = /vesc/odom twist; pf_speed from /pf/pose/odom if running).
"""

import csv
import math


def _make_node():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Imu
    from nav_msgs.msg import Odometry
    from ackermann_msgs.msg import AckermannDriveStamped

    class DataLogger(Node):
        FIELDS = ['t', 'steer_cmd', 'speed_cmd', 'gyro_z', 'accel_x',
                  'wheel_speed', 'pf_speed']

        def __init__(self):
            super().__init__('data_logger')
            self.declare_parameter('out', '/tmp/sysid_run.csv')
            self.declare_parameter('drive_topic', '/drive')
            self.declare_parameter('imu_topic', '/oakd/imu')
            self.declare_parameter('wheel_odom_topic', '/vesc/odom')
            self.declare_parameter('pf_odom_topic', '/pf/pose/odom')
            p = lambda n: self.get_parameter(n).value   # noqa: E731
            self._f = open(p('out'), 'w', newline='')
            self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS)
            self._w.writeheader()
            self._n = 0
            self.create_subscription(AckermannDriveStamped, p('drive_topic'),
                                     self._drive, 10)
            self.create_subscription(Imu, p('imu_topic'), self._imu, 50)
            self.create_subscription(Odometry, p('wheel_odom_topic'),
                                     self._wheel, 10)
            self.create_subscription(Odometry, p('pf_odom_topic'),
                                     self._pf, 10)
            self.get_logger().info(f"logging to {p('out')}")

        def _t(self):
            return self.get_clock().now().nanoseconds / 1e9

        def _row(self, **kw):
            row = {k: math.nan for k in self.FIELDS}
            row.update(kw, t=self._t())
            self._w.writerow(row)
            self._n += 1
            if self._n % 500 == 0:
                self._f.flush()
                self.get_logger().info(f'{self._n} rows')

        def _drive(self, m):
            self._row(steer_cmd=m.drive.steering_angle,
                      speed_cmd=m.drive.speed)

        def _imu(self, m):
            self._row(gyro_z=m.angular_velocity.z,
                      accel_x=m.linear_acceleration.x)

        def _wheel(self, m):
            self._row(wheel_speed=m.twist.twist.linear.x)

        def _pf(self, m):
            self._row(pf_speed=math.hypot(m.twist.twist.linear.x,
                                          m.twist.twist.linear.y))

        def shutdown(self):
            self._f.flush()
            self._f.close()

    return rclpy, DataLogger


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
