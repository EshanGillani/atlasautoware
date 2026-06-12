"""
Unified drive node — one /drive endpoint, two interchangeable backends.
=======================================================================

Subscribes to `AckermannDriveStamped` and actuates the car through whichever
hardware path it finds at startup:

  backend "pca9685"  I2C PWM board -> VESC PPM input (+ optional servo ch)
  backend "vesc"     direct VESC UART -> SET_RPM / SET_SERVO_POS,
                     plus free telemetry: GET_VALUES -> /vesc/odom (wheel
                     speed for the particle filter / MPC speed state)

With `backend: auto` (default) the node probes both: reads the PCA9685's
MODE1 register over I2C, and asks the serial port for the VESC firmware
version.  Whichever answers wins (PCA9685 first — if you wired the PWM board
you meant to use it); the same launch file therefore runs unchanged on either
wiring.  Set `backend` explicitly to skip probing.

Safety on both paths:
  - arming hold (neutral for `arm_time` s before commands are accepted),
  - command watchdog (`cmd_timeout` s without /drive -> neutral / zero
    current; the VESC's own app timeout is a second line of defence),
  - neutral on clean shutdown.

Pure pulse maths lives in pca9685.py, the UART protocol in vesc_protocol.py —
both unit-tested without hardware (tests/test_hardware.py).

Run:
    ros2 run f1tenth_gym_ros drive_node --ros-args --params-file config/hardware.yaml
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pca9685 as pca
import vesc_protocol as vp


# ─────────────────────────────────────────────────────────────────────────────
# Backends — same interface, picked at startup
# ─────────────────────────────────────────────────────────────────────────────

class PCA9685Backend:
    """Throttle pulses into the VESC PPM input, steering on a second channel."""

    name = 'pca9685'
    has_telemetry = False

    def __init__(self, bus, cfg):
        self.cfg = cfg
        self.dev = pca.PCA9685(bus, cfg['i2c_address'], cfg['pwm_hz'])
        self.ch_thr = int(cfg['throttle_channel'])
        self.ch_str = int(cfg['steer_channel'])

    def command(self, speed, steer):
        c = self.cfg
        self.dev.set_pulse_us(self.ch_thr, pca.speed_to_us(
            speed, c['max_speed'], c['neutral_us'],
            c['full_fwd_us'], c['full_rev_us']))
        if self.ch_str >= 0:
            self.dev.set_pulse_us(self.ch_str, pca.steer_to_us(
                steer, c['max_steer'], c['steer_center_us'],
                c['steer_half_range_us'], c['steer_invert'], c['steer_trim_us']))

    def neutral(self):
        self.dev.set_pulse_us(self.ch_thr, self.cfg['neutral_us'])
        if self.ch_str >= 0:
            self.dev.set_pulse_us(
                self.ch_str,
                self.cfg['steer_center_us'] + self.cfg['steer_trim_us'])

    def stop(self):
        self.neutral()
        time.sleep(0.05)
        self.dev.set_off(self.ch_thr)
        if self.ch_str >= 0:
            self.dev.set_off(self.ch_str)


class VescSerialBackend:
    """Direct UART: closed-loop SET_RPM + the VESC's own servo header."""

    name = 'vesc'
    has_telemetry = True

    def __init__(self, ser, cfg):
        self.ser = ser
        self.cfg = cfg
        self.erpm_gain = float(cfg['erpm_gain'])
        self.parser = vp.PacketParser()

    def command(self, speed, steer):
        self.ser.write(vp.pkt_set_rpm(speed * self.erpm_gain))
        c = self.cfg
        frac = max(-1.0, min(1.0, float(steer) / float(c['max_steer'])))
        if c['steer_invert']:
            frac = -frac
        # trim expressed in pulse us for parity with the PCA path: ~1000us span
        pos = 0.5 + 0.5 * frac + float(c['steer_trim_us']) / 1000.0
        self.ser.write(vp.pkt_set_servo_pos(pos))

    def neutral(self):
        self.ser.write(vp.pkt_set_current(0.0))

    def stop(self):
        self.neutral()

    def poll_telemetry(self):
        """Request GET_VALUES, drain the port; returns latest dict or None."""
        self.ser.write(vp.pkt_request(vp.COMM_GET_VALUES))
        values = None
        waiting = self.ser.in_waiting
        if waiting:
            for payload in self.parser.feed(self.ser.read(waiting)):
                parsed = vp.parse_values(payload)
                if parsed is not None:
                    values = parsed
        return values


# ─────────────────────────────────────────────────────────────────────────────
# Detection — probe I2C for a PCA9685, the serial port for a VESC
# ─────────────────────────────────────────────────────────────────────────────

def probe_pca9685(cfg, log):
    try:
        bus = pca.open_i2c(cfg['i2c_bus'])
        bus.read_byte_data(int(cfg['i2c_address']), pca.PCA9685.MODE1)
        return bus
    except Exception as e:
        log(f"no PCA9685 on i2c-{cfg['i2c_bus']} @0x{int(cfg['i2c_address']):02x}: {e}")
        return None


def probe_vesc(cfg, log):
    try:
        import serial
        ser = serial.Serial(cfg['serial_port'], int(cfg['serial_baud']),
                            timeout=0.1)
        parser = vp.PacketParser()
        for _ in range(3):                       # fw-version handshake
            ser.write(vp.pkt_request(vp.COMM_FW_VERSION))
            time.sleep(0.1)
            for payload in parser.feed(ser.read(ser.in_waiting or 1)):
                if payload and payload[0] == vp.COMM_FW_VERSION:
                    return ser
        ser.close()
        log(f"no VESC reply on {cfg['serial_port']}")
    except Exception as e:
        log(f"no VESC on {cfg['serial_port']}: {e}")
    return None


def pick_backend(prefer, cfg, log):
    """prefer in ('auto', 'pca9685', 'vesc') -> backend instance or None."""
    if prefer in ('auto', 'pca9685'):
        bus = probe_pca9685(cfg, log)
        if bus is not None:
            return PCA9685Backend(bus, cfg)
        if prefer == 'pca9685':
            return None
    if prefer in ('auto', 'vesc'):
        ser = probe_vesc(cfg, log)
        if ser is not None:
            return VescSerialBackend(ser, cfg)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ROS node
# ─────────────────────────────────────────────────────────────────────────────

def _make_node():
    import rclpy
    from rclpy.node import Node
    from ackermann_msgs.msg import AckermannDriveStamped
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Bool

    class DriveNode(Node):
        def __init__(self):
            super().__init__('drive_node')
            self.declare_parameter('drive_topic', '/drive')
            self.declare_parameter('backend', 'auto')   # auto | pca9685 | vesc
            # shared actuation limits / calibration
            self.declare_parameter('max_speed', 7.0)    # m/s at full throttle
            self.declare_parameter('max_steer', 0.41)   # rad at full servo throw
            self.declare_parameter('steer_invert', False)
            self.declare_parameter('steer_trim_us', 0.0)
            self.declare_parameter('cmd_timeout', 0.5)
            self.declare_parameter('arm_time', 2.0)
            # pca9685 path
            self.declare_parameter('i2c_bus', 1)        # check `i2cdetect -l`
            self.declare_parameter('i2c_address', 0x40)
            self.declare_parameter('pwm_hz', 50.0)      # VESC PPM is happy <=200
            self.declare_parameter('throttle_channel', 0)
            self.declare_parameter('steer_channel', 1)  # -1 = no steering servo
            self.declare_parameter('neutral_us', 1500.0)
            self.declare_parameter('full_fwd_us', 2000.0)
            self.declare_parameter('full_rev_us', 1000.0)
            self.declare_parameter('steer_center_us', 1500.0)
            self.declare_parameter('steer_half_range_us', 400.0)
            # vesc-uart path
            self.declare_parameter('serial_port', '/dev/ttyACM0')
            self.declare_parameter('serial_baud', 115200)
            self.declare_parameter('erpm_gain', 4614.0)  # erpm per m/s
            self.declare_parameter('odom_topic', '/vesc/odom')
            self.declare_parameter('odom_frame', 'odom')
            self.declare_parameter('base_frame', 'base_link')
            self.declare_parameter('telemetry_hz', 20.0)

            cfg = {n: self.get_parameter(n).value for n in (
                'max_speed', 'max_steer', 'steer_invert', 'steer_trim_us',
                'i2c_bus', 'i2c_address', 'pwm_hz', 'throttle_channel',
                'steer_channel', 'neutral_us', 'full_fwd_us', 'full_rev_us',
                'steer_center_us', 'steer_half_range_us',
                'serial_port', 'serial_baud', 'erpm_gain')}
            prefer = self.get_parameter('backend').value
            self.backend = pick_backend(
                prefer, cfg, lambda m: self.get_logger().info(m))
            if self.backend is None:
                raise RuntimeError(
                    f"no actuation hardware found (backend={prefer}) — "
                    f"checked PCA9685 on i2c-{cfg['i2c_bus']} and VESC on "
                    f"{cfg['serial_port']}")
            self.get_logger().info(f'backend: {self.backend.name}')

            self.timeout = float(self.get_parameter('cmd_timeout').value)
            self.arm_until = time.monotonic() + \
                float(self.get_parameter('arm_time').value)
            self.last_cmd = 0.0
            self._wd_warned = False
            self.backend.neutral()
            # RC kill-switch gate: while /autonomy_enabled is False or stale,
            # /drive is ignored and the throttle holds neutral.  '' = no gate
            # (bench use); ALWAYS gate when driving outdoors/around people.
            self.declare_parameter('enable_topic', '')
            self.declare_parameter('enable_timeout', 1.0)
            en_topic = self.get_parameter('enable_topic').value
            self.gated = bool(en_topic)
            self.enabled = not self.gated
            self._last_enable = 0.0
            if self.gated:
                self.create_subscription(Bool, en_topic, self._enable_cb, 10)
                self.get_logger().info(f'actuation gated on {en_topic}')
            self.create_subscription(
                AckermannDriveStamped,
                self.get_parameter('drive_topic').value, self._drive_cb, 1)
            self.create_timer(0.04, self._watchdog)

            if self.backend.has_telemetry:
                self.odom_pub = self.create_publisher(
                    Odometry, self.get_parameter('odom_topic').value, 10)
                self.erpm_gain = float(cfg['erpm_gain'])
                self._telem_n = 0
                self.create_timer(
                    1.0 / float(self.get_parameter('telemetry_hz').value),
                    self._telemetry)

        def _enable_cb(self, msg):
            was = self.enabled
            self.enabled = bool(msg.data)
            self._last_enable = time.monotonic()
            if was and not self.enabled:                # disarm NOW, not at WD
                try:
                    self.backend.neutral()
                except Exception:
                    pass
                self.get_logger().warning('autonomy disarmed — neutral')

        def _gate_open(self):
            if not self.gated:
                return True
            stale = time.monotonic() - self._last_enable > \
                float(self.get_parameter('enable_timeout').value)
            return self.enabled and not stale

        def _drive_cb(self, msg):
            self.last_cmd = time.monotonic()
            self._wd_warned = False
            if self.last_cmd < self.arm_until:          # arming: hold neutral
                return
            if not self._gate_open():                   # RC switch says no
                return
            try:
                self.backend.command(float(msg.drive.speed),
                                     float(msg.drive.steering_angle))
            except Exception as e:
                self.get_logger().error(f'actuation write failed: {e}')

        def _watchdog(self):
            if self.last_cmd > 0.0 and \
                    time.monotonic() - self.last_cmd > self.timeout:
                try:
                    self.backend.neutral()
                except Exception:
                    pass
                if not self._wd_warned:
                    self.get_logger().warning(
                        f'no /drive for {self.timeout:.1f}s — neutral')
                    self._wd_warned = True

        def _telemetry(self):
            try:
                values = self.backend.poll_telemetry()
            except Exception as e:
                self.get_logger().warning(f'telemetry read failed: {e}')
                return
            if values is None:
                return
            odom = Odometry()
            odom.header.stamp = self.get_clock().now().to_msg()
            odom.header.frame_id = self.get_parameter('odom_frame').value
            odom.child_frame_id = self.get_parameter('base_frame').value
            odom.twist.twist.linear.x = values['erpm'] / self.erpm_gain
            self.odom_pub.publish(odom)
            self._telem_n += 1
            if self._telem_n % 200 == 0:                 # ~every 10 s at 20 Hz
                self.get_logger().info(
                    f"vesc: {values['v_in']:.1f}V fet {values['temp_fet']:.0f}C "
                    f"fault {values['fault']}")
            if values['fault']:
                self.get_logger().error(f"VESC FAULT code {values['fault']}")

        def shutdown(self):
            try:
                self.backend.stop()
            except Exception:
                pass

    return rclpy, DriveNode


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
