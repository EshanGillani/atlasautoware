"""
RC mode toggle — a 2.4 GHz transmitter switch arms/disarms autonomy.
====================================================================

Reads one channel of the existing RC receiver (a spare switch channel on
the plane transmitter) as a PWM pulse on a Jetson GPIO pin and publishes
`/autonomy_enabled` (std_msgs/Bool).  The drive node refuses to actuate
while this is False or stale — transmitter off, out of range, switch down,
node crashed: all land on "disabled", which the drive node turns into
neutral.  Fail-safe by construction on the software side.

  receiver AUX channel ──► Jetson GPIO (3.3 V logic!) ──► /autonomy_enabled
  pulse < `manual_below` us  -> manual (autonomy OFF)
  pulse > `auto_above`  us   -> autonomous (ON)
  in between / no pulses     -> keep previous / OFF when stale

IMPORTANT — software gating is the *second* line of defence.  For true
manual override wire a hardware PWM multiplexer (e.g. Pololu 4-channel RC
mux): receiver steer/throttle into input A, PCA9685 outputs into input B,
mux SELECT from the same transmitter switch.  Then the switch hands the
servos to the human even if every computer on the car is dead.  See
docs/selfdriving.md for wiring.  Most receivers output 5 V pulses — level
shift (or use a 1k/2k divider) before the Jetson's 3.3 V GPIO.

Pulse classification is pure and unit-tested; the GPIO edge timing uses
Jetson.GPIO and tolerates user-space jitter (we only need a 3-position
switch, not precision).
"""

import time


class PulseClassifier:
    """Hysteretic 2-state classifier over RC pulse widths, stale-aware."""

    def __init__(self, manual_below=1300.0, auto_above=1700.0,
                 stale_after=0.5):
        self.manual_below = float(manual_below)
        self.auto_above = float(auto_above)
        self.stale_after = float(stale_after)
        self.enabled = False                       # boot disarmed
        self._last_t = None

    def feed(self, width_us, t):
        """One measured pulse (us) at time t (s) -> enabled state."""
        if width_us >= self.auto_above:
            self.enabled = True
        elif width_us <= self.manual_below:
            self.enabled = False
        # widths between the thresholds keep the previous state (hysteresis)
        self._last_t = float(t)
        return self.enabled

    def state(self, now):
        """Stale pulses (transmitter off / out of range) read as disabled."""
        if self._last_t is None or now - self._last_t > self.stale_after:
            return False
        return self.enabled


def _make_node():
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Bool

    class RCMonitor(Node):
        def __init__(self):
            super().__init__('rc_monitor')
            self.declare_parameter('gpio_pin', 18)      # BOARD numbering
            self.declare_parameter('enable_topic', '/autonomy_enabled')
            self.declare_parameter('manual_below_us', 1300.0)
            self.declare_parameter('auto_above_us', 1700.0)
            self.declare_parameter('stale_after', 0.5)
            self.declare_parameter('publish_hz', 20.0)
            p = lambda n: self.get_parameter(n).value   # noqa: E731

            self.clf = PulseClassifier(float(p('manual_below_us')),
                                       float(p('auto_above_us')),
                                       float(p('stale_after')))
            self.pub = self.create_publisher(Bool, p('enable_topic'), 10)
            self._rise_ns = None
            self._last_state = None
            try:
                import Jetson.GPIO as GPIO
                self.GPIO = GPIO
                GPIO.setmode(GPIO.BOARD)
                self.pin = int(p('gpio_pin'))
                GPIO.setup(self.pin, GPIO.IN)
                GPIO.add_event_detect(self.pin, GPIO.BOTH,
                                      callback=self._edge)
                self.get_logger().info(
                    f"rc_monitor ready — pin {self.pin} -> {p('enable_topic')} "
                    f"(<{p('manual_below_us'):.0f}us manual, "
                    f">{p('auto_above_us'):.0f}us auto)")
            except Exception as e:                      # not on a Jetson
                self.GPIO = None
                self.get_logger().error(
                    f'GPIO unavailable ({e}) — publishing DISABLED; '
                    f'autonomy stays off until rc_monitor runs on the car')
            self.create_timer(1.0 / float(p('publish_hz')), self._tick)

        def _edge(self, _ch):
            now = time.monotonic_ns()
            if self.GPIO.input(self.pin):               # rising edge
                self._rise_ns = now
            elif self._rise_ns is not None:             # falling: full pulse
                width_us = (now - self._rise_ns) / 1e3
                if 500.0 < width_us < 2500.0:           # plausible RC pulse
                    self.clf.feed(width_us, now / 1e9)

        def _tick(self):
            from std_msgs.msg import Bool
            state = self.clf.state(time.monotonic_ns() / 1e9)
            self.pub.publish(Bool(data=bool(state)))
            if state != self._last_state:
                self.get_logger().warning(
                    f"autonomy {'ARMED' if state else 'DISARMED'} (RC switch)")
                self._last_state = state

        def shutdown(self):
            if self.GPIO is not None:
                try:
                    self.GPIO.cleanup()
                except Exception:
                    pass

    return rclpy, RCMonitor


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
