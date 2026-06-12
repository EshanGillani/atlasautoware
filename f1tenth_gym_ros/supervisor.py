"""
Supervisor — the watchdog-of-watchdogs: whole-system health monitor.
====================================================================

Safety in this stack is distributed and local: drive_node has a command
watchdog and the RC gate, raceline_mpc carries the AEB and the traction
governor, the velocity EKF flags wheel slip, the particle filter reports its
own confidence in its published covariance.  Each guards its own corner;
nothing watches the WHOLE system.  A dead lidar, a diverged particle filter,
a stalled control loop or a stale IMU degrade silently until the AEB — or a
wall — catches it.

This node closes that gap (ForzaETH-state-machine flavored):

  scan/pose/imu/drive arrival ──► HealthMonitor ──► OK        pass through
  PF pose covariance          ──► (pure, tested)──► DEGRADED  speed-scale 0.4
  EKF slip flag (optional)    ──►                ──► EMERGENCY car neutrals

  - per-channel rate/staleness tracking against configured minimum rates
    (scan >= 5 Hz, pose >= 5 Hz, imu >= 50 Hz, drive output >= 10 Hz): a
    dead channel is caught fast by a staleness gap (~2 beat periods at the
    minimum rate), a slow-but-alive one by a windowed rate measurement;
  - value monitors: PF position covariance above threshold = localization
    degraded (hysteretic trip/release so a value riding the threshold can't
    chatter); EKF slip flag sustained beyond `slip_degraded_after` =
    traction degraded, escalating to EMERGENCY if it never clears;
  - recovery hysteresis: any alarm holds until the system has been fully
    healthy for `recover_after` s — an input oscillating around its
    threshold produces ONE alarm and one recovery, not flapping.

Policy mapping (the thin node around the pure core):

  OK        -> /supervisor/speed_scale 1.0, /supervisor/enable True
  DEGRADED  -> speed_scale `degraded_scale` (default 0.4), reasons at 1 Hz
  EMERGENCY -> /supervisor/enable False — drive_node's enable gate accepts
               any Bool topic, so pointing it here neutrals the car

raceline_mpc consumes the scale through its optional `speed_scale_topic`
parameter (an extra v_scale factor).  If `rc_topic` is set (e.g.
`/autonomy_enabled` from rc_monitor), the RC state is ANDed into
`/supervisor/enable`, so drive_node keeps the RC kill switch while gaining
the supervisor — it only has the one Bool gate.

Pure logic in HealthMonitor below (unit-tested, no ROS); scripted fault
scenarios with latency/false-alarm numbers in tools/benchmark_supervisor.py.

Run:
    ros2 run f1tenth_gym_ros supervisor --ros-args --params-file config/hardware.yaml
"""

import time
from collections import deque

# severity / system-state codes (DEGRADED and EMERGENCY double as both)
OK, DEGRADED, EMERGENCY = 'OK', 'DEGRADED', 'EMERGENCY'
_SEV = {OK: 0, DEGRADED: 1, EMERGENCY: 2}

# channel -> (minimum rate Hz, severity when it fails).  A silent lidar,
# localization or actuation pipeline means the car is blind/uncontrolled:
# EMERGENCY.  A silent IMU only blinds the traction governor / de-skew: the
# car can still drive, slowly: DEGRADED.
DEFAULT_RATES = {
    'scan':  (5.0,  EMERGENCY),
    'pose':  (5.0,  EMERGENCY),
    'imu':   (50.0, DEGRADED),
    'drive': (10.0, EMERGENCY),
}


class HealthMonitor:
    """Pure system-health core: heartbeats + scalar reports -> state.

    Feed it `note(channel, t)` on every message arrival and
    `report('pf_cov'|'slip', value, t)` for the value monitors; call
    `update(t)` periodically (any rate >= a few Hz) to get
    `(state, reasons)` with state in {'OK', 'DEGRADED', 'EMERGENCY'}.
    All times are seconds on one monotonic clock; no wall clock is read.
    """

    def __init__(self, rates=None,
                 cov_threshold=0.5, cov_release=0.6,
                 slip_degraded_after=0.5, slip_emergency_after=3.0,
                 recover_after=2.0, startup_grace=5.0,
                 window=2.0, stale_factor=2.0, min_stale=0.1):
        self.rates = dict(DEFAULT_RATES if rates is None else rates)
        self.cov_threshold = float(cov_threshold)
        self.cov_release = float(cov_release)        # release at thr*release
        self.slip_degraded_after = float(slip_degraded_after)
        self.slip_emergency_after = float(slip_emergency_after)
        self.recover_after = float(recover_after)
        self.startup_grace = float(startup_grace)
        self.window = float(window)                  # rate-measurement window
        self.stale_factor = float(stale_factor)      # beats of silence = dead
        self.min_stale = float(min_stale)            # floor for fast channels
        self._beats = {ch: deque() for ch in self.rates}
        self._first = {ch: None for ch in self.rates}  # first beat ever
        self._last = {ch: None for ch in self.rates}   # newest beat
        self._t0 = None                              # first time ever seen
        self._cov_bad = False
        self._cov = 0.0
        self._slip_since = None
        self._last_fault_t = None                    # newest DEGRADED+ fault
        self._last_emerg_t = None                    # newest EMERGENCY fault
        self._last_reasons = []
        self.state = OK
        self.reasons = []

    # ── inputs ──────────────────────────────────────────────────────────────
    def note(self, channel, t):
        """One message arrived on `channel` at time t (heartbeat)."""
        self._touch(t)
        if channel not in self._beats:
            return
        if self._first[channel] is None:
            self._first[channel] = t
        self._last[channel] = t
        dq = self._beats[channel]
        dq.append(t)
        horizon = t - self.window
        while dq and dq[0] <= horizon:
            dq.popleft()

    def report(self, channel, value, t):
        """Scalar monitor input: 'pf_cov' (PF position variance, m^2) or
        'slip' (truthy = EKF slip flag currently raised)."""
        self._touch(t)
        if channel == 'pf_cov':
            self._cov = float(value)
            if self._cov > self.cov_threshold:
                self._cov_bad = True                 # trips above threshold,
            elif self._cov < self.cov_threshold * self.cov_release:
                self._cov_bad = False                # releases well below it
        elif channel == 'slip':
            if value:
                if self._slip_since is None:
                    self._slip_since = t
            else:
                self._slip_since = None

    # ── evaluation ──────────────────────────────────────────────────────────
    def _touch(self, t):
        if self._t0 is None:
            self._t0 = t

    def _stale_after(self, min_hz):
        return max(self.stale_factor / min_hz, self.min_stale)

    def measured_rate(self, channel, t):
        """Beats/s over the window (None until a full window of history)."""
        first = self._first[channel]
        if first is None or t - first < self.window:
            return None
        dq = self._beats[channel]
        horizon = t - self.window
        while dq and dq[0] <= horizon:
            dq.popleft()
        return len(dq) / self.window

    def _faults(self, t):
        out = []                                     # (severity, reason)
        for ch, (min_hz, sev) in self.rates.items():
            last = self._last[ch]
            if last is None:                         # never heard from it
                if t - self._t0 > self.startup_grace:
                    out.append((sev, f'{ch}: no messages '
                                     f'{t - self._t0:.1f} s after start'))
                continue
            gap = t - last
            if gap > self._stale_after(min_hz):
                out.append((sev, f'{ch}: stale ({gap:.2f} s silent, '
                                 f'min {min_hz:g} Hz)'))
                continue
            rate = self.measured_rate(ch, t)
            # one-beat tolerance so running exactly at the minimum is legal
            if rate is not None and rate * self.window < min_hz * self.window - 1.0:
                out.append((sev, f'{ch}: rate {rate:.1f} Hz '
                                 f'< min {min_hz:g} Hz'))
        if self._cov_bad:
            out.append((DEGRADED, f'localization degraded (pf position var '
                                  f'{self._cov:.3f} > {self.cov_threshold:g})'))
        if self._slip_since is not None:
            dur = t - self._slip_since
            if dur >= self.slip_emergency_after:
                out.append((EMERGENCY,
                            f'slip sustained {dur:.1f} s — traction emergency'))
            elif dur >= self.slip_degraded_after:
                out.append((DEGRADED, f'wheel slip sustained {dur:.1f} s'))
        return out

    def update(self, t):
        """Evaluate at time t -> (state, reasons).  recover_after s of full
        health are required before an alarm releases (no flapping)."""
        self._touch(t)
        faults = self._faults(t)
        if faults:
            self._last_fault_t = t
            self._last_reasons = [r for _, r in faults]
            if any(_SEV[s] >= _SEV[EMERGENCY] for s, _ in faults):
                self._last_emerg_t = t
        held = lambda since: (since is not None                  # noqa: E731
                              and t - since < self.recover_after)
        if held(self._last_emerg_t):
            self.state = EMERGENCY
        elif held(self._last_fault_t):
            self.state = DEGRADED
        else:
            self.state = OK
        if faults:
            self.reasons = self._last_reasons
        elif self.state != OK:
            self.reasons = ['recovering (%.1f s hold): %s' % (
                self.recover_after, '; '.join(self._last_reasons))]
        else:
            self.reasons = []
        return self.state, self.reasons


def policy(state, degraded_scale=0.4):
    """State -> (enable, speed_scale).  OK passes through, DEGRADED caps
    speed, EMERGENCY drops the drive_node enable gate (car neutrals)."""
    if state == EMERGENCY:
        return False, 0.0
    if state == DEGRADED:
        return True, float(degraded_scale)
    return True, 1.0


# ─────────────────────────────────────────────────────────────────────────────
# ROS node (rclpy imported lazily — pure core stays testable anywhere)
# ─────────────────────────────────────────────────────────────────────────────

def _make_node():
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Bool, Float32, String
    from sensor_msgs.msg import LaserScan, Imu
    from nav_msgs.msg import Odometry
    from ackermann_msgs.msg import AckermannDriveStamped

    class SupervisorNode(Node):
        def __init__(self):
            super().__init__('supervisor')
            # watched channels ('' disables a channel and its rate rule)
            self.declare_parameter('scan_topic', '/scan')
            self.declare_parameter('pose_topic', '/pf/pose/odom')
            self.declare_parameter('imu_topic', '/oakd/imu')
            self.declare_parameter('drive_topic', '/drive')
            self.declare_parameter('slip_topic', '')   # Bool slip flag; '' off
            self.declare_parameter('rc_topic', '')     # AND /autonomy_enabled
            self.declare_parameter('rc_timeout', 1.0)  # stale RC reads False
            # minimum rates (Hz)
            self.declare_parameter('scan_min_hz', 5.0)
            self.declare_parameter('pose_min_hz', 5.0)
            self.declare_parameter('imu_min_hz', 50.0)
            self.declare_parameter('drive_min_hz', 10.0)
            # value monitors / hysteresis
            self.declare_parameter('cov_threshold', 0.5)   # m^2 position var
            self.declare_parameter('slip_degraded_after', 0.5)
            self.declare_parameter('slip_emergency_after', 3.0)
            self.declare_parameter('recover_after', 2.0)
            self.declare_parameter('startup_grace', 5.0)
            # policy / outputs
            self.declare_parameter('degraded_scale', 0.4)
            self.declare_parameter('publish_hz', 20.0)
            self.declare_parameter('enable_out', '/supervisor/enable')
            self.declare_parameter('scale_out', '/supervisor/speed_scale')
            self.declare_parameter('diag_out', '/supervisor/diagnostics')
            p = lambda n: self.get_parameter(n).value   # noqa: E731

            subs = (('scan', p('scan_topic'), LaserScan, p('scan_min_hz'),
                     EMERGENCY),
                    ('pose', p('pose_topic'), Odometry, p('pose_min_hz'),
                     EMERGENCY),
                    ('imu', p('imu_topic'), Imu, p('imu_min_hz'), DEGRADED),
                    ('drive', p('drive_topic'), AckermannDriveStamped,
                     p('drive_min_hz'), EMERGENCY))
            rates = {ch: (float(hz), sev)
                     for ch, topic, _, hz, sev in subs if topic}
            self.hm = HealthMonitor(
                rates=rates,
                cov_threshold=float(p('cov_threshold')),
                slip_degraded_after=float(p('slip_degraded_after')),
                slip_emergency_after=float(p('slip_emergency_after')),
                recover_after=float(p('recover_after')),
                startup_grace=float(p('startup_grace')))
            for ch, topic, msg_t, _, _ in subs:
                if topic:
                    cb = self._pose_cb if ch == 'pose' else \
                        (lambda m, c=ch: self.hm.note(c, time.monotonic()))
                    self.create_subscription(msg_t, topic, cb, 10)
            if p('slip_topic'):
                self.create_subscription(
                    Bool, p('slip_topic'),
                    lambda m: self.hm.report('slip', m.data,
                                             time.monotonic()), 10)
            self._rc_ok = True                       # no RC topic = no veto
            self._rc_t = 0.0
            if p('rc_topic'):
                self._rc_ok = False                  # gated: disarmed at boot
                self.create_subscription(Bool, p('rc_topic'),
                                         self._rc_cb, 10)

            self.degraded_scale = float(p('degraded_scale'))
            self.pub_enable = self.create_publisher(Bool, p('enable_out'), 10)
            self.pub_scale = self.create_publisher(Float32, p('scale_out'), 10)
            self.pub_diag = self.create_publisher(String, p('diag_out'), 10)
            self._prev_state = None
            self._log_every = max(1, int(float(p('publish_hz'))))  # 1 Hz
            self._tick_n = 0
            self.create_timer(1.0 / float(p('publish_hz')), self._tick)
            self.get_logger().info(
                'supervisor ready — watching ' + ', '.join(
                    f'{ch}({hz:g}Hz)' for ch, (hz, _) in rates.items())
                + f" -> {p('enable_out')}, {p('scale_out')}")

        def _pose_cb(self, m):
            t = time.monotonic()
            self.hm.note('pose', t)
            # PF publishes its weight-spread confidence as position variance
            self.hm.report('pf_cov', m.pose.covariance[0], t)

        def _rc_cb(self, m):
            self._rc_ok = bool(m.data)
            self._rc_t = time.monotonic()

        def _tick(self):
            now = time.monotonic()
            state, reasons = self.hm.update(now)
            enable, scale = policy(state, self.degraded_scale)
            if self.get_parameter('rc_topic').value:
                rc = self._rc_ok and now - self._rc_t <= float(
                    self.get_parameter('rc_timeout').value)
                enable = enable and rc               # stale RC reads False
            self.pub_enable.publish(Bool(data=bool(enable)))
            self.pub_scale.publish(Float32(data=float(scale)))
            text = state + (': ' + '; '.join(reasons) if reasons else '')
            if state != self._prev_state:
                log = self.get_logger().error if state == EMERGENCY else \
                    self.get_logger().warning if state == DEGRADED else \
                    self.get_logger().info
                log(f'supervisor: {text}')
                self._prev_state = state
            if self._tick_n % self._log_every == 0:  # 1 Hz
                self.pub_diag.publish(String(data=text))
                if state != OK:
                    (self.get_logger().error if state == EMERGENCY
                     else self.get_logger().warning)(f'supervisor: {text}')
            self._tick_n += 1

    return rclpy, SupervisorNode


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
