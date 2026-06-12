# Hardware bringup — Jetson, OAK-D Pro, RPLidar, VESC (PCA9685 or UART)

How the sim-tested stack runs on the physical car. One launch file brings up
sensors, actuation, and the competition racing node:

```bash
ros2 launch f1tenth_gym_ros car_bringup_launch.py
# pieces individually:
ros2 launch f1tenth_gym_ros car_bringup_launch.py use_racing:=false   # bench test
```

```
RPLidar ──► rplidar_node ──► /scan ─────────────┐
OAK-D Pro ─► oakd_camera ──► /oakd/rgb ──► camera_perception (optional)
                        └──► /oakd/imu ─────────┤
localization (particle filter) ─► /pf/pose/odom ┤
                                                ▼
                                          raceline_mpc ──► /drive ──► drive_node
                                                                        │ auto-detect
                                                          ┌─────────────┴────────────┐
                                                   PCA9685 (I2C)              VESC (UART)
                                                   PWM ► VESC PPM input       SET_RPM/SET_SERVO
```

## Dependencies (on the Jetson)

```bash
pip3 install depthai rplidar-roboticia smbus2 pyserial
# depthai udev rule (once, then replug the camera):
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | \
  sudo tee /etc/udev/rules.d/80-movidius.rules && sudo udevadm control --reload-rules
```

Everything is plain pip — no extra ROS packages needed, so it works on Foxy.

## Actuation: one node, two backends

`drive_node` subscribes to `/drive` (AckermannDriveStamped) and probes for
hardware at startup (`backend: auto` in `config/hardware.yaml`):

1. **PCA9685** — reads the chip's MODE1 register on `i2c_bus`/`i2c_address`.
   If present: throttle pulses on channel 0 go to the **VESC PPM input**,
   steering pulses on channel 1 to the servo (set `steer_channel: -1` if the
   car has none). Wire SDA/SCL to the Jetson header (pins 3/5; confirm the bus
   number with `i2cdetect -l`, then `i2cdetect -y <bus>` should show `40`).
   In VESC Tool, enable the PPM app and run pulse calibration so
   1000/1500/2000 µs = full-brake/neutral/full-throttle.
2. **VESC UART** — asks `serial_port` for the firmware version. If it answers:
   closed-loop `SET_RPM` for speed (`erpm_gain` converts m/s → erpm; tune for
   your motor poles/gearing/wheel diameter), `SET_SERVO_POS` for steering on
   the VESC's own servo header. This backend additionally polls `GET_VALUES`
   and publishes wheel-speed odometry on `/vesc/odom` (feed it to your
   particle filter) plus battery/temperature/fault logging.

Set `backend:` to `pca9685` or `vesc` to pin one explicitly. Either way you
get: a 2 s neutral **arming hold** at startup, a **watchdog** that snaps to
neutral if `/drive` goes quiet for `cmd_timeout`, and neutral on shutdown.

### Running without a VESC (PCA9685 + dumb hobby ESC)

The PCA9685 backend emits standard RC servo pulses — identical to a
receiver — so an ordinary hobby ESC works as a drop-in until the VESC
arrives. Wire: PCA9685 channel 0 signal → ESC signal input (1000/1500/2000
µs full-brake-or-reverse / neutral / full), channel 1 → steering servo.
Nothing else changes; `backend: auto` finds the PCA9685, or pin
`backend: 'pca9685'`. Dumb-ESC specifics:

- **Run the ESC's own throttle calibration** first (per its manual) so its
  endpoints match 1000/1500/2000 µs, then trim `full_fwd_us` /
  `full_rev_us` / `neutral_us` if needed.
- **Reverse lockout / brake semantics**: most RC car ESCs treat the first
  below-neutral pulse as *brake*, and only reverse after returning to
  neutral. The racing stack never commands reverse, so this is usually
  what you want. If your ESC reverses immediately and you want it
  disabled, set `full_rev_us: 1500.0` (negative speeds then clamp to
  neutral).
- **Stops are coasts**: speed 0 maps to the neutral pulse — a dumb ESC
  freewheels there rather than actively braking like a VESC in
  current-brake mode. Budget for it with a larger `aeb_dist` /
  smaller `aeb_decel` until the VESC arrives.
- **Arming** is covered by the existing `arm_time` neutral hold.
- **Lost without the VESC**: `/vesc/odom` wheel-speed telemetry. The
  velocity EKF automatically falls back to the *commanded* speed as a
  weakly-trusted measurement (it bounds IMU drift; it is not real
  odometry — expect a coarser estimate). `erpm_gain`, `serial_port`, and
  `telemetry_hz` are simply unused.

### Measured calibration (recommended over hand-tuning)

Log a short drive and let the fitters produce the numbers:

```bash
ros2 run f1tenth_gym_ros data_logger --ros-args -p out:=/tmp/run1.csv
# drive ~30 s: gentle S-turns (delay), straights (trim), steady laps (gain)
python3 tools/sysid_report.py /tmp/run1.csv
```

It prints `actuation_delay` (steering↔yaw-rate cross-correlation, with a
correlation-quality gate), an `erpm_gain` correction factor (wheel vs PF
speed), and `steer_trim_us` (straight-commanded yaw drift) — paste them
into `config/hardware.yaml`.

**First-drive calibration order** (in `config/hardware.yaml`):
`steer_invert`/`steer_trim_us` until it drives straight → `max_steer` against
real wheel angle → `max_speed`/`erpm_gain` against measured speed → raise
`v_scale` in the `raceline_mpc` section last.

## Sensors

- **rplidar_node** — Slamtec serial protocol via `rplidar-roboticia`, binned
  into a fixed 720-beam, 360° `LaserScan` on `/scan` (same shape the sim
  publishes, so AEB/gap-following/SLAM run unmodified). A1/A2 use baud
  115200; A3/S-series 256000. `angle_offset` corrects mounting yaw. The read
  loop auto-reconnects if the USB link hiccups.
- **oakd_camera** — `depthai` pipeline publishing interleaved-BGR frames
  (`/oakd/rgb`, default 640×360@30) with factory-calibrated `CameraInfo`
  (what `camera_perception` needs for back-projection) and the onboard IMU at
  200 Hz on `/oakd/imu`. No cv_bridge, no per-frame conversions.

## IMU in the control loop (racing performance)

`raceline_mpc` gained two hardware-aware behaviours:

- **Traction governor** (`imu_topic`, `max_lat_accel`): the raceline's speed
  profile assumes a friction budget; the IMU measures what the car actually
  pulls (a_lat ≈ |gyro z|·v). Past the budget, commanded speed scales down and
  recovers gradually — an optimistic raceline or dusty patch costs a little
  pace instead of the wall. Sign/mounting agnostic (any flat mounting works).
- **Dynamic AEB range** (`aeb_decel`): the emergency-brake trigger distance
  grows with v²/(2·a) so the stop physically fits at speed, instead of using
  only the standstill margin.

Both are inert in sim (no `imu_topic` set; speeds low).

## Frames

`car_bringup_launch.py` publishes static `base_link → laser` and
`base_link → oakd_rgb` transforms — measure your mounts and edit the values.
