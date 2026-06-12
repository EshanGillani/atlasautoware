# Racing tech — what's implemented and where it comes from

The control stack adopts the techniques the winning F1TENTH/RoboRacer teams
(ForzaETH / ETH Zurich PBL, TUM) published in 2023–2025. Compute is split by
what benefits: control/planning run in pure numpy on CPU
(microseconds-to-milliseconds per tick), while neural perception uses the
Jetson's GPU or the OAK-D's onboard VPU — see "GPU-accelerated perception"
below.

## MAP controller (`f1tenth_gym_ros/map_controller.py`)

**Model- and Acceleration-based Pursuit** — Becker et al., ICRA 2023
([arXiv:2209.04346](https://arxiv.org/abs/2209.04346)), the controller that
won the 2022 F1TENTH AGP with ~4× lower lateral error than pure pursuit at
speed. It keeps pure pursuit's lookahead geometry but converts the lookahead
angle to a *desired lateral acceleration* (L1 guidance) and inverts a
tire-aware steady-state map `a_lat(steer, v) → steer`:

```
L     = clip(0.15 + 0.3·v_target, 0.3, 5.0)
eta   = asin( lateral component of the lookahead vector )
a_des = 2·v_target²·sin(eta) / L
steer = LUT⁻¹(a_des, v)          # kinematic atan(L_wb·a/v²) below 1 m/s
```

The LUT is generated at startup (~0.1 s, vectorized) by integrating a dynamic
single-track model with the f1tenth_gym tire parameters to steady state over
a steer×speed grid; cells that never converge are past the grip limit and the
inversion clamps there — which is the grip-limit steering. In this stack MAP
is `raceline_mpc`'s **fallback**: any tick the OSQP solve fails (or osqp is
missing entirely) the car gets MAP instead of plain pure pursuit, so the
degraded mode is itself competition-grade.

## Friction-limited velocity profile (`f1tenth_gym_ros/velocity_profiler.py`)

The TUMFTM forward-backward pass (Heilmeier et al., Vehicle System Dynamics
2020; [TUMFTM/trajectory_planning_helpers](https://github.com/TUMFTM/trajectory_planning_helpers)),
also used by the ForzaETH race stack: start from the lateral-grip speed
`v = √(a_lat_max/|κ|)`, then a forward pass bounded by engine acceleration and
a backward pass bounded by braking, with longitudinal and lateral demands
coupled through the friction ellipse. Closed tracks run each pass twice
around so the start/finish constraint wraps the loop.

Use it two ways:

```bash
# bake into the CSV (prints the estimated lap-time change):
python3 tools/reprofile_raceline.py racelines/best_raceline.csv --a-lat 6.0
# or at runtime: raceline_mpc parameter `reprofile_speeds: true`
# (on by default in config/hardware.yaml, budget = max_lat_accel)
```

This guarantees the speed the controller asks for is *achievable* — no more
mid-corner overspeed for the traction governor to catch after the fact. The
governor (live, IMU) and the profiler (planned, curvature) now share one
`max_lat_accel` budget: the profiler plans to it, the governor enforces it.

## Already in the stack (same lineage)

- **Kinematic LTV-MPC** raceline tracker with a persistent warm-started OSQP
  problem (~0.6 ms/solve) — `docs/mpc.md`.
- **IMU traction governor** (a_lat ≈ |gyro z|·v) and speed-aware AEB —
  `docs/hardware.md`.

## State estimation (`velocity_ekf.py`, `scan_deskew.py`)

- **Velocity EKF with slip rejection** — fuses the OAK-D IMU (200 Hz
  prediction) with VESC wheel speed, plus a nonholonomic pseudo-measurement
  that makes lateral velocity observable. Slip is detected by comparing the
  wheel against an IMU-anchored velocity integral (innovation gating alone
  is defeated by slip *ramps*), with hysteresis and a rejection timeout.
  Synthetic benchmark (25% slip pulses): speed RMSE during slip 1.05 → 0.05
  m/s (20×), 100% detection, 0.2% false positives. Publishes `/ekf/odom`
  for the controller, the latency predictor, and a future particle filter.
- **Lidar de-skew** — a 10 Hz scan sweeps for ~100 ms; at 7 m/s that is
  0.7 m of travel *within one scan*. `scan_deskew` rotates/translates every
  beam by the EKF twist times the beam's age (exact on synthetic truth:
  wall-distance RMS 0.31 m → 0.00). Wired into `rplidar_node` via
  `deskew_odom_topic`.

## GPU-accelerated perception (`camera_perception.py`, `oakd_camera.py`)

YOLO opponent detection picks the fastest available backend at startup
(`backend: auto` in `config/hardware.yaml`):

1. **TensorRT (Jetson GPU, preferred)** — build the engine once on the car:
   `trtexec --onnx=models/car_yolov8.onnx --saveEngine=models/car_yolov8.engine --fp16`
   (engines are device-specific; rebuild after JetPack upgrades). `auto`
   picks up the `.engine` automatically.
2. **cv2.dnn CUDA** — the ONNX through OpenCV's CUDA FP16 target (Jetson
   OpenCV builds ship CUDA-enabled).
3. **CPU** — always-works fallback.

Alternatively set `yolo_blob` in the `oakd_camera` section to run the
detector **on the camera's Myriad X VPU** (convert the ONNX with
`blobconverter`, e.g. 416×416): the host receives only boxes, back-projected
to relative opponent poses on `/oakd/opponents_rel`, costing the Jetson
nothing. The control loop stays on CPU deliberately — a 76-variable QP at
0.6 ms gains nothing from offload.

## Adopted-next candidates (researched, not yet implemented)

1. **Frenet-frame "spliner" overtaker** (ForzaETH race stack, JFR 2024,
   [arXiv:2403.11784](https://arxiv.org/abs/2403.11784)) — cubic spline in
   (s, d) around a detected opponent; plugs into the existing YOLO/lidar
   detection. ~100 lines + a Frenet conversion.
2. **Race-time localization swap**: frozen map + particle filter (SynPF,
   [arXiv:2401.07658](https://arxiv.org/abs/2401.07658)) fed by an EKF fusing
   VESC wheel odometry (`/vesc/odom`) + OAK-D IMU — robust to wheel slip
   where scan-matching SLAM degrades, 1.25 ms CPU.
3. **Predictive spliner** (RA-L 2025) once 1 works.

(Considered and skipped: MPCC — needs acados and duplicates the working
LTV-MPC; TinyLidarNet end-to-end CNN — inferior to model-based on a mapped
track; RL-tuned lookahead — MAP's linear schedule captures most of it.)
