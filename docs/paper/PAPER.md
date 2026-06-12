# AtlasAutoware: An Integrated Autonomous Racing Stack for 1/10-Scale Vehicles with GPU-Accelerated Perception and Component-Wise Closed-Loop Evaluation

**Author:** Eshan Iyer
**Affiliation:** Thomas Jefferson High School for Science and Technology (TJHSST), Alexandria, VA
**Contact:** 2027eiyer@tjhsst.edu
**Code:** https://github.com/AtlasAutoware/atlasautoware

---

## Abstract

We present AtlasAutoware, an integrated software stack for 1/10-scale
autonomous racing (F1TENTH/RoboRacer class) designed for
simulation-to-hardware parity: the same control, planning, and safety nodes
drive both the F1TENTH gym simulator and a physical car built from commodity
hardware (NVIDIA Jetson, OAK-D Pro camera with integrated IMU, Slamtec
RPLidar, VESC motor controller). Compute is partitioned deliberately: the
real-time control and planning path runs on CPU in Python/numpy — its
quadratic programs are too small to benefit from offload — while neural
perception is accelerated, with three interchangeable inference paths
(a TensorRT FP16 engine on the Jetson GPU, OpenCV's CUDA DNN backend, or
fully on-camera inference on the OAK-D's Myriad X VPU at zero host cost).
The stack combines a linear-time-varying model-predictive raceline tracker
with a persistent warm-started quadratic-program formulation (0.6 ms mean
solve time, a 2.3× reduction over per-tick problem construction), a
tire-aware Model- and Acceleration-based Pursuit (MAP) fallback controller,
minimum-curvature raceline refinement, friction-ellipse-coupled velocity
profiling, IMU-based traction governance, slip-rejecting velocity
estimation, lidar motion de-skew, actuation-latency compensation, and a
Frenet-frame overtaking planner. A hardware abstraction layer auto-detects
the available actuation path (I2C PWM bridge into the motor controller's RC
input, or direct UART) at startup. Each component was
evaluated independently against a shared deterministic closed-loop
benchmark; we report component-wise results, including a 9% lap-time
reduction from raceline refinement (43.76 s → 39.78 s), a 37% reduction in
mean tracking error from curvature-aware lookahead scheduling, recovery of
near-zero-latency tracking performance under 100–140 ms of injected actuator
delay, and conversion of an opponent-blocked DNF into a completed overtake
at a cost of 0.30 s. We discuss the engineering methodology — parallel,
isolated candidate evaluation with adopt/reject decisions gated on
measurable closed-loop gains — and the limitations of kinematic-plant
validation.

**Keywords:** autonomous racing, F1TENTH, model predictive control, pure
pursuit, velocity profiling, latency compensation, low-cost robotics

---

## 1. Introduction

Small-scale autonomous racing has become a standard proving ground for
high-speed robot autonomy: 1/10-scale Ackermann-steered cars with planar
lidar and embedded compute race at speeds where control latency, tire
limits, and software robustness dominate outcomes [1, 2]. The platform's
accessibility, however, hides a practical gap. The published
state of the art — exemplified by the ForzaETH race stack [2], the MAP
controller [3], and the TUM minimum-curvature trajectory toolchain [4] — is
distributed across separate repositories, firmware-pinned drivers, and
GPU-assumed perception pipelines, while most fielded student and hobbyist
cars still run geometric followers with hand-tuned speed tables.

This paper describes AtlasAutoware, a stack built to close that gap under
three constraints:

1. **Compute partitioned by what actually benefits.** The real-time control
   and planning path is pure Python/numpy (plus the OSQP solver [5]) on CPU
   and consumes about 3% of a 50 Hz budget — its QPs are far too small for
   GPU offload to help. Neural perception, the genuinely parallel workload,
   runs on whatever accelerator is present: a TensorRT FP16 engine on the
   Jetson GPU, OpenCV's CUDA backend, or entirely on the camera's Myriad X
   VPU, selected automatically at startup with a CPU fallback. No GPU is
   *required* anywhere, and the control loop never depends on one.
2. **Simulation-to-hardware parity.** The hardware drivers publish the same
   message shapes the simulator does, so every algorithm runs unmodified on
   both. Hardware specifics (actuation path, latency, steering trim, grip
   budget) are isolated in configuration.
3. **Measured adoption.** Every candidate technique — including those drawn
   from the recent literature — was implemented behind a shared,
   deterministic closed-loop benchmark and adopted only when it produced a
   measurable gain. We report negative and trade-off results alongside
   positive ones.

Our contributions are: (i) an integrated, tested, openly available racing
stack meeting the constraints above; (ii) a persistent-QP formulation of the
standard kinematic LTV-MPC racing tracker that reduces solve cost 2.3× and,
by removing solver-tolerance noise, exposed and corrected a latent tuning
deficiency; (iii) a CPU-vectorized procedure for generating the MAP
controller's tire-compensation lookup table from a dynamic single-track
model in under a second at startup; (iv) component-wise closed-loop
measurements of five recent racing techniques on one benchmark, with
adopt/reject analysis; and (v) a dual-backend, auto-detecting actuation
architecture covering both PWM-bridge and direct-UART motor-controller
wiring with common safety envelopes.

## 2. Related Work

**Integrated race stacks.** The ForzaETH race stack [2] is the most complete
published system for this vehicle class: EKF velocity estimation, particle
filter localization on a frozen map [6], MAP control [3], sectored velocity
scaling, and a Frenet-frame "spliner" overtaker, later extended with
opponent-trajectory prediction [7]. Our stack adopts its controller and
overtaker designs (re-implemented in numpy) and differs in its dual-backend
actuation layer, its single-package pip-installable footprint, and its
explicit benchmark-gated adoption methodology.

**Tracking control.** Geometric pure pursuit remains the field default;
Becker et al. [3] showed that replacing its kinematic steering inversion
with a lateral-acceleration-based inversion of a steady-state tire map (the
MAP controller) reduces high-speed lateral error roughly fourfold.
Optimization-based trackers span LTV-MPC (this work), model-predictive
contouring [8], and learned end-to-end policies such as TinyLidarNet [9]; we
deliberately retain a model-based primary/fallback pair for inspectability.

**Trajectory generation.** Minimum-curvature raceline optimization and
forward-backward friction-limited velocity profiling follow Heilmeier et
al. [4] and the TUMFTM trajectory-planning tools [10], which the ForzaETH
stack also builds on.

**Latency.** Forward-predicting the vehicle state across the
sensor-to-actuator delay before each control solve is standard practice in
fielded race stacks [2]; we quantify its closed-loop value under controlled,
injected delay.

## 3. System Overview

```
                ┌── sim:  gym_bridge (f1tenth_gym) ───────────────┐
sensors ────────┤                                                 ├─► /scan, odom, /oakd/*
                └── car:  rplidar_node · oakd_camera · drive_node ┘
                                          ▲
map ─► SLAM ─► raceline fit ─► min-curvature refinement ─► velocity profile
                                          │
                     raceline_mpc ────────┴───────────────► /drive
                     (LTV-MPC │ MAP fallback │ AEB │ traction governor
                      │ latency compensation │ spliner overtake)
```

The racing node consumes a raceline (x, y, heading, curvature, speed),
a planar laser scan, a map-frame pose estimate, and (on hardware) an IMU
stream; it publishes Ackermann drive commands at 50 Hz. Offline, the
raceline is fit from driven laps, optionally refined for minimum curvature
(§4.3), and re-profiled for friction-feasible speeds (§4.4).

### 3.1 Hardware abstraction and actuation

The physical platform is an NVIDIA Jetson with an OAK-D Pro (RGB camera and
200 Hz IMU over the `depthai` Python API), an RPLidar (binned to a
fixed-grid 360° scan identical in shape to the simulator's), and a VESC
motor controller. A single *drive node* exposes the `/drive` topic through
one of two interchangeable backends selected by probing at startup:

- **PWM bridge:** a PCA9685 I2C PWM generator produces RC-style pulses into
  the VESC's PPM input (and optionally a steering servo channel). The probe
  reads the chip's MODE1 register.
- **Direct UART:** the VESC serial protocol (closed-loop RPM and servo
  position commands), probed via a firmware-version handshake. This backend
  additionally polls telemetry and publishes wheel-speed odometry.

Both paths share a safety envelope: a neutral arming hold at startup, a
command watchdog that returns the throttle to neutral when the command
stream stalls, and neutral-on-shutdown. All register sequencing, pulse
arithmetic, and protocol framing (CRC-16/XMODEM packetization) are pure
functions covered by hardware-free unit tests.

## 4. Methods

### 4.1 Persistent-QP LTV-MPC

The primary tracker is a standard kinematic-bicycle LTV-MPC: states
$z=[x,y,\psi,v]$, inputs $u=[\delta,a]$, horizon $N{=}12$ at
$\Delta t{=}80$ ms, linearized about a time-parametrized raceline reference
each tick and solved as a sparse QP with OSQP [5], with box constraints on
steering, acceleration, braking, and speed, and input-rate penalties for
actuator smoothness.

The optimization contribution is structural: because the horizon, weights,
and constraint topology never change between ticks, the cost matrix $P$ and
the *sparsity pattern* of the constraint matrix $A$ are built exactly once.
The pattern includes every entry the linearization can ever touch (explicit
zeros where, e.g., $\sin\psi=0$), satisfying OSQP's fixed-pattern
requirement for `update()`. Each tick then performs only a vectorized
refill of the linear cost $q$, the dynamics values in $A$, and the bounds,
followed by a warm-started solve. On our benchmark machine this reduced
mean solve time from 1.4 ms to 0.6 ms (§6.1) and allowed tightening the
solver tolerance from $10^{-3}$ to $10^{-4}$ with polishing at negligible
cost.

Tightening the tolerance exposed a methodological hazard worth reporting:
the original loose-tolerance configuration *passed* the closed-loop
tracking gate (0.394 m steady-state maximum error) that the numerically
identical high-accuracy configuration *failed* (0.475 m) — the pass was an
artifact of inaccurate solves at one corner. The honest remedy was
re-tuning (position weight 14 → 28), which improved true steady-state mean
error from 0.136 m to 0.122 m and mean lap speed from 5.91 to 6.01 m/s.

### 4.2 MAP fallback controller

Any tick the QP fails (or if the solver is absent), control falls through to
a Model- and Acceleration-based Pursuit controller [3] rather than plain
pure pursuit. MAP retains the lookahead-point construction with a
speed-scheduled lookahead $L=\mathrm{clip}(q_{la}+m_{la}v_t,\,L_{\min},
L_{\max})$, converts the lookahead angle $\eta$ to a desired lateral
acceleration $a_{des}=2v_t^2\sin\eta/L$ (L1 guidance), and inverts a
steady-state map $a_{lat}(\delta,v)\!\to\!\delta$ that encodes tire slip.

We generate that map at startup by forward-integrating a dynamic
single-track model (linear tire with friction saturation, parameters
matching the f1tenth_gym vehicle) to steady state over a 40×35
steer-by-speed grid, *vectorized across the entire grid simultaneously*, in
under one second of CPU time. Non-converging cells mark the grip limit;
the runtime inversion interpolates the strictly-increasing stable branch
and clamps at the limit, which is precisely the maximum useful steering.
Below 1 m/s the map coincides with the kinematic inversion and the latter
is used directly.

### 4.3 Minimum-curvature raceline refinement

Racelines fit from driven laps are not curvature-optimal. Following
Heilmeier et al. [4], we shift each waypoint laterally by $d_i$ along its
normal $\mathbf n_i$, minimizing the squared second arc-length difference
of the shifted positions — a QP in $d$ — subject to $|d_i|\le$ a corridor
bound, with ridge and offset-smoothness regularization, re-linearized over
three passes with the per-point budget shrunk by displacement already
spent (so total excursion from the *original* line provably respects the
corridor). The corridor (default 0.25 m) is the safety knob: wall clearance
of the input line is assumed, not known, so the parameter defaults off in
deployment until clearance is validated against the map.

### 4.4 Friction-limited velocity profile

Speed columns are recomputed by the standard forward-backward pass [4, 10]:
a quasi-steady-state lateral-limit profile
$v=\min(\sqrt{a_{lat,\max}/|\kappa|},\,v_{\max})$, a forward pass limited by
engine acceleration, and a backward pass limited by braking, with
longitudinal budget coupled to lateral usage through a friction ellipse
($p{=}2$). Closed tracks run each pass twice around so the start/finish
constraint propagates across the line. The profiler shares its
$a_{lat,\max}$ budget with the runtime traction governor (§4.6): one plans
to the budget, the other enforces it.

### 4.5 Curvature-aware lookahead scheduling

We extend MAP's speed-only lookahead schedule with upcoming curvature:
$L = \mathrm{clip}\!\left(L_{base}/(1+k_\kappa\bar\kappa),\,L_{\min},
L_{\max}\right)$, where $\bar\kappa$ is the mean absolute curvature over the
arc $[s, s+L_{base}]$, computed in O(1) per step from a precomputed prefix
integral of $|\kappa|\,ds$. With $k_\kappa=0$ the original controller is
recovered exactly.

### 4.6 Traction governance and speed-adaptive emergency braking

Two runtime safety mechanisms use only proprioception. The *traction
governor* estimates realized lateral acceleration as
$\hat a_{lat}=|\dot\psi|\,v$ from the gyroscope (mounting-sign agnostic),
low-passes it, and scales the commanded speed by
$\min(1, a_{lat,\max}/\hat a_{lat})$ with gradual recovery — so an
optimistic raceline or a low-grip patch costs pace rather than the wall.
The *automatic emergency brake* monitors a cached forward cone of the laser
scan and triggers a full stop when the minimum range falls below
$d_0 + v^2/(2a_{brake})$, i.e., the trigger distance grows with the true
braking requirement rather than a standstill margin.

### 4.7 Actuation-latency compensation

Real cars exhibit 50–150 ms of sensor-to-actuator delay (serial links,
servo lag, ESC ramping); a controller solving from the last measured state
is therefore steering the car's past. Before each control step we
forward-integrate the kinematic bicycle by the measured delay under the
*last published* command and solve from the predicted state. The delay
parameter defaults to zero (simulation) and is configured per-vehicle from a
step-steer measurement.

### 4.8 Velocity estimation with slip rejection, and lidar de-skew

A three-state EKF ($v_x, v_y, \omega$, body frame) mechanizes the 200 Hz IMU
as its prediction model and updates from the gyroscope, the wheel speed, and
a nonholonomic pseudo-measurement ($v_y-\omega l_r\approx 0$, which makes
lateral velocity observable). Wheel slip defeats classical innovation
gating: a slip *ramp* drags the filter estimate along, keeping innovations
inside any gate (we measured this failure directly). We therefore gate the
wheel measurement against a pure-IMU velocity integral anchored at the last
accepted update — the anchored residual exposes the full wheel offset at
once — with hysteresis re-entry at 60% of the trip threshold and a rejection
timeout that prevents the gate from latching on a biased IMU.

Downstream, every beam of the 10 Hz mechanically-spinning lidar is motion-
corrected ("de-skewed") by the EKF twist times the beam's age: at 7 m/s the
car translates 0.7 m and, in corners, rotates >10° *within one sweep*,
which biases scan matching and shifts obstacle centroids by decimetres.
The correction is a vectorized rotate-translate per beam, applied between
the driver and all geometry consumers.

### 4.9 Frenet-frame overtaking ("spliner")

For head-to-head racing we re-implement the ForzaETH spliner [2]: a
detected opponent is projected into the raceline's Frenet frame; a passing
side is chosen toward the outside of the local turn (sign of summed signed
curvature); and a clamped cubic spline in $(s,d)$ is fit through seven
control points at arc offsets $\{-4,-3,-1.5,0,+2,+3,+4\}$ m around the
apex, where the apex offset is the opponent's lateral position plus an
evasion distance (0.65 m default). The clamped end conditions ($d=0$ and
$d'=0$) guarantee smooth rejoin. The output is a full same-indexing
deformed raceline, so any tracker consumes it unchanged; the spline is
re-planned per step against moving opponents.

### 4.10 Accelerated perception

Opponent detection (a YOLOv8 model trained on car imagery) is the one
genuinely parallel workload in the stack, and the only component that
touches an accelerator. Three interchangeable inference paths share one
output parser and back-projection geometry, selected by a `backend`
parameter with automatic fallback:

1. **TensorRT (Jetson GPU).** The ONNX model is compiled once on the target
   device (`trtexec --onnx=... --saveEngine=... --fp16`) and executed at
   FP16 through a thin runtime wrapper (page-locked host buffers,
   asynchronous device copies, one execution stream). `backend: auto`
   prefers an `.engine` file found beside the configured model.
2. **OpenCV CUDA DNN.** The same ONNX through `cv2.dnn` with the CUDA FP16
   target, for Jetson images with CUDA-enabled OpenCV but no TensorRT
   Python bindings.
3. **On-camera VPU.** The OAK-D Pro's Myriad X runs the detector *inside
   the camera* (depthai `YoloDetectionNetwork`, fed by an on-device
   planar-format converter); the host receives only detection boxes, which
   the camera node back-projects to relative opponent positions using the
   factory intrinsics. Host CPU and GPU cost is zero, freeing the Jetson's
   GPU entirely — at the cost of running a smaller (e.g. 416×416) model.

The control loop deliberately stays on CPU: profiling the persistent-QP
solve at 0.6 ms (§6.1) shows the 76-variable problem is dominated by
fixed overheads that GPU offload would add to, not remove.

## 5. Experimental Setup

All closed-loop results use one shared, deterministic benchmark harness: a
kinematic-bicycle plant integrated at 50 Hz around the 300-point, 240.3 m
competition raceline used by the team, starting from a deliberate 0.36 m /
14° pose error, with bounded longitudinal acceleration (±4/−8 m/s²) and,
where stated, an actuator-delay FIFO between controller and plant. Metrics
are lap time and perpendicular cross-track error (mean and maximum) after a
2 s settle period. Unless noted, speed columns are identical across
compared configurations so differences isolate the component under test.

Candidate techniques were each implemented and benchmarked in an isolated
copy of the repository (independent git worktrees, evaluated in parallel)
against this harness, with adoption gated on measurable gains; the
benchmark scripts ship in the repository (`tools/benchmark_*.py`) and the
harness doubles as the regression-test plant (52 hardware-free unit and
closed-loop tests).

**Reproducibility.** `python3 -m pytest tests/ -q` runs the full suite;
`python3 tests/test_mpc.py` reproduces the MPC validation;
`tools/benchmark_refiner.py`, `tools/benchmark_lookahead.py`, and
`tools/benchmark_spliner.py` reproduce Tables 3–5.

## 6. Results

### 6.1 Solver performance and tracking (persistent QP)

| Configuration | mean solve | p95 | max | steady XTE mean | steady XTE max |
|---|---|---|---|---|---|
| Per-tick setup, $\varepsilon{=}10^{-3}$ (baseline) | 1.4 ms | 1.6 ms | 2.5 ms | 0.135 m | 0.394 m* |
| Persistent QP, $\varepsilon{=}10^{-4}$, polish | 0.5 ms | 0.6 ms | 1.4 ms | 0.136 m | 0.475 m |
| + retuned weights ($q_{pos}$ 14→28) | **0.6 ms** | 0.7 ms | 1.3 ms | **0.122 m** | **0.438 m** |

\*Artifact of loose solver tolerance (§4.1); the same construction at high
accuracy yields 0.475 m. Zero solver failures in all configurations; mean
lap speed rose from 5.91 to 6.01 m/s with the retuned weights. The 0.6 ms
solve occupies 3% of the 20 ms control budget.

### 6.2 Controller comparison

One lap, identical (original CSV) speed columns:

| Controller | lap time | XTE mean | XTE max |
|---|---|---|---|
| Pure pursuit (prior fallback) | 42.20 s | 0.114 m | 0.587 m |
| MAP (adopted fallback) | 41.60 s | **0.089 m** | **0.340 m** |
| LTV-MPC (primary) | **41.48 s** | 0.123 m | 0.406 m |

MAP halves the fallback's worst-case error relative to pure pursuit; the
kinematic plant (no tire slip) understates MAP's advantage, which is
precisely tire-slip compensation.

![Figure 1: closed-loop simulation overview — raceline with friction-limited speeds; controller cross-track comparison; speed profiles; lateral-acceleration demand vs. grip budget](figures/sim_comparison.png)

**Figure 1.** Closed-loop simulation overview: raceline with friction-limited
speeds (top left); cross-track error around the lap for the three
controllers (top right); speed profiles (bottom left); and the
lateral-acceleration demand of each speed profile against the 6 m/s² grip
budget (bottom right).

### 6.3 Velocity profile feasibility

The hand-tuned CSV speed column demands more than the 6 m/s² lateral
budget on **22% of the lap** (peak demand exceeding 8 m/s²); the
friction-limited profile never exceeds it, at an estimated lap-time cost of
2.6 s at that budget. The profile is therefore a feasibility guarantee
whose pace depends on the configured budget, not a free speedup — raising
the budget recovers the pace as tire confidence grows.

### 6.4 Minimum-curvature raceline refinement

Speeds reprofiled identically (6 m/s² budget) on every line; closed loop =
MAP controller:

| Line | max $\lvert\kappa\rvert$ | est. lap | closed-loop lap | XTE max | max displacement |
|---|---|---|---|---|---|
| Original | 2.53 m⁻¹ | 44.27 s | 43.76 s | 0.418 m | — |
| Refined, 0.15 m corridor | 1.10 m⁻¹ | 41.19 s | 40.84 s (−2.92) | 0.326 m | 0.150 m |
| Refined, 0.25 m corridor | 1.04 m⁻¹ | 39.99 s | **39.78 s (−3.98)** | 0.285 m | 0.250 m |
| Refined, 0.35 m corridor | 0.90 m⁻¹ | 38.88 s | 38.78 s (−4.98) | 0.243 m | 0.350 m |

A **9% lap-time reduction** at the conservative default corridor, with
tracking error *improving* simultaneously; the gain is monotone in the
corridor, making it a clean safety/pace knob. Profiler estimates agree
with closed-loop laps within 0.5 s, confirming the gain is geometric
(Fig. 2: the refinement clips the curvature spikes — max |κ| 2.53 → 1.04
m⁻¹ — which is exactly where the friction-limited profile was forced to
slow down).

![Figure 2: minimum-curvature refinement — geometry, curvature, and resulting speed profiles](figures/raceline_refinement.png)

**Figure 2.** Minimum-curvature refinement at the 0.25 m corridor: raceline
geometry (left), curvature along the lap (center), and the friction-limited
speed profiles that produce the 3.98 s closed-loop gain (right).

### 6.5 Curvature-aware lookahead

| Speed scale | $k_\kappa{=}0$ XTE mean / lap | $k_\kappa{=}2$ XTE mean / lap |
|---|---|---|
| 1.00× | 0.089 m / 41.60 s | **0.056 m (−37%)** / 42.76 s |
| 1.15× | 0.110 m / 36.44 s | **0.075 m (−31%)** / 37.88 s |
| 1.30× | 0.197 m / 32.34 s | **0.158 m (−20%)** / 34.14 s |

The tracking gain is monotone in $k_\kappa$ and consistent across speeds;
the lap-time *increase* (~3%) is the honest cost of no longer cutting
corners — in the wall-less harness corner-cutting is free, whereas on a
bounded track the baseline's 1.30 m worst-case excursion at 1.3× speed
would be off-track. We adopt $k_\kappa{=}2$ for the fallback controller,
where tracking robustness dominates.

### 6.6 Actuation-latency compensation

Injected actuator delay; compensation = forward prediction by the true
delay (Fig. 2):

| Delay | MAP XTE max (plain → comp.) | MPC lap (plain → comp.) |
|---|---|---|
| 60 ms | 0.384 → 0.336 m | 39.92 → 41.56 s |
| 100 ms | 0.565 → **0.335 m** | 50.48 → **42.16 s** |
| 140 ms | 1.096 → **0.505 m** | 69.76 → **42.88 s** |

At realistic delays (100–140 ms) compensation restores MAP's worst-case
tracking to its zero-delay level (0.340 m) and recovers the MPC from severe
degradation (the receding-horizon tracker is markedly more
delay-sensitive than the geometric one). At 60 ms, effects are within
noise — the uncompensated MPC's faster lap there results from
delay-induced corner-cutting, not better control. Figure 3 summarizes.

![Figure 3: actuation-latency compensation under injected actuator delay](figures/delay_compensation.png)

**Figure 3.** Worst-case tracking (MAP, left) and lap time (MPC, right)
under 60/100/140 ms injected actuator delay, with and without
forward-prediction compensation, against the zero-delay references (dashed).

### 6.7 Overtaking

Static and moving (40% of local raceline speed) opponents placed on the
raceline; clean-lap reference 41.60 s:

| Scenario | Outcome | Lap | Min clearance |
|---|---|---|---|
| Opponent, AEB only (status quo) | **DNF** (stops 0.52 m behind) | — | — |
| Static opponent + spliner | completes | 41.90 s (+0.30) | 0.467 m |
| Moving opponent + spliner (re-planned per step) | completes | 41.90 s (+0.30) | 0.580 m |

The planner converts a guaranteed non-finish into a 0.30 s overtake.
Clearances are center-to-center between point models; subtracting two
vehicle half-widths (~0.15 m each) leaves ~0.17 m of physical margin at the
default evasion distance — thin, and the parameter should be raised for
physical cars (§7).

### 6.8 Velocity estimation and de-skew (synthetic truth)

On a 30 s synthetic drive (200 Hz IMU with noise and bias, 50 Hz wheel
speed with 25% slip pulses during hard acceleration and braking,
kinematically consistent cornering):

| Estimator | speed RMSE, clean | speed RMSE, during slip |
|---|---|---|
| Raw wheel odometry | 0.065 m/s | 1.052 m/s |
| EKF, no gating | 0.060 m/s | 1.041 m/s (follows the slipping wheel) |
| EKF, anchored slip gate | **0.042 m/s** | **0.052 m/s (20×)** |

Slip detection: 100% of slipping samples flagged, 0.2% false-positive rate
on clean segments. Notably, the *ungated* EKF is barely better than raw
wheel during slip (Fig. 4: it visibly rides the slipping wheel signal) —
quantifying that the gating, not the fusion, carries the value. De-skew
restores a scan synthesized from a sensor moving at 7 m/s and 2.3 rad/s
(per-beam honest raycast) to its static-pose geometry exactly
(wall-distance RMS 0.307 m → numerically zero); these are open-loop,
synthetic-truth validations rather than closed-loop lap results.

![Figure 4: velocity estimation through wheel-slip pulses](figures/ekf_slip_rejection.png)

**Figure 4.** Velocity estimation through two 25% wheel-slip pulses (shaded):
the ungated EKF follows the slipping wheel almost exactly, while the
anchored slip gate coasts on the IMU and tracks the true speed.

## 7. Discussion and Limitations

**What the methodology bought.** Gating adoption on one shared benchmark
produced three kinds of outcome that a less disciplined integration would
have blurred: clean wins (raceline refinement: faster *and* tighter),
explicit trade-offs (curvature-aware lookahead: tighter but slower — adopted
for the fallback, where the trade is correct), and exposed artifacts (the
loose-tolerance MPC "pass" of §4.1, and the uncompensated 60 ms MPC lap
that is faster for the wrong reason). We consider reporting the latter two
as important as the wins.

**Limitations.** (i) All closed-loop results use a kinematic-bicycle plant:
no tire dynamics, no track walls, no sensor noise, and a single track. This
biases some results conservative (MAP's tire compensation is unexercised;
the refiner's corridor cannot trade against walls) and some optimistic
(corner-cutting is unpunished; clearances are point-to-point). (ii) The
stack has not yet been validated end-to-end on the physical vehicle; the
hardware layer is unit-tested against protocol and register-level
specifications, and all parameters (latency, trim, ERPM gain, grip budget)
are exposed for the calibration procedure documented in the repository, but
on-car lap results are future work and a precondition for any claims
transferring to hardware. (iii) The accelerated perception paths (TensorRT,
cv2-CUDA, on-camera VPU) are implemented and selected automatically, but
their throughput has not yet been benchmarked on a physical Jetson/OAK-D —
we report no GPU timing numbers rather than estimate them. (iv) The refiner
assumes the input line's wall clearance; integrating occupancy-grid
clearance checks is mechanical but not yet done. (v) The overtaker chooses
sides from curvature only; the ForzaETH original consults measured per-side
track width [2].

**Future work.** On-vehicle validation; a particle filter on a frozen map
for race-time localization [6], consuming the velocity EKF of §4.8 as its
motion prior; occupancy-aware corridors for the refiner and spliner; and
opponent-trajectory prediction for the overtaker [7].

## 8. Conclusion

AtlasAutoware demonstrates that the published state of the art in
small-scale autonomous racing — MPC tracking, tire-aware geometric control,
minimum-curvature trajectory optimization, friction-limited profiling,
slip-rejecting state estimation, latency compensation, and Frenet-frame
overtaking — fits in a single pip-installable, hardware-adaptive stack
whose real-time path needs only a CPU and whose perception exploits
whatever accelerator the platform offers, and that each component's
contribution can be isolated and measured on a shared closed-loop
benchmark. The measured component gains (9% lap time from
trajectory refinement; 37% tracking error from lookahead scheduling; full
recovery of tracking under 100 ms actuation delay; DNF-to-overtake
capability for 0.30 s) compose into a system whose degraded modes are
themselves competition-grade.

## References

[1] M. O'Kelly, H. Zheng, D. Karthik, and R. Mangharam, "F1TENTH: An
open-source evaluation environment for continuous control and reinforcement
learning," *NeurIPS 2019 Competition and Demonstration Track*, PMLR, 2020.

[2] N. Baumann, E. Ghignone, J. Kühne, et al., "ForzaETH Race Stack —
Scaled autonomous head-to-head racing on fully commercial-off-the-shelf
hardware," *Journal of Field Robotics*, 2024. arXiv:2403.11784.

[3] J. Becker, N. Imholz, L. Schwarzenbach, E. Ghignone, N. Baumann, and
M. Magno, "Model- and acceleration-based pursuit controller for
high-performance autonomous racing," *Proc. IEEE ICRA*, 2023.
arXiv:2209.04346.

[4] A. Heilmeier, A. Wischnewski, L. Hermansdorfer, J. Betz, M. Lienkamp,
and B. Lohmann, "Minimum curvature trajectory planning and control for an
autonomous race car," *Vehicle System Dynamics*, 58(10), 2020.

[5] B. Stellato, G. Banjac, P. Goulart, A. Bemporad, and S. Boyd, "OSQP:
An operator splitting solver for quadratic programs," *Mathematical
Programming Computation*, 12(4), 2020.

[6] T. Y. Lim, E. Ghignone, N. Baumann, and M. Magno, "Robustness
evaluation of localization techniques for autonomous racing,"
arXiv:2401.07658, 2024.

[7] N. Baumann et al., "Predictive spliner: Data-driven overtaking in
autonomous racing using opponent trajectory prediction," *IEEE Robotics and
Automation Letters*, 2025.

[8] A. Liniger, A. Domahidi, and M. Morari, "Optimization-based autonomous
racing of 1:43 scale RC cars," *Optimal Control Applications and Methods*,
36(5), 2015.

[9] M. M. Zarrar, Q. Weng, B. Yerjan, A. Soyyigit, and H. Yun,
"TinyLidarNet: 2D LiDAR-based end-to-end deep learning model for F1TENTH
autonomous racing," *Proc. IEEE/RSJ IROS*, 2024. arXiv:2410.07447.

[10] TUM Institute of Automotive Technology, "trajectory_planning_helpers"
and "global_racetrajectory_optimization," open-source repositories,
github.com/TUMFTM.

---

## Appendix A: Figures

All four figures are embedded inline (§6.2, §6.4, §6.6, §6.8) and generated
from the repository's own benchmark code; sources live in
`docs/paper/figures/` (`sim_comparison.png`, `raceline_refinement.png`,
`delay_compensation.png`, `ekf_slip_rejection.png`).

## Appendix B: Reproduction

```bash
pip3 install numpy scipy "osqp<1" pytest matplotlib
python3 -m pytest tests/ -q             # 52 unit + closed-loop tests
python3 tests/test_mpc.py               # MPC validation + solve timing (§6.1)
python3 tools/benchmark_refiner.py      # §6.4
python3 tools/benchmark_lookahead.py    # §6.5
python3 tools/benchmark_spliner.py      # §6.7
python3 tools/reprofile_raceline.py racelines/comp_raceline.csv --a-lat 6.0 -o /tmp/r.csv  # §6.3
```
