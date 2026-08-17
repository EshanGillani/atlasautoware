# VTC 2026 — RoboRacer Autonomous Racing: campaign plan

Objectives, in priority order, straight from the competition: **(1) don't
crash, (2) minimize laptime.** Two entry paths: the in-person 30th RoboRacer
race (physical 1/10 car) and the virtual **RoboRacer Sim Racing League**
(AutoDRIVE Simulator). Every phase below gates on a *completed* lap before any
speed push — skipping the basics (localization, one clean lap) is what crashes
the car.

## Phase 0 — Intel & registration (off-car, now)
- Pull exact rules, track/car spec, and **deadlines** from
  `vtc2026-race.roboracer.ai` + the RoboRacer Slack. Register both paths.
- Gate: known format (time-trial + head-to-head), spec, dates → one-page brief.

## Phase 1 — Sim path (AutoDRIVE port) — primary while the car is down
- **Bridge:** `f1tenth_gym_ros/autodrive_bridge.py` + `autodrive_bringup_launch.py`
  translate AutoDRIVE ↔ stack topics (see [docs/autodrive.md](autodrive.md)).
  Built from devkit source; **validate against the live sim** (calibrate
  `steer_sign`, `max_speed`, `max_steer`; confirm topic rates).
- **Time-trial qualification:** `raceline_mpc` tracks a raceline generated for
  the AutoDRIVE track. Sim odom is ground truth → no particle filter needed.
- **Unseen-track readiness** (qualification races a previously-unseen track):
  `tools/build_raceline.py <map.yaml>` is the one-command pipeline — auto-picks
  a drivable seed (widest point on the track), runs the min-curvature
  optimizer, and **validates in the closed-loop MPC benchmark** with a
  feasibility gate (completes? lap time, XTE, planned a_lat vs grip budget,
  wall clearance). It never green-lights a line that cuts walls or exceeds the
  budget — on tight tracks it returns REVIEW (bump `--margin` / add a refine
  pass). Needs properly-scaled RoboRacer track maps to produce a race line.
- **Head-to-head:** `race_agent`/`spliner` overtaking.
- Gate: completes laps in AutoDRIVE with bounded XTE before any speed push;
  cross-check in the deterministic f1tenth_gym benchmark.

## Phase 2 — Physical-car parity (when hardware is stable)
Hardware checklist (each a gate):
- Lidar streaming `/scan`; VESC FOC config (sensorless, detected λ) → **motor spins**.
- `odom→base_link` TF flowing (vesc_to_odom + a steering command).
- **Particle filter converged** on the track map — the #1 cause of a
  raceline car hitting a wall; never skip the convergence check.
- Map the track (SLAM or push-by-hand); generate the raceline.
- Gate: one clean **conservative** autonomous lap (`v_scale 0.4`, AEB on, e-stop ready).

## Phase 3 — Minimize laptime (only after a clean lap)
- Feasibility-gated raceline (min-curvature refined line). On `comp_track` the
  **fastest *feasible* lap is ~45.9 s** at `v_scale 0.85` / grip 8 m/s²; the
  40.5 s "baseline" cheats grip (~11 m/s²) and would spin out. There is no free
  lap time below ~46 s without more tire grip.
- **Safe speed ladder:** `v_scale 0.40 → 0.55 → 0.65 → 0.75 → 0.80 → 0.85`,
  one rung per clean lap. Stop the instant the IMU traction governor cuts or
  the car drifts — that's the real grip ceiling, which sim cannot know.
- Chassis (Traxxas Slash): wheelbase 0.33 m, **`max_steer` 0.36 rad** (down
  from 0.41), traction `max_lat_accel` tuned to the real surface.

## Phase 4 — Race-day ops
- Pre-flight checklist; e-stop on the deadman.
- **Fallback:** reactive `mapping_driver` (lidar-only, no localization) is a
  no-crash backup — a completed slow lap beats a fast DNF.

## Open unknowns to close
- Exact VTC 2026 deadlines / track format (Phase 0).
- AutoDRIVE calibration (`steer_sign`, `max_speed`, `max_steer`) — confirm in sim.
- Real tire–surface friction μ — only knowable on track; the governor is the net.
