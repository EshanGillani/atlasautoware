# Racing in the AutoDRIVE Simulator (RoboRacer Sim Racing League)

A bridge that lets the **same** AtlasAutoware nodes (`raceline_mpc`,
`race_agent`, …) race in the [AutoDRIVE Simulator](https://autodrive-ecosystem.github.io/)
used by the RoboRacer Sim Racing League — no changes to the controllers.

> **Status: built from the AutoDRIVE devkit source, not yet run against the
> live simulator.** The interface below is taken from the devkit
> `config.py`/`bridge.py` (ground truth), but a few items must be confirmed
> in a running sim (see [Calibration](#calibration)). Treat the first sim
> session as a bring-up/validation run.

## How it fits together

```
AutoDRIVE Simulator (Unity)
        │  Socket.IO
AutoDRIVE devkit bridge  ── /autodrive/<veh>/{lidar,imu,odom,...}  ──┐
   (autodrive_roboracer)  ◄─ /autodrive/<veh>/{throttle,steering}_command
                                                                     │
        autodrive_bridge  (this package) ── /scan, /ego_racecar/odom, /oakd/imu ──► raceline_mpc
                          ◄─────────────────── /drive (AckermannDriveStamped) ─────┘
```

The AutoDRIVE side speaks **namespaced topics** and **normalized Float32
[-1,1]** commands; our stack speaks `/scan`, `nav_msgs/Odometry`, and
`ackermann_msgs/AckermannDriveStamped` on `/drive`. `autodrive_bridge`
translates both directions.

## Interface mapping (RoboRacer generation, `vehicle:=roboracer_1`)

Sim → stack:

| AutoDRIVE topic | Type | → republished as | Notes |
|---|---|---|---|
| `/autodrive/roboracer_1/lidar` | `sensor_msgs/LaserScan` | `/scan` | 270° FOV, 1080 rays, 0.06–10 m. Bridge re-frames to `laser` and (option `mask_max_range`) converts clamped maxima → `inf` for AEB/gap-following. |
| `/autodrive/roboracer_1/imu` | `sensor_msgs/Imu` | `/oakd/imu` | feeds the traction governor (uses yaw-rate magnitude). |
| `/autodrive/roboracer_1/odom` | `nav_msgs/Odometry` | `/ego_racecar/odom` | ground-truth global pose; re-stamped `odom`→`base_link`. **raceline_mpc tracks directly — no particle filter needed in sim.** |

Stack → sim:

| `/drive` field | → AutoDRIVE topic | Conversion |
|---|---|---|
| `drive.speed` (m/s) | `…/throttle_command` (Float32) | `clip(speed / max_speed, -1, 1)` |
| `drive.steering_angle` (rad) | `…/steering_command` (Float32) | `clip(steer_sign · angle / max_steer, -1, 1)` |

Legacy build: set `vehicle:=f1tenth_1` (namespace `/autodrive/f1tenth_1/*`).
The legacy devkit has **no native `/odom`** — derive it from the encoders
(`…/left_encoder`,`…/right_encoder` JointState) + `…/ips` if you're on that build.

## Run it

```bash
# 1) AutoDRIVE devkit bridge (separate process/container, connected to the sim):
ros2 launch autodrive_roboracer bringup_headless.launch.py     # RoboRacer
#   (legacy: ros2 launch autodrive_f1tenth simulator_bringup_headless.launch.py)

# 2) AtlasAutoware bridge + time-trial controller:
ros2 launch f1tenth_gym_ros autodrive_bringup_launch.py vehicle:=roboracer_1 v_scale:=0.4
```

You need a **raceline for the AutoDRIVE track** (world coords). Generate it
offline from the track map with `raceline_optimizer.py` → `refine` →
`reprofile`, then point `raceline_mpc` at it (or `F1_RACELINE`).

## Calibration (NEEDS CONFIRMATION in the running sim)

These defaults come from a community port and **may differ per sim build** —
verify on the first run, they directly affect whether the car drives correctly:

1. **`steer_sign`** — does `+steering_command` turn **left**? Publish a small
   `+` steering command, watch the wheels. Flip `steer_sign:=-1.0` if reversed.
2. **`max_steer`** (default `0.5236` rad ≈ 30°) — the angle that maps to
   `steering_command = 1.0`.
3. **`max_speed`** (default `22.8` m/s) — the speed that maps to
   `throttle_command = 1.0`.
4. **Topic rates** — `ros2 topic hz /scan /ego_racecar/odom` (Socket.IO/sim
   step driven; documented Hz not guaranteed).
5. **Generation** — `ros2 topic list | grep autodrive` to confirm
   `roboracer_1` vs `f1tenth_1` and whether `/odom` exists.

## Verify the bridge (first sim session)

```bash
ros2 topic hz /scan                         # ~sim lidar rate, frame 'laser'
ros2 topic echo /ego_racecar/odom --once    # global pose flowing
# command a gentle creep and watch the car move the RIGHT way:
ros2 topic pub -r 20 /drive ackermann_msgs/msg/AckermannDriveStamped \
  "{drive: {speed: 1.0, steering_angle: 0.1}}"
```
If it drives forward and turns the expected direction, set `v_scale` low and
launch the full stack. Raise `v_scale` per clean lap (don't crash first,
minimize laptime second).

Pure conversion helpers are unit-tested in `tests/test_autodrive_bridge.py`.
