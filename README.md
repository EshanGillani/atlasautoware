# AtlasAutoware — F1TENTH/RoboRacer autonomous racing stack

A complete ROS 2 racing stack for 1/10-scale autonomous racing: simulation
bridge, SLAM mapping, raceline optimization, an OSQP-based MPC racing
controller with lidar AEB and an IMU traction governor, camera opponent
detection, and a hardware bringup layer (Jetson + OAK-D Pro + RPLidar + VESC)
so the same nodes that win in sim drive the physical car.

Built on the [f1tenth_gym_ros](https://github.com/f1tenth/f1tenth_gym_ros)
simulation bridge (ROS 2 Humble).

## Start here

Everything runs through one command, which behaves the same whether you are on
the Jetson, a Linux workstation with ROS, or a Windows laptop driving a Docker
container — it detects what the machine has and routes accordingly.

```bash
python3 tools/atlas.py list      # everything you can run, and what is ready
python3 tools/atlas.py doctor    # check every connection before you drive
python3 tools/atlas.py ui        # the dashboard at http://localhost:8000
python3 tools/atlas.py run sim   # start the simulator
```

New here? **[docs/GUIDE.md](docs/GUIDE.md)** covers setup, a practice session
start to finish, and competition day.

```
                 ┌── sim:  gym_bridge (f1tenth_gym) ──────────────┐
sensors ─────────┤                                                ├──► /scan, odom, /oakd/*
                 └── car:  rplidar_node · oakd_camera · drive_node┘
                                            ▲
map ─► slam_toolbox ─► raceline_optimizer ─► raceline_mpc (MPC + AEB + traction governor) ─► /drive
                                            │
                       camera_perception ───┘ (YOLO opponent detection, optional)
```

## The racing pipeline

1. **Map** the track: `ros2 launch f1tenth_gym_ros slam_mapping_launch.py`
   driving slowly (or `mapping_driver` autonomously) — see
   [docs/mapping.md](docs/mapping.md).
2. **Optimize** a raceline: `raceline_optimizer` / `track_learner` produce
   `racelines/*.csv` (x, y, heading, curvature, speed); inspect with
   `tools/draw_raceline.py`.
3. **Profile** the speeds: `tools/reprofile_raceline.py` (or
   `reprofile_speeds: true` at runtime) replaces the speed column with the
   TUMFTM friction-limited forward-backward profile, so commanded speeds
   provably fit the grip budget — see
   [docs/racing_tech.md](docs/racing_tech.md).
4. **Race**: `raceline_mpc` tracks the line with a kinematic LTV-MPC
   (persistent warm-started OSQP solve, ~0.6 ms/tick), falls back to the
   MAP controller (Becker et al., ICRA 2023 — tire-aware pursuit) on any
   solver hiccup, brakes for obstacles with a speed-aware AEB, and scales
   speed when the IMU says grip is running out — see
   [docs/mpc.md](docs/mpc.md) and [docs/racing_tech.md](docs/racing_tech.md).

5. **Learn**: `rl_agent` adds a neural policy that outputs a *bounded correction*
   to the MPC command — safe by construction, since zero authority is exactly
   the MPC — see [docs/rl.md](docs/rl.md).

Racing nodes: `raceline_mpc` (competition time-trial), `race_agent` /
`racing_agent` (full strategy + opponents), `rl_agent` (learned policy),
`opponent_driver`, `camera_perception`
([docs/camera_perception.md](docs/camera_perception.md)).

## Learning and tuning between sessions

Practice time is the scarce resource at a competition, so three tools exist to
spend it well:

- **`atlas run practice`** — records a run, then reads off the lateral
  acceleration the car *actually* sustained on that surface and re-profiles the
  raceline against it. Every speed on your line rests on an assumed `a_lat`;
  this replaces the guess with a measurement.
- **`atlas run bayes-tune`** — a Gaussian process with Expected Improvement over
  the grip, speed and MPC-weight parameters, scored on *expected time to
  complete a lap including restarts after a crash*, so it will not trade
  reliability for lap time. Far fewer runs than coordinate descent
  (`tools/auto_tune.py`, kept for overnight sim runs).
- **`atlas run train-rl`** — trains the residual policy in sim, warm-started
  from MPC demonstrations. `atlas run eval-rl` races it against the MPC from
  identical starts; if it does not win there it does not go on the track.

## Quickstart — simulation

Supported: Ubuntu 22.04 native with ROS 2 Humble, or any OS via Docker
(NVIDIA GPU with `rocker`, or noVNC without).

**Native (Ubuntu 22.04 + ROS 2 Humble):**
```bash
# dependencies
git clone https://github.com/f1tenth/f1tenth_gym && cd f1tenth_gym && pip3 install -e . && cd ..
# workspace
mkdir -p $HOME/sim_ws/src && cd $HOME/sim_ws/src
git clone <this repo>
# point map_path in config/sim.yaml at <your_home>/sim_ws/src/atlasautoware/maps/levine
cd $HOME/sim_ws
rosdep install -i --from-path src --rosdistro humble -y
colcon build
source /opt/ros/humble/setup.bash && source install/local_setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

**Docker (NVIDIA GPU):**
```bash
docker build -t f1tenth_gym_ros -f Dockerfile .
rocker --nvidia --x11 --volume .:/sim_ws/src/f1tenth_gym_ros -- f1tenth_gym_ros
```

**Docker (no GPU, noVNC):**
```bash
docker-compose up
# second terminal:
docker exec -it f1tenth_gym_ros-sim-1 /bin/bash
# browser: http://localhost:8080/vnc.html → Connect, then launch as above
```

Use `headless_bridge_launch.py` instead of `gym_bridge_launch.py` for
benchmarking/auto-tuning without rviz.

## Quickstart — real car

Jetson + OAK-D Pro (RGB + 200 Hz IMU) + RPLidar + VESC. Actuation goes
through a PCA9685 PWM board into the VESC's PPM input **or** direct VESC
UART — `drive_node` probes both at startup and uses what it finds:

```bash
pip3 install depthai rplidar-roboticia smbus2 pyserial
ros2 launch f1tenth_gym_ros car_bringup_launch.py
```

Calibration (steering trim, `erpm_gain`, speed scaling) lives in
`config/hardware.yaml`. Wiring, VESC Tool setup, and the first-drive
calibration order are in [docs/hardware.md](docs/hardware.md). Both actuation
paths carry an arming hold, a command watchdog, and neutral-on-shutdown.
YOLO opponent detection auto-selects its accelerator — TensorRT on the
Jetson GPU, cv2-CUDA, the OAK-D's onboard VPU, or CPU fallback — see
[docs/racing_tech.md](docs/racing_tech.md).

## Configuration

- `config/sim.yaml` — simulation: `map_path`, `num_agent` (1 or 2), start
  poses, `kb_teleop` (then `ros2 run teleop_twist_keyboard
  teleop_twist_keyboard`: `i`/`u`/`o` forward, `,`/`m`/`.` reverse, `k` stop).
- `config/hardware.yaml` — drive backend + sensor + racing parameters for the
  car.
- `config/slam_mapping.yaml` — mapping session settings.

## Topics

| Topic | Type | Direction |
|---|---|---|
| `/scan` | LaserScan | sim bridge / rplidar_node → stack |
| `/ego_racecar/odom` (sim) · `/pf/pose/odom` (car) | Odometry | localization → controllers |
| `/oakd/rgb`, `/oakd/camera_info` | Image, CameraInfo | oakd_camera → camera_perception |
| `/oakd/imu` | Imu | oakd_camera → traction governor |
| `/vesc/odom` | Odometry | drive_node (UART backend) → particle filter |
| `/drive` | AckermannDriveStamped | controllers → sim bridge / drive_node |
| `/map`, `tf` | — | map server / SLAM |

Two-agent sim additionally has `/opp_scan`, `/opp_drive`,
`/opp_racecar/odom`, and mirrored `opp_odom` topics; reset poses via RViz's
*2D Pose Estimate* (`/initialpose`) and *2D Goal Pose* tools.

## Tests

```bash
python3 tools/atlas.py run tests   # or: python3 -m pytest tests/ -q
python3 tests/test_mpc.py          # closed-loop MPC validation + solve-time budget
```

167 tests, none of which need ROS, hardware or a GPU (torch-dependent ones skip
cleanly if torch is absent):

- `test_hardware.py` — PCA9685 register/pulse maths, the VESC UART protocol,
  drive-backend auto-detection, lidar scan binning, the traction governor.
- `test_mpc.py`, `test_mpcc.py`, `test_map_controller.py` — the controllers.
- `test_rl.py` — the observation contract and, most importantly, the residual
  safety envelope: the clamp that guarantees an untrained or diverged policy
  still drives exactly as well as the MPC.
- `test_bayes_opt.py` — GP inference, the acquisition function, and that the
  search actually beats random search on the same budget.
- `test_atlas.py` — every registry entry points at a file that exists and a ROS
  node that is registered, so no launcher button is a dead one.

## License

MIT — see [LICENSE](LICENSE).
