"""
Atlas command registry — the single source of truth for "what can I run".
=========================================================================

Every runnable piece of this stack is described once, here, as data: what it is,
where it runs (sim / car / host), what environment it needs (ROS sourced,
f110_gym, or plain Python), and which arguments matter.  Both front ends read
this file:

    tools/atlas.py   the CLI          (`atlas list`, `atlas run raceline_mpc`)
    ui/server.py     the dashboard    (the Launcher tab is generated from it)

Adding a script to the stack means adding one COMMAND entry — it then appears in
the CLI and the web UI automatically, with no other edits.

Field reference
---------------
id      short handle used on the command line
group   section it appears under (see GROUPS for ordering + labels)
title   one line: what it does
kind    'launch'  ros2 launch f1tenth_gym_ros <target>
        'node'    ros2 run f1tenth_gym_ros <target>
        'python'  python3 <target>            (path relative to repo root)
        'shell'   bash <target>
env     'ros'   needs /opt/ros/humble sourced + the workspace overlay
        'gym'   needs the f110_gym package, but NOT ROS
        'plain' plain Python + numpy — runs anywhere, including the Windows host
where   'sim' | 'car' | 'either' | 'host'  — where it is meant to be executed
args    [(flag, help, default_or_None)] — the arguments worth surfacing in a UI
needs   pip packages that must be importable for it to work
danger  True if it moves a real car; the UI double-confirms these
long    a paragraph of "when do I actually use this"
"""

GROUPS = [
    ('sim',      'Simulation',        'Run the f1tenth_gym sim and drive in it.'),
    ('car',      'Real car',          'Bring up hardware and race the physical car.'),
    ('mapping',  'Mapping',           'Build a map of the track during practice.'),
    ('raceline', 'Raceline',          'Generate, refine, profile and inspect racing lines.'),
    ('learning', 'Learning (RL)',     'Train and deploy the neural driving policy.'),
    ('tune',     'Tuning',            'Search the parameter space for a faster lap.'),
    ('bench',    'Benchmarks',        'Offline validation — no ROS, no hardware.'),
    ('util',     'Utilities',         'Diagnostics and one-off helpers.'),
]

COMMANDS = [
    # ── simulation ────────────────────────────────────────────────────────────
    dict(
        id='sim', group='sim', kind='launch', target='gym_bridge_launch.py',
        env='ros', where='sim', needs=['rclpy', 'f110_gym'],
        title='Start the simulator (with RViz)',
        args=[],
        long='The f1tenth_gym bridge plus RViz. Start this first, then run a '
             'driving agent (raceline_mpc / race_agent) in a second terminal. '
             'Map and start poses come from config/sim.yaml.',
    ),
    dict(
        id='sim-headless', group='sim', kind='launch',
        target='headless_bridge_launch.py',
        env='ros', where='sim', needs=['rclpy', 'f110_gym'],
        title='Start the simulator (no RViz — for benchmarking)',
        args=[],
        long='Same bridge without the visualiser. Use it for auto-tuning and '
             'benchmark sweeps, where rendering just costs frames.',
    ),
    dict(
        id='opponent', group='sim', kind='node', target='opponent_driver',
        env='ros', where='sim', needs=['rclpy'],
        title='Drive the second (opponent) car',
        args=[('--cap', 'speed cap in m/s — how hard the opponent races', '3.0')],
        long='Needs num_agent: 2 in config/sim.yaml. Run alongside race_agent '
             'to exercise overtaking and defending.',
    ),

    # ── real car ──────────────────────────────────────────────────────────────
    dict(
        id='car', group='car', kind='launch', target='car_bringup_launch.py',
        env='ros', where='car', needs=['rclpy'], danger=True,
        title='Bring up the whole car (lidar + camera + drive + EKF)',
        args=[],
        long='The one command that starts the physical car: rplidar_node, '
             'oakd_camera, velocity_ekf and drive_node, all parameterised from '
             'config/hardware.yaml. Run `atlas doctor` FIRST — it checks every '
             'device this launch expects to find. The car stays neutral until '
             'a /drive command arrives.',
    ),
    dict(
        id='race', group='car', kind='node', target='raceline_mpc',
        env='ros', where='either', needs=['rclpy'], danger=True,
        title='Race the optimized line (MPC + AEB + traction governor)',
        args=[('-p v_scale:=', 'global speed multiplier — START AT 0.3', '0.5'),
              ('-p odom_topic:=', 'pose source; /pf/pose/odom on the car',
               '/ego_racecar/odom'),
              ('-p raceline:=', 'explicit raceline CSV; empty = auto-find', '')],
        long='The competition time-trial node. On the car always begin at '
             'v_scale 0.3 and raise it a step at a time once you have watched a '
             'full clean lap at the current setting.',
    ),
    dict(
        id='race-strategy', group='car', kind='node', target='race_agent',
        env='ros', where='either', needs=['rclpy'], danger=True,
        title='Race with opponent strategy (CRUISE/ATTACK/DEFEND/EVADE)',
        args=[],
        long='Adds the race brain — lidar+camera opponent fusion and overtaking '
             'decisions — on top of raceline tracking. Heavier than `race`; use '
             '`race` for a clean time trial.',
    ),
    dict(
        id='camera', group='car', kind='node', target='camera_perception',
        env='ros', where='car', needs=['rclpy', 'cv2'],
        title='YOLO opponent detection from the OAK-D',
        args=[],
        long='Optional. Publishes camera-detected opponents for the race brain. '
             'Auto-selects TensorRT / cv2-CUDA / OAK-D VPU / CPU.',
    ),

    # ── mapping ───────────────────────────────────────────────────────────────
    dict(
        id='map-session', group='mapping', kind='launch',
        target='slam_mapping_launch.py',
        env='ros', where='either', needs=['rclpy', 'slam_toolbox'],
        title='Start a SLAM mapping session',
        args=[],
        long='Drive the track slowly (teleop or `map-drive`) while this runs, '
             'then save with `map-finish`. This is step 1 of a practice session.',
    ),
    dict(
        id='map-drive', group='mapping', kind='node', target='mapping_driver',
        env='ros', where='either', needs=['rclpy'], danger=True,
        title='Drive the track autonomously to build the map',
        args=[],
        long='A slow, cautious wall-follower that explores the track so you do '
             'not have to teleop a full lap by hand. Run it with map-session.',
    ),
    dict(
        id='map-finish', group='mapping', kind='shell',
        target='tools/finish_mapping.sh',
        env='ros', where='either', needs=[],
        title='Save the SLAM map to maps/',
        args=[],
        long='Serialises the slam_toolbox map to maps/<name>.png + .yaml, ready '
             'for the raceline optimizer.',
    ),
    dict(
        id='practice', group='mapping', kind='python',
        target='tools/practice_session.py',
        env='plain', where='host', needs=['numpy'],
        title='Practice-session manager — record, map, optimize, report',
        args=[('--name', 'session name (folder under practice/)', 'session'),
              ('--stage', 'record | build | all', 'all'),
              ('--duration', 'seconds to record when recording', '120')],
        long='The track-side workflow in one command. Records a driven lap, '
             'extracts the driven line, optimizes and re-profiles a raceline '
             'from it, and writes a report you can read between runs. Also '
             'produces the demonstration data the RL policy warm-starts from.',
    ),

    # ── raceline ──────────────────────────────────────────────────────────────
    dict(
        id='optimize', group='raceline', kind='python',
        target='f1tenth_gym_ros/raceline_optimizer.py',
        env='plain', where='host', needs=['numpy', 'PIL', 'yaml'],
        title='Generate a raceline from a map',
        args=[('--map', 'map YAML', 'maps/comp_track.yaml'),
              ('--output', 'raceline CSV to write', 'racelines/best_raceline.csv'),
              ('--margin', 'wall clearance in m', '0.35'),
              ('--apex-bias', 'late-apex bias (>1 = later apex)', '1.0'),
              ('--a-lat', 'lateral grip budget m/s^2', '6.5'),
              ('--v-max', 'top speed m/s', '7.0')],
        long='Turns an occupancy map into a speed-profiled racing line. The four '
             'knobs above are exactly what the Bayesian tuner searches over.',
    ),
    dict(
        id='reprofile', group='raceline', kind='python',
        target='tools/reprofile_raceline.py',
        env='plain', where='host', needs=['numpy'],
        title='Recompute a raceline\'s speeds (friction-limited)',
        args=[('--a-lat', 'lateral grip budget m/s^2', '6.5')],
        long='Replaces the speed column with the TUMFTM forward-backward '
             'profile, so commanded speeds provably fit the grip budget. Do this '
             'whenever you change tyres or surface.',
    ),
    dict(
        id='refine', group='raceline', kind='python',
        target='tools/refine_comp_raceline.py',
        env='plain', where='host', needs=['numpy'],
        title='Refine + validate + install the competition line',
        args=[],
        long='Minimum-curvature refinement with wall-clearance validation, then '
             'installs the result as the competition raceline.',
    ),
    dict(
        id='draw', group='raceline', kind='python', target='tools/draw_raceline.py',
        env='plain', where='host', needs=['PIL'],
        title='Draw a raceline over the track image',
        args=[('--image', 'track image', 'racetrackForComp.png'),
              ('--csv', 'raceline CSV', 'racelines/best_raceline.csv')],
        long='Quick visual check that a generated line actually stays on track.',
    ),
    dict(
        id='annotate', group='raceline', kind='python',
        target='tools/annotate_raceline.py',
        env='plain', where='host', needs=['PIL'],
        title='Annotate corners, apex speeds and overtake zones',
        args=[('--image', 'track image', 'racetrackForComp.png'),
              ('--csv', 'raceline CSV', 'racelines/best_raceline.csv')],
        long='The picture to put in front of the team before a run — numbered '
             'corners with the speed the car will actually carry through them.',
    ),
    dict(
        id='image-to-map', group='raceline', kind='python',
        target='tools/image_to_map.py',
        env='plain', where='host', needs=['PIL', 'numpy'],
        title='Convert a track drawing into an occupancy map',
        args=[('--image', 'source drawing', 'racetrackForComp.png')],
        long='When the organisers publish a track diagram before the event, this '
             'turns it into a map you can optimize a line on days early.',
    ),

    # ── learning ──────────────────────────────────────────────────────────────
    dict(
        id='train-rl', group='learning', kind='python', target='tools/train_rl.py',
        env='gym', where='host', needs=['numpy', 'torch'],
        title='Train the neural driving policy (SAC, residual on MPC)',
        args=[('--steps', 'environment steps to train for', '300000'),
              ('--raceline', 'line the policy learns around',
               'racelines/comp_raceline.csv'),
              ('--map', 'map to train on (no extension)', 'maps/comp_track'),
              ('--warm-start', 'imitate the MPC for N steps first', '20000'),
              ('--out', 'checkpoint directory', 'runtime/rl')],
        long='Trains a LiDAR-driven policy that outputs a CORRECTION to the MPC '
             'command rather than replacing it, so an untrained policy still '
             'drives (as the MPC) and a trained one can only bend the line '
             'within a bounded envelope. Warm-starting on MPC demonstrations '
             'gets it racing in far fewer steps than learning from scratch.',
    ),
    dict(
        id='train-duel', group='learning', kind='python',
        target='tools/train_duel.py',
        env='gym', where='host', needs=['numpy', 'torch'],
        title='Train the style-conditioned overtaking policy (SAC)',
        args=[('--steps', 'environment steps to train for', '200000'),
              ('--authority', 'how much the policy may bend the decision (0..1)',
               '1.0'),
              ('--style', 'fix the style instead of sampling it', None),
              ('--out', 'checkpoint directory', 'runtime/duel')],
        long='Learns a bounded correction to the race brain\'s DECISION -- the '
             'lateral offset and speed factor it already emits -- rather than to '
             'raw control. The mode (CRUISE/ATTACK/DEFEND/EVADE) stays '
             'rule-based and readable, spliner still generates the geometry, '
             'and AEB is still downstream, so an untrained policy races exactly '
             'as the existing brain does. Style is resampled every episode so '
             'one network covers conservative through aggressive; check the '
             'per-style columns actually separate before trusting the knob.',
    ),
    dict(
        id='eval-rl', group='learning', kind='python', target='tools/eval_rl.py',
        env='gym', where='host', needs=['numpy', 'torch'],
        title='Evaluate a trained policy against the MPC baseline',
        args=[('--checkpoint', 'policy checkpoint', 'runtime/rl/policy.pt'),
              ('--episodes', 'evaluation episodes', '10')],
        long='Head-to-head lap times and crash rate, policy vs pure MPC, on the '
             'real gym dynamics. Never field a policy that loses this.',
    ),
    dict(
        id='rl-drive', group='learning', kind='node', target='rl_agent',
        env='ros', where='either', needs=['rclpy', 'torch'], danger=True,
        title='Race using the trained policy (MPC fallback + AEB)',
        args=[('-p checkpoint:=', 'policy file', 'runtime/rl/policy.pt'),
              ('-p residual_scale:=', 'how much authority the policy gets (0..1)',
               '0.5'),
              ('-p v_scale:=', 'global speed multiplier', '0.5')],
        long='Deploys the policy on sim or car. residual_scale 0 is pure MPC, '
             '1 is full policy authority — raise it gradually. Any missing '
             'checkpoint, stale inference or bad output silently reverts to the '
             'MPC for that tick.',
    ),
    dict(
        id='train-detector', group='learning', kind='python',
        target='tools/train_car_detector.py',
        env='plain', where='host', needs=['ultralytics'],
        title='Train the YOLO opponent detector',
        args=[('--data', 'dataset YAML', 'data/car_dataset/data.yaml')],
        long='Run on a GPU machine, then export to ONNX/TensorRT for the Jetson '
             'or a .blob for the OAK-D VPU.',
    ),
    dict(
        id='collect-images', group='learning', kind='python',
        target='tools/collect_camera_data.py',
        env='ros', where='car', needs=['rclpy'],
        title='Collect camera frames for detector training',
        args=[('--topic', 'image topic', '/oakd/rgb')],
        long='Grab frames of the other teams\' cars during practice — that is '
             'the dataset that makes the detector work at the actual event.',
    ),

    # ── tuning ────────────────────────────────────────────────────────────────
    dict(
        id='bayes-tune', group='tune', kind='python', target='tools/bayes_tune.py',
        env='plain', where='host', needs=['numpy'],
        title='Bayesian optimization of raceline + controller parameters',
        args=[('--iters', 'evaluations to spend', '40'),
              ('--init', 'random designs before the model takes over', '8'),
              ('--trials', 'perturbed-start laps scoring each candidate', '12'),
              ('--backend', 'grip (fast, runs anywhere) | gym (real dynamics)',
               'grip'),
              ('--mu', 'tyre friction — sweep to bracket the real surface',
               '1.0489'),
              ('--min-success', 'reliability floor; below it, penalized hard',
               '0.9'),
              ('--resume', 'continue from runtime/bayes_log.jsonl', None),
              ('--apply', 'write the winner into config/hardware.yaml', None)],
        long='A Gaussian-process surrogate with Expected Improvement: it models '
             'lap time across the ten grip, speed and MPC-weight parameters and '
             'spends each run where the model says the most is to be learned. '
             'Scored on expected time to COMPLETE a lap including restarts '
             'after a crash, so it will not trade reliability for lap time. Far '
             'fewer runs than the coordinate-descent tuner for the same gain, '
             'which matters when each evaluation costs practice time.',
    ),
    dict(
        id='auto-tune', group='tune', kind='python', target='tools/auto_tune.py',
        env='ros', where='sim', needs=['numpy'],
        title='Coordinate-descent tuning loop (the older tuner)',
        args=[('--minutes', 'wall-clock budget', '55'),
              ('--bench-time', 'seconds per evaluation', '60')],
        long='Kept for long unattended overnight runs in sim. Prefer bayes-tune '
             'when evaluations are expensive.',
    ),

    # ── benchmarks ────────────────────────────────────────────────────────────
    dict(
        id='validate', group='bench', kind='python', target='tools/gym_validate.py',
        env='gym', where='host', needs=['numpy', 'f110_gym'],
        title='Drive the real gym dynamics and check for collisions',
        args=[('--v-scale', 'speed multiplier to test', '1.1'),
              ('--laps', 'laps to attempt', '3'),
              ('--render', 'open a window and watch', None)],
        long='The honest test before raising speed on the car: full dynamic '
             'single-track physics with real collision detection. If it crashes '
             'here it will crash on the track.',
    ),
    dict(
        id='sweep-reliability', group='bench', kind='python',
        target='tools/reliability_sweep.py',
        env='plain', where='host', needs=['numpy'],
        title='Success rate over perturbed starts x speed scales',
        args=[('--raceline', 'raceline CSV', 'racelines/comp_raceline.csv')],
        long='One clean lap proves nothing. This asks how often you finish from '
             'slightly different starting states — the number that decides how '
             'fast you dare go in a real race.',
    ),
    dict(
        id='sweep-grip', group='bench', kind='python', target='tools/dynamic_sweep.py',
        env='plain', where='host', needs=['numpy'],
        title='Grip-aware sweep — which corner breaks first',
        args=[('--raceline', 'raceline CSV', 'racelines/comp_raceline.csv')],
        long='Adds lateral-grip and steer-rate limits, and names the corner that '
             'fails first so you know exactly where to slow down.',
    ),
    dict(
        id='bench-lap', group='bench', kind='python', target='tools/benchmark_lap.py',
        env='ros', where='sim', needs=['rclpy'],
        title='Score one solo lap in sim (used by the tuners)',
        args=[('--time', 'seconds to run', '60')],
        long='Prints a JSON score line. This is the objective function the '
             'tuners optimize.',
    ),
    dict(
        id='bench-delay', group='bench', kind='python',
        target='tools/benchmark_delay.py',
        env='plain', where='host', needs=['numpy'],
        title='Naive vs delay-compensated MPC across latencies',
        args=[],
        long='Justifies the actuation_delay parameter. Measure your car\'s real '
             'latency, then read off what it costs you uncompensated.',
    ),
    dict(
        id='bench-mpcc', group='bench', kind='python', target='tools/benchmark_mpcc.py',
        env='plain', where='host', needs=['numpy'],
        title='MPCC vs tracking MPC under identical physics',
        args=[],
        long='Contour-following control (MPCC maximises progress along the '
             'track) against the reference-tracking MPC the stack races, under '
             'the same physics budget. Run it when deciding whether the extra '
             'complexity of MPCC is buying anything on your track.',
    ),
    dict(
        id='bench-spliner', group='bench', kind='python',
        target='tools/benchmark_spliner.py',
        env='plain', where='host', needs=['numpy'],
        title='Frenet overtaker vs brake-only behaviour',
        args=[],
        long='How much time the Frenet overtaking planner actually saves '
             'against simply braking behind the car in front. The answer '
             'decides whether to run race-strategy or the simpler race node in '
             'a head-to-head.',
    ),
    dict(
        id='bench-refiner', group='bench', kind='python',
        target='tools/benchmark_refiner.py',
        env='plain', where='host', needs=['numpy'],
        title='Minimum-curvature refinement gain',
        args=[],
        long='What the minimum-curvature refiner buys in lap time, and — the '
             'part that matters — whether the refined line still clears the '
             'walls. Check this before enabling refine_corridor on a car.',
    ),
    dict(
        id='bench-map', group='bench', kind='python', target='tools/benchmark_map.py',
        env='plain', where='host', needs=['numpy'],
        title='MAP controller steer-LUT interpolation',
        args=[],
        long='Validates the fallback controller\'s steering lookup table, '
             'including interpolation along the speed axis. The MAP controller '
             'is what drives whenever an MPC solve fails, so it has to be '
             'correct even though it rarely runs.',
    ),
    dict(
        id='bench-lookahead', group='bench', kind='python',
        target='tools/benchmark_lookahead.py',
        env='plain', where='host', needs=['numpy'],
        title='Curvature-aware lookahead scheduling',
        args=[],
        long='Compares fixed against curvature-scheduled lookahead for the MAP '
             'controller: long lookahead is smooth on straights and cuts '
             'corners in hairpins, so the schedule is what makes the fallback '
             'usable at racing speed.',
    ),

    # ── utilities ─────────────────────────────────────────────────────────────
    dict(
        id='doctor', group='util', kind='python', target='tools/hw_doctor.py',
        env='plain', where='either', needs=[],
        title='Check every connection before you drive',
        args=[('--fix', 'print the exact fix command for each failure', None),
              ('--json', 'machine-readable output for the dashboard', None)],
        long='Probes the I2C bus and PCA9685, the VESC over UART, the RPLidar, '
             'the OAK-D, the ROS graph and the raceline files, then tells you '
             'what is wrong and how to fix it. Run this before every session — '
             'it is faster than debugging a dead car on the grid.',
    ),
    dict(
        id='ui', group='util', kind='python', target='ui/server.py',
        env='plain', where='host', needs=[],
        title='Open the Race Control dashboard',
        args=[('--port', 'port to serve on', '8000')],
        long='The browser UI: hardware status, one-click launching of everything '
             'in this registry, the raceline studio, live telemetry and the '
             'tuning monitor. Standard library only — nothing to install.',
    ),
    dict(
        id='tests', group='util', kind='python', target='-m pytest',
        env='plain', where='host', needs=['pytest'],
        title='Run the test suite',
        args=[('tests/', 'which tests', 'tests/'), ('-q', 'quiet', None)],
        long='Pure-logic tests: controllers, RL env, Bayesian optimizer, '
             'hardware protocol maths. No ROS, no hardware, no GPU needed.',
    ),
]

# ── lookup helpers (used by the CLI and the dashboard) ───────────────────────
BY_ID = {c['id']: c for c in COMMANDS}


def get(cmd_id):
    """Look up one command, or None."""
    return BY_ID.get(cmd_id)


def by_group():
    """[(group_id, label, blurb, [commands...])] in display order."""
    out = []
    for gid, label, blurb in GROUPS:
        out.append((gid, label, blurb, [c for c in COMMANDS if c['group'] == gid]))
    return out


def ids():
    return [c['id'] for c in COMMANDS]
