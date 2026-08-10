# Operator guide

Everything in this stack runs through one command. If you remember nothing else,
remember these four:

```bash
python3 tools/atlas.py list        # what can I run?
python3 tools/atlas.py env         # what does this machine have?
python3 tools/atlas.py doctor      # is the car ready?
python3 tools/atlas.py ui          # the dashboard, in a browser
```

Everything below is a longer explanation of those.

---

## 1. Setup, once per machine

`atlas` figures out where things can run and does the right thing, so the setup
depends only on what you want to do on that machine.

| I want to… | Install |
|---|---|
| Generate racelines, tune, analyse practice runs | Python 3.8+, `numpy scipy pyyaml pillow osqp==0.6.3` |
| Run the simulator | the above + Docker, **or** ROS 2 Humble + `f1tenth_gym` |
| Train the RL policy | the above + `torch` + `f1tenth_gym` |
| Drive the real car | ROS 2 Humble on the Jetson + `depthai rplidar-roboticia smbus2 pyserial` |

Check what you have:

```bash
python3 tools/atlas.py env
```

It prints whether ROS is native here, whether a sim container is running, and
whether `torch` and `f110_gym` are importable — then tells you what that means
for what you can run. A command shown as `x` in `atlas list` is not broken; it
just needs something this machine does not have, and `atlas info <id>` says what.

Add a shortcut so you can type `atlas` instead of the full path:

```bash
echo "alias atlas='python3 $PWD/tools/atlas.py'" >> ~/.bashrc && source ~/.bashrc
```

---

## 2. Before every session: the doctor

```bash
atlas doctor            # or: python3 tools/atlas.py doctor
atlas doctor -- --fix   # also print the fix for warnings, not just failures
```

It walks the whole chain — Python packages, the I2C bus and PCA9685, the VESC
over a real UART handshake, the RPLidar, the OAK-D, the ROS graph, and every map
and raceline file — and for anything broken prints the command that fixes it.

Every probe is passive. It opens ports, reads registers and listens; nothing in
it ever commands throttle, so it is safe to run with the car powered and on the
ground.

**Read the result like this:**

- **FAIL** — do not power the motor. A dead lidar or an invalid raceline means
  the car does not know where it is.
- **WARN** — usually fine. A missing OAK-D disables camera opponent detection
  and nothing else. Missing `osqp` is the one warning worth fixing: the MPC
  falls back to the MAP controller and you lose lap time.
- **skip** — not applicable here, e.g. Linux device paths checked from a Windows
  laptop.

The exit code is 0 unless something failed, so you can gate a launch script on it.

---

## 3. The dashboard

```bash
atlas run ui
# open http://localhost:8000
```

Standard library only — nothing to install. It is a front end to the same
registry and the same environment detection the CLI uses, so the two cannot
disagree.

| Tab | What it is for |
|---|---|
| **Pre-flight** | The hardware check, live, plus the order of operations |
| **Launcher** | Every command in the stack, with its arguments, one click to run, live output |
| **Raceline** | Sliders for the optimizer, with the annotated overlay |
| **Race** | Start/stop the two-car demo, live telemetry and map |
| **Tuning** | Bayesian tuning history, the convergence curve, the best setup |
| **Practice** | Record a run, rebuild the speed profile, compare sessions |

Commands that can move a real car are marked and require an explicit
confirmation, in the UI and in the CLI both.

> **On network access.** The server binds to loopback by default. `--host
> 0.0.0.0` makes it reachable from the pit, which is genuinely useful and also
> means anyone on that network can start the car — there is no authentication.
> Fine on an isolated pit LAN; not on venue wifi.

---

## 4. A practice session, start to finish

This is the loop that turns track time into lap time.

### 4.1 Map the track

```bash
atlas run map-session            # SLAM, in one terminal
atlas run map-drive              # optional: drive it autonomously, slowly
atlas run map-finish -- mytrack  # save to maps/mytrack.{png,yaml}
```

### 4.2 Generate a racing line

```bash
atlas run optimize -- --map maps/mytrack.yaml \
                      --output racelines/best_raceline.csv \
                      --margin 0.35 --apex-bias 1.0 --a-lat 6.5 --v-max 7.0
atlas run annotate               # numbered corners with apex speeds — print this
```

Or use the **Raceline** tab and move the sliders.

### 4.3 Drive it, slowly

```bash
atlas run race -- -p v_scale:=0.3
```

Wheels off the ground for the first run. Then on the ground, and raise
`v_scale` **one step per clean lap** — 0.3, 0.4, 0.5. Never two steps.

### 4.4 Measure what the car actually did

```bash
atlas run practice -- --name friday-am --stage record --duration 120
atlas run practice -- --name friday-am --stage build
```

The `build` stage reads the run and extracts the thing you cannot get any other
way: **the lateral acceleration the car actually sustained on this surface
today**. Every speed in your raceline rests on an assumed `a_lat`; guess it too
high and the car understeers off at the first fast corner, too low and you leave
seconds on the table. This reads the real number off the run and re-profiles the
raceline against it.

It writes `practice/friday-am/report.md` with the measured grip, the new speed
profile, and the estimated lap time. `atlas run practice -- --compare` ranks
every session you have recorded.

### 4.5 Search for a faster setup

```bash
atlas run bayes-tune -- --iters 40
```

A Gaussian process models lap time as a function of ten parameters (grip budget,
accel and brake limits, speed ceiling and scale, and the five MPC cost weights),
and each run is spent where the model says the most is to be learned. It is
scored on **expected time to complete a lap including restarts after a crash**,
so it will not sell you half a second in exchange for spinning one lap in four.

Everything is logged to `runtime/bayes_log.jsonl`; `--resume` continues where a
previous session stopped, which matters when a practice slot ends mid-search.

Confirm the winner on the real dynamics before you race it:

```bash
atlas run validate -- --v-scale 1.10 --render
```

Then put it on the car about 10% under the winning `v_scale`.

---

## 5. The learned policy

The neural policy never drives the car directly. It outputs a **bounded
correction** to what the MPC already decided — at most ±0.10 rad of steering and
±1.5 m/s, scaled by an authority you control. That single design decision is
what makes it safe to field:

- an **untrained** policy drives exactly as well as the MPC,
- a **diverged or NaN-producing** policy is caught and ignored for that tick,
- **`residual_scale:=0`** is bit-for-bit the MPC,
- the AEB and traction governor are applied *after* the policy, so it can never
  talk the car out of braking.

```bash
atlas run train-rl -- --steps 300000       # trains in sim; needs torch + f110_gym
atlas run eval-rl -- --authority-sweep     # policy vs MPC, identical starts
atlas run rl-drive -- -p residual_scale:=0.25 -p v_scale:=0.4
```

**Do not skip `eval-rl`.** It races the policy and the pure MPC from the same
starting poses and reports both. A policy that does not beat the MPC there has
no business on the track, and it is completely normal for early training runs to
lose — that is the measurement working.

Then climb the authority ladder on the car the same way you climb `v_scale`:
0.25, 0.5, 0.75, 1.0, one clean lap at each. `/rl/status` publishes what the
policy is actually doing — how often it is being used, and how far it is bending
the command — so you can watch it earn its place.

See [rl.md](rl.md) for how it works and why it is built this way.

---

## 6. Competition day

```bash
atlas doctor                                     # every connection
atlas run practice -- --compare                  # which setup was fastest
atlas run validate -- --v-scale <winner>         # confirm on real dynamics
atlas run race -- -p v_scale:=<winner minus 10%>
```

Keep the dashboard open on a second screen. If anything misbehaves, the fallback
ladder is always available and always in this order:

1. lower `v_scale`
2. `residual_scale:=0` (drop the policy, keep the MPC)
3. `atlas run race` instead of `race-strategy` (drop opponent logic)
4. a raceline you have already driven clean

---

## Troubleshooting

**"cannot run 'x': needs ROS 2"** — no local ROS and no sim container. Run
`docker-compose up -d`, or run it on the Jetson. `atlas env` shows what was found.

**The car does not move.** `atlas doctor --only vesc`. Most often: the battery is
not connected (the VESC reads ~0 V on USB power alone), the arming hold has not
elapsed, or no `/drive` messages are arriving.

**The car runs wide in fast corners.** Your assumed `a_lat` is above the real
grip. Record a practice run and rebuild — the report tells you the real number.

**The car is slow and jerky.** Check that `osqp` is installed; without it the MPC
falls back to the MAP controller full-time. `atlas doctor --only python`.

**The policy is not being used.** The node logs why at startup and
`/rl/status` carries the state: a missing checkpoint, no torch, or an
*observation mismatch* — that last one means the checkpoint was trained against
a different observation layout and the node is refusing to run it on wrong
inputs. Retrain against the current setup.

**Tuning says everything crashes.** Your ranges may be wrong for this track, or
`--mu` is optimistic. Lower it and re-run; `atlas run sweep-grip` names the
corner that fails first.
