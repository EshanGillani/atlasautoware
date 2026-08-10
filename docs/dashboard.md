# Race Control dashboard

A web UI for the whole stack — no RViz, no CLI, nothing to install.

```bash
python3 tools/atlas.py run ui        # http://localhost:8000
python3 ui/server.py --port 8080     # or directly
```

Standard library only. It is a front end to the *same* command registry
(`tools/atlas_registry.py`) and the *same* environment detection
(`tools/atlas.py`) that the CLI uses, so the two cannot drift apart: adding a
script to the registry makes it appear in both.

## Tabs

**Pre-flight** — runs `tools/hw_doctor.py` and renders every check with its fix
hint. This is the first thing to look at before a session; the header pills also
show at a glance whether ROS, `f110_gym` and `torch` are available here.

**Launcher** — every command in the stack, grouped, with its arguments as form
fields and a one-click Run. Output streams live and jobs keep running while you
switch tabs. Commands that can move a real car are marked and require an
explicit confirmation.

**Raceline** — sliders for wall clearance, late-apex bias, lateral grip and top
speed; *Generate* re-runs the optimizer and annotator and shows the annotated
overlay (numbered corners, apex speeds, overtake zones).

**Race** — start/stop the two-car opponent demo, with live telemetry polled 4×/s:
the mode badge (CRUISE/ATTACK/DEFEND/EVADE), the agent's reasoning, speed, lap,
opponent count, and a live map with the ego (white triangle) and opponents
(red = lidar, blue = camera, green = fused).

**Tuning** — the Bayesian tuning history: a convergence curve, the best
configuration, and a table of every evaluation with its lap time and reliability.
Crashed candidates are drawn in red along the bottom, since plotting their raw
scores would flatten every real lap time into one line.

**Practice** — record a session, rebuild the speed profile from the grip that
was actually measured, and compare every session you have recorded.

## Endpoints

| | |
|---|---|
| `GET /api/env` | what this machine has (ROS, docker, torch, gym) |
| `GET /api/commands` | the registry, grouped, with availability per command |
| `GET /api/doctor` | hardware check as JSON |
| `GET /api/state` | live race telemetry |
| `GET /api/raceline` | current raceline polyline |
| `GET /api/tuning` | Bayesian tuning history + best |
| `GET /api/sessions` | practice sessions |
| `GET /api/jobs`, `/api/jobs/<id>` | running jobs, with output |
| `POST /api/run` | `{id, args, confirm}` → start a registry command |
| `POST /api/jobs/<id>/stop` | stop it |
| `POST /api/generate` | regenerate the raceline |
| `POST /api/race/{start,stop}` | the opponent demo |

## Network binding

The server binds to **loopback only** by default.

```bash
python3 ui/server.py --host 0.0.0.0     # reachable from the pit
```

That is genuinely useful — a laptop or phone in the pit can watch telemetry and
launch runs — and it also means anyone on that network can start a command that
moves the car, because there is no authentication. Fine on an isolated pit LAN;
not on venue wifi. The server prints a warning when you do it.

## How it is wired

- `race_agent.py` writes `runtime/race_state.json` 10×/s; the server reads it.
- Launched jobs are `subprocess.Popen` with a bounded rolling output tail, built
  through `atlas.build()` so they run natively, in the sim container, or locally,
  exactly as the CLI would run them.
- For the **real car**, run the server where it can reach the car's ROS graph.
