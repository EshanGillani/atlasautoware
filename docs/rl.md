# The learned driving policy

A neural policy that reads the lidar and drives — but never on its own.

```
scan + odom ──► MPC ──► + policy residual ──► traction governor ──► AEB ──► /drive
                 └────────── fallback ──────────┘
```

## The one decision that matters

The policy outputs a **bounded correction to the MPC's command**, not a command
of its own:

```python
steer = mpc_steer + authority * a[0] * d_steer     # d_steer = 0.10 rad
speed = mpc_speed + authority * a[1] * d_speed     # d_speed = 1.5 m/s
```

Everything good about this design follows from that line.

**It is safe by construction.** An untrained network outputs roughly zero, so it
drives exactly as well as the MPC. A diverged one is clamped to ±0.10 rad. A
NaN-producing one is detected and ignored for that tick. `residual_scale:=0` is
bit-for-bit the MPC. There is no configuration in which the policy can do
something the MPC could not almost do already.

**It learns fast.** A from-scratch policy on a race track spends its first
hundred thousand steps discovering that walls are bad. Starting from the MPC,
the very first episode completes laps, so every sample is collected in the
region of state space the car will actually race in. And "where does the MPC
leave time on the table" is a far lower-variance learning target than "how do I
drive a car".

**Training and deployment share one clamp.** `ResidualAction` in
`f1tenth_gym_ros/rl/features.py` is used by the training environment and by the
ROS node, so behaviour cannot diverge between them.

**The safety layers stay downstream.** The traction governor and the lidar AEB
are applied *after* the policy. It cannot override a brake command.

## What the policy sees

Built by `build_observation` — 128 values, all normalized to roughly [-1, 1]:

| Block | Size | Why |
|---|---|---|
| lidar | 108 | Downsampled by **min-pooling**, so a thin obstacle survives |
| kinematics | 4 | Speed, yaw rate, a slip proxy, lateral-g demand |
| line frame | 2 | **Signed** cross-track and heading error |
| preview | 12 | Curvature and target speed 0.5–7 m ahead |
| baseline | 2 | What the MPC intends to do |

Three of these are load-bearing:

- **Min-pooling, not averaging.** The value that matters for not hitting
  something is the *closest* return in each sector. Averaging smooths a cone or
  another car's wheel out of existence.
- **Preview.** A lidar-only policy cannot know a hairpin follows the fast
  right-hander, so it brakes reactively and late. Curvature ahead lets it plan.
- **The baseline.** Telling the network what the MPC intends is what turns the
  problem from "drive" into "correct", which is the easier problem by a wide
  margin.

### The observation contract

A policy is only valid on the observation distribution it was trained on. Feed
it a different beam count or a different normalization and it still *runs* —
silently, on wrong inputs, into a wall.

So `ObsSpec.fingerprint()` identifies the layout, every checkpoint stores it, and
`rl_agent` refuses to load a policy whose fingerprint does not match what the
node builds. It logs the mismatch and races on the MPC instead.

## The network

A 1-D CNN over the beams, then an MLP.

Adjacent lidar beams are spatially adjacent in the world, and the features that
matter — a wall receding, a gap opening — are translation-equivariant: a gap two
metres left means the same thing as a gap two metres right, mirrored. A
convolution learns one gap detector and applies it at every bearing. A
fully-connected layer has to learn the same detector 108 times, from far more
data.

The actor — the only part that ships on the car — is ~113k parameters and
measures **~0.9 ms per tick** on a laptop CPU, against a 20 ms budget at 50 Hz.
The twin critics (~357k) exist only during training. If inference ever overruns
its budget the node skips the policy for that tick, because in a 50 Hz loop a
late command built from a stale state is worse than the MPC's fresh one.

## The algorithm: SAC

- **Continuous actions.** Discretising steering throws away exactly the fine
  control that wins lap time.
- **Off-policy.** Every step ever collected stays in the replay buffer and gets
  reused — which also means you can train on logged real driving, not only fresh
  rollouts.
- **Maximum entropy.** SAC maximises reward *plus* policy entropy, with the
  trade-off tuned automatically. It keeps exploring alternative lines instead of
  locking onto the first one that completes a lap, and it is far less
  hyperparameter-sensitive than PPO — which matters when you have a week before
  a competition, not a month of tuning runs.

## The reward

```
+ progress      metres advanced along the raceline this step
- deviation     quadratic, so small corrections are nearly free
- effort        for using the residual at all
- crash         large, one-off, ends the episode
+ finish        for completing the lap
```

Progress is *distance*, not velocity, so the policy gets nothing for pointing
fast at a wall. Everything else is shaping that stops it buying speed with risk
it cannot see.

Two details that are easy to get wrong and expensive to miss:

- **The start/finish wrap.** A raw arc-length difference goes hugely negative
  when the car crosses the line, handing the policy a giant penalty for the one
  thing it is supposed to do. `_advance` treats large backward jumps as the wrap
  and leaves genuine reversing negative.
- **Timeouts are not terminal.** A timeout means the car could have kept going.
  Bootstrapping through it, rather than treating it as an absorbing state worth
  zero, is what keeps the value function honest.

## Training

```bash
atlas run train-rl -- --steps 300000
```

Three phases:

1. **Warm start** (20k steps). The car drives on the MPC while the policy is
   trained, supervised, to output the MPC's action. Because the action space is
   a residual, the expert demonstration is a constant — behaviour cloning
   reduces to teaching the policy to output zero. That fills the replay buffer
   with on-track states at racing speed before a single reward gradient is taken.
2. **Exploration** (5k steps of random residuals).
3. **SAC**, with periodic deterministic evaluation against the pure MPC.

`best.pt` is the one to deploy. `policy.pt` is the latest and can easily be
mid-regression.

Random starting positions matter more here than in most RL problems: a fixed
start lets the policy memorise one lap as a sequence rather than learn a control
law, and it then falls apart the moment the car is nudged off that trajectory —
which is exactly what a real start line, or contact, does.

## Evaluating

```bash
atlas run eval-rl -- --authority-sweep
```

Runs the policy and the pure MPC over the **same** starting poses and reports
lap time, completion rate and crashes for each. Identical starts are the point:
a policy that draws easier spawns looks better than it is, and the difference
you are measuring is often under a second.

A policy that does not beat the MPC here does not go on the track.

## Deploying

```bash
atlas run rl-drive -- -p checkpoint:=runtime/rl/best.pt \
                      -p residual_scale:=0.25 -p v_scale:=0.4
```

Climb `residual_scale` — 0, 0.25, 0.5, 1.0 — one clean lap at each.
`/rl/status` publishes what the policy is doing: whether it is being consulted,
how far it is bending the steer and speed, and what fraction of ticks it is
actually used on.

Every failure is non-fatal and logged with its reason. A car that will not start
because a checkpoint is missing is worse than a car that races on the controller
which was already good enough to qualify.

## Files

| File | What |
|---|---|
| `rl/features.py` | Observation + residual envelope. Pure numpy, no torch |
| `rl/networks.py` | CNN encoder, actor, twin critics |
| `rl/env.py` | f110_gym wrapped for RL, plus the reward |
| `rl/sac.py` | SAC, replay buffer, checkpoints |
| `rl_agent.py` | The ROS node |
| `tools/train_rl.py` | Training |
| `tools/eval_rl.py` | Policy vs MPC |
| `tests/test_rl.py` | Tests — the envelope ones are the load-bearing set |
