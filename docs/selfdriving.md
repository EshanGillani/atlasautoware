# Self-driving modes & the RC kill switch

Beyond racing, the stack has driving modes for everyday autonomy — and a
hard rule: **no autonomous mode runs without the RC kill switch armed.**

## RC kill switch (your 2.4 GHz plane transmitter)

Two layers, use both:

**1. Hardware mux (true manual override — strongly recommended).**
A PWM multiplexer board (e.g. Pololu 4-channel RC multiplexer, ~$6) sits
between everything and the servos:

```
receiver CH1 (steer) ──► mux IN-A ch1          PCA9685 ch1 ──► mux IN-B ch1
receiver CH2 (throttle) ► mux IN-A ch2         PCA9685 ch0 ──► mux IN-B ch2
receiver CH5 (a switch) ► mux SELECT
mux OUT ch1 ──► steering servo        mux OUT ch2 ──► ESC
```

Flip the transmitter switch: the *hardware* hands steering+throttle to the
human sticks, even if the Jetson is frozen, on fire, or mid-kernel-panic.
Most plane transmitters put a 3-position switch on CH5/CH6 — assign it in
the transmitter menu.

**2. Software gate (the stack knows which mode it's in).**
Wire the same (or another) spare receiver channel to a Jetson GPIO pin —
**level-shift the receiver's 5 V signal to 3.3 V** (1k/2k divider works).
`rc_monitor` measures the pulse width and publishes `/autonomy_enabled`:

- pulse > 1700 µs → armed; < 1300 µs → disarmed; in between → hysteresis
- no pulses for 0.5 s (transmitter off / out of range) → **disarmed**
- boots disarmed

`drive_node` (with `enable_topic: '/autonomy_enabled'`, on by default in
`config/hardware.yaml`) snaps to neutral the instant it disarms and ignores
`/drive` until re-armed. Set `enable_topic: ''` only for bench testing.

## Modes

| Mode | Node | Where | Needs |
|---|---|---|---|
| Racing | `raceline_mpc` | mapped track | map + raceline + localization |
| Goal navigation | `navigator` | indoor / mapped (hallways) | map + localization + a goal pose |
| Sidewalk following | `sidewalk_follow` | outdoors, map-free | segmentation model + supervision |

**Goal navigation** — A* on the inflated occupancy map, smoothed,
speed-profiled with the friction profiler at gentle limits, followed at
≤2 m/s with the lidar AEB; replans if the path gets blocked. Set goals with
RViz's *2D Goal Pose*. Works anywhere you've SLAM-mapped — school hallways
included.

**Sidewalk following** — map-free outdoor mode: a Cityscapes-pretrained
segmentation model (`tools/get_pretrained_models.py`) marks
road/sidewalk pixels; the car steers toward the drivable centroid of the
near-field image band, slows as the visible surface shrinks, and stops if
the surface runs out or the lidar sees an obstacle.

**Honest scope:** this is *follow-the-paved-ribbon* driving. It does not
understand intersections, driveways, traffic, signals, or pedestrians
beyond stop-when-close. It is a supervised, line-of-sight mode: kill switch
armed, walking pace (`v_cruise: 1.2`), you behind it. Treat anything more
(road crossings, unsupervised runs) as out of scope for this sensor suite —
that's GPS + prediction territory.

## Pretrained models (no training needed)

```bash
python3 tools/get_pretrained_models.py            # both models -> models/
# on the Jetson, compile the detector to TensorRT:
trtexec --onnx=models/yolov8n_coco.onnx --saveEngine=models/yolov8n_coco.engine --fp16
```

- **Detector**: COCO-pretrained YOLOv8n. `camera_perception` is configured
  with `car_class: 2` (COCO "car") — it sees RC cars passably at close
  range because they look like cars; fine-tune with
  `tools/train_car_detector.py` when you want competition-grade detection.
- **Segmentation**: the tool exports a torchvision LR-ASPP as a smoke-test
  model; for real sidewalk driving substitute a Cityscapes-trained ONNX
  (PIDNet-S / Fast-SCNN releases) where road=0, sidewalk=1 and set
  `drivable_classes` accordingly (drop class 0 to refuse roadways).
