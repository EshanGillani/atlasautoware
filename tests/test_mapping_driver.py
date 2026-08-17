"""
Tests + sim harness for the mapping driver's pure follow-the-gap control law.
=============================================================================

Three things live here, all ROS-free and deterministic:

1. ``original_scan_command`` — a faithful port of the ORIGINAL per-beam
   disparity-extender loop, kept as the equivalence oracle.  The new vectorized
   ``gap_follow_command`` (with smoothing+center-bias disabled) must reproduce
   its *processed ranges* and gap target exactly on arbitrary scans.

2. ``RaycastLidar`` + ``run_mapping_lap`` — a deterministic 2D raycast lidar
   over the competition occupancy grid (loaded with the same conventions as
   tools/build_raceline.py) driven closed-loop by a kinematic bicycle (same
   plant family as tests/closed_loop.py).  Measures lap completion, min wall
   clearance, steering smoothness, and free-track *coverage* per lap.

3. Unit tests: vectorized == loop, smoothing bounds steering rate, center-bias
   centers the car in a symmetric corridor, and the closed-loop sim is
   deterministic and safe.

Run:  python3 -m pytest tests/test_mapping_driver.py -q
"""

import math
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))

from mapping_driver import GapParams, gap_follow_command  # noqa: E402

# The closed-loop raycast-sim tests are accurate but slow (~150 s total); skip
# them in the default suite so it stays fast. Run them with RUN_MAPPING_SIM=1,
# or use tools/benchmark_mapping.py for the full before/after coverage benchmark.
slow_sim = pytest.mark.skipif(
    not os.environ.get('RUN_MAPPING_SIM'),
    reason='slow raycast closed-loop sim — set RUN_MAPPING_SIM=1 to run')


# ════════════════════════════════════════════════════════════════════════════
# 1.  Original disparity-extender (oracle) — verbatim port of the old _scan loop
# ════════════════════════════════════════════════════════════════════════════
def original_processed_ranges(ranges, angle_inc, car_half, disparity, max_range):
    """The exact pre-vectorization Python loop, returning the processed ranges."""
    r = np.asarray(ranges, np.float32)
    r = np.where(np.isfinite(r), r, max_range)
    r = np.clip(r, 0.0, max_range)
    n = len(r)
    proc = r.copy()
    for i in range(1, n):
        if abs(r[i] - r[i - 1]) > disparity:
            near = min(r[i], r[i - 1])
            span = int(np.arctan2(car_half, max(near, 0.1)) / angle_inc)
            if r[i] < r[i - 1]:
                proc[i:min(n, i + span)] = np.minimum(proc[i:min(n, i + span)], near)
            else:
                proc[max(0, i - span):i] = np.minimum(proc[max(0, i - span):i], near)
    return proc


def original_scan_command(ranges, angle_min, angle_inc, params):
    """The full ORIGINAL control law (no smoothing, no center-bias)."""
    p = params
    r = np.asarray(ranges, np.float32)
    r = np.where(np.isfinite(r), r, p.max_range)
    r = np.clip(r, 0.0, p.max_range)
    n = len(r)
    ang = angle_min + np.arange(n) * angle_inc
    proc = original_processed_ranges(ranges, angle_inc, p.car_half,
                                     p.disparity, p.max_range)
    fwd = np.abs(ang) < (np.pi / 2)
    proc_fwd = np.where(fwd, proc, 0.0)
    target = int(np.argmax(proc_fwd))
    steer = float(np.clip(ang[target], -p.max_steer, p.max_steer))
    front = proc[np.abs(ang) < 0.26]
    clear = float(front.min()) if len(front) else p.max_range
    spd = p.speed * float(np.clip((clear - 0.5) / 2.0, 0.3, 1.0))
    return steer, max(spd, 0.5)


# extract just the processed-ranges out of the new vectorized path so we can
# compare the disparity-extender stage directly (the public fn only returns the
# command).  We re-run the new internals via a tiny shim that mirrors them.
def vectorized_processed_ranges(ranges, angle_inc, params):
    from mapping_driver import _apply_windows
    p = params
    r = np.asarray(ranges, np.float32)
    r = np.where(np.isfinite(r), r, p.max_range)
    r = np.clip(r, 0.0, p.max_range)
    n = len(r)
    proc = r.copy()
    if n < 2:
        return proc
    diff = r[1:] - r[:-1]
    big = np.abs(diff) > p.disparity
    idx = np.nonzero(big)[0] + 1
    if not idx.size:
        return proc
    near = np.minimum(r[idx], r[idx - 1]).astype(np.float64)
    span = (np.arctan2(p.car_half, np.maximum(near, 0.1)) / angle_inc).astype(np.int64)
    forward = r[idx] < r[idx - 1]
    fwd_start = idx[forward]
    fwd_val = near[forward]
    fwd_len = np.minimum(span[forward], n - fwd_start)
    bwd_start = np.maximum(idx[~forward] - span[~forward], 0)
    bwd_val = near[~forward]
    bwd_len = idx[~forward] - bwd_start
    starts = np.concatenate([fwd_start, bwd_start])
    vals = np.concatenate([fwd_val, bwd_val])
    lens = np.concatenate([fwd_len, bwd_len])
    keep = lens > 0
    starts, vals, lens = starts[keep], vals[keep], lens[keep]
    if not starts.size:
        return proc
    return _apply_windows(proc, starts, vals, lens, n)


def no_extras_params(**kw):
    """GapParams with center-bias and smoothing turned off (pure gap-follow)."""
    base = dict(center_gain=0.0, steer_lp=0.0)
    base.update(kw)
    return GapParams(**base)


# ════════════════════════════════════════════════════════════════════════════
# 2.  Deterministic raycast lidar + kinematic-bicycle closed-loop mapping sim
# ════════════════════════════════════════════════════════════════════════════
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402


def load_map(yaml_path):
    """Load occupancy grid with the SAME conventions as tools/build_raceline.py."""
    meta = yaml.safe_load(open(yaml_path))
    res = float(meta['resolution'])
    origin = meta['origin']
    img_path = meta['image']
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), img_path)
    img = np.array(Image.open(img_path).convert('L'))
    H, W = img.shape
    occ = (255 - img) / 255.0 > float(meta.get('occupied_thresh', 0.65))
    return res, origin, occ, H, W


def world_to_px(x, y, origin, res, H):
    px = (np.asarray(x) - origin[0]) / res
    py = (H - 1) - (np.asarray(y) - origin[1]) / res
    return px, py


class RaycastLidar:
    """Deterministic 2D raycast over an occupancy grid (no noise).

    Casts ``num_beams`` rays from a pose across [angle_min, angle_max], marching
    in fixed ``step`` increments until an occupied (or out-of-bounds) cell, the
    classic grid-lidar model.  Returns ranges in metres clipped to max_range.
    Cells the rays pass through are recorded for the coverage metric.
    """

    def __init__(self, occ, res, origin, H, W,
                 num_beams=1080, fov=np.deg2rad(270.0), max_range=10.0,
                 step=None):
        self.occ = occ
        self.res = res
        self.origin = origin
        self.H, self.W = H, W
        self.num_beams = num_beams
        self.angle_min = -fov / 2.0
        self.angle_max = fov / 2.0
        self.angle_inc = fov / (num_beams - 1)
        self.max_range = max_range
        self.step = step if step is not None else res * 0.5
        self.angles = self.angle_min + np.arange(num_beams) * self.angle_inc

    def _occupied_px(self, px, py):
        cx = np.round(px).astype(np.int64)
        cy = np.round(py).astype(np.int64)
        oob = (cx < 0) | (cx >= self.W) | (cy < 0) | (cy >= self.H)
        out = np.ones(px.shape, bool)
        valid = ~oob
        out[valid] = self.occ[cy[valid], cx[valid]]
        return out, oob

    def scan(self, x, y, yaw, seen=None):
        """Return ranges (m) at pose; optionally mark seen free cells in `seen`."""
        beam_ang = yaw + self.angles
        cos, sin = np.cos(beam_ang), np.sin(beam_ang)
        ranges = np.full(self.num_beams, self.max_range, np.float64)
        n_steps = int(self.max_range / self.step) + 1
        alive = np.ones(self.num_beams, bool)
        for k in range(1, n_steps):
            d = k * self.step
            wx = x + cos * d
            wy = y + sin * d
            px, py = world_to_px(wx, wy, self.origin, self.res, self.H)
            hit, oob = self._occupied_px(px, py)
            # record free cells the (still-alive, in-bounds, free) beams traverse
            if seen is not None:
                rec = alive & ~oob & ~hit
                cx = np.round(px[rec]).astype(np.int64)
                cy = np.round(py[rec]).astype(np.int64)
                seen[cy, cx] = True
            stop = alive & (hit | oob)
            ranges[stop] = d
            alive &= ~(hit | oob)
            if not alive.any():
                break
        return ranges


def run_mapping_lap(map_yaml, params, start=None, laps=2, dt=0.05,
                    wheelbase=0.33, max_steps=20000, num_beams=1080,
                    lidar_max_range=10.0):
    """Drive the pure mapping controller closed-loop; return metrics + traces.

    Kinematic bicycle plant.  Records, per lap: completion, min wall clearance,
    steering trace (for smoothness), and cumulative free-track coverage.
    """
    res, origin, occ, H, W = load_map(map_yaml)
    edt = ndimage.distance_transform_edt(~occ) * res          # clearance field (m)
    free = ~occ
    n_free = int(free.sum())
    # rough track-perimeter estimate (m): free area / mean half-corridor width,
    # i.e. centerline length ≈ area / (2 * mean clearance on the ridge).  Used
    # only as a minimum-arc gate for the lap counter, so an estimate is fine.
    free_area = n_free * res * res
    mean_w = 2.0 * float(edt[free].mean())
    lap_perimeter = free_area / max(mean_w, res)
    lidar = RaycastLidar(occ, res, origin, H, W, num_beams=num_beams,
                         max_range=lidar_max_range)

    if start is None:
        # auto start: widest free point + heading along the local free direction
        py0, px0 = np.unravel_index(int(np.argmax(edt)), edt.shape)
        sx = origin[0] + (px0 + 0.5) * res
        sy = origin[1] + (H - 1 - py0 + 0.5) * res
        # pick a heading that has the most clearance ahead (deterministic sweep)
        best_yaw, best_clr = 0.0, -1.0
        for deg in range(0, 360, 5):
            yaw = math.radians(deg)
            tx = sx + math.cos(yaw) * 0.8
            ty = sy + math.sin(yaw) * 0.8
            ppx, ppy = world_to_px(tx, ty, origin, res, H)
            ci, cj = int(round(float(ppy))), int(round(float(ppx)))
            if 0 <= ci < H and 0 <= cj < W and not occ[ci, cj]:
                if edt[ci, cj] > best_clr:
                    best_clr, best_yaw = edt[ci, cj], yaw
        x, y, yaw = sx, sy, best_yaw
    else:
        x, y, yaw = start

    seen = np.zeros((H, W), bool)
    prev_steer = 0.0
    v = max(params.speed * 0.5, 0.5)
    steers = []
    clearances = []
    coverage_per_lap = []

    # Lap = leave a wide radius around the start, then come back near it.  On a
    # twisty closed circuit the car re-centers, so it need not pass through the
    # exact widest-point seed; a generous return radius (and a minimum arc to
    # have actually gone round) makes this robust on any track shape.
    start_xy = (x, y)
    R_leave, R_return = 5.0, 2.5
    min_lap_arc = 0.4 * lap_perimeter if lap_perimeter else 20.0
    lap = 0
    crashed = False
    moved_away = False
    path_len = 0.0
    last_lap_len = 0.0
    px_prev, py_prev = x, y
    a_accel, a_brake = 4.0, 8.0

    for step in range(max_steps):
        ranges = lidar.scan(x, y, yaw, seen=seen)
        steer, v_t = gap_follow_command(ranges, lidar.angle_min,
                                        lidar.angle_inc, prev_steer, params)
        prev_steer = steer
        steers.append(steer)

        # clearance at current cell (collision check)
        ppx, ppy = world_to_px(x, y, origin, res, H)
        ci, cj = int(round(float(ppy))), int(round(float(ppx)))
        if 0 <= ci < H and 0 <= cj < W:
            clr = float(edt[ci, cj])
        else:
            clr = 0.0
        clearances.append(clr)
        if clr <= 0.0:                           # ran into / outside a wall
            crashed = True
            break

        # kinematic bicycle integration
        a = float(np.clip((v_t - v) / dt, -a_brake, a_accel))
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += v * math.tan(steer) / wheelbase * dt
        v = max(0.0, v + a * dt)
        path_len += math.hypot(x - px_prev, y - py_prev)
        px_prev, py_prev = x, y

        d_start = math.hypot(x - start_xy[0], y - start_xy[1])
        if d_start > R_leave:
            moved_away = True
        if (moved_away and d_start < R_return
                and (path_len - last_lap_len) > min_lap_arc):
            lap += 1
            moved_away = False
            last_lap_len = path_len
            coverage_per_lap.append(float(seen.sum()) / float(n_free))
            if lap >= laps:
                break

    # ensure we always report `laps` coverage entries (pad with final state)
    while len(coverage_per_lap) < laps:
        coverage_per_lap.append(float(seen.sum()) / float(n_free))

    steers = np.array(steers)
    rate = np.diff(steers) / dt if len(steers) > 1 else np.array([0.0])
    return dict(
        completed_laps=lap,
        completed=lap >= laps,
        steps=step + 1,
        lap_time=(step + 1) * dt,
        min_clearance=float(np.min(clearances)) if clearances else 0.0,
        steer_rate_std=float(rate.std()),
        steer_rate_max=float(np.abs(rate).max()),
        coverage_per_lap=coverage_per_lap,
        coverage_final=float(seen.sum()) / float(n_free),
    )


MAP_YAML = os.path.join(REPO, 'maps', 'comp_track.yaml')


# ════════════════════════════════════════════════════════════════════════════
# 3.  Unit tests
# ════════════════════════════════════════════════════════════════════════════
def _sample_scans():
    """A deterministic battery of scans covering many disparity geometries."""
    rng = np.random.default_rng(12345)
    n = 1080
    inc = np.deg2rad(270.0) / (n - 1)
    amin = -np.deg2rad(135.0)
    scans = []
    # smooth wall
    base = 3.0 + 0.5 * np.sin(np.linspace(0, 4 * np.pi, n))
    scans.append(base.copy())
    # single near step (right-near and left-near variants)
    s = base.copy(); s[500:] = 1.0; scans.append(s.copy())
    s = base.copy(); s[:500] = 1.0; scans.append(s.copy())
    # many random steps
    s = rng.uniform(0.3, 6.0, n).astype(np.float32); scans.append(s)
    # corridor with two walls and a gap straight ahead
    s = np.full(n, 1.2); s[n // 2 - 60:n // 2 + 60] = 6.0; scans.append(s.copy())
    # inf / nan handling
    s = base.copy(); s[100:110] = np.inf; s[700:705] = np.nan; scans.append(s.copy())
    # tight pinch
    s = np.full(n, 0.6); s[n // 2 - 10:n // 2 + 10] = 4.0; scans.append(s.copy())
    # overlapping disparities (staircase)
    s = np.repeat(np.arange(1.0, 1.0 + n // 40) * 0.3, 40)[:n].astype(np.float32)
    scans.append(s)
    return amin, inc, scans


def test_vectorized_processed_ranges_match_original_loop():
    amin, inc, scans = _sample_scans()
    p = no_extras_params()
    for k, ranges in enumerate(scans):
        orig = original_processed_ranges(ranges, inc, p.car_half,
                                         p.disparity, p.max_range)
        vec = vectorized_processed_ranges(ranges, inc, p)
        assert np.allclose(orig, vec, atol=1e-5), f"processed ranges differ on scan {k}"


def test_vectorized_command_matches_original_gap_target():
    """With smoothing+center-bias off, the new fn must equal the original law."""
    amin, inc, scans = _sample_scans()
    p = no_extras_params()
    for k, ranges in enumerate(scans):
        s_orig, v_orig = original_scan_command(ranges, amin, inc, p)
        s_new, v_new = gap_follow_command(ranges, amin, inc, 0.0, p)
        assert abs(s_orig - s_new) < 1e-5, f"steer differs on scan {k}: {s_orig} vs {s_new}"
        assert abs(v_orig - v_new) < 1e-5, f"speed differs on scan {k}"


def test_disparity_extender_inflates_near_edge():
    """Sanity: a near step shrinks neighbouring beams to the near value."""
    n = 1080
    inc = np.deg2rad(270.0) / (n - 1)
    ranges = np.full(n, 4.0)
    ranges[600:] = 1.0                       # sharp drop -> right side is near
    p = no_extras_params()
    proc = vectorized_processed_ranges(ranges, inc, p)
    # beams just BEFORE the step (i-span..i) get pulled to 1.0 (backward fill),
    # because the step is r[600]-r[599] < 0 -> r[i] < r[i-1] -> forward fill;
    # check the forward side around the edge is inflated to the near value.
    assert proc[600] <= 1.0 + 1e-6
    assert proc[610] <= 1.0 + 1e-6          # within car-half span of the edge


def test_smoothing_bounds_steering_rate():
    """The low-pass must make a step input converge gradually (rate-limited)."""
    n = 1080
    inc = np.deg2rad(270.0) / (n - 1)
    amin = -np.deg2rad(135.0)
    # scan that wants a hard right turn
    ranges = np.full(n, 1.0)
    ranges[: n // 4] = 6.0                   # deep gap far to one side
    p = GapParams(steer_lp=0.7, center_gain=0.0)
    p_raw = no_extras_params()
    target, _ = gap_follow_command(ranges, amin, inc, 0.0, p_raw)

    prev = 0.0
    rates = []
    vals = []
    for _ in range(60):
        s, _ = gap_follow_command(ranges, amin, inc, prev, p)
        rates.append(abs(s - prev))
        prev = s
        vals.append(s)
    # each step moves at most (1-lp) of the remaining error -> bounded & shrinking
    assert max(rates) <= abs(target) * (1.0 - p.steer_lp) + 1e-6
    # and it converges toward the unsmoothed target
    assert abs(vals[-1] - target) < 1e-2
    # smoothed first step is strictly gentler than jumping straight to target
    assert rates[0] < abs(target) - 1e-6


@slow_sim
def test_smoothing_reduces_steering_rate_vs_unsmoothed_closed_loop():
    # smoothing reduces steering jitter; compare at a fixed (safe) center gain
    # so both runs follow a comparable path and only the low-pass differs.
    smooth = GapParams(center_gain=0.6, steer_lp=0.6)
    sharp = GapParams(center_gain=0.6, steer_lp=0.0)
    r_sharp = run_mapping_lap(MAP_YAML, sharp, laps=1, **FAST_KW)
    r_smooth = run_mapping_lap(MAP_YAML, smooth, laps=1, **FAST_KW)
    assert r_smooth['steer_rate_std'] <= r_sharp['steer_rate_std'] + 1e-9


def test_center_bias_centers_in_symmetric_corridor():
    """In a perfectly symmetric corridor the bias must produce ~zero net steer
    and must actively re-center when the car is off to one side."""
    n = 1080
    inc = np.deg2rad(270.0) / (n - 1)
    amin = -np.deg2rad(135.0)
    ang = amin + np.arange(n) * inc

    # symmetric corridor: both walls at equal distance, gap straight ahead.
    # left clearance == right clearance -> center term contributes ~0.
    sym = np.full(n, 1.5)
    sym[np.abs(ang) < 0.05] = 6.0            # narrow deep gap dead ahead
    p = GapParams(center_gain=0.3, steer_lp=0.0)
    s_sym, _ = gap_follow_command(sym, amin, inc, 0.0, p)
    assert abs(s_sym) < 0.05                 # stays centered

    # car closer to the RIGHT wall: right beams short, left beams long ->
    # more clearance on the left -> bias must steer LEFT (positive).
    off = np.full(n, 1.5)
    off[ang < 0] = 0.7                       # right wall close
    off[ang > 0] = 2.3                       # left wall far
    s_off, _ = gap_follow_command(off, amin, inc, 0.0, p)
    assert s_off > 0.02, f"expected leftward re-centering, got {s_off}"

    # mirror: closer to LEFT wall -> steer RIGHT (negative)
    off2 = np.full(n, 1.5)
    off2[ang > 0] = 0.7
    off2[ang < 0] = 2.3
    s_off2, _ = gap_follow_command(off2, amin, inc, 0.0, p)
    assert s_off2 < -0.02


def test_center_bias_does_not_override_gap_in_tight_section():
    """A strong deep gap to one side must still dominate the mild center term."""
    n = 1080
    inc = np.deg2rad(270.0) / (n - 1)
    amin = -np.deg2rad(135.0)
    ang = amin + np.arange(n) * inc
    ranges = np.full(n, 0.8)
    ranges[(ang > 0.3) & (ang < 0.5)] = 6.0  # only escape is a hard left turn
    p = GapParams(center_gain=0.18, steer_lp=0.0)
    s, _ = gap_follow_command(ranges, amin, inc, 0.0, p)
    assert s > 0.2                            # still turns toward the gap


def test_empty_and_degenerate_scans():
    p = GapParams()
    s, v = gap_follow_command([], 0.0, 0.01, 0.0, p)
    assert v >= 0.5 and abs(s) < 1e-9
    s, v = gap_follow_command([3.0], -0.1, 0.01, 0.0, p)
    assert np.isfinite(s) and np.isfinite(v)


def test_output_bounds():
    amin, inc, scans = _sample_scans()
    p = GapParams()
    for ranges in scans:
        s, v = gap_follow_command(ranges, amin, inc, 0.0, p)
        assert -p.max_steer - 1e-9 <= s <= p.max_steer + 1e-9
        assert v >= 0.5


# ── closed-loop sim tests ───────────────────────────────────────────────────
# A reduced beam count keeps the pytest suite quick while still exercising the
# full closed loop; the controller is beam-density-robust so behaviour matches.
FAST_KW = dict(num_beams=181, max_steps=5500)


@slow_sim
def test_closed_loop_completes_lap_without_crash():
    p = GapParams()
    r = run_mapping_lap(MAP_YAML, p, laps=1, **FAST_KW)
    assert r['completed_laps'] >= 1, "did not complete a lap"
    assert r['min_clearance'] > p.car_half, (
        f"min clearance {r['min_clearance']:.3f} <= car half {p.car_half}")


@slow_sim
def test_closed_loop_deterministic():
    p = GapParams()
    r1 = run_mapping_lap(MAP_YAML, p, laps=1, **FAST_KW)
    r2 = run_mapping_lap(MAP_YAML, p, laps=1, **FAST_KW)
    assert r1['steps'] == r2['steps']
    assert r1['coverage_final'] == r2['coverage_final']
    assert r1['min_clearance'] == r2['min_clearance']
    assert r1['steer_rate_std'] == r2['steer_rate_std']


def test_pure_function_is_deterministic():
    amin, inc, scans = _sample_scans()
    p = GapParams()
    for ranges in scans:
        a = gap_follow_command(ranges, amin, inc, 0.1, p)
        b = gap_follow_command(ranges, amin, inc, 0.1, p)
        assert a == b


@slow_sim
def test_center_bias_improves_or_keeps_coverage_per_lap():
    """The whole point: center-bias + smoothing should not REDUCE one-lap
    coverage vs the bare argmax follower (ideally improves it)."""
    raw = GapParams(center_gain=0.0, steer_lp=0.0)
    tuned = GapParams()           # defaults: center-bias + smoothing on
    r_raw = run_mapping_lap(MAP_YAML, raw, laps=1, **FAST_KW)
    r_tuned = run_mapping_lap(MAP_YAML, tuned, laps=1, **FAST_KW)
    # tuned must still complete safely
    assert r_tuned['min_clearance'] > tuned.car_half
    # coverage per lap should be at least as good (allow tiny slack)
    assert r_tuned['coverage_per_lap'][0] >= r_raw['coverage_per_lap'][0] - 0.02


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
