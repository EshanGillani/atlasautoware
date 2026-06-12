"""
Head-to-head race behavior — ForzaETH-style state machine (pure, no ROS).
=========================================================================

The stack can already *plan* an overtake (spliner.py deforms the raceline
around an opponent) but nothing decides WHEN: raceline_mpc's only response to
traffic is the AEB stopping the car.  This module is the missing orchestrator,
the three racing states of the ForzaETH state machine:

  GB_TRACK   free running: track the global raceline at full speed.
  TRAILING   opponent ahead within `trail_range` and no safe gap to pass:
             follow at a speed-dependent gap with a proportional gap
             controller   v_cmd = v_opp + k_gap * (gap - gap_des),
             gap_des = gap_min + t_gap * v_ego, clamped to [0, raceline
             speed] — sit in the tow, never rear-end, wait for a gap.
  OVERTAKE   safe gap exists: activate the spliner evasion line (re-planned
             as the opponent moves) until the opponent is `clear_margin`
             behind, then rejoin the raceline.

Transitions are debounced the ForzaETH way: a candidate state must hold for a
per-target hysteresis time before it is committed, and the engagement range
has a spatial band (engage < trail_range, disengage > trail_range *
disengage_factor), so a noisy opponent estimate never makes the machine
chatter.

Overtake feasibility = spliner side preference (outside of the local turn,
falling back to the inside) + a lateral-room check: the apex must stay within
`d_max` of the raceline and — when an occupancy GridMap is supplied — keep
`wall_clearance` metres of free space along the evasion corridor.  A speed
check (raceline profile at the opponent vs the opponent's measured speed)
keeps us from diving on an opponent we cannot actually out-run there.

Inputs per tick: ego pose + speed + nearest raceline index, and the opponent
as (s, d, v_s) in the raceline Frenet frame (or None).  Output: which line to
track (raceline or spliner-deformed — same length/indexing so any controller
takes it unchanged), a speed cap, and the state.  Pure numpy; the ROS wiring
lives in raceline_mpc.py behind the `enable_behavior` parameter.

Reference: Baumann et al., "ForzaETH Race Stack", Journal of Field Robotics
2024 (arXiv:2403.11784) — state machine in sec. 6 (their Global Tracking /
Trailing / Overtaking).
"""

import math

import numpy as np

from spliner import (arc_lengths, cartesian_to_frenet, frenet_to_cartesian,
                     choose_side, plan_overtake)

GB_TRACK = 'GB_TRACK'
TRAILING = 'TRAILING'
OVERTAKE = 'OVERTAKE'


class RaceBehavior:
    """ForzaETH-style 3-state racing behavior over a closed raceline.

    raceline: (x, y, v) equal-length closed-lap arrays.  Call `update(...)`
    once per control tick; it returns a dict with

      state        GB_TRACK | TRAILING | OVERTAKE
      line         (x, y, v) arrays to track this tick (raceline or deformed)
      line_changed True when `line` was swapped/re-planned this tick — only
                   then do controllers need a set_raceline()
      speed_cap    absolute m/s cap (inf outside TRAILING)
      gap          along-track gap to the opponent (m, None without one)
      side         locked overtake side while OVERTAKE is active
    """

    def __init__(self, raceline, *,
                 trail_range=8.0,        # m, engage when opponent closer ahead
                 disengage_factor=1.25,  # spatial hysteresis on trail_range
                 lateral_engage=1.0,     # m, opponent |d| beyond this = ignore
                 gap_min=1.2,            # m, standstill following gap
                 t_gap=0.35,             # s, time-gap term of gap_des
                 k_gap=0.8,              # 1/s, gap-controller proportional gain
                 clear_margin=1.5,       # m behind ego before rejoining
                 evasion_dist=0.65,      # m, spliner apex offset
                 d_max=1.2,              # m, max |d| of the apex (no-map bound)
                 wall_clearance=0.30,    # m, min map clearance on the corridor
                 dv_overtake=0.3,        # m/s required pace advantage
                 abort_gap=4.0,          # m, only abort OVERTAKE when this far
                 v_max=8.0,              # m/s, spliner window speed scaling
                 replan_ds=0.5,          # m of opponent motion between replans
                 hyst=None,              # {target_state: seconds} overrides
                 grid_map=None):         # optional GridMap for the room check
        self.rx, self.ry, self.rv = (np.asarray(a, float) for a in raceline)
        self.s, self.total = arc_lengths(self.rx, self.ry)
        self.trail_range = float(trail_range)
        self.disengage = float(trail_range) * float(disengage_factor)
        self.lateral_engage = float(lateral_engage)
        self.gap_min = float(gap_min)
        self.t_gap = float(t_gap)
        self.k_gap = float(k_gap)
        self.clear_margin = float(clear_margin)
        self.evasion_dist = float(evasion_dist)
        self.d_max = float(d_max)
        self.wall_clearance = float(wall_clearance)
        self.dv_overtake = float(dv_overtake)
        self.abort_gap = float(abort_gap)
        self.v_max = float(v_max)
        self.replan_ds = float(replan_ds)
        self.grid_map = grid_map
        self.hyst = {TRAILING: 0.15, OVERTAKE: 0.40, GB_TRACK: 0.30}
        if hyst:
            self.hyst.update(hyst)

        self.state = GB_TRACK
        self.transitions = []           # (t, from_state, to_state) trace
        self._t = 0.0
        self._pending = None            # candidate state under hysteresis
        self._pending_t = 0.0
        self._side = None               # locked overtake side
        self._line = (self.rx, self.ry, self.rv)
        self._plan_s = None             # opponent s of the active spliner plan

    # ── Frenet helpers (public: the ROS node reuses them) ────────────────────
    def to_frenet(self, px, py):
        """World (x, y) -> (s, d) in the raceline Frenet frame."""
        return cartesian_to_frenet(px, py, self.rx, self.ry, self.s, self.total)

    def wrap(self, ds):
        """Signed along-track difference into [-total/2, total/2)."""
        return (float(ds) + self.total / 2.0) % self.total - self.total / 2.0

    # ── overtake feasibility ──────────────────────────────────────────────────
    def _side_feasible(self, opp_s, opp_d, side, lookahead=4.0):
        """Room for the evasion apex on `side` of the opponent?

        Samples the corridor from 1 m behind to `lookahead` m ahead of the
        opponent — the pass takes a while, so the room must exist a little
        down the road too, not just where the opponent is right now.
        """
        sign = 1.0 if side == 'left' else -1.0
        d_apex = float(opp_d) + sign * self.evasion_dist
        if abs(d_apex) > self.d_max:
            return False
        if self.grid_map is not None:
            for ds in np.arange(-1.0, lookahead + 0.5, 1.0):
                ax, ay = frenet_to_cartesian(
                    float(opp_s) + float(ds), d_apex,
                    self.rx, self.ry, self.s, self.total)
                if float(self.grid_map.distance_to_wall(ax, ay)) \
                        < self.wall_clearance:
                    return False
        return True

    def _pace_advantage(self, opp_s, v_opp):
        """Can the raceline profile out-run the opponent where it is?"""
        k = int(np.searchsorted(self.s, float(opp_s) % self.total,
                                side='right') - 1)
        return float(self.rv[k]) > float(v_opp) + self.dv_overtake

    def overtake_feasible(self, opp_s, opp_d, v_opp):
        """(feasible, side): outside of the turn first, then the inside."""
        if not self._pace_advantage(opp_s, v_opp):
            return False, None
        pref = choose_side(self.rx, self.ry, float(opp_s))
        other = 'right' if pref == 'left' else 'left'
        for side in (pref, other):
            if self._side_feasible(opp_s, opp_d, side):
                return True, side
        return False, None

    # ── state machine ─────────────────────────────────────────────────────────
    def _commit(self, new):
        self.transitions.append((self._t, self.state, new))
        self.state = new
        self._pending, self._pending_t = None, 0.0
        if new != OVERTAKE:
            self._side = None
            self._plan_s = None

    def update(self, px, py, ego_speed, nearest_idx, opponent, dt):
        """One behavior tick.  opponent = (s, d, v_s) Frenet or None."""
        self._t += dt
        ego_s, _ = self.to_frenet(px, py)

        gap = None
        engaged = False
        feasible, side = False, None
        if opponent is not None:
            opp_s, opp_d, v_opp = (float(v) for v in opponent)
            gap = self.wrap(opp_s - ego_s)
            band = self.trail_range if self.state == GB_TRACK else self.disengage
            engaged = 0.0 < gap < band and abs(opp_d) <= self.lateral_engage
            if engaged and self.state != OVERTAKE:
                feasible, side = self.overtake_feasible(opp_s, opp_d, v_opp)

        # desired state from the raw (un-debounced) conditions
        if self.state == GB_TRACK:
            desired = TRAILING if engaged else GB_TRACK
        elif self.state == TRAILING:
            if not engaged:
                desired = GB_TRACK
            elif feasible:
                desired = OVERTAKE
            else:
                desired = TRAILING
        else:                                            # OVERTAKE
            cleared = (opponent is None or gap < -self.clear_margin
                       or gap > self.disengage)
            if cleared:
                desired = GB_TRACK
            elif (gap > self.abort_gap and self._side is not None
                  and not (self._pace_advantage(opp_s, v_opp)
                           and self._side_feasible(opp_s, opp_d, self._side))):
                # abort ONLY while still well behind (no lateral deviation
                # yet): swapping back to the raceline while alongside would
                # steer straight into the opponent
                desired = TRAILING
            else:
                desired = OVERTAKE

        # hysteresis: the candidate must persist before it is committed
        if desired == self.state:
            self._pending, self._pending_t = None, 0.0
        else:
            if desired != self._pending:
                self._pending, self._pending_t = desired, 0.0
            self._pending_t += dt
            if self._pending_t >= self.hyst.get(desired, 0.2):
                if desired == OVERTAKE:
                    self._side = side or choose_side(self.rx, self.ry, opp_s)
                self._commit(desired)

        # ── outputs per state ────────────────────────────────────────────────
        line_changed = False
        speed_cap = float('inf')
        if self.state == OVERTAKE and opponent is not None:
            if self._plan_s is None or \
                    abs(self.wrap(opp_s - self._plan_s)) > self.replan_ds:
                # hold the apex offset past the opponent for roughly the
                # ego's lookahead so a slow pass never converges back onto
                # the opponent while still alongside
                hold = float(np.clip(0.5 * float(ego_speed), 2.0, 5.0))
                self._line = plan_overtake(
                    (self.rx, self.ry, self.rv), opp_s, opp_d, self._side,
                    evasion_dist=self.evasion_dist,
                    ego_speed=float(ego_speed), v_max=self.v_max,
                    hold_dist=hold)
                self._plan_s = opp_s
                line_changed = True
        elif self._line[0] is not self.rx:               # rejoin the raceline
            self._line = (self.rx, self.ry, self.rv)
            line_changed = True

        if self.state == TRAILING and opponent is not None and gap is not None:
            gap_des = self.gap_min + self.t_gap * float(ego_speed)
            v_cmd = v_opp + self.k_gap * (gap - gap_des)
            rl_v = float(self.rv[int(nearest_idx) % len(self.rv)])
            speed_cap = float(np.clip(v_cmd, 0.0, rl_v))

        return dict(state=self.state, line=self._line,
                    line_changed=line_changed, speed_cap=speed_cap,
                    gap=gap, side=self._side, ego_s=float(ego_s))
