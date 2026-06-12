"""
Navigator — slow indoor "drive to a goal" mode (pure follower core + ROS node).
===============================================================================

The non-racing autonomy mode: given the static map and an RViz "2D Goal Pose",
plan a route with grid_planner and drive it gently (hallway scale, v <= 2 m/s),
stopping at the goal.  Reuses the house foundations: GridMap (distance field,
clearance), grid_planner (inflated Dijkstra + smoothing + friction-limited
speed profile), and the same lidar AEB as raceline_mpc.

  /goal_pose ──► GridPlanner.plan ──► PathFollower ──► /drive
  /pf/pose/odom ──┘        /scan: AEB + blocked-path replan ──┘

**Why a dedicated open-path follower instead of MAPController:** the MAP
controller treats racelines as CLOSED loops — its lookahead arclength wraps
past the last point back to the first, so on an open A-to-B path the car would
chase a phantom target back at the start the moment it nears the goal.  At
navigator speeds (<= 2 m/s) MAP's tire-aware inversion is identical to the
kinematic relation anyway (it falls back to atan(L·a/v²) below 1 m/s), so a
plain pure-pursuit lookahead follower with explicit goal termination is the
same geometry minus the wrap hazard — simpler and safer here.

Safety behaviours (node):
  - AEB exactly as raceline_mpc: min lidar range in a narrow forward cone vs
    a speed-dependent stop distance  aeb_dist + v²/(2·aeb_decel)  -> hard 0.
  - Blocked-path replan: scan returns that are NOT map walls (distance field
    says free space) and sit within half a car width of the path ahead, for
    longer than `blocked_replan_s` (1 s), trigger a replan that treats those
    returns as extra obstacles.
  - Goal stop: inside `goal_tol` the follower latches done and the node
    publishes zero speed until a new goal arrives.

Run (after registering the entry point — integrator: add
    'navigator = f1tenth_gym_ros.navigator:main'
to setup.py console_scripts):

    ros2 run f1tenth_gym_ros navigator --ros-args \
        -p map_yaml:=/path/to/maps/levine.yaml -p odom_topic:=/pf/pose/odom
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_map import GridMap                                       # noqa: E402
from grid_planner import (GridPlanner, load_drivable, path_curvature,  # noqa: E402
                          DEFAULT_INFLATION)

try:                                            # pure parts importable sans ROS
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseStamped
    from ackermann_msgs.msg import AckermannDriveStamped
    from transforms3d.euler import quat2euler
    HAVE_ROS = True
except ImportError:                             # pragma: no cover
    HAVE_ROS = False
    Node = object


# ─────────────────────────── pure follower core ──────────────────────────────
class PathFollower:
    """Open-path pure-pursuit follower with explicit goal termination.

    set_path() takes the (x, y, v) arrays from GridPlanner.plan; control()
    maps pose -> (steer, speed, done).  done latches once the car is within
    goal_tol of the final point (or has overrun it), after which the command
    is (0, 0) forever — the caller decides when a new path resets it.
    """

    def __init__(self, wheelbase=0.33, max_steer=0.41,
                 q_la=0.35, m_la=0.45, la_min=0.4, la_max=1.6, k_curv=1.5,
                 goal_tol=0.3, v_floor=0.2, a_stop=1.0):
        self.L_wb = float(wheelbase)
        self.max_steer = float(max_steer)
        self.q_la, self.m_la = float(q_la), float(m_la)
        self.la_min, self.la_max = float(la_min), float(la_max)
        # curvature-aware lookahead shrink (as MAPController's k_curv):
        # shorter lookahead in corners = far less corner cutting at low speed
        self.k_curv = float(k_curv)
        self.goal_tol = float(goal_tol)
        self.v_floor = float(v_floor)
        self.a_stop = float(a_stop)             # final-approach decel law
        self.path = None
        self.done = False
        self._i = 0

    def set_path(self, x, y, v, kappa=None):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        v = np.asarray(v, float)
        if kappa is None:
            kappa = path_curvature(x, y)
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        self.path = dict(x=x, y=y, v=v, s=s, n=len(x),
                         kabs=np.abs(np.asarray(kappa, float)))
        self._i = 0
        self.done = False

    def clear(self):
        self.path = None
        self.done = False

    @property
    def goal(self):
        if self.path is None:
            return None
        return float(self.path['x'][-1]), float(self.path['y'][-1])

    def nearest_index(self):
        return self._i

    def control(self, x, y, yaw, v):
        """Pose + speed -> (steer, v_cmd, done)."""
        if self.path is None or self.done:
            return 0.0, 0.0, True
        p = self.path
        # nearest path point — windowed, monotonically advancing
        lo = max(0, self._i - 5)
        hi = min(p['n'], self._i + 60)
        seg = np.hypot(p['x'][lo:hi] - x, p['y'][lo:hi] - y)
        self._i = j = lo + int(np.argmin(seg))

        gx, gy = p['x'][-1], p['y'][-1]
        d_goal = math.hypot(gx - x, gy - y)
        behind = ((gx - x) * math.cos(yaw) + (gy - y) * math.sin(yaw)) < 0.0
        if d_goal < self.goal_tol or (j >= p['n'] - 2 and behind
                                      and d_goal < 2.0 * self.goal_tol):
            self.done = True                    # at (or just past) the goal
            return 0.0, 0.0, True

        # lookahead point by arclength (clamped to the open path's end),
        # shrunk by upcoming curvature so corners aren't cut
        v_ref = float(p['v'][j])
        L = float(np.clip(self.q_la + self.m_la * max(v, v_ref),
                          self.la_min, self.la_max))
        if self.k_curv > 0.0:
            k1 = min(int(np.searchsorted(p['s'], p['s'][j] + L)), p['n'] - 1)
            kap = float(p['kabs'][j:k1 + 1].max()) if k1 > j else 0.0
            L = float(np.clip(L / (1.0 + self.k_curv * kap),
                              self.la_min, self.la_max))
        k = int(np.searchsorted(p['s'], p['s'][j] + L))
        k = min(k, p['n'] - 1)
        tx, ty = p['x'][k], p['y'][k]
        L_d = max(math.hypot(tx - x, ty - y), 0.15)
        eta = math.atan2(ty - y, tx - x) - yaw
        eta = math.atan2(math.sin(eta), math.cos(eta))
        steer = math.atan2(2.0 * self.L_wb * math.sin(eta), L_d)
        steer = float(np.clip(steer, -self.max_steer, self.max_steer))

        # speed: profile value, braked for the final approach, floored so the
        # car keeps creeping until it is actually inside the tolerance
        v_cmd = min(v_ref, math.sqrt(2.0 * self.a_stop *
                                     max(d_goal - 0.5 * self.goal_tol, 0.0)))
        if abs(eta) > 1.0:                      # grossly misaligned: crawl
            v_cmd = min(v_cmd, 0.5)
        v_cmd = max(v_cmd, self.v_floor)
        return steer, float(v_cmd), False


def drive_to_goal(follower, x0, y0, yaw0, v0=0.0, dt=0.02,
                  wheelbase=0.33, a_accel=2.0, a_brake=4.0, t_max=180.0):
    """Open-path kinematic-bicycle sim loop (50 Hz pattern of
    tests/closed_loop.run_lap, but goal-terminated instead of lap-counted).

    Runs follower.control until it reports done and the plant has stopped.
    Returns dict(reached, t, x, y, v arrays, final_dist, final_speed).
    """
    px, py, yaw, v = float(x0), float(y0), float(yaw0), float(v0)
    gx, gy = follower.goal
    xs, ys, vs = [px], [py], [v]
    t = 0.0
    reached = False
    while t < t_max:
        steer, v_t, done = follower.control(px, py, yaw, v)
        if done and v < 0.05:
            reached = math.hypot(gx - px, gy - py) < 2.0 * follower.goal_tol
            break
        a = float(np.clip((v_t - v) / dt, -a_brake, a_accel))
        px += v * math.cos(yaw) * dt
        py += v * math.sin(yaw) * dt
        yaw += v * math.tan(float(steer)) / wheelbase * dt
        v = max(0.0, v + a * dt)
        t += dt
        xs.append(px)
        ys.append(py)
        vs.append(v)
    return dict(reached=reached, t=t, x=np.array(xs), y=np.array(ys),
                v=np.array(vs), final_dist=math.hypot(gx - px, gy - py),
                final_speed=v)


# ─────────────────────────────── ROS node ────────────────────────────────────
class Navigator(Node):
    def __init__(self):
        super().__init__('navigator')

        self.declare_parameter('map_yaml', '')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/pf/pose/odom')
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('max_steer', 0.41)
        self.declare_parameter('control_hz', 50.0)
        self.declare_parameter('inflation', DEFAULT_INFLATION)
        self.declare_parameter('v_max', 2.0)        # hallway-gentle
        self.declare_parameter('a_lat_max', 3.0)
        self.declare_parameter('v_goal', 0.5)       # arrival creep speed
        self.declare_parameter('goal_tol', 0.3)
        self.declare_parameter('aeb_dist', 0.45)    # as raceline_mpc
        self.declare_parameter('aeb_cone', 0.20)
        self.declare_parameter('aeb_decel', 6.0)
        self.declare_parameter('blocked_replan_s', 1.0)
        self.declare_parameter('obstacle_clear', 0.25)  # scan pt not a wall if
        self.declare_parameter('corridor_radius', 0.35)  # blocking the path if

        map_yaml = self.get_parameter('map_yaml').value
        if not map_yaml or not os.path.exists(map_yaml):
            self.get_logger().error(
                'map_yaml parameter missing or not found — pass the static '
                'map, e.g. -p map_yaml:=maps/levine.yaml')
            raise FileNotFoundError('map_yaml')
        self.L = float(self.get_parameter('wheelbase').value)
        self.max_steer = float(self.get_parameter('max_steer').value)
        self.inflation = float(self.get_parameter('inflation').value)
        self.aeb_dist = float(self.get_parameter('aeb_dist').value)
        self.aeb_cone = float(self.get_parameter('aeb_cone').value)
        self.aeb_decel = float(self.get_parameter('aeb_decel').value)
        self.blocked_replan_s = float(
            self.get_parameter('blocked_replan_s').value)
        self.obstacle_clear = float(self.get_parameter('obstacle_clear').value)
        self.corridor_radius = float(
            self.get_parameter('corridor_radius').value)

        self.gm = GridMap.load(map_yaml)
        self.gm.distance_field()                       # warm the cache
        self.planner = GridPlanner(self.gm, self.inflation,
                                   drivable=load_drivable(map_yaml))
        self.follower = PathFollower(
            wheelbase=self.L, max_steer=self.max_steer,
            goal_tol=float(self.get_parameter('goal_tol').value))
        self.get_logger().info(
            f'navigator ready — map={os.path.basename(map_yaml)} '
            f'({self.gm.h}x{self.gm.w} @ {self.gm.res:.3f} m), '
            f'inflation={self.inflation:.2f} m, '
            f'{self.planner.n_nodes} drivable cells')

        self.x = self.y = self.yaw = self.speed = 0.0
        self.have_odom = False
        self.scan = None
        self.goal = None
        self._pending_goal = False
        self._blocked_since = None
        self._goal_logged = False
        self._log = 0
        self._cone_key = None                          # cached AEB cone mask
        self._cone_mask = None

        self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value,
            self._scan_cb, 10)
        self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value,
            self._odom_cb, 10)
        self.create_subscription(
            PoseStamped, self.get_parameter('goal_topic').value,
            self._goal_cb, 10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.get_parameter('drive_topic').value, 10)
        self.create_timer(
            1.0 / float(self.get_parameter('control_hz').value), self._loop)

    # ── callbacks ────────────────────────────────────────────────────────────
    def _scan_cb(self, m):
        self.scan = m

    def _odom_cb(self, m):
        self.x = m.pose.pose.position.x
        self.y = m.pose.pose.position.y
        self.speed = float(np.hypot(m.twist.twist.linear.x,
                                    m.twist.twist.linear.y))
        q = m.pose.pose.orientation
        _, _, self.yaw = quat2euler([q.w, q.x, q.y, q.z])
        self.have_odom = True

    def _goal_cb(self, m):
        self.goal = (float(m.pose.position.x), float(m.pose.position.y))
        self._goal_logged = False
        if not self.have_odom:
            self.get_logger().warning('goal received before odom — deferred')
            self._pending_goal = True
            return
        self._plan(extra_obstacles=None)

    # ── planning ─────────────────────────────────────────────────────────────
    def _plan(self, extra_obstacles=None):
        import time
        t0 = time.perf_counter()
        result = self.planner.plan(
            (self.x, self.y), self.goal,
            v_max=float(self.get_parameter('v_max').value),
            a_lat_max=float(self.get_parameter('a_lat_max').value),
            v_goal=float(self.get_parameter('v_goal').value),
            extra_obstacles=extra_obstacles)
        ms = (time.perf_counter() - t0) * 1e3
        self._blocked_since = None
        if result is None:
            self.follower.clear()
            self.get_logger().warning(
                f'no path to goal {self.goal} ({ms:.0f} ms) — stopping')
            return
        self.follower.set_path(result['x'], result['y'], result['v'],
                               kappa=result['kappa'])
        self.get_logger().info(
            f"planned {result['length']:.1f} m, "
            f"v<= {result['v'].max():.1f} m/s ({ms:.0f} ms)")

    # ── AEB: min range in a narrow forward cone (as raceline_mpc) ───────────
    def _forward_clear(self):
        s = self.scan
        r = np.asarray(s.ranges, np.float32)
        key = (len(r), s.angle_min, s.angle_increment)
        if key != self._cone_key:
            ang = s.angle_min + np.arange(len(r)) * s.angle_increment
            self._cone_mask = np.abs(ang) < self.aeb_cone
            self._cone_key = key
        r = np.where(np.isfinite(r) & (r > 0.03), r, 30.0)
        cone = self._cone_mask
        return float(r[cone].min()) if cone.any() else 30.0

    # ── blocked-path detection: scan obstacles on the route ahead ───────────
    def _scan_obstacles(self):
        """World points of scan returns that are NOT map walls."""
        s = self.scan
        r = np.asarray(s.ranges, np.float32)[::4]
        ang = (s.angle_min
               + np.arange(len(s.ranges), dtype=np.float32)
               * s.angle_increment)[::4]
        ok = np.isfinite(r) & (r > 0.05) & (r < 6.0)
        r, ang = r[ok], ang[ok]
        px = self.x + r * np.cos(self.yaw + ang)
        py = self.y + r * np.sin(self.yaw + ang)
        new = self.gm.distance_to_wall(px, py) > self.obstacle_clear
        return np.column_stack([px[new], py[new]])

    def _path_blocked(self, obstacles):
        if len(obstacles) == 0 or self.follower.path is None:
            return False
        p = self.follower.path
        i = self.follower.nearest_index()
        j = int(np.searchsorted(p['s'], p['s'][i] + 5.0))   # next 5 m
        ax = p['x'][i:j + 1]
        ay = p['y'][i:j + 1]
        if len(ax) == 0:
            return False
        d2 = ((obstacles[:, 0, None] - ax[None, :]) ** 2
              + (obstacles[:, 1, None] - ay[None, :]) ** 2)
        return bool(d2.min() < self.corridor_radius ** 2)

    # ── control loop ─────────────────────────────────────────────────────────
    def _loop(self):
        self._log += 1
        if not self.have_odom:
            return
        if self._pending_goal:                  # goal arrived before odom
            self._pending_goal = False
            self._plan(extra_obstacles=None)
        steer, v_cmd, done = self.follower.control(
            self.x, self.y, self.yaw, self.speed)
        if done and self.follower.path is not None and not self._goal_logged:
            self.get_logger().info('goal reached — stopping')
            self._goal_logged = True

        if self.scan is not None:
            # blocked-path replan (checked at ~10 Hz; replan after 1 s solid)
            if not done and self._log % 5 == 0:
                obstacles = self._scan_obstacles()
                if self._path_blocked(obstacles):
                    now = self.get_clock().now().nanoseconds * 1e-9
                    if self._blocked_since is None:
                        self._blocked_since = now
                    elif now - self._blocked_since > self.blocked_replan_s:
                        self.get_logger().warning(
                            'path blocked > %.1f s — replanning around it'
                            % self.blocked_replan_s)
                        self._plan(extra_obstacles=obstacles)
                        steer, v_cmd, done = self.follower.control(
                            self.x, self.y, self.yaw, self.speed)
                else:
                    self._blocked_since = None
            # AEB — speed-dependent stop distance, exactly as raceline_mpc
            stop_dist = (self.aeb_dist
                         + self.speed ** 2 / (2.0 * self.aeb_decel))
            if self._forward_clear() < stop_dist:
                steer, v_cmd = 0.0, 0.0
                if self._log % 10 == 0:
                    self.get_logger().warning('AEB — obstacle ahead, stopping')

        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(
            np.clip(steer, -self.max_steer, self.max_steer))
        msg.drive.speed = float(v_cmd)
        self.drive_pub.publish(msg)
        if self._log % 100 == 0 and self.follower.path is not None and not done:
            self.get_logger().info(
                f"wp {self.follower.nearest_index()}/{self.follower.path['n']}"
                f' v={v_cmd:.1f} steer={math.degrees(steer):.0f}deg')


def main(args=None):                            # pragma: no cover
    if not HAVE_ROS:
        raise RuntimeError('rclpy not available — navigator node needs ROS 2')
    rclpy.init(args=args)
    try:
        node = Navigator()
        rclpy.spin(node)
    except (FileNotFoundError, KeyboardInterrupt):
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
