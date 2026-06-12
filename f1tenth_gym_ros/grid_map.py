"""
Occupancy-grid map — load, query, and distance fields (pure, no ROS).
=====================================================================

One shared foundation for everything that needs the track map off-line or
on-CPU: particle-filter localization (likelihood field), occupancy-aware
raceline corridors (clearance along normals), and grid path planning
(inflated free space).  Reads the standard ROS map_server pair (YAML +
image) and exposes:

  GridMap.occupied      bool (H, W), row 0 = top of the image
  world_to_grid / grid_to_world      ROS convention: origin = lower-left
  distance_field()      metres to the nearest occupied cell, per cell
  clearance(x, y, dx, dy, max_d)     free distance along a ray (vectorized)

The distance field is computed once (scipy EDT) and cached — likelihood
lookups and clearance checks are then O(1) per query.
"""

import math
import os

import numpy as np
import yaml


def _read_image(path):
    try:
        import cv2
        img = cv2.imread(path, 0)
        if img is None:
            raise IOError(f'cannot read {path}')
        return img
    except ImportError:                            # pragma: no cover
        from PIL import Image
        return np.asarray(Image.open(path).convert('L'))


class GridMap:
    def __init__(self, occupied, resolution, origin=(0.0, 0.0)):
        self.occupied = np.asarray(occupied, bool)         # (H, W), row 0 = top
        self.res = float(resolution)
        self.ox, self.oy = float(origin[0]), float(origin[1])
        self.h, self.w = self.occupied.shape
        self._dist = None

    @classmethod
    def load(cls, yaml_path):
        """ROS map_server YAML + image -> GridMap."""
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        img_path = meta['image']
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.path.dirname(yaml_path), img_path)
        img = _read_image(img_path).astype(float) / 255.0
        occ_p = (1.0 - img) if not meta.get('negate', 0) else img
        occupied = occ_p > float(meta.get('occupied_thresh', 0.65))
        return cls(occupied, meta['resolution'], meta.get('origin', [0, 0])[:2])

    # ── coordinate transforms (ROS: world origin at the image's lower-left) ──
    def world_to_grid(self, x, y):
        """World (m) -> (row, col) arrays; may fall outside the map."""
        col = np.floor((np.asarray(x) - self.ox) / self.res).astype(int)
        row = (self.h - 1) - np.floor((np.asarray(y) - self.oy)
                                      / self.res).astype(int)
        return row, col

    def grid_to_world(self, row, col):
        x = self.ox + (np.asarray(col) + 0.5) * self.res
        y = self.oy + ((self.h - 1) - np.asarray(row) + 0.5) * self.res
        return x, y

    def in_bounds(self, row, col):
        return (np.asarray(row) >= 0) & (np.asarray(row) < self.h) & \
               (np.asarray(col) >= 0) & (np.asarray(col) < self.w)

    def is_occupied(self, x, y):
        """World coords -> occupied? (out-of-map counts as occupied)."""
        r, c = self.world_to_grid(x, y)
        inb = self.in_bounds(r, c)
        out = np.ones(np.shape(r), bool)
        out[inb] = self.occupied[np.asarray(r)[inb], np.asarray(c)[inb]]
        return out if out.shape else bool(out)

    # ── distance field: metres to nearest occupied cell ──────────────────────
    def distance_field(self):
        if self._dist is None:
            from scipy.ndimage import distance_transform_edt
            self._dist = distance_transform_edt(~self.occupied) * self.res
        return self._dist

    def distance_to_wall(self, x, y):
        """World coords -> metres to the nearest wall (0 outside the map)."""
        d = self.distance_field()
        r, c = self.world_to_grid(x, y)
        inb = self.in_bounds(r, c)
        out = np.zeros(np.shape(r), float)
        out[inb] = d[np.asarray(r)[inb], np.asarray(c)[inb]]
        return out if out.shape else float(out)

    # ── free distance along a ray (sphere tracing on the distance field) ─────
    def clearance(self, x, y, dx, dy, max_d=5.0):
        """Distance from (x, y) along unit (dx, dy) to the first wall, capped
        at max_d.  Sphere tracing: steps by the distance-field value, so a
        handful of iterations per ray."""
        d = 0.0
        for _ in range(64):
            step = self.distance_to_wall(x + d * dx, y + d * dy)
            if step < self.res * 0.5:
                return min(d, max_d)
            d += float(step)
            if d >= max_d:
                return max_d
        return min(d, max_d)


def map_path_for(raceline_csv_or_name, maps_dir=None):
    """Best-effort: find the map YAML matching a track name."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    maps_dir = maps_dir or os.path.join(repo, 'maps')
    name = os.path.basename(str(raceline_csv_or_name))
    for cand in (name, name.replace('_raceline', '_track'),
                 'comp_track.yaml'):
        cand = os.path.splitext(cand)[0] + '.yaml'
        p = os.path.join(maps_dir, cand)
        if os.path.exists(p):
            return p
    return None


def grid_raycast(grid_map, x, y, angles, max_range=12.0):
    """Simulated lidar from pose (x, y): range along each world-frame angle.

    Sphere-traces the distance field per beam (math.cos/sin loop — meant for
    benchmarks and tests, not the 50 Hz path)."""
    out = np.empty(len(angles))
    for i, a in enumerate(angles):
        out[i] = grid_map.clearance(x, y, math.cos(a), math.sin(a), max_range)
    return out
