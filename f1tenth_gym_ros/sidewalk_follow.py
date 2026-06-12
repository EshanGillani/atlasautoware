"""
Sidewalk follower — vision-based outdoor driving mode (mini self-driving).
==========================================================================

Map-free outdoor mode: a pretrained semantic-segmentation model marks the
drivable surface (sidewalk/path/road class) in the camera image; the car
steers toward the drivable region's centroid in a lower image band, governed
by the lidar AEB and a drivable-fraction sanity stop.  No map, no GPS —
this follows the paved ribbon in front of it.

  /oakd/rgb ─► segmentation (TensorRT / cv2.dnn CUDA / CPU) ─► drivable mask
  mask ─► band centroid ─► steer        /scan ─► AEB ─► /drive (low speed)

Honest scope: this is supervised, line-of-sight, "follow the sidewalk"
driving — it does not understand intersections, crossings, traffic, or
people beyond stop-if-something-is-close.  Run it ONLY with the RC
kill-switch gate armed (`enable_topic` on the drive node + rc_monitor) and
walk behind the car.

Model: any segmentation ONNX whose output is (1, C, h, w) class logits.
Cityscapes-pretrained models work out of the box (class 1 = sidewalk,
0 = road); see tools/get_pretrained_models.py.  The mask -> steering core
is pure numpy and unit-tested on synthetic masks.
"""

import math

import numpy as np

try:
    import cv2
except Exception:                      # pragma: no cover
    cv2 = None


# ─────────────────────────────────────────────────────────────────────────────
# Pure core: drivable mask -> steering (unit-tested, no ROS, no model)
# ─────────────────────────────────────────────────────────────────────────────

def mask_to_steering(mask, band=(0.55, 0.95), min_fraction=0.06,
                     gain=1.8, max_steer=0.41):
    """Boolean drivable mask (H, W) -> (steer rad, drivable_fraction, ok).

    Looks at the image band `band` (fractions of height — near the car,
    below the horizon), takes the drivable pixels' column centroid weighted
    toward nearer rows, and steers proportionally to its offset from the
    image centre (left of centre -> positive/left steer, REP 103).  `ok`
    goes False when less than `min_fraction` of the band is drivable —
    the surface ran out: stop instead of guessing.
    """
    m = np.asarray(mask, bool)
    h, w = m.shape
    r0, r1 = int(band[0] * h), int(band[1] * h)
    band_m = m[r0:r1]
    frac = float(band_m.mean()) if band_m.size else 0.0
    if frac < float(min_fraction):
        return 0.0, frac, False
    rows, cols = np.nonzero(band_m)
    row_w = 0.5 + rows / max(r1 - r0 - 1, 1)        # nearer rows weigh more
    centroid = float(np.average(cols, weights=row_w))
    offset = (centroid - (w - 1) / 2.0) / (w / 2.0)  # [-1, 1], + = right
    steer = float(np.clip(-gain * offset * max_steer, -max_steer, max_steer))
    return steer, frac, True


def speed_from_confidence(frac, v_cruise=1.2, v_min=0.4, full_at=0.30):
    """Less visible sidewalk -> slower.  frac >= full_at gives cruise speed."""
    scale = min(1.0, frac / float(full_at))
    return max(float(v_min), float(v_cruise) * scale)


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation backend (same accelerator story as camera_perception)
# ─────────────────────────────────────────────────────────────────────────────

class Segmenter:
    """ONNX semantic segmentation via cv2.dnn (CUDA FP16 when available).

    Output (1, C, h, w) logits; `drivable_classes` are OR-ed into the mask
    (Cityscapes: road=0, sidewalk=1 — default follows sidewalks AND road,
    drop 0 from the list to refuse roadways).
    """

    def __init__(self, model_path, input_size=(512, 256),
                 drivable_classes=(0, 1), use_cuda=False):
        if cv2 is None:
            raise RuntimeError('OpenCV not available')
        self.net = cv2.dnn.readNetFromONNX(model_path)
        if use_cuda:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        self.size = tuple(input_size)
        self.classes = tuple(drivable_classes)

    def drivable_mask(self, img_bgr):
        blob = cv2.dnn.blobFromImage(img_bgr, 1 / 255.0, self.size,
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()                    # (1, C, h, w)
        labels = np.argmax(out[0], axis=0)
        mask = np.isin(labels, self.classes)
        return cv2.resize(mask.astype(np.uint8),
                          (img_bgr.shape[1], img_bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
# ROS node
# ─────────────────────────────────────────────────────────────────────────────

def _make_node():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image, LaserScan
    from ackermann_msgs.msg import AckermannDriveStamped

    class SidewalkFollow(Node):
        def __init__(self):
            super().__init__('sidewalk_follow')
            self.declare_parameter('image_topic', '/oakd/rgb')
            self.declare_parameter('scan_topic', '/scan')
            self.declare_parameter('drive_topic', '/drive')
            self.declare_parameter('model_path', '')
            self.declare_parameter('drivable_classes', [0, 1])  # road+sidewalk
            self.declare_parameter('use_cuda', True)
            self.declare_parameter('v_cruise', 1.2)    # m/s — walking pace
            self.declare_parameter('max_steer', 0.41)
            self.declare_parameter('steer_gain', 1.8)
            self.declare_parameter('aeb_dist', 0.8)    # generous outdoors
            self.declare_parameter('aeb_cone', 0.35)
            p = lambda n: self.get_parameter(n).value   # noqa: E731
            self.v_cruise = float(p('v_cruise'))
            self.max_steer = float(p('max_steer'))
            self.gain = float(p('steer_gain'))
            self.aeb_dist = float(p('aeb_dist'))
            self.aeb_cone = float(p('aeb_cone'))

            try:
                self.seg = Segmenter(
                    p('model_path'),
                    drivable_classes=tuple(p('drivable_classes')),
                    use_cuda=bool(p('use_cuda')))
                self.get_logger().info('segmentation model loaded')
            except Exception as e:
                self.seg = None
                self.get_logger().error(
                    f'no segmentation model ({e}) — run '
                    f'tools/get_pretrained_models.py; node idle')
            self.scan_clear = True
            qos = QoSProfile(depth=1,
                             reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(Image, p('image_topic'),
                                     self._image_cb, qos)
            self.create_subscription(LaserScan, p('scan_topic'),
                                     self._scan_cb, qos)
            self.pub = self.create_publisher(AckermannDriveStamped,
                                             p('drive_topic'), 1)
            self.get_logger().warning(
                'sidewalk_follow is a SUPERVISED outdoor mode — arm the RC '
                'kill switch (rc_monitor + drive_node enable_topic) and '
                'stay within reach')

        def _scan_cb(self, m):
            r = np.asarray(m.ranges, np.float32)
            ang = m.angle_min + np.arange(len(r)) * m.angle_increment
            cone = np.abs(ang) < self.aeb_cone
            r = np.where(np.isfinite(r) & (r > 0.05), r, 30.0)
            self.scan_clear = float(r[cone].min()) > self.aeb_dist \
                if cone.any() else True

        def _image_cb(self, msg):
            if self.seg is None:
                return
            img = np.frombuffer(msg.data, np.uint8).reshape(
                msg.height, msg.width, 3)
            if msg.encoding == 'rgb8':
                img = img[:, :, ::-1]
            mask = self.seg.drivable_mask(img)
            steer, frac, ok = mask_to_steering(
                mask, gain=self.gain, max_steer=self.max_steer)
            out = AckermannDriveStamped()
            if ok and self.scan_clear:
                out.drive.steering_angle = steer
                out.drive.speed = speed_from_confidence(
                    frac, v_cruise=self.v_cruise)
            else:
                out.drive.steering_angle = 0.0
                out.drive.speed = 0.0               # surface lost / obstacle
            self.pub.publish(out)

    return rclpy, SidewalkFollow


def main(args=None):
    rclpy, NodeCls = _make_node()
    rclpy.init(args=args)
    try:
        rclpy.spin(NodeCls())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
