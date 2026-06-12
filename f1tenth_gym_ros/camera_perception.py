"""
Camera perception — detect other cars with a trained YOLO model and feed the
same race brain the lidar does.
=============================================================================

This runs on the **real car** (the f1tenth_gym sim has no camera).  Inference
picks the fastest available backend at startup (`backend: auto`):

  1. **TensorRT** — a `.engine` file built on the Jetson (`trtexec
     --onnx=car_yolov8.onnx --saveEngine=car_yolov8.engine --fp16`) runs on
     the GPU at FP16; the right way to use the Jetson for perception.
  2. **cv2.dnn CUDA** — the ONNX through OpenCV's CUDA backend (needs the
     Jetson's CUDA-enabled OpenCV build).
  3. **cv2.dnn CPU** — always-works fallback; no GPU required.

(Training happens separately — see `tools/train_car_detector.py`.  The
control loop stays on CPU on purpose: its QP is far too small to benefit
from a GPU; the camera pipeline is where the Jetson's GPU earns its keep.)

Pipeline:  image  ->  YOLO car boxes  ->  back-project each box to a position
relative to the car (pinhole + known car width)  ->  `race_brain.Opponent`
objects, in the **same format the lidar detector emits**, so they drop straight
into `RaceStrategist` or fuse with the lidar tracks.

The output is sensor-agnostic on purpose: camera gives "what + bearing", lidar
gives precise range; fusing them (camera class/bearing + lidar range) is the
robust setup, and both already speak the `Opponent` type.

Standalone (no ROS) test of the detector + geometry:
    from camera_perception import CarDetector, box_to_relative
"""

import math
import os

import numpy as np

try:
    import cv2
except Exception:                      # pragma: no cover
    cv2 = None


# ─────────────────────────────────────────────────────────────────────────────
# Detectors — TensorRT (Jetson GPU) or YOLOv8 ONNX via cv2.dnn (CUDA/CPU)
# ─────────────────────────────────────────────────────────────────────────────

def parse_yolo_output(out, w0, h0, img_size, conf, nms, car_class):
    """Raw YOLOv8 head (1, 4+nc, N) -> [(x, y, w, h, confidence), ...] px."""
    out = np.squeeze(np.asarray(out))                   # (4+nc, N)
    if out.ndim != 2:
        return []
    if out.shape[0] < out.shape[1]:                     # (4+nc, N) -> (N, 4+nc)
        out = out.T
    cls = out[:, 4:]
    cids = np.argmax(cls, axis=1)
    confs = cls[np.arange(len(cls)), cids]
    keep = (confs > conf) & (cids == car_class)
    if not keep.any():
        return []
    rows, confs = out[keep], confs[keep]
    sx, sy = w0 / img_size, h0 / img_size
    cx, cy, bw, bh = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    boxes = np.stack([(cx - bw/2)*sx, (cy - bh/2)*sy, bw*sx, bh*sy], 1)
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), confs.tolist(), conf, nms)
    if len(idx) == 0:
        return []
    idx = np.array(idx).flatten()
    return [(*boxes[i], float(confs[i])) for i in idx]


class CarDetector:
    """ONNX via cv2.dnn — CUDA target on the Jetson GPU, or CPU fallback."""

    def __init__(self, model_path, img_size=640, conf=0.35, nms=0.45,
                 car_class=0, use_cuda=False):
        if cv2 is None:
            raise RuntimeError('OpenCV not available')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f'model not found: {model_path} — train one with '
                f'tools/train_car_detector.py and export to ONNX')
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.backend = 'cpu'
        if use_cuda:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            self.backend = 'cuda'
        self.sz, self.conf, self.nms, self.car_class = img_size, conf, nms, car_class

    def detect(self, img):
        """BGR image -> list of (x, y, w, h, confidence) car boxes in image px."""
        h0, w0 = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (self.sz, self.sz),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()                       # (1, 4+nc, N)
        return parse_yolo_output(out, w0, h0, self.sz, self.conf, self.nms,
                                 self.car_class)


class TRTDetector:
    """YOLOv8 TensorRT engine on the Jetson GPU (FP16).

    Build once on the target device (TensorRT engines are not portable):
        trtexec --onnx=car_yolov8.onnx --saveEngine=car_yolov8.engine --fp16
    Same detect() interface and output parsing as CarDetector.
    """

    backend = 'tensorrt'

    def __init__(self, engine_path, img_size=640, conf=0.35, nms=0.45,
                 car_class=0):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit                          # noqa: F401 — CUDA ctx
        self._cuda = cuda
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f'engine not found: {engine_path}')
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.sz, self.conf, self.nms, self.car_class = img_size, conf, nms, car_class
        # one input, one output binding; allocate page-locked host + device
        self._host, self._dev, self._shapes = [], [], []
        for i in range(self.engine.num_bindings):
            shape = tuple(self.ctx.get_binding_shape(i))
            n = int(np.prod(shape))
            self._host.append(cuda.pagelocked_empty(n, np.float32))
            self._dev.append(cuda.mem_alloc(self._host[-1].nbytes))
            self._shapes.append(shape)
        self.stream = cuda.Stream()

    def detect(self, img):
        h0, w0 = img.shape[:2]
        if cv2 is not None:
            blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (self.sz, self.sz),
                                         swapRB=True, crop=False)
        else:                                           # pragma: no cover
            raise RuntimeError('OpenCV needed for preprocessing')
        np.copyto(self._host[0], blob.ravel())
        cuda = self._cuda
        cuda.memcpy_htod_async(self._dev[0], self._host[0], self.stream)
        self.ctx.execute_async_v2([int(d) for d in self._dev], self.stream.handle)
        cuda.memcpy_dtoh_async(self._host[1], self._dev[1], self.stream)
        self.stream.synchronize()
        out = self._host[1].reshape(self._shapes[1])
        return parse_yolo_output(out, w0, h0, self.sz, self.conf, self.nms,
                                 self.car_class)


def make_detector(model_path, backend='auto', **kw):
    """backend: auto | tensorrt | cuda | cpu.  `auto` prefers a TensorRT
    .engine next to (or instead of) the ONNX, then cv2-CUDA, then CPU."""
    engine = model_path if model_path.endswith(('.engine', '.trt')) \
        else os.path.splitext(model_path)[0] + '.engine'
    if backend in ('auto', 'tensorrt') and os.path.exists(engine):
        try:
            return TRTDetector(engine, **kw)
        except Exception:
            if backend == 'tensorrt':
                raise
    if backend == 'tensorrt':
        raise FileNotFoundError(f'no TensorRT engine at {engine}')
    has_cuda = (cv2 is not None
                and cv2.cuda.getCudaEnabledDeviceCount() > 0) \
        if backend == 'auto' else (backend == 'cuda')
    return CarDetector(model_path, use_cuda=has_cuda, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry — image box -> position relative to the car
# ─────────────────────────────────────────────────────────────────────────────

def box_to_relative(box, fx, cx_img, car_width=0.30):
    """
    Pinhole back-projection with a known object width.
    box = (x, y, w, h) px.  Returns (x_forward, y_left, range) in metres, in the
    camera/base frame (ROS: +X forward, +Y left).
    """
    x, y, w, h = box[:4]
    u_center = x + w / 2.0
    depth = car_width * fx / max(w, 1.0)               # along camera optical axis
    y_left = -(u_center - cx_img) * depth / fx         # right of centre -> -Y
    rng = math.hypot(depth, y_left)
    return depth, y_left, rng


def relative_to_world(x_fwd, y_left, ego):
    ex, ey, eyaw = ego
    wx = ex + x_fwd * math.cos(eyaw) - y_left * math.sin(eyaw)
    wy = ey + x_fwd * math.sin(eyaw) + y_left * math.cos(eyaw)
    return wx, wy


# ─────────────────────────────────────────────────────────────────────────────
# ROS node (only imported/used on the car)
# ─────────────────────────────────────────────────────────────────────────────

def _make_node():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from nav_msgs.msg import Odometry
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import PoseArray, Pose
    from std_msgs.msg import ColorRGBA
    from transforms3d.euler import quat2euler

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from race_brain import Opponent, OpponentDetector  # noqa: reuse the tracker

    class CameraPerception(Node):
        def __init__(self):
            super().__init__('camera_perception')
            self.declare_parameter('image_topic', '/camera/color/image_raw')
            self.declare_parameter('model_path', os.path.join(
                os.path.dirname(__file__), '..', 'models', 'car_yolov8.onnx'))
            self.declare_parameter('fx', 600.0)        # camera intrinsics
            self.declare_parameter('cx', 320.0)
            self.declare_parameter('car_width', 0.30)
            self.declare_parameter('conf', 0.35)
            self.declare_parameter('odom_topic', '/ego_racecar/odom')
            self.declare_parameter('backend', 'auto')  # auto|tensorrt|cuda|cpu
            self.declare_parameter('car_class', 0)     # 0 custom model; 2 COCO
            topic = self.get_parameter('image_topic').value
            self.fx = float(self.get_parameter('fx').value)
            self.cx = float(self.get_parameter('cx').value)
            self.car_w = float(self.get_parameter('car_width').value)

            try:
                self.detector = make_detector(
                    self.get_parameter('model_path').value,
                    backend=self.get_parameter('backend').value,
                    conf=float(self.get_parameter('conf').value),
                    car_class=int(self.get_parameter('car_class').value))
                self.get_logger().info(
                    f'YOLO car detector loaded (backend: {self.detector.backend})')
            except Exception as e:
                self.detector = None
                self.get_logger().error(f'No detector ({e}); node idle until a model exists')

            # Reuse the lidar tracker purely for smoothing camera detections.
            self.tracker = OpponentDetector()
            self.ego = (0.0, 0.0, 0.0)
            odom_topic = self.get_parameter('odom_topic').value
            self.create_subscription(Odometry, odom_topic, self._odom, 10)
            self.create_subscription(Image, topic, self._image, 5)
            self.opp_pub = self.create_publisher(MarkerArray, '/camera_opponents', 5)
            self.pose_pub = self.create_publisher(PoseArray, '/camera_opponents_poses', 5)
            self.Marker, self.MarkerArray = Marker, MarkerArray
            self.PoseArray, self.Pose = PoseArray, Pose
            self.ColorRGBA = ColorRGBA
            self.get_logger().info(f'camera_perception subscribed to {topic}')

        def _odom(self, m):
            q = m.pose.pose.orientation
            from transforms3d.euler import quat2euler as q2e
            _, _, yaw = q2e([q.w, q.x, q.y, q.z])
            self.ego = (m.pose.pose.position.x, m.pose.pose.position.y, yaw)

        def _image(self, msg):
            if self.detector is None:
                return
            img = self._decode(msg)
            if img is None:
                return
            opps = []
            for box in self.detector.detect(img):
                xf, yl, rng = box_to_relative(box, self.fx, self.cx, self.car_w)
                wx, wy = relative_to_world(xf, yl, self.ego)
                o = Opponent(wx, wy, rng, self.car_w, math.atan2(yl, xf))
                opps.append(o)
            self._publish(opps)

        @staticmethod
        def _decode(msg):
            try:
                from cv_bridge import CvBridge
                return CvBridge().imgmsg_to_cv2(msg, 'bgr8')
            except Exception:
                if msg.encoding in ('rgb8', 'bgr8'):
                    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
                    return a[:, :, ::-1] if msg.encoding == 'rgb8' else a
                return None

        def _publish(self, opps):
            pa = self.PoseArray()
            pa.header.frame_id = 'map'
            for o in opps:
                p = self.Pose()
                p.position.x, p.position.y = float(o.x), float(o.y)
                p.orientation.w = 1.0
                pa.poses.append(p)
            self.pose_pub.publish(pa)

            arr = self.MarkerArray()
            for k, o in enumerate(opps):
                m = self.Marker()
                m.header.frame_id = 'map'; m.ns = 'camera_opp'; m.id = k
                m.type = self.Marker.CUBE; m.action = self.Marker.ADD
                m.pose.position.x, m.pose.position.y, m.pose.position.z = o.x, o.y, 0.1
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = 0.4; m.scale.z = 0.2
                m.color = self.ColorRGBA(r=0.1, g=0.6, b=0.9, a=0.9)
                arr.markers.append(m)
            self.opp_pub.publish(arr)

    return CameraPerception


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = _make_node()()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
