#!/usr/bin/env python3
"""
Fetch precompiled/pretrained perception models — no training required.
======================================================================

Two models, exported to ONNX into models/ (then optionally compiled to
TensorRT on the Jetson):

  detector      YOLOv8n pretrained on COCO.  Detects full-size 'car'
                (class 2) and 'person' (0); detects RC cars passably at
                close range since they look like cars — set
                `car_class: 2` in camera_perception.  For best RC-car
                accuracy, fine-tune later with tools/train_car_detector.py.
  segmentation  Cityscapes-pretrained semantic segmentation for
                sidewalk_follow (road=0, sidewalk=1).

Usage (needs internet; installs ultralytics/torch into the current env —
run on a dev machine and copy models/ to the car if the Jetson is tight):

    python3 tools/get_pretrained_models.py [--detector] [--segmentation]

Then on the Jetson, compile to TensorRT (engines are device-specific):

    trtexec --onnx=models/yolov8n_coco.onnx \
            --saveEngine=models/yolov8n_coco.engine --fp16
"""

import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(REPO, 'models')


def get_detector():
    os.makedirs(MODELS, exist_ok=True)
    out = os.path.join(MODELS, 'yolov8n_coco.onnx')
    if os.path.exists(out):
        print(f'already present: {out}')
        return
    try:
        from ultralytics import YOLO
    except ImportError:
        print('installing ultralytics (pulls torch — a few minutes)...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                               'ultralytics'])
        from ultralytics import YOLO
    model = YOLO('yolov8n.pt')                     # downloads COCO weights
    path = model.export(format='onnx', imgsz=640, opset=12)
    os.replace(path, out)
    print(f'wrote {out}')
    print('configure camera_perception with: model_path -> this file, '
          'car_class: 2 (COCO car)')


def get_segmentation():
    os.makedirs(MODELS, exist_ok=True)
    out = os.path.join(MODELS, 'segmentation_cityscapes.onnx')
    if os.path.exists(out):
        print(f'already present: {out}')
        return
    # PIDNet/Fast-SCNN ONNX exports float around; the most reproducible
    # CPU-friendly route is exporting torchvision's LR-ASPP MobileNetV3
    # (trained on COCO+VOC classes) or a Cityscapes PIDNet checkpoint.
    try:
        import torch
        import torchvision
    except ImportError:
        print('installing torch/torchvision for the one-time export...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                               'torch', 'torchvision'])
        import torch
        import torchvision
    m = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
        weights='DEFAULT').eval()

    class Logits(torch.nn.Module):                 # plain (1, C, h, w) output
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x)['out']

    torch.onnx.export(Logits(m), torch.zeros(1, 3, 256, 512), out,
                      input_names=['image'], output_names=['logits'],
                      opset_version=12)
    print(f'wrote {out}')
    print('NOTE: torchvision LR-ASPP is VOC/COCO-class (no sidewalk class) — '
          'it is a wiring/smoke model.  For real sidewalk driving use a '
          'Cityscapes-trained export (e.g. PIDNet-S or Fast-SCNN ONNX from '
          'their releases) and set drivable_classes: [0, 1].')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--detector', action='store_true')
    ap.add_argument('--segmentation', action='store_true')
    args = ap.parse_args()
    if not (args.detector or args.segmentation):
        args.detector = args.segmentation = True
    if args.detector:
        get_detector()
    if args.segmentation:
        get_segmentation()


if __name__ == '__main__':
    main()
