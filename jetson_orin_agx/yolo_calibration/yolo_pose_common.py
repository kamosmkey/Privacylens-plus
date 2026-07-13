#!/usr/bin/env python3
import os

import cv2
import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from ultralytics import YOLO


def resolve_device(device):
    if device != "auto":
        return device
    return 0 if torch.cuda.is_available() else "cpu"


def load_yolo_pose(model_path, device="auto", fuse=False):
    model = YOLO(model_path)
    if fuse:
        try:
            model.fuse()
        except Exception:
            pass
    return model, resolve_device(device)


def open_rgb_camera(dev, width, height):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened() and dev.startswith("/dev/video"):
        cap = cv2.VideoCapture(int(dev.replace("/dev/video", "")))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RGB camera {dev}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def run_pose(model, bgr_image, img_size, conf_thres, iou_thres, device, quantize=None):
    result = run_pose_result(model, bgr_image, img_size, conf_thres, iou_thres, device, quantize)
    if result is None:
        return []
    return postprocess_ultralytics_pose(result, bgr_image.shape)


def run_pose_result(model, bgr_image, img_size, conf_thres, iou_thres, device, quantize=None):
    predict_kwargs = {
        "source": bgr_image,
        "imgsz": img_size,
        "conf": conf_thres,
        "iou": iou_thres,
        "device": device,
        "verbose": False,
    }
    if quantize is not None:
        predict_kwargs["quantize"] = quantize

    results = model.predict(**predict_kwargs)
    if not results:
        return None
    return results[0]


def postprocess_ultralytics_pose(result, image_shape):
    if result.boxes is None or result.keypoints is None:
        return []

    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    kpt_xy = result.keypoints.xy.detach().cpu().numpy().astype(np.float32)

    if result.keypoints.conf is None:
        kpt_conf = np.ones(kpt_xy.shape[:2], dtype=np.float32)
    else:
        kpt_conf = result.keypoints.conf.detach().cpu().numpy().astype(np.float32)

    h, w = image_shape[:2]
    detections = []
    for box, score, xy, conf in zip(boxes_xyxy, scores, kpt_xy, kpt_conf):
        x1, y1, x2, y2 = box
        x1 = np.clip(x1, 0, w - 1)
        y1 = np.clip(y1, 0, h - 1)
        x2 = np.clip(x2, 0, w - 1)
        y2 = np.clip(y2, 0, h - 1)
        if x2 <= x1 or y2 <= y1:
            continue

        xy[:, 0] = np.clip(xy[:, 0], 0, w - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)
        kpts = np.concatenate((xy, conf[:, None]), axis=1).astype(np.float32)
        detections.append((np.array([x1, y1, x2, y2], dtype=np.float32), float(score), kpts))

    return detections


def select_primary_person(detections):
    if not detections:
        return None

    def rank(det):
        box, score, _ = det
        x1, y1, x2, y2 = box
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return score * max(1.0, area)

    return max(detections, key=rank)
