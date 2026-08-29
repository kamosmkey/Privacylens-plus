#!/usr/bin/env python3
"""Static beamsplitter thermal-to-RGB privacy warp for RK3588/RKNNLite.

The thermal image is aligned with a fixed flip/scale/translation.  RGB pose is
run on the NPU; hot pixels form the privacy mask and the detected skeleton is
drawn over a captured empty-scene background.  Press ``b`` to capture the
background and ``q``/Esc to quit.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from .thermal_mask import (
    COCO_SKELETON,
    decode_yolo_pose,
    letterbox,
    load_rknn,
    nms,
    open_thermal_camera,
    parse_temperature,
    temp_to_display,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default=str(
            SCRIPT_DIR
            / "model/yolo26n-pose_rknn_model/best.sanitized-rk3588.rknn"
        ),
    )
    p.add_argument("--thermal-dev", default="/dev/video4")
    p.add_argument("--thermal-fps", type=float, default=25.0)
    p.add_argument("--rgb-dev", default="/dev/video1")
    p.add_argument("--rgb-width", type=int, default=640)
    p.add_argument("--rgb-height", type=int, default=480)
    p.add_argument("--rgb-fps", type=float, default=30.0)
    p.add_argument("--rgb-fourcc", default="YUYV")
    p.add_argument("--rgb-calibration", default=str(SCRIPT_DIR / "calibration_standard.npz"))
    p.add_argument("--rgb-balance", type=float, default=0.0)
    p.add_argument("--no-undistort", action="store_true", help="run without RGB calibration")
    p.add_argument("--thermal-scale", type=float, default=0.740)
    p.add_argument("--thermal-x", type=float, default=5.0)
    p.add_argument("--thermal-y", type=float, default=15.0)
    flip = p.add_mutually_exclusive_group()
    flip.add_argument("--thermal-flip", dest="thermal_flip", action="store_true")
    flip.add_argument("--no-thermal-flip", dest="thermal_flip", action="store_false")
    p.set_defaults(thermal_flip=True)
    p.add_argument("--crop-to-thermal", action="store_true")
    p.add_argument("--thermal-rgb-offset-ms", type=float, default=50.0)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--kpt-conf", type=float, default=0.4)
    p.add_argument("--mask-min-temp", type=float, default=26.0)
    p.add_argument("--mask-max-temp", type=float, default=36.0)
    p.add_argument("--mask-percentile", type=float)
    p.add_argument("--mask-alpha", type=float, default=0.45)
    p.add_argument("--warped-mask-dilate-px", type=int, default=10)
    p.add_argument("--print-every", type=int, default=30)
    p.add_argument("--display-scale", type=float, default=1.0)
    p.add_argument("--no-display", action="store_true", help="benchmark without GUI")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--test", action="store_true", help="show aligned thermal and pose diagnostics")
    return p.parse_args()


def fps_from_times(times, now):
    while times and now - times[0] > 3.0:
        times.popleft()
    return (len(times) - 1) / max(1e-6, times[-1] - times[0]) if len(times) > 1 else 0.0


def open_rgb_camera(args):
    cap = cv2.VideoCapture(args.rgb_dev, cv2.CAP_V4L2)
    if not cap.isOpened() and args.rgb_dev.startswith("/dev/video"):
        cap = cv2.VideoCapture(int(args.rgb_dev[10:]))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RGB camera {args.rgb_dev}")
    if len(args.rgb_fourcc) != 4:
        cap.release()
        raise ValueError("--rgb-fourcc must contain four characters")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.rgb_fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.rgb_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.rgb_height)
    cap.set(cv2.CAP_PROP_FPS, args.rgb_fps)
    # Keep the same V4L2 capture settings as the original beamsplitter script.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)
    return cap


def capture_description(cap):
    value = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((value >> (8 * i)) & 0xff) for i in range(4))
    return (
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
        f"{fourcc} @ {cap.get(cv2.CAP_PROP_FPS):g} fps"
    )


def verify_camera_read(cap, name, attempts=10):
    """Read synchronously before handing a VideoCapture to its worker thread."""
    for _ in range(attempts):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            return frame
        time.sleep(0.03)
    raise RuntimeError(
        f"{name} opened but cannot read a frame "
        f"(negotiated {capture_description(cap)})"
    )


class TimestampedCapture:
    def __init__(self, cap, name, maxlen):
        self.cap, self.name = cap, name
        self.frames, self.times = deque(maxlen=maxlen), deque()
        self.condition, self.stop_event = threading.Condition(), threading.Event()
        self.sequence = self.failed = self.consecutive_failed = 0
        self.capture_fps = 0.0
        self.thread = threading.Thread(target=self._run, name=f"{name}-capture", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            now = time.perf_counter()
            with self.condition:
                if ok:
                    self.sequence += 1
                    self.consecutive_failed = 0
                    self.times.append(now)
                    self.capture_fps = fps_from_times(self.times, now)
                    self.frames.append((now, frame, self.sequence))
                else:
                    self.failed += 1
                    self.consecutive_failed += 1
                self.condition.notify_all()
            if not ok:
                # Some V4L2 drivers return immediately on a bad node/format.
                # Avoid a CPU-burning retry loop and an unhelpfully huge count.
                self.stop_event.wait(0.02)

    def wait_newest(self, sequence, timeout=1.0):
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence != sequence
                or self.consecutive_failed >= 30
                or self.stop_event.is_set(),
                timeout,
            )
            if self.consecutive_failed >= 30:
                raise RuntimeError(
                    f"{self.name} camera failed to read "
                    f"{self.consecutive_failed} consecutive frames"
                )
            return self.frames[-1] if self.frames and self.sequence != sequence else None

    def snapshot(self):
        with self.condition:
            return list(self.frames)

    def stats(self):
        with self.condition:
            return self.capture_fps, self.failed

    def raise_if_failed(self):
        with self.condition:
            if self.consecutive_failed >= 30:
                raise RuntimeError(
                    f"{self.name} camera failed to read "
                    f"{self.consecutive_failed} consecutive frames"
                )

    def stop(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()


def select_frame_by_time(samples, target):
    return min(samples, key=lambda sample: abs(sample[0] - target)) if samples else None


def load_calibration(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"RGB calibration not found: {path}; copy calibration_standard.npz "
            "next to warp.py or pass --no-undistort"
        )
    with np.load(path) as data:
        def first(*names):
            for name in names:
                if name in data:
                    return np.asarray(data[name])
            raise KeyError(f"{path} lacks any of: {', '.join(names)}")
        K = first("K", "camera_matrix", "mtx").astype(np.float64)
        D = first("D", "dist_coeffs", "dist").astype(np.float64)
        size = first("image_size", "size", "calibrated_size").reshape(-1)
        rms = float(np.asarray(data["rms"]).reshape(-1)[0]) if "rms" in data else float("nan")
    return path, K, D, (int(size[0]), int(size[1])), rms


def make_standard_maps(K, D, calibrated_size, output_size, balance):
    sx, sy = output_size[0] / calibrated_size[0], output_size[1] / calibrated_size[1]
    scaled = K.copy()
    scaled[0, :] *= sx
    scaled[1, :] *= sy
    scaled[2, 2] = 1.0
    if D.size == 4:  # fisheye calibration
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            scaled, D.reshape(-1, 1), output_size, np.eye(3), balance=balance
        )
        return cv2.fisheye.initUndistortRectifyMap(
            scaled, D.reshape(-1, 1), np.eye(3), new_K, output_size, cv2.CV_32FC1
        )
    new_K, _ = cv2.getOptimalNewCameraMatrix(scaled, D, output_size, balance, output_size)
    return cv2.initUndistortRectifyMap(scaled, D, None, new_K, output_size, cv2.CV_32FC1)


def alignment_matrix(size, scale, x, y):
    cx, cy = (size[0] - 1) * 0.5, (size[1] - 1) * 0.5
    return np.array([[scale, 0, (1 - scale) * cx + x],
                     [0, scale, (1 - scale) * cy + y]], np.float32)


def align(image, size, matrix, interpolation, flip):
    if flip:
        image = cv2.flip(image, 1)
    image = cv2.resize(image, size, interpolation=interpolation)
    return cv2.warpAffine(image, matrix, size, flags=interpolation,
                          borderMode=cv2.BORDER_CONSTANT)


def valid_roi(size, matrix):
    valid = np.full((size[1], size[0]), 255, np.uint8)
    valid = cv2.warpAffine(valid, matrix, size, flags=cv2.INTER_NEAREST)
    points = cv2.findNonZero(valid)
    if points is None:
        raise RuntimeError("Thermal alignment has no valid output pixels")
    return cv2.boundingRect(points)


def crop(frame, roi):
    x, y, w, h = roi
    return frame[y:y + h, x:x + w]


def make_mask(temp_c, min_temp, max_temp, percentile):
    threshold = float(min_temp)
    valid = temp_c[np.isfinite(temp_c)]
    if percentile is not None and valid.size:
        threshold = max(threshold, float(np.percentile(valid, percentile)))
    mask = np.zeros(temp_c.shape, np.uint8)
    mask[np.isfinite(temp_c) & (temp_c >= threshold) & (temp_c <= max_temp)] = 255
    return mask, threshold


def dilate(mask, pixels):
    if pixels <= 0:
        return mask
    size = pixels * 2 + 1
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))


def overlay(image, mask, color=(0, 255, 255), alpha=0.45, contour=False):
    out = image.copy()
    hit = mask.astype(bool)
    if np.any(hit):
        out[hit] = np.clip(
            out[hit].astype(np.float32) * (1 - alpha) + np.asarray(color) * alpha, 0, 255
        ).astype(np.uint8)
        if contour:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, color, 1, cv2.LINE_AA)
    return out


def run_pose(rknn, bgr, args):
    inp, scale, left, top = letterbox(bgr, args.img_size)
    tensor = np.expand_dims(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB), 0)
    t0 = time.perf_counter()
    outputs = rknn.inference(inputs=[tensor])
    infer_ms = (time.perf_counter() - t0) * 1000
    boxes, scores, kpts = decode_yolo_pose(outputs, args.img_size, args.conf)
    keep = scores >= args.conf
    boxes, scores, kpts = boxes[keep], scores[keep], kpts[keep]
    if not len(scores):
        return [], infer_ms
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale
    kpts[:, :, 0] = (kpts[:, :, 0] - left) / scale
    kpts[:, :, 1] = (kpts[:, :, 1] - top) / scale
    h, w = bgr.shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
    kpts[:, :, 0] = kpts[:, :, 0].clip(0, w - 1)
    kpts[:, :, 1] = kpts[:, :, 1].clip(0, h - 1)
    detections = [(boxes[i], float(scores[i]), kpts[i]) for i in nms(boxes, scores, args.iou)]
    return detections, infer_ms


def draw_skeleton(image, detections, min_conf, lower_body_min_conf=None):
    lower_body_min_conf = (
        min_conf if lower_body_min_conf is None else lower_body_min_conf
    )

    def visible(points, index):
        threshold = lower_body_min_conf if index >= 11 else min_conf
        return points[index, 2] >= threshold

    for _, _, points in detections:
        for a, b in COCO_SKELETON:
            if visible(points, a) and visible(points, b):
                cv2.line(image, tuple(points[a, :2].astype(int)),
                         tuple(points[b, :2].astype(int)), (255, 220, 60), 3, cv2.LINE_AA)
        # Connect the head to the torso through the midpoint of both shoulders.
        if all(points[index, 2] >= min_conf for index in (0, 5, 6)):
            shoulder_midpoint = ((points[5, :2] + points[6, :2]) / 2).astype(int)
            cv2.line(image, tuple(points[0, :2].astype(int)),
                     tuple(shoulder_midpoint), (255, 220, 60), 3, cv2.LINE_AA)
        for index, (x, y, _) in enumerate(points):
            if visible(points, index):
                cv2.circle(image, (int(x), int(y)), 4, (40, 80, 255), -1, cv2.LINE_AA)


def stickfigure(
    rgb,
    mask,
    detections,
    background,
    kpt_conf,
    lower_body_kpt_conf=None,
):
    if background is None or background.shape != rgb.shape:
        out = rgb.copy()
        out[mask.astype(bool)] = 0
    else:
        out = rgb.copy()
        out[mask.astype(bool)] = background[mask.astype(bool)]
    draw_skeleton(out, detections, kpt_conf, lower_body_kpt_conf)
    return out


def label(image, text):
    cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 4)
    cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)


def show(name, image, scale):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, max(1, round(image.shape[1] * scale)),
                     max(1, round(image.shape[0] * scale)))
    cv2.imshow(name, image)


def avg(history, key):
    return sum(history[key]) / len(history[key]) if history[key] else 0.0


def main():
    args = parse_args()
    if args.mask_percentile is not None and not 0 <= args.mask_percentile <= 100:
        raise ValueError("--mask-percentile must be from 0 to 100")
    model = Path(args.model).expanduser().resolve()
    if not model.exists() or model.suffix != ".rknn":
        raise FileNotFoundError(f"RKNN model not found: {model}")

    calibration = None if args.no_undistort else load_calibration(args.rgb_calibration)
    rknn = load_rknn(str(model))
    thermal_cap = open_thermal_camera(args.thermal_dev)
    thermal_cap.set(cv2.CAP_PROP_FPS, args.thermal_fps)
    thermal_cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)
    try:
        rgb_cap = open_rgb_camera(args)
    except Exception:
        # Opening the RGB device can fail when the device number/format is
        # wrong or another process owns it.  Do not leave thermal captured.
        thermal_cap.release()
        rknn.release()
        raise
    try:
        thermal_probe = verify_camera_read(thermal_cap, "thermal")
        # Validate that the raw frame really is the 256x384 thermal payload.
        parse_temperature(thermal_probe)
        rgb_probe = verify_camera_read(rgb_cap, "rgb")
        if rgb_probe.ndim != 3 or rgb_probe.shape[2] != 3:
            raise RuntimeError(f"unexpected RGB frame shape: {rgb_probe.shape}")
    except Exception:
        thermal_cap.release()
        rgb_cap.release()
        rknn.release()
        raise
    thermal_reader = TimestampedCapture(thermal_cap, "thermal", max(8, int(args.thermal_fps * 2)))
    rgb_reader = TimestampedCapture(rgb_cap, "rgb", max(8, int(args.rgb_fps * 4)))
    thermal_reader.start()
    rgb_reader.start()

    maps = matrix = roi = background = None
    last_sequence = frame_count = 0
    frame_times = deque()
    history = defaultdict(lambda: deque(maxlen=120))
    offset = args.thermal_rgb_offset_ms / 1000
    print(f"RKNN pose model: {model}")
    print(f"Thermal: {args.thermal_dev} @ {args.thermal_fps:g} fps; "
          f"RGB: {args.rgb_dev} {args.rgb_width}x{args.rgb_height} @ {args.rgb_fps:g} fps")
    print(f"RGB negotiated: {capture_description(rgb_cap)}")
    print(f"Alignment: flip={args.thermal_flip} scale={args.thermal_scale:.3f} "
          f"x={args.thermal_x:.1f} y={args.thermal_y:.1f}")
    print("Press b with an empty scene to capture background; q/Esc to quit.")

    try:
        while not args.max_frames or frame_count < args.max_frames:
            loop_start = time.perf_counter()
            thermal_sample = thermal_reader.wait_newest(last_sequence)
            if thermal_sample is None:
                continue
            thermal_time, thermal_raw, last_sequence = thermal_sample
            rgb_reader.raise_if_failed()
            rgb_sample = select_frame_by_time(rgb_reader.snapshot(), thermal_time - offset)
            if rgb_sample is None:
                continue
            rgb_time, rgb_raw, _ = rgb_sample
            prepare_start = time.perf_counter()
            size = (rgb_raw.shape[1], rgb_raw.shape[0])
            if matrix is None:
                matrix = alignment_matrix(size, max(.05, args.thermal_scale),
                                          args.thermal_x, args.thermal_y)
                if calibration:
                    maps = make_standard_maps(calibration[1], calibration[2],
                                              calibration[3], size, args.rgb_balance)
                if args.crop_to_thermal:
                    roi = valid_roi(size, matrix)
                print(f"Actual RGB: {size[0]}x{size[1]}, crop={roi or 'off'}")
            rgb = cv2.remap(rgb_raw, maps[0], maps[1], cv2.INTER_LINEAR) if maps else rgb_raw
            temp_c = parse_temperature(thermal_raw)
            thermal_mask, threshold = make_mask(
                temp_c, args.mask_min_temp, args.mask_max_temp, args.mask_percentile
            )
            aligned_mask = dilate(
                align(thermal_mask, size, matrix, cv2.INTER_NEAREST, args.thermal_flip),
                args.warped_mask_dilate_px,
            )
            aligned_thermal = align(
                temp_to_display(temp_c), size, matrix, cv2.INTER_LINEAR, args.thermal_flip
            )
            if roi:
                rgb, aligned_mask, aligned_thermal = (
                    crop(rgb, roi), crop(aligned_mask, roi), crop(aligned_thermal, roi)
                )
            prepare_ms = (time.perf_counter() - prepare_start) * 1000
            detections, infer_ms = run_pose(rknn, rgb, args)
            post_start = time.perf_counter()
            thermal_view = overlay(aligned_thermal, aligned_mask, alpha=args.mask_alpha, contour=True)
            color_view = overlay(rgb, aligned_mask, alpha=args.mask_alpha)
            privacy_view = stickfigure(
                rgb, aligned_mask, detections, background, args.kpt_conf
            )
            post_ms = (time.perf_counter() - post_start) * 1000
            frame_count += 1
            now = time.perf_counter()
            total_ms = (now - loop_start) * 1000
            frame_times.append(now)
            fps = fps_from_times(frame_times, now)
            for key, value in (("prepare", prepare_ms), ("infer", infer_ms),
                               ("post", post_ms), ("total", total_ms)):
                history[key].append(value)

            if not args.no_display:
                label(privacy_view, f"RKNN {fps:.1f} FPS  NPU {avg(history, 'infer'):.1f} ms")
                label(thermal_view, f"{np.nanmin(temp_c):.1f}-{np.nanmax(temp_c):.1f} C")
                show("thermal", thermal_view, args.display_scale)
                show("color", color_view, args.display_scale)
                show("stickfigure", privacy_view, args.display_scale)
                if args.test:
                    pose_view = rgb.copy()
                    draw_skeleton(pose_view, detections, args.kpt_conf)
                    show("thermal_aligned", aligned_thermal, args.display_scale)
                    show("rgb_pose", pose_view, args.display_scale)

            if frame_count % max(1, args.print_every) == 0:
                thermal_fps, thermal_failed = thermal_reader.stats()
                rgb_fps, rgb_failed = rgb_reader.stats()
                print(
                    f"frame={frame_count} fps={fps:.1f} "
                    f"prepare={avg(history,'prepare'):.1f}ms "
                    f"npu={avg(history,'infer'):.1f}ms "
                    f"post={avg(history,'post'):.1f}ms "
                    f"total={avg(history,'total'):.1f}ms det={len(detections)} "
                    f"threshold={threshold:.2f}C pair_delta={(thermal_time-rgb_time)*1000:+.1f}ms "
                    f"capture_fps=thermal:{thermal_fps:.1f}/rgb:{rgb_fps:.1f} "
                    f"read_failed={thermal_failed}/{rgb_failed}"
                )
            key = -1 if args.no_display else cv2.waitKey(1) & 0xff
            if key in (ord("q"), 27):
                break
            if key == ord("b"):
                background = rgb.copy()
                print("Background captured.")
    finally:
        thermal_reader.stop()
        rgb_reader.stop()
        thermal_reader.thread.join(timeout=2)
        rgb_reader.thread.join(timeout=2)
        thermal_cap.release()
        rgb_cap.release()
        rknn.release()
        cv2.destroyAllWindows()
        if frame_count:
            print(f"FINAL frames={frame_count} fps={fps_from_times(frame_times, time.perf_counter()):.1f} "
                  f"prepare={avg(history,'prepare'):.1f}ms npu={avg(history,'infer'):.1f}ms "
                  f"post={avg(history,'post'):.1f}ms total={avg(history,'total'):.1f}ms")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
