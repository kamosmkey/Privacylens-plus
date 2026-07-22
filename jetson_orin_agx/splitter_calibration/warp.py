#!/usr/bin/env python3
"""Static beamsplitter thermal-to-RGB warp privacy pipeline.

This is the splitter equivalent of yolo_calibration/jetson_thermal_rgb_warp.py.
The RGB frame is flattened with a standard camera calibration.  Thermal-to-RGB
alignment is fixed (flip + resize + scale + translation), so YOLO is not used
to estimate calibration at runtime.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
YOLO_DIR = PROJECT_DIR / "yolo_calibration"
if str(YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_DIR))

from background_stickfigure import BackgroundStickFigure
from jetson_thermal_rgb_warp import (
    detections_from_result,
    dilate_mask,
    fps_from_times,
    predict_pose,
    resolve_device,
    select_frame_by_time,
)
from thermal_common import (
    make_temperature_mask,
    open_thermal_camera,
    overlay_mask,
    parse_temperature,
    put_text_top_right,
    temp_to_display,
)
from ultralytics import YOLO

# Reuse the exact standard-calibration loading/map behavior tested in test.py.
from test import load_standard_calibration, make_standard_maps


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=str(YOLO_DIR / "model/yolo26n-pose.engine"),
        help="YOLO pose TensorRT engine",
    )
    parser.add_argument(
        "--seg-model",
        default=str(YOLO_DIR / "model/yolo26n-seg.engine"),
        help="YOLO segmentation TensorRT engine used for the RGB person mask",
    )
    parser.add_argument(
        "--seg",
        action="store_true",
        help="run RGB person segmentation and union it with the thermal mask",
    )
    parser.add_argument("--thermal-dev", default="/dev/video0")
    parser.add_argument("--thermal-fps", type=float, default=25.0)
    parser.add_argument("--rgb-dev", default="/dev/video2")
    parser.add_argument("--rgb-width", type=int, default=640)
    parser.add_argument("--rgb-height", type=int, default=480)
    parser.add_argument("--rgb-fps", type=float, default=30.0)
    parser.add_argument("--rgb-fourcc", default="YUYV")
    parser.add_argument(
        "--rgb-calibration",
        default=str(SCRIPT_DIR / "calibration_standard.npz"),
    )
    parser.add_argument("--rgb-balance", type=float, default=0.0)

    parser.add_argument("--thermal-scale", type=float, default=0.740)
    parser.add_argument("--thermal-x", type=float, default=5.0)
    parser.add_argument("--thermal-y", type=float, default=15.0)
    thermal_flip_group = parser.add_mutually_exclusive_group()
    thermal_flip_group.add_argument(
        "--thermal-flip",
        dest="thermal_flip",
        action="store_true",
        help="horizontally flip the thermal frame (default)",
    )
    thermal_flip_group.add_argument(
        "--no-thermal-flip",
        dest="thermal_flip",
        action="store_false",
        help="do not horizontally flip the thermal frame",
    )
    parser.set_defaults(thermal_flip=True)
    parser.add_argument(
        "--crop-to-thermal",
        action="store_true",
        help="crop RGB and thermal outputs to the valid aligned thermal region",
    )
    parser.add_argument("--thermal-rgb-offset-ms", type=float, default=50.0)

    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--kpt-conf", type=float, default=0.4)
    parser.add_argument("--mask-min-temp", type=float, default=26.5)
    parser.add_argument("--mask-max-temp", type=float, default=27.8)
    parser.add_argument(
        "--mask-percentile",
        type=float,
        default=None,
        help=(
            "optional percentile from 0 to 100; the effective lower threshold "
            "is max(mask-min-temp, the current frame percentile)"
        ),
    )
    parser.add_argument("--mask-alpha", type=float, default=0.45)
    parser.add_argument("--warped-mask-dilate-px", type=int, default=10)
    parser.add_argument(
        "--test",
        action="store_true",
        help="also show aligned thermal and RGB pose diagnostic windows",
    )
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument(
        "--display-scale",
        type=float,
        default=1,
        help="output window size multiplier (default: 1.5)",
    )
    return parser.parse_args()


def resolve_engine(path):
    model_path = Path(path).expanduser().resolve()
    if model_path.suffix != ".engine":
        raise ValueError("--model must point to a TensorRT .engine file")
    if not model_path.exists():
        raise FileNotFoundError(f"TensorRT engine not found: {model_path}")
    return model_path


def open_rgb_camera(args):
    cap = cv2.VideoCapture(args.rgb_dev, cv2.CAP_V4L2)
    if not cap.isOpened() and args.rgb_dev.startswith("/dev/video"):
        cap = cv2.VideoCapture(int(args.rgb_dev[len("/dev/video"):]))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RGB camera {args.rgb_dev}")
    if len(args.rgb_fourcc) != 4:
        cap.release()
        raise RuntimeError("--rgb-fourcc must contain exactly four characters")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.rgb_fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.rgb_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.rgb_height)
    if args.rgb_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, args.rgb_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)
    return cap


def static_alignment_matrix(out_size, scale, offset_x, offset_y):
    width, height = out_size
    center_x = (width - 1) * 0.5
    center_y = (height - 1) * 0.5
    return np.asarray(
        [
            [scale, 0.0, (1.0 - scale) * center_x + offset_x],
            [0.0, scale, (1.0 - scale) * center_y + offset_y],
        ],
        dtype=np.float32,
    )


def align_thermal(image, out_size, matrix, interpolation, thermal_flip=True):
    """Optionally flip, then resize and apply the centered affine alignment."""
    if thermal_flip:
        image = cv2.flip(image, 1)
    resized = cv2.resize(image, out_size, interpolation=interpolation)
    return cv2.warpAffine(
        resized,
        matrix,
        out_size,
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
    )


def valid_thermal_roi(out_size, matrix):
    """Return the bounding box of valid thermal pixels after affine alignment."""
    width, height = out_size
    valid = np.full((height, width), 255, dtype=np.uint8)
    warped = cv2.warpAffine(
        valid,
        matrix,
        out_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    points = cv2.findNonZero(warped)
    if points is None:
        raise RuntimeError("Thermal alignment leaves no valid pixels in the RGB frame")
    return cv2.boundingRect(points)


def crop_to_roi(frame, roi):
    x, y, width, height = roi
    return frame[y:y + height, x:x + width]


class TimestampedCapture:
    """Continuously read a camera into a timestamped, thread-safe buffer."""

    def __init__(self, cap, name, maxlen):
        self.cap = cap
        self.name = name
        self.frames = deque(maxlen=maxlen)
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.sequence = 0
        self.failed_reads = 0
        self.capture_fps = 0.0
        self.frame_times = deque()
        self.thread = threading.Thread(
            target=self._run, name=f"{name}-capture", daemon=True
        )

    def start(self):
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            captured_at = time.perf_counter()
            if not ok:
                with self.condition:
                    self.failed_reads += 1
                    self.condition.notify_all()
                continue

            self.frame_times.append(captured_at)
            capture_fps = fps_from_times(self.frame_times, captured_at)
            with self.condition:
                self.sequence += 1
                self.frames.append((captured_at, frame, self.sequence))
                self.capture_fps = capture_fps
                self.condition.notify_all()

    def wait_for_newest(self, previous_sequence, timeout=1.0):
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence != previous_sequence
                or self.stop_event.is_set(),
                timeout=timeout,
            )
            if not self.frames or self.sequence == previous_sequence:
                return None
            return self.frames[-1]

    def snapshot(self):
        with self.condition:
            return list(self.frames)

    def stats(self):
        with self.condition:
            return self.capture_fps, self.failed_reads, self.sequence

    def stop(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()


_WINDOW_SIZES = {}


def show_labeled(name, frame, text, display_scale=1.0):
    scale = max(0.1, float(display_scale))
    window_size = (
        max(1, int(round(frame.shape[1] * scale))),
        max(1, int(round(frame.shape[0] * scale))),
    )
    if _WINDOW_SIZES.get(name) != window_size:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, *window_size)
        _WINDOW_SIZES[name] = window_size
    display = put_text_top_right(frame, text) if text else frame
    cv2.imshow(name, display)


def make_optional_percentile_mask(temp_c, min_temp, max_temp, percentile):
    threshold = float(min_temp)
    if percentile is not None:
        valid = temp_c[np.isfinite(temp_c)]
        if valid.size:
            threshold = max(threshold, float(np.percentile(valid, percentile)))
    return make_temperature_mask(temp_c, threshold, max_temp), threshold


def person_mask_from_segmentation(result, image_shape):
    height, width = image_shape[:2]
    person_mask = np.zeros((height, width), dtype=np.uint8)
    if result is None or result.boxes is None or result.masks is None:
        return person_mask, 0

    classes = result.boxes.cls.detach().cpu().numpy().astype(np.int32)
    masks = result.masks.data.detach().cpu().numpy()
    names = result.names
    person_count = 0
    for class_id, mask in zip(classes, masks):
        class_name = names.get(int(class_id), str(class_id)) if isinstance(names, dict) else names[int(class_id)]
        if class_name != "person":
            continue
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        person_mask[mask > 0.5] = 255
        person_count += 1
    return person_mask, person_count


def main():
    args = parse_args()
    if args.mask_percentile is not None and not 0.0 <= args.mask_percentile <= 100.0:
        raise ValueError("--mask-percentile must be between 0 and 100")
    needs_pose = True
    device = resolve_device(args.device)
    pose_engine_path = resolve_engine(args.model) if needs_pose else None
    seg_engine_path = resolve_engine(args.seg_model) if args.seg else None
    calibration_path, K, D, calibrated_size, calibration_rms = (
        load_standard_calibration(args.rgb_calibration)
    )

    pose_model = YOLO(str(pose_engine_path)) if needs_pose else None
    seg_model = YOLO(str(seg_engine_path)) if args.seg else None
    thermal_cap = open_thermal_camera(args.thermal_dev)
    if args.thermal_fps > 0:
        thermal_cap.set(cv2.CAP_PROP_FPS, args.thermal_fps)
    thermal_cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)
    rgb_cap = None
    thermal_reader = None
    rgb_reader = None
    try:
        rgb_cap = open_rgb_camera(args)
        thermal_reader = TimestampedCapture(
            thermal_cap, "thermal", maxlen=max(8, int(args.thermal_fps * 2))
        )
        rgb_reader = TimestampedCapture(
            rgb_cap, "rgb", maxlen=max(8, int(args.rgb_fps * 4))
        )
        thermal_reader.start()
        rgb_reader.start()
        compositor = BackgroundStickFigure(args.kpt_conf)
        rgb_maps = None
        alignment_matrix = None
        thermal_roi = None
        frame_times = deque()
        frame_count = 0
        last_thermal_sequence = 0
        offset_s = args.thermal_rgb_offset_ms / 1000.0

        if args.seg:
            print(f"Segmentation model: {seg_engine_path}")
        else:
            print("Segmentation model: disabled (enable with --seg)")
        print(f"Pose model: {pose_engine_path}")
        print(f"Device: {device}")
        print(
            f"Thermal camera: {args.thermal_dev}, requested "
            f"256x192 temperature at {args.thermal_fps:g} fps"
        )
        print(
            f"RGB camera: {args.rgb_dev}, requested {args.rgb_width}x{args.rgb_height} "
            f"{args.rgb_fourcc} at {args.rgb_fps:g} fps"
        )
        print("Capture: independent thermal/RGB threads, V4L2 buffers=4")
        print(
            f"RGB calibration: {calibration_path} standard "
            f"{calibrated_size[0]}x{calibrated_size[1]} RMS={calibration_rms:.4f}px"
        )
        print(
            f"Static thermal alignment: flip={args.thermal_flip}, "
            f"scale={args.thermal_scale:.3f}, x={args.thermal_x:.1f}, "
            f"y={args.thermal_y:.1f}"
        )
        print(f"Thermal/RGB offset: {args.thermal_rgb_offset_ms:+.1f} ms")
        print("Windows: thermal, color, stickfigure")
        print("Trust view is retained internally but not displayed")
        if args.test:
            print("Test windows: thermal_aligned, rgb_yolo")
        print("Press b with an empty scene to capture background; q to quit.")

        while True:
            t0 = time.perf_counter()
            thermal_sample = thermal_reader.wait_for_newest(last_thermal_sequence)
            if thermal_sample is None:
                print("waiting for thermal frame", file=sys.stderr)
                continue
            thermal_read_done, thermal_frame, last_thermal_sequence = thermal_sample

            rgb_samples = rgb_reader.snapshot()
            selected_rgb = select_frame_by_time(
                rgb_samples, thermal_read_done - offset_s
            )
            if selected_rgb is None:
                print("waiting for RGB frame", file=sys.stderr)
                continue
            rgb_read_done, rgb_raw, _ = selected_rgb
            pair_delta_ms = (thermal_read_done - rgb_read_done) * 1000.0
            t_prepare_start = time.perf_counter()

            rgb_size = (rgb_raw.shape[1], rgb_raw.shape[0])
            if rgb_maps is None:
                rgb_maps = make_standard_maps(
                    K, D, calibrated_size, rgb_size, args.rgb_balance
                )
                alignment_matrix = static_alignment_matrix(
                    rgb_size,
                    max(0.05, args.thermal_scale),
                    args.thermal_x,
                    args.thermal_y,
                )
                print(f"Actual RGB output: {rgb_size[0]}x{rgb_size[1]}")
                if args.crop_to_thermal:
                    thermal_roi = valid_thermal_roi(rgb_size, alignment_matrix)
                    roi_x, roi_y, roi_w, roi_h = thermal_roi
                    print(
                        f"Crop to thermal ROI: {roi_w}x{roi_h} "
                        f"at x={roi_x}, y={roi_y}"
                    )

            rgb_flat = cv2.remap(
                rgb_raw,
                rgb_maps[0],
                rgb_maps[1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            temp_c = parse_temperature(thermal_frame)
            thermal_mask, mask_threshold = make_optional_percentile_mask(
                temp_c,
                args.mask_min_temp,
                args.mask_max_temp,
                args.mask_percentile,
            )
            aligned_mask = align_thermal(
                thermal_mask,
                rgb_size,
                alignment_matrix,
                cv2.INTER_NEAREST,
                args.thermal_flip,
            )
            aligned_mask = dilate_mask(aligned_mask, args.warped_mask_dilate_px)
            aligned_thermal = align_thermal(
                temp_to_display(temp_c),
                rgb_size,
                alignment_matrix,
                cv2.INTER_LINEAR,
                args.thermal_flip,
            )
            if thermal_roi is not None:
                rgb_flat = crop_to_roi(rgb_flat, thermal_roi)
                aligned_mask = crop_to_roi(aligned_mask, thermal_roi)
                aligned_thermal = crop_to_roi(aligned_thermal, thermal_roi)
            t_prepare = time.perf_counter()

            # RGB segmentation contributes to the mask only when --seg is set.
            if args.seg:
                seg_result = predict_pose(seg_model, rgb_flat, args, device)
                rgb_person_mask, seg_person_count = person_mask_from_segmentation(
                    seg_result, rgb_flat.shape
                )
                union_mask = cv2.bitwise_or(aligned_mask, rgb_person_mask)
            else:
                seg_result = None
                seg_person_count = 0
                union_mask = aligned_mask
            if needs_pose:
                pose_result = predict_pose(pose_model, rgb_flat, args, device)
                rgb_dets = detections_from_result(pose_result, rgb_flat.shape)
            else:
                pose_result = None
                rgb_dets = []
            t_yolo = time.perf_counter()

            thermal_mask_view = overlay_mask(
                aligned_thermal,
                aligned_mask,
                alpha=args.mask_alpha,
                draw_contour=True,
            )
            color_view = overlay_mask(
                rgb_flat,
                union_mask,
                color=(0, 255, 255),
                alpha=args.mask_alpha,
            )
            stickfigure_view = compositor.render(
                rgb_flat, union_mask, rgb_dets, update_uncertainty=False
            )

            frame_count += 1
            now = time.perf_counter()
            frame_times.append(now)
            fps = fps_from_times(frame_times, now)
            fps_text = f"{fps:.1f} fps"
            min_temp = float(np.nanmin(temp_c))
            max_temp = float(np.nanmax(temp_c))
            thermal_text = f"{min_temp:.1f}-{max_temp:.1f} C"
            show_labeled(
                "thermal", thermal_mask_view, thermal_text, args.display_scale
            )
            show_labeled(
                "color", color_view, None, args.display_scale
            )
            show_labeled(
                "stickfigure",
                stickfigure_view,
                f"STICKFIGURE  {fps_text}",
                args.display_scale,
            )
            if args.test:
                rgb_yolo = pose_result.plot() if pose_result is not None else rgb_flat
                rgb_seg = seg_result.plot() if seg_result is not None else rgb_flat
                show_labeled(
                    "thermal_aligned", aligned_thermal, None, args.display_scale
                )
                show_labeled(
                    "rgb_yolo",
                    rgb_yolo,
                    f"{len(rgb_dets)} det",
                    args.display_scale,
                )
                show_labeled(
                    "rgb_seg",
                    rgb_seg,
                    f"{seg_person_count} person",
                    args.display_scale,
                )

            if frame_count % max(1, args.print_every) == 0:
                total_ms = (now - t0) * 1000.0
                print(
                    f"frame={frame_count} fps={fps:.1f} total={total_ms:.1f}ms "
                    f"rgb_pose_det={len(rgb_dets)} rgb_seg_person={seg_person_count} "
                    f"mask_threshold={mask_threshold:.2f}C "
                    f"offset_target={args.thermal_rgb_offset_ms:+.1f}ms "
                    f"paired_th_minus_rgb={pair_delta_ms:+.1f}ms "
                    f"alignment=static"
                )
                if args.timing:
                    thermal_capture_fps, thermal_failed, _ = thermal_reader.stats()
                    rgb_capture_fps, rgb_failed, _ = rgb_reader.stats()
                    print(
                        f"timing prepare={(t_prepare - t_prepare_start) * 1000.0:.1f}ms "
                        f"yolo={(t_yolo - t_prepare) * 1000.0:.1f}ms "
                        f"capture_fps=thermal:{thermal_capture_fps:.1f}/rgb:{rgb_capture_fps:.1f} "
                        f"read_failed=thermal:{thermal_failed}/rgb:{rgb_failed} "
                        f"temp={float(np.min(temp_c)):.1f}/"
                        f"{float(np.mean(temp_c)):.1f}/{float(np.max(temp_c)):.1f}C "
                        f"thermal_mask_hit={100.0 * np.count_nonzero(aligned_mask) / aligned_mask.size:.1f}% "
                        f"union_mask_hit={100.0 * np.count_nonzero(union_mask) / union_mask.size:.1f}%"
                    )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("b"):
                compositor.capture(rgb_flat)
                print("Background captured.")
    finally:
        if thermal_reader is not None:
            thermal_reader.stop()
        if rgb_reader is not None:
            rgb_reader.stop()
        if thermal_reader is not None:
            thermal_reader.thread.join(timeout=2.0)
        if rgb_reader is not None:
            rgb_reader.thread.join(timeout=2.0)
        thermal_cap.release()
        if rgb_cap is not None:
            rgb_cap.release()
        if thermal_reader is not None and thermal_reader.thread.is_alive():
            thermal_reader.thread.join(timeout=1.0)
        if rgb_reader is not None and rgb_reader.thread.is_alive():
            rgb_reader.thread.join(timeout=1.0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
