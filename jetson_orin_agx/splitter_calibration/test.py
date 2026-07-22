#!/usr/bin/env python3
"""Display the thermal camera and a calibrated/flattened RGB camera."""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
THERMAL_MODULE_DIR = PROJECT_DIR / "yolo_calibration"
if str(THERMAL_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(THERMAL_MODULE_DIR))

# Import the exact thermal capture and conversion functions used by
# yolo_calibration/jetson_thermal_rgb_warp.py.
from thermal_common import open_thermal_camera, parse_temperature, temp_to_display


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show /dev/video0 thermal and flattened /dev/video2 RGB."
    )
    parser.add_argument("--thermal-dev", default="/dev/video0")
    parser.add_argument("--rgb-dev", default="/dev/video2")
    parser.add_argument("--rgb-width", type=int, default=640)
    parser.add_argument("--rgb-height", type=int, default=480)
    parser.add_argument("--rgb-fps", type=float, default=0.0)
    parser.add_argument("--rgb-fourcc", default="YUYV")
    parser.add_argument(
        "--calibration",
        default=str(SCRIPT_DIR / "calibration_standard.npz"),
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=0.0,
        help="0 crops invalid borders; 1 retains more field of view",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.5,
        help="thermal contribution in the RGB/thermal overlay (default: 0.5)",
    )
    parser.add_argument(
        "--thermal-scale",
        type=float,
        default=1.0,
        help="thermal content scale around image center (default: 1.0)",
    )
    parser.add_argument(
        "--thermal-flip",
        action="store_true",
        help="horizontally flip the thermal frame",
    )
    parser.add_argument("--thermal-x", type=float, default=0.0,
                        help="thermal horizontal offset in RGB pixels")
    parser.add_argument("--thermal-y", type=float, default=0.0,
                        help="thermal vertical offset in RGB pixels")
    return parser.parse_args()


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
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def load_standard_calibration(path):
    calibration_path = Path(path).expanduser().resolve()
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    with np.load(calibration_path) as data:
        required = {"K", "D", "image_size", "model"}
        missing = required.difference(data.files)
        if missing:
            raise RuntimeError(
                f"Calibration file is missing: {', '.join(sorted(missing))}"
            )
        model = str(data["model"].item())
        if model != "standard":
            raise RuntimeError(
                f"Expected a standard calibration, but file contains {model!r}"
            )
        K = data["K"].astype(np.float64).copy()
        D = data["D"].astype(np.float64).copy()
        image_size = tuple(int(value) for value in data["image_size"])
        rms = float(data["rms"]) if "rms" in data.files else float("nan")
    return calibration_path, K, D, image_size, rms


def make_standard_maps(K, D, calibrated_size, frame_size, balance):
    calibrated_w, calibrated_h = calibrated_size
    frame_w, frame_h = frame_size
    if calibrated_size != frame_size:
        calibrated_ratio = calibrated_w / calibrated_h
        frame_ratio = frame_w / frame_h
        if abs(calibrated_ratio - frame_ratio) > 1e-3:
            raise RuntimeError(
                f"Calibration resolution is {calibrated_w}x{calibrated_h}, but RGB "
                f"camera is {frame_w}x{frame_h}; recalibrate at the runtime aspect ratio"
            )
        K = K.copy()
        K[0, :] *= frame_w / calibrated_w
        K[1, :] *= frame_h / calibrated_h

    balance = float(np.clip(balance, 0.0, 1.0))
    new_K, _ = cv2.getOptimalNewCameraMatrix(
        K, D, frame_size, balance, frame_size
    )
    return cv2.initUndistortRectifyMap(
        K, D, None, new_K, frame_size, cv2.CV_16SC2
    )


def measured_fps(timestamps, now):
    timestamps.append(now)
    while timestamps and now - timestamps[0] > 2.0:
        timestamps.popleft()
    if len(timestamps) < 2:
        return 0.0
    return (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])


def label(frame, text):
    output = frame.copy()
    cv2.putText(
        output,
        text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def transform_thermal(frame, scale, offset_x, offset_y):
    """Scale around the frame center, then translate in RGB pixel units."""
    height, width = frame.shape[:2]
    center_x = (width - 1) * 0.5
    center_y = (height - 1) * 0.5
    matrix = np.array(
        [
            [scale, 0.0, (1.0 - scale) * center_x + offset_x],
            [0.0, scale, (1.0 - scale) * center_y + offset_y],
        ],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def main():
    args = parse_args()
    calibration_path, K, D, calibrated_size, rms = load_standard_calibration(
        args.calibration
    )

    thermal_cap = open_thermal_camera(args.thermal_dev)
    rgb_cap = None
    try:
        rgb_cap = open_rgb_camera(args)
        rgb_maps = None
        frame_times = deque()
        thermal_scale = max(0.05, float(args.thermal_scale))
        thermal_x = float(args.thermal_x)
        thermal_y = float(args.thermal_y)

        print(f"Thermal: {args.thermal_dev} (same processing as jetson_thermal_rgb_warp.py)")
        print(
            f"RGB: {args.rgb_dev}, requested {args.rgb_width}x{args.rgb_height} "
            f"{args.rgb_fourcc}"
        )
        print(
            f"Calibration: {calibration_path}, standard, "
            f"{calibrated_size[0]}x{calibrated_size[1]}, RMS={rms:.4f}px"
        )
        thermal_orientation = "horizontally flipped" if args.thermal_flip else "not flipped"
        print(f"Thermal display: {thermal_orientation}, then resized to the RGB frame")
        print("Windows: thermal, rgb_flat, overlay")
        print("Keys: +/- scale | a/d left/right | w/s up/down | r reset | p print | q quit")

        while True:
            ok_thermal, thermal_frame = thermal_cap.read()
            ok_rgb, rgb_frame = rgb_cap.read()
            if not ok_thermal or not ok_rgb:
                print(
                    f"Camera read failed: thermal={ok_thermal}, rgb={ok_rgb}",
                    file=sys.stderr,
                )
                continue

            # Identical thermal conversion path to jetson_thermal_rgb_warp.py.
            temp_c = parse_temperature(thermal_frame)
            thermal_view = temp_to_display(temp_c)
            if args.thermal_flip:
                thermal_view = cv2.flip(thermal_view, 1)

            rgb_size = (rgb_frame.shape[1], rgb_frame.shape[0])
            if rgb_maps is None:
                rgb_maps = make_standard_maps(
                    K, D, calibrated_size, rgb_size, args.balance
                )
                print(f"Actual RGB output: {rgb_size[0]}x{rgb_size[1]}")
            rgb_flat = cv2.remap(
                rgb_frame,
                rgb_maps[0],
                rgb_maps[1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            thermal_resized = cv2.resize(
                thermal_view,
                rgb_size,
                interpolation=cv2.INTER_LINEAR,
            )
            thermal_aligned = transform_thermal(
                thermal_resized, thermal_scale, thermal_x, thermal_y
            )
            overlay_alpha = float(np.clip(args.overlay_alpha, 0.0, 1.0))
            overlay = cv2.addWeighted(
                rgb_flat,
                1.0 - overlay_alpha,
                thermal_aligned,
                overlay_alpha,
                0.0,
            )

            fps = measured_fps(frame_times, time.perf_counter())
            cv2.imshow(
                "thermal",
                label(
                    thermal_aligned,
                    f"thermal s={thermal_scale:.3f} x={thermal_x:.0f} y={thermal_y:.0f}",
                ),
            )
            cv2.imshow("rgb_flat", label(rgb_flat, f"RGB flat {fps:.1f}fps"))
            cv2.imshow(
                "overlay",
                label(
                    overlay,
                    f"overlay s={thermal_scale:.3f} x={thermal_x:.0f} y={thermal_y:.0f}",
                ),
            )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("+"), ord("=")):
                thermal_scale += 0.01
            elif key in (ord("-"), ord("_")):
                thermal_scale = max(0.05, thermal_scale - 0.01)
            elif key == ord("a"):
                thermal_x -= 1.0
            elif key == ord("d"):
                thermal_x += 1.0
            elif key == ord("w"):
                thermal_y -= 1.0
            elif key == ord("s"):
                thermal_y += 1.0
            elif key == ord("r"):
                thermal_scale, thermal_x, thermal_y = 1.0, 0.0, 0.0
            elif key == ord("p"):
                print(
                    f"Alignment: --thermal-scale {thermal_scale:.3f} "
                    f"--thermal-x {thermal_x:.0f} --thermal-y {thermal_y:.0f}"
                )
    finally:
        thermal_cap.release()
        if rgb_cap is not None:
            rgb_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
