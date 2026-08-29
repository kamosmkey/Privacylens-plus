#!/usr/bin/env python3
"""Preview synchronized RGB/thermal frames and save each privacy-pipeline stage.

Keys:
  b  capture the current (empty) processed RGB frame as the background
  s  save all six stages into a timestamped directory
  q/Esc  quit
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RKNN_DIR = PROJECT_DIR / "rknn"
if str(RKNN_DIR) not in sys.path:
    sys.path.insert(0, str(RKNN_DIR))

import warp as core  # noqa: E402
from thermal_mask import (  # noqa: E402
    open_thermal_camera,
    parse_raw_temperature,
    temp_to_display,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_DIR
        / "ui/model/yolo26n-pose_rknn_model/best.sanitized-rk3588.rknn",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_DIR / "ui/calibration_standard.npz",
    )
    parser.add_argument(
        "--thermal-dev",
        default="/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0",
    )
    parser.add_argument(
        "--rgb-dev",
        default="/dev/v4l/by-id/usb-HD_USB_Camera_HD_USB_Camera-video-index0",
    )
    parser.add_argument("--thermal-fps", type=float, default=25.0)
    parser.add_argument("--rgb-width", type=int, default=640)
    parser.add_argument("--rgb-height", type=int, default=480)
    parser.add_argument("--rgb-fps", type=float, default=30.0)
    parser.add_argument("--rgb-fourcc", default="YUYV")
    parser.add_argument("--thermal-scale", type=float, default=0.740)
    parser.add_argument("--thermal-x", type=float, default=2.0)
    parser.add_argument("--thermal-y", type=float, default=15.0)
    parser.add_argument("--thermal-rgb-offset-ms", type=float, default=50.0)
    parser.add_argument("--mask-min-temp", type=float, default=25.0)
    parser.add_argument("--mask-max-temp", type=float, default=36.0)
    parser.add_argument("--mask-percentile", type=float)
    parser.add_argument("--mask-dilate-px", type=int, default=15)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--kpt-conf", type=float, default=0.2)
    parser.add_argument("--lower-body-kpt-conf", type=float, default=0.7)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "screenshots")
    parser.add_argument("--no-undistort", action="store_true")
    parser.add_argument("--no-crop", action="store_true")
    return parser.parse_args()


def make_preview(rgb, thermal):
    """Put raw RGB and raw thermal side by side without changing saved data."""
    target_h = rgb.shape[0]
    scale = target_h / thermal.shape[0]
    thermal = cv2.resize(
        thermal,
        (max(1, round(thermal.shape[1] * scale)), target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    preview = np.hstack((rgb, thermal))
    split = rgb.shape[1]
    cv2.putText(preview, "RAW RGB", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(preview, "RAW THERMAL", (split + 12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(preview, "b: capture background   s: screenshot   q: quit",
                (12, target_h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return preview


class PreviewWindow(QWidget):
    """Small Qt preview; it does not depend on OpenCV's optional HighGUI."""

    def __init__(self):
        super().__init__()
        self.pending_key = None
        self.setWindowTitle("Raw RGB + Raw Thermal")
        self.image_label = QLabel("Waiting for camera frames...")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)
        self.setFocusPolicy(Qt.StrongFocus)

    def set_frame(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image = QImage(
            rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888
        ).copy()
        self.image_label.setPixmap(QPixmap.fromImage(image))

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Q, Qt.Key_Escape):
            self.pending_key = "q"
        elif event.key() == Qt.Key_B:
            self.pending_key = "b"
        elif event.key() == Qt.Key_S:
            self.pending_key = "s"
        else:
            super().keyPressEvent(event)

    def take_key(self):
        key, self.pending_key = self.pending_key, None
        return key

    def closeEvent(self, event):
        self.pending_key = "q"
        event.accept()


def save_stages(output_root, raw_rgb, raw_thermal, aligned_mask, processed_rgb,
                detections, background, args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    target = output_root.expanduser().resolve() / timestamp
    target.mkdir(parents=True, exist_ok=False)

    # "Color mode (black)": replace every final aligned-mask pixel with black,
    # but do not draw a skeleton yet.
    color_mode_black = processed_rgb.copy()
    color_mode_black[aligned_mask.astype(bool)] = 0
    stick_no_background = core.stickfigure(
        processed_rgb, aligned_mask, detections, None,
        args.kpt_conf, args.lower_body_kpt_conf,
    )
    stick_with_background = core.stickfigure(
        processed_rgb, aligned_mask, detections, background,
        args.kpt_conf, args.lower_body_kpt_conf,
    )

    images = {
        "01_raw_rgb.png": raw_rgb,
        "02_raw_thermal.png": raw_thermal,
        "03_thermal_mask.png": aligned_mask,
        "04_rgb_colormode_black.png": color_mode_black,
        "05_stickfigure_no_background.png": stick_no_background,
        "06_stickfigure_with_background.png": stick_with_background,
    }
    for name, image in images.items():
        if not cv2.imwrite(str(target / name), image):
            raise RuntimeError(f"Failed to save {target / name}")
    return target


def main():
    args = parse_args()
    if args.mask_percentile is not None and not 0 <= args.mask_percentile <= 100:
        raise ValueError("--mask-percentile must be between 0 and 100")
    if not args.model.is_file():
        raise FileNotFoundError(f"RKNN model not found: {args.model}")

    calibration = None
    if not args.no_undistort:
        calibration = core.load_calibration(args.calibration)

    rknn = thermal_cap = rgb_cap = None
    thermal_reader = rgb_reader = None
    background = None
    app = QApplication.instance() or QApplication(sys.argv)
    window = PreviewWindow()
    window.show()
    app.processEvents()
    try:
        rknn = core.load_rknn(str(args.model))
        thermal_cap = open_thermal_camera(args.thermal_dev)
        thermal_cap.set(cv2.CAP_PROP_FPS, args.thermal_fps)
        thermal_cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)
        rgb_cap = core.open_rgb_camera(args)

        parse_raw_temperature(core.verify_camera_read(thermal_cap, "thermal"))
        rgb_probe = core.verify_camera_read(rgb_cap, "rgb")
        if rgb_probe.ndim != 3 or rgb_probe.shape[2] != 3:
            raise RuntimeError(f"Unexpected RGB frame shape: {rgb_probe.shape}")

        thermal_reader = core.TimestampedCapture(
            thermal_cap, "thermal", max(8, int(args.thermal_fps * 2))
        )
        rgb_reader = core.TimestampedCapture(
            rgb_cap, "rgb", max(8, int(args.rgb_fps * 4))
        )
        thermal_reader.start()
        rgb_reader.start()

        maps = matrix = roi = None
        last_sequence = 0
        offset = args.thermal_rgb_offset_ms / 1000.0
        print("Preview: raw RGB + raw thermal")
        print("Keys: b=capture background, s=save six stages, q/Esc=quit")

        while True:
            thermal_sample = thermal_reader.wait_newest(last_sequence, timeout=0.5)
            if thermal_sample is None:
                continue
            thermal_time, thermal_payload, last_sequence = thermal_sample
            rgb_reader.raise_if_failed()
            rgb_sample = core.select_frame_by_time(
                rgb_reader.snapshot(), thermal_time - offset
            )
            if rgb_sample is None:
                continue
            _, raw_rgb, _ = rgb_sample

            size = (raw_rgb.shape[1], raw_rgb.shape[0])
            if matrix is None:
                matrix = core.alignment_matrix(
                    size, max(0.05, args.thermal_scale),
                    args.thermal_x, args.thermal_y,
                )
                if calibration is not None:
                    maps = core.make_standard_maps(
                        calibration[1], calibration[2], calibration[3], size, 0.0
                    )
                if not args.no_crop:
                    roi = core.valid_roi(size, matrix)

            processed_rgb = (
                cv2.remap(raw_rgb, maps[0], maps[1], cv2.INTER_LINEAR)
                if maps is not None else raw_rgb.copy()
            )
            thermal_raw16 = parse_raw_temperature(thermal_payload)
            temp_c = thermal_raw16.astype(np.float32) / 64.0 - 273.15
            raw_thermal = temp_to_display(temp_c)
            native_mask, _ = core.make_mask(
                temp_c, args.mask_min_temp, args.mask_max_temp,
                args.mask_percentile,
            )
            aligned_mask = core.dilate(
                core.align(native_mask, size, matrix, cv2.INTER_NEAREST, False),
                args.mask_dilate_px,
            )
            if roi is not None:
                processed_rgb = core.crop(processed_rgb, roi)
                aligned_mask = core.crop(aligned_mask, roi)

            detections, _ = core.run_pose(rknn, processed_rgb, args)
            window.set_frame(make_preview(raw_rgb, raw_thermal))
            app.processEvents()
            key = window.take_key()
            if key == "q":
                break
            if key == "b":
                background = processed_rgb.copy()
                print("Background captured. The scene should have been empty.")
            elif key == "s":
                if background is None:
                    print("Background has not been captured; press b with an empty scene first.")
                    continue
                target = save_stages(
                    args.output_dir, raw_rgb, raw_thermal, aligned_mask,
                    processed_rgb, detections, background, args,
                )
                print(f"Saved six stage images: {target}")
    finally:
        if thermal_reader is not None:
            thermal_reader.stop()
        if rgb_reader is not None:
            rgb_reader.stop()
        if thermal_reader is not None:
            thermal_reader.thread.join(timeout=2)
        if rgb_reader is not None:
            rgb_reader.thread.join(timeout=2)
        if thermal_cap is not None:
            thermal_cap.release()
        if rgb_cap is not None:
            rgb_cap.release()
        if rknn is not None:
            rknn.release()
        window.close()
        app.processEvents()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
