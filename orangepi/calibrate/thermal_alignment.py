#!/usr/bin/env python3
"""Interactively align a thermal camera image over an RGB camera image."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--thermal-dev", default="/dev/video0")
    parser.add_argument("--rgb-dev", default="/dev/video2")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--scale", type=float, default=0.740)
    parser.add_argument("--x", type=float, default=-5.0)
    parser.add_argument("--y", type=float, default=15.0)
    parser.add_argument("--output", type=Path, default=root / "ui/thermal_alignment.json")
    return parser.parse_args()


def device(value):
    return int(value) if value.isdigit() else value


def parse_thermal(frame):
    array = np.asarray(frame)
    if array.shape == (1, 384, 256, 2):
        array = array[0]
    elif array.shape != (384, 256, 2):
        data = array.reshape(-1).view(np.uint8)
        if data.size != 256 * 384 * 2:
            raise RuntimeError(f"Unexpected thermal frame shape: {array.shape}")
        array = data.reshape(384, 256, 2)
    bottom = array[192:384].astype(np.uint16)
    raw = bottom[:, :, 0] | (bottom[:, :, 1] << 8)
    return raw.astype(np.float32) / 64.0 - 273.15


def thermal_color(temperatures):
    low, high = np.percentile(temperatures, (2, 98))
    normalized = np.clip((temperatures - low) * 255.0 / max(high - low, 0.01), 0, 255)
    return cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)


class Window(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.scale, self.x, self.y = args.scale, args.x, args.y
        self.label = QLabel(alignment=Qt.AlignCenter)
        self.setCentralWidget(self.label)
        self.setWindowTitle("Thermal alignment")
        self.rgb = cv2.VideoCapture(device(args.rgb_dev))
        self.thermal = cv2.VideoCapture(device(args.thermal_dev))
        self.rgb.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        self.rgb.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        self.thermal.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        self.thermal.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
        self.thermal.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)
        self.thermal.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        if not self.rgb.isOpened() or not self.thermal.isOpened():
            raise RuntimeError("Could not open both cameras")
        self.timer = QTimer(self, timeout=self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ok_rgb, rgb = self.rgb.read()
        ok_thermal, raw = self.thermal.read()
        if not ok_rgb or not ok_thermal:
            return
        color = thermal_color(parse_thermal(raw))
        height, width = rgb.shape[:2]
        color = cv2.resize(color, (width, height), interpolation=cv2.INTER_LINEAR)
        cx, cy = width / 2.0, height / 2.0
        matrix = np.array(
            [[self.scale, 0, (1 - self.scale) * cx + self.x],
             [0, self.scale, (1 - self.scale) * cy + self.y]], np.float32
        )
        aligned = cv2.warpAffine(color, matrix, (width, height))
        mask = cv2.warpAffine(np.full(color.shape[:2], 255, np.uint8), matrix, (width, height))
        view = rgb.copy()
        hit = mask > 0
        view[hit] = (
            rgb[hit].astype(np.float32) * 0.5
            + aligned[hit].astype(np.float32) * 0.5
        ).astype(np.uint8)
        cv2.putText(
            view, f"scale={self.scale:.3f} x={self.x:.1f} y={self.y:.1f} | arrows move, +/- scale, S save",
            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        )
        view = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        image = QImage(view.data, width, height, view.strides[0], QImage.Format_RGB888).copy()
        self.label.setPixmap(QPixmap.fromImage(image).scaled(self.label.size(), Qt.KeepAspectRatio))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.x -= 1
        elif event.key() == Qt.Key_Right:
            self.x += 1
        elif event.key() == Qt.Key_Up:
            self.y -= 1
        elif event.key() == Qt.Key_Down:
            self.y += 1
        elif event.text() in ("+", "="):
            self.scale += 0.005
        elif event.text() in ("-", "_"):
            self.scale = max(0.05, self.scale - 0.005)
        elif event.key() == Qt.Key_S:
            self.args.output.parent.mkdir(parents=True, exist_ok=True)
            self.args.output.write_text(
                json.dumps({"scale": self.scale, "x": self.x, "y": self.y}, indent=2) + "\n",
                encoding="utf-8",
            )
            self.statusBar().showMessage(f"Saved {self.args.output}", 5000)
        elif event.key() in (Qt.Key_Q, Qt.Key_Escape):
            self.close()

    def closeEvent(self, event):
        self.timer.stop()
        self.rgb.release()
        self.thermal.release()
        event.accept()


def main():
    args = parse_args()
    app = QApplication(sys.argv)
    window = Window(args)
    window.resize(args.width, args.height)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
