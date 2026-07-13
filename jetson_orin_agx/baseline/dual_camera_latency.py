#!/usr/bin/env python3
import argparse
import threading
import time
from collections import deque

import cv2
import numpy as np


def size(text):
    w, h = text.split("x")
    return int(w), int(h)


def open_cam(dev, wh, thermal=False):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wh[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, wh[1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if thermal:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {dev}")
    return cap


class Grabber(threading.Thread):
    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.ts = None
        self.frame_id = 0
        self.running = True

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            ts = time.perf_counter()
            if ok:
                with self.lock:
                    self.frame = frame
                    self.ts = ts
                    self.frame_id += 1

    def get(self):
        with self.lock:
            if self.frame is None:
                return None, None, 0
            return self.frame.copy(), self.ts, self.frame_id


def thermal_raw(frame):
    if frame.dtype == np.uint8 and frame.ndim == 2:
        return np.ascontiguousarray(frame).view("<u2").reshape(frame.shape[0], frame.shape[1] // 2)
    if frame.dtype == np.uint8 and frame.ndim == 3:
        return frame[:, :, 0].astype(np.uint16) | (frame[:, :, 1].astype(np.uint16) << 8)
    return frame


def thermal_display(frame):
    raw = thermal_raw(frame)
    lo, hi = np.percentile(raw, (2, 98))
    gray = np.clip((raw.astype(np.float32) - lo) * 255 / max(1, hi - lo), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def put_text(img, text):
    out = img.copy()
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rgb-dev", default="/dev/video2")
    p.add_argument("--thermal-dev", default="/dev/video0")
    p.add_argument("--rgb-size", default="640x480")
    p.add_argument("--thermal-size", default="256x192")
    p.add_argument("--print-every", type=int, default=30)
    args = p.parse_args()

    rgb_cap = open_cam(args.rgb_dev, size(args.rgb_size))
    th_cap = open_cam(args.thermal_dev, size(args.thermal_size), thermal=True)
    rgb = Grabber(rgb_cap)
    th = Grabber(th_cap)
    rgb.start()
    th.start()

    diffs = deque(maxlen=120)
    last_pair = None
    print("No-event mode. This measures frame arrival timestamp difference, not sensor exposure timestamp.")
    print("Press q in an OpenCV window to quit.")

    try:
        while True:
            rgb_frame, rgb_ts, rgb_id = rgb.get()
            th_frame, th_ts, th_id = th.get()
            if rgb_frame is None or th_frame is None:
                time.sleep(0.01)
                continue

            pair = (rgb_id, th_id)
            diff_ms = (th_ts - rgb_ts) * 1000.0
            diffs.append(diff_ms)

            cv2.imshow("rgb", put_text(rgb_frame, f"id {rgb_id}"))
            cv2.imshow("thermal", put_text(thermal_display(th_frame), f"thermal-rgb {diff_ms:.1f} ms"))

            if pair != last_pair and min(rgb_id, th_id) % args.print_every == 0:
                print(
                    f"rgb_id={rgb_id} thermal_id={th_id} "
                    f"thermal_minus_rgb={diff_ms:.2f}ms avg={np.mean(diffs):.2f}ms"
                )
                last_pair = pair

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        rgb.running = False
        th.running = False
        rgb.join(1)
        th.join(1)
        rgb_cap.release()
        th_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
