#!/usr/bin/env python3
"""Display one V4L2 camera and measure its delivered frame rate."""

from __future__ import annotations

import argparse
import threading
import time
from collections import deque

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", default="/dev/video2", help="camera device")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="requested camera FPS; 0 keeps the driver default",
    )
    parser.add_argument("--fourcc", default="YUYV")
    parser.add_argument("--print-every", type=int, default=30)
    return parser.parse_args()


def open_camera(args):
    cap = cv2.VideoCapture(args.dev, cv2.CAP_V4L2)
    if not cap.isOpened() and args.dev.startswith("/dev/video"):
        cap = cv2.VideoCapture(int(args.dev[len("/dev/video"):]))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.dev}")
    if len(args.fourcc) != 4:
        cap.release()
        raise ValueError("--fourcc must contain exactly four characters")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps > 0:
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    # Keep enough V4L2 buffers for continuous capture. A dedicated reader
    # thread below prevents these buffers from adding processing latency.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)
    return cap


def measured_fps(timestamps, now):
    timestamps.append(now)
    while timestamps and now - timestamps[0] > 3.0:
        timestamps.popleft()
    if len(timestamps) < 2:
        return 0.0
    return (len(timestamps) - 1) / max(1e-6, timestamps[-1] - timestamps[0])


class LatestFrameCapture:
    """Continuously capture frames and expose only the newest one."""

    def __init__(self, cap):
        self.cap = cap
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.frame = None
        self.sequence = 0
        self.capture_fps = 0.0
        self.read_ms = 0.0
        self.failed_reads = 0
        self.timestamps = deque()
        self.thread = threading.Thread(target=self._run, name="camera-reader", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            read_start = time.perf_counter()
            ok, frame = self.cap.read()
            read_done = time.perf_counter()
            if not ok:
                with self.condition:
                    self.failed_reads += 1
                    self.condition.notify_all()
                continue

            capture_fps = measured_fps(self.timestamps, read_done)
            with self.condition:
                self.frame = frame
                self.sequence += 1
                self.capture_fps = capture_fps
                self.read_ms = (read_done - read_start) * 1000.0
                self.condition.notify_all()

    def latest(self, previous_sequence, timeout=1.0):
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence != previous_sequence
                or self.stop_event.is_set(),
                timeout=timeout,
            )
            return (
                self.sequence,
                self.frame,
                self.capture_fps,
                self.read_ms,
                self.failed_reads,
            )

    def stop(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()


def main():
    args = parse_args()
    cap = open_camera(args)
    display_timestamps = deque()
    displayed_count = 0
    last_sequence = 0

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    reported_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_text = "".join(chr((reported_fourcc >> (8 * i)) & 0xFF) for i in range(4))
    print(f"Camera: {args.dev}")
    print(
        f"Driver output: {actual_width}x{actual_height}, "
        f"FPS={reported_fps:.2f}, FOURCC={fourcc_text!r}"
    )
    print("Capture: dedicated reader thread, V4L2 buffers=4, latest frame only")
    print("Press q or Esc to quit.")

    reader = LatestFrameCapture(cap)
    reader.start()
    try:
        while True:
            sequence, frame, capture_fps, read_ms, failed_reads = reader.latest(
                last_sequence
            )
            if sequence == last_sequence or frame is None:
                print("Waiting for camera frame...")
                continue
            dropped_for_display = max(0, sequence - last_sequence - 1)
            last_sequence = sequence

            displayed_count += 1
            display_fps = measured_fps(display_timestamps, time.perf_counter())
            view = frame.copy()
            cv2.putText(
                view,
                f"capture {capture_fps:.1f} fps  display {display_fps:.1f} fps",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("camera_fps", view)

            if displayed_count % max(1, args.print_every) == 0:
                print(
                    f"captured={sequence} capture_fps={capture_fps:.2f} "
                    f"display_fps={display_fps:.2f} last_read={read_ms:.1f}ms "
                    f"display_dropped={dropped_for_display} read_failed={failed_reads}"
                )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        reader.stop()
        reader.thread.join(timeout=2.0)
        cap.release()
        if reader.thread.is_alive():
            reader.thread.join(timeout=1.0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
