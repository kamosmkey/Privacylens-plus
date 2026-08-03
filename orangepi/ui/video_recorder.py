"""Timestamped variable-frame-rate recording for UI pipeline frames."""

from __future__ import annotations

import time
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

try:
    import av
except ImportError:  # Report the optional dependency only when recording starts.
    av = None


RECORDING_MODES = (
    ("Stick Figure", "stickfigure"),
    ("Raw Thermal", "raw_thermal"),
    ("Raw RGB", "raw_rgb"),
    ("Thermal Mask", "thermal_mask"),
    ("Color Mode", "color_mode"),
)
VALID_MODE_KEYS = frozenset(key for _, key in RECORDING_MODES)


class VideoRecorder:
    """Write each produced frame once with its real monotonic timestamp."""

    def __init__(self, output_dir: Path, fps: float, fourcc: str):
        self.output_dir = Path(output_dir)
        self.fps = float(fps)
        self.fourcc = fourcc
        self.container = None
        self.stream = None
        self.path = None
        self.mode = None
        self.frame_size = None
        self.started_at = None
        self.last_pts = -1
        self.last_frame = None
        self.time_base = Fraction(1, 1_000)

    @property
    def active(self):
        return self.container is not None

    def start(self, mode: str, frame, now=None):
        if self.active:
            raise RuntimeError("A recording is already active")
        if mode not in VALID_MODE_KEYS:
            raise ValueError(f"Unknown recording mode: {mode}")
        if av is None:
            raise RuntimeError(
                "Variable-frame-rate recording requires PyAV. "
                "Install it with: sudo apt install python3-av"
            )

        frame = self._prepare_frame(frame)
        height, width = frame.shape[:2]
        mode_output_dir = self.output_dir / mode
        mode_output_dir.mkdir(parents=True, exist_ok=True)
        self.path = mode_output_dir / (
            f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp4"
        )
        try:
            self.container = av.open(str(self.path), mode="w")
            self.stream = self.container.add_stream("mpeg4", rate=max(1, round(self.fps)))
            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = "yuv420p"
            self.stream.time_base = self.time_base
            self.stream.codec_context.time_base = self.time_base
            # Avoid several selected-mode encoders competing for every CPU core.
            self.stream.codec_context.thread_count = 1
        except Exception:
            if self.container is not None:
                self.container.close()
            self.container = self.stream = None
            self.path = None
            raise

        self.frame_size = (width, height)
        self.mode = mode
        self.started_at = time.perf_counter() if now is None else now
        self.last_pts = -1
        self.last_frame = None
        return self.path

    def write(self, frame, now=None):
        if not self.active:
            raise RuntimeError("No recording is active")

        frame = self._prepare_frame(frame)
        if (frame.shape[1], frame.shape[0]) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)

        current_time = time.perf_counter() if now is None else now
        pts = max(self.last_pts + 1, round((current_time - self.started_at) * 1_000))
        self._encode(frame, pts)
        self.last_pts = pts
        self.last_frame = frame.copy()

    def _encode(self, frame, pts):
        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = self.time_base
        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)

    def stop(self, now=None):
        saved_path = self.path
        if self.container is not None:
            if self.last_frame is not None:
                stopped_at = time.perf_counter() if now is None else now
                stop_pts = round((stopped_at - self.started_at) * 1_000)
                if stop_pts > self.last_pts:
                    self._encode(self.last_frame, stop_pts)
            for packet in self.stream.encode():
                self.container.mux(packet)
            self.container.close()
        self.container = self.stream = None
        self.path = None
        self.mode = None
        self.frame_size = None
        self.started_at = None
        self.last_pts = -1
        self.last_frame = None
        return saved_path

    @staticmethod
    def _prepare_frame(frame):
        if frame is None:
            raise ValueError("Cannot record an empty frame")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Unsupported recording frame shape: {frame.shape}")
        return np.ascontiguousarray(frame)


class MultiVideoRecorder:
    """Record selected views using one shared timestamp per UI pipeline frame."""

    def __init__(self, output_dir: Path, fps: float, fourcc: str):
        self.output_dir = Path(output_dir)
        self.fps = float(fps)
        self.fourcc = fourcc
        self.recorders = {}

    @property
    def active(self):
        return bool(self.recorders)

    @property
    def modes(self):
        return tuple(self.recorders)

    def start(self, modes, frames):
        if self.active:
            raise RuntimeError("A recording session is already active")
        modes = tuple(dict.fromkeys(modes))
        if not modes:
            raise ValueError("Select at least one recording mode")

        now = time.perf_counter()
        try:
            for mode in modes:
                if frames.get(mode) is None:
                    raise ValueError(f"Cannot record an empty frame for mode: {mode}")
                recorder = VideoRecorder(self.output_dir, self.fps, self.fourcc)
                recorder.start(mode, frames[mode], now=now)
                self.recorders[mode] = recorder
        except Exception:
            self.stop()
            raise
        return tuple(recorder.path for recorder in self.recorders.values())

    def write(self, frames):
        now = time.perf_counter()
        for mode, recorder in self.recorders.items():
            recorder.write(frames[mode], now=now)

    def stop(self):
        now = time.perf_counter()
        paths = tuple(
            recorder.stop(now=now) for recorder in self.recorders.values()
        )
        self.recorders.clear()
        return paths
