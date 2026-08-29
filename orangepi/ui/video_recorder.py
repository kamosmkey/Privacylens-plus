"""Timestamped variable-frame-rate recording for UI pipeline frames."""

from __future__ import annotations

import json
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


RESEARCH_RECORDING_MODES = (
    "stickfigure",
    "raw_rgb",
    "raw_thermal",
    "pose_metadata",
)
VALID_VIDEO_MODES = frozenset(RESEARCH_RECORDING_MODES[:-1])


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
        if mode not in VALID_VIDEO_MODES:
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
            self.stream = self.container.add_stream(
                self.fourcc, rate=max(1, round(self.fps))
            )
            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = "yuv420p"
            if self.fourcc == "libx264":
                # Favor real-time recording on the embedded device. yuv420p
                # keeps the resulting H.264 stream broadly browser-compatible.
                self.stream.options = {"preset": "ultrafast", "crf": "23"}
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


class ResearchDataRecorder:
    """Record synchronized stick-figure/RGB/raw-thermal videos and metadata."""

    def __init__(self, output_dir: Path, fps: float, fourcc: str, flush_frames=250):
        self.output_dir = Path(output_dir)
        self.fps = float(fps)
        self.fourcc = fourcc
        self.flush_frames = max(1, int(flush_frames))
        self.session_dir = None
        self.stickfigure_recorder = None
        self.rgb_recorder = None
        self.thermal_recorder = None
        self.metadata_file = None
        self.metadata_path = None
        self.frame_index = 0
        self.started_at = None

    @property
    def active(self):
        return self.session_dir is not None

    def start(self, stickfigure_frame, rgb_frame, raw_thermal_frame, session_metadata=None):
        if self.active:
            raise RuntimeError("A research recording session is already active")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.session_dir = self.output_dir / f"session_{stamp}"
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.metadata_path = self.session_dir / "pose_metadata.jsonl"
        self.metadata_file = self.metadata_path.open("w", encoding="utf-8")
        self.started_at = time.perf_counter()
        self.frame_index = 0

        info = {
            "format_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "thermal_storage": "unaligned 8-bit Inferno preview without a thermal mask overlay",
            "thermal_video_contains_absolute_temperature": False,
            "stickfigure_video": "the processed stick-figure view displayed by the UI",
            "pose_metadata": "one JSON object per processed frame; detections are unsmoothed model outputs",
        }
        info.update(session_metadata or {})
        (self.session_dir / "session.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        try:
            self.stickfigure_recorder = VideoRecorder(
                self.session_dir, self.fps, self.fourcc
            )
            self.rgb_recorder = VideoRecorder(
                self.session_dir, self.fps, self.fourcc
            )
            self.thermal_recorder = VideoRecorder(
                self.session_dir, self.fps, self.fourcc
            )
            stickfigure_path = self.stickfigure_recorder.start(
                "stickfigure", stickfigure_frame, now=self.started_at
            )
            rgb_path = self.rgb_recorder.start(
                "raw_rgb", rgb_frame, now=self.started_at
            )
            thermal_path = self.thermal_recorder.start(
                "raw_thermal", raw_thermal_frame, now=self.started_at
            )
        except Exception:
            self.stop()
            raise

        return (stickfigure_path, rgb_path, thermal_path, self.metadata_path)

    def write(
        self,
        stickfigure_frame,
        rgb_frame,
        raw_thermal_frame,
        detections,
        thermal_timestamp,
        rgb_timestamp,
        inference_ms=None,
        inference_performed=True,
    ):
        if not self.active:
            raise RuntimeError("No research recording is active")

        now = time.perf_counter()
        self.stickfigure_recorder.write(stickfigure_frame, now=now)
        self.rgb_recorder.write(rgb_frame, now=now)
        self.thermal_recorder.write(raw_thermal_frame, now=now)

        people = []
        for box, score, keypoints in detections:
            points = np.asarray(keypoints, dtype=np.float32)
            people.append(
                {
                    "bbox_xyxy": np.asarray(box, dtype=np.float32).tolist(),
                    "person_confidence": float(score),
                    "keypoints_xy_confidence": points.tolist(),
                }
            )
        record = {
            "frame_index": self.frame_index,
            "elapsed_seconds": now - self.started_at,
            "thermal_timestamp": float(thermal_timestamp),
            "rgb_timestamp": float(rgb_timestamp),
            "pair_delta_ms": (float(thermal_timestamp) - float(rgb_timestamp)) * 1000.0,
            "inference_performed": bool(inference_performed),
            "inference_ms": None if inference_ms is None else float(inference_ms),
            "person_detected": bool(people),
            "person_count": len(people),
            "people": people,
        }
        self.metadata_file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.frame_index += 1

        if self.frame_index % self.flush_frames == 0:
            self.metadata_file.flush()

    def stop(self):
        if not self.active:
            return ()
        session_dir = self.session_dir
        try:
            if self.stickfigure_recorder is not None:
                self.stickfigure_recorder.stop()
            if self.rgb_recorder is not None:
                self.rgb_recorder.stop()
            if self.thermal_recorder is not None:
                self.thermal_recorder.stop()
            if self.metadata_file is not None:
                self.metadata_file.flush()
                self.metadata_file.close()
        finally:
            self.session_dir = None
            self.stickfigure_recorder = None
            self.rgb_recorder = None
            self.thermal_recorder = None
            self.metadata_file = None
            self.metadata_path = None
            self.started_at = None
        return (session_dir,)
