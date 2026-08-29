"""Configuration and background processing worker for the thermal pose UI."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

UI_DIR = Path(__file__).resolve().parent

from core import warp as core
from core.thermal_mask import (
    open_thermal_camera,
    parse_raw_temperature,
    temp_to_display,
)
from pose_smoother import PoseSmoother
from video_recorder import RESEARCH_RECORDING_MODES, ResearchDataRecorder


def _load_alignment():
    defaults = {"scale": 0.740, "x": -5.0, "y": 15.0}
    path = UI_DIR / "thermal_alignment.json"
    if not path.exists():
        return defaults
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        return {key: float(saved.get(key, value)) for key, value in defaults.items()}
    except (OSError, ValueError, TypeError):
        return defaults


ALIGNMENT = _load_alignment()


@dataclass(frozen=True)
class PipelineConfig:
    # Parameters below are program-only settings by design.
    model: Path = (
        UI_DIR / "model/yolo26n-pose_rknn_model/best.sanitized-rk3588.rknn"
    )
    calibration: Path = UI_DIR / "calibration_standard.npz"
    thermal_dev: str = "/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0"
    thermal_fps: float = 25.0
    rgb_dev: str = "/dev/v4l/by-id/usb-HD_USB_Camera_HD_USB_Camera-video-index0"
    rgb_width: int = 640
    rgb_height: int = 480
    rgb_fps: float = 30.0
    rgb_fourcc: str = "YUYV"
    rgb_balance: float = 0.0
    use_undistort: bool = True
    thermal_scale: float = ALIGNMENT["scale"]
    thermal_x: float = ALIGNMENT["x"]
    thermal_y: float = ALIGNMENT["y"]
    # No beamsplitter is used: keep thermal and RGB in the same native
    # left-to-right orientation. Both flip stages must remain disabled.
    mirror_thermal_on_read: bool = False
    thermal_flip_during_alignment: bool = False
    crop_to_thermal: bool = True
    thermal_rgb_offset_ms: float = 50.0
    img_size: int = 640
    # Run RKNN once every N processed frames and reuse the latest pose between runs.
    # Set to 1 for inference on every frame, 2 for every other frame, and so on.
    pose_inference_interval: int = 2
    confidence: float = 0.25
    iou: float = 0.45
    keypoint_confidence: float = 0.2
    lower_body_keypoint_confidence: float = 0.7
    mask_alpha: float = 0.45
    mask_percentile: float | None = None
    warped_mask_dilate_px: int = 15
    # Update the captured background only after a pixel has remained outside
    # both the thermal foreground and the guarded person boxes for a while.
    background_confirm_frames: int = 20
    background_update_alpha: float = 0.01
    background_motion_threshold: int = 10
    background_person_guard_ratio: float = 0.20
    recording_dir: Path = Path("/mnt/ssd/videos")
    # MP4 is only the container. Use H.264 for playback compatibility with
    # Chromium/VS Code instead of the former MPEG-4 Part 2 (mp4v) stream.
    recording_fourcc: str = "libx264"
    recording_max_seconds: float = 8 * 60 * 60


class PoseArgs:
    """Small adapter for warp.run_pose's argument interface."""

    def __init__(self, config: PipelineConfig):
        self.img_size = config.img_size
        self.conf = config.confidence
        self.iou = config.iou


class PipelineWorker(QThread):
    frames_ready = pyqtSignal(object, object, object)
    stats_ready = pyqtSignal(dict)
    state_changed = pyqtSignal(str)
    background_captured = pyqtSignal()
    recording_changed = pyqtSignal(bool, object)
    failed = pyqtSignal(str)

    def __init__(self, config: PipelineConfig, temperature_getter, alignment_getter):
        super().__init__()
        self.config = config
        self.temperature_getter = temperature_getter
        self.alignment_getter = alignment_getter
        self.stop_event = threading.Event()
        self.capture_background_event = threading.Event()
        self.recording_requested = threading.Event()
        self.recording_lock = threading.Lock()
        self.requested_recording_modes = RESEARCH_RECORDING_MODES
        self.thermal_view_requested = threading.Event()
        self.color_view_requested = threading.Event()

    def request_stop(self):
        self.stop_event.set()

    def request_background_capture(self):
        self.capture_background_event.set()

    def request_view(self, mode, enabled):
        event = {
            "raw_thermal": self.thermal_view_requested,
            "raw_rgb": self.color_view_requested,
        }[mode]
        if enabled:
            event.set()
        else:
            event.clear()

    def request_recording(self, enabled, modes=None):
        if enabled:
            with self.recording_lock:
                self.requested_recording_modes = tuple(
                    modes or RESEARCH_RECORDING_MODES
                )
            self.recording_requested.set()
        else:
            self.recording_requested.clear()

    def recording_modes(self):
        with self.recording_lock:
            return self.requested_recording_modes

    def run(self):
        config = self.config
        thermal_cap = rgb_cap = rknn = None
        thermal_reader = rgb_reader = None
        frame_times = deque()
        inference_times = deque(maxlen=120)
        recorder = ResearchDataRecorder(
            config.recording_dir, config.thermal_fps, config.recording_fourcc
        )
        recording_started_at = None

        try:
            if not config.model.exists():
                raise FileNotFoundError(f"RKNN model not found: {config.model}")

            self.state_changed.emit("Loading RKNN model…")
            rknn = core.load_rknn(str(config.model))

            self.state_changed.emit("Connecting thermal camera…")
            thermal_cap = open_thermal_camera(config.thermal_dev)
            thermal_cap.set(cv2.CAP_PROP_FPS, config.thermal_fps)
            thermal_cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)

            class RGBArgs:
                rgb_dev = config.rgb_dev
                rgb_width = config.rgb_width
                rgb_height = config.rgb_height
                rgb_fps = config.rgb_fps
                rgb_fourcc = config.rgb_fourcc

            self.state_changed.emit("Connecting RGB camera…")
            rgb_cap = core.open_rgb_camera(RGBArgs)
            thermal_probe = core.verify_camera_read(thermal_cap, "thermal")
            parse_raw_temperature(thermal_probe)
            rgb_probe = core.verify_camera_read(rgb_cap, "rgb")
            if rgb_probe.ndim != 3 or rgb_probe.shape[2] != 3:
                raise RuntimeError(f"Unexpected RGB frame shape: {rgb_probe.shape}")

            thermal_reader = core.TimestampedCapture(
                thermal_cap, "thermal", max(8, int(config.thermal_fps * 2))
            )
            rgb_reader = core.TimestampedCapture(
                rgb_cap, "rgb", max(8, int(config.rgb_fps * 4))
            )
            thermal_reader.start()
            rgb_reader.start()

            calibration = (
                core.load_calibration(config.calibration)
                if config.use_undistort
                else None
            )
            pose_args = PoseArgs(config)
            maps = matrix = roi = None
            alignment_xy = None
            background = None
            background_model = None
            background_safe_count = None
            previous_gray = None
            detections = []
            pose_smoother = PoseSmoother()
            processed_frame_index = 0
            last_sequence = 0
            offset = config.thermal_rgb_offset_ms / 1000.0
            self.state_changed.emit("Running")

            while not self.stop_event.is_set():
                thermal_sample = thermal_reader.wait_newest(last_sequence, timeout=0.25)
                if thermal_sample is None:
                    continue
                thermal_time, thermal_raw, last_sequence = thermal_sample
                rgb_reader.raise_if_failed()
                rgb_sample = core.select_frame_by_time(
                    rgb_reader.snapshot(), thermal_time - offset
                )
                if rgb_sample is None:
                    continue
                rgb_time, rgb_raw, _ = rgb_sample

                size = (rgb_raw.shape[1], rgb_raw.shape[0])
                current_alignment_xy = self.alignment_getter()
                if matrix is None or current_alignment_xy != alignment_xy:
                    alignment_xy = current_alignment_xy
                    matrix = core.alignment_matrix(
                        size,
                        max(0.05, config.thermal_scale),
                        alignment_xy[0],
                        alignment_xy[1],
                    )
                    if calibration and maps is None:
                        maps = core.make_standard_maps(
                            calibration[1],
                            calibration[2],
                            calibration[3],
                            size,
                            config.rgb_balance,
                        )
                    if config.crop_to_thermal:
                        roi = core.valid_roi(size, matrix)

                rgb = (
                    cv2.remap(rgb_raw, maps[0], maps[1], cv2.INTER_LINEAR)
                    if maps
                    else rgb_raw
                )
                thermal_raw16 = parse_raw_temperature(thermal_raw)
                temp_c = thermal_raw16.astype(np.float32) / 64.0 - 273.15
                # Optional sensor-orientation correction. This is disabled for
                # the current no-beamsplitter layout, so the thermal mask keeps
                # the same left-to-right orientation as the RGB frame.
                if config.mirror_thermal_on_read:
                    temp_c = cv2.flip(temp_c, 1)
                min_temp, max_temp = self.temperature_getter()
                thermal_mask, threshold = core.make_mask(
                    temp_c, min_temp, max_temp, config.mask_percentile
                )
                aligned_mask = core.dilate(
                    core.align(
                        thermal_mask,
                        size,
                        matrix,
                        cv2.INTER_NEAREST,
                        config.thermal_flip_during_alignment,
                    ),
                    config.warped_mask_dilate_px,
                )
                # Use one recording-state snapshot for this entire frame so
                # view generation and recorder actions cannot disagree.
                recording_requested = self.recording_requested.is_set()
                recording_modes = self.recording_modes() if recording_requested else ()
                need_thermal_view = (
                    self.thermal_view_requested.is_set()
                    or "thermal_mask" in recording_modes
                )
                need_color_view = (
                    self.color_view_requested.is_set()
                    or "color_mode" in recording_modes
                )
                need_raw_thermal = (
                    need_thermal_view or "raw_thermal" in recording_modes
                )
                raw_thermal_view = temp_to_display(temp_c) if need_raw_thermal else None
                if roi:
                    rgb = core.crop(rgb, roi)
                    aligned_mask = core.crop(aligned_mask, roi)
                if self.capture_background_event.is_set():
                    background = rgb.copy()
                    background_model = rgb.astype(np.float32)
                    background_safe_count = np.zeros(rgb.shape[:2], np.uint16)
                    previous_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                    self.capture_background_event.clear()
                    self.background_captured.emit()
                inference_interval = max(1, int(config.pose_inference_interval))
                infer_ms = None
                inference_performed = processed_frame_index % inference_interval == 0
                # Metadata contains only fresh model output. If the inference
                # interval is increased later, skipped frames are explicitly
                # marked instead of presenting reused/smoothed poses as new
                # detections.
                raw_detections = []
                if inference_performed:
                    raw_detections, infer_ms = core.run_pose(rknn, rgb, pose_args)
                    detections = pose_smoother.update(raw_detections)
                processed_frame_index += 1

                # Slowly learn only pixels that have been consistently safe.
                # A one-frame hole in the thermal mask must never be enough to
                # copy a person or moving bedding into the background model.
                current_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                if (
                    background_model is not None
                    and background_safe_count is not None
                    and background_model.shape == rgb.shape
                    and background_safe_count.shape == rgb.shape[:2]
                ):
                    person_guard = np.zeros(rgb.shape[:2], np.uint8)
                    height, width = rgb.shape[:2]
                    guard_ratio = max(0.0, float(config.background_person_guard_ratio))
                    for box, _, _ in detections:
                        x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
                        pad_x = (x2 - x1) * guard_ratio
                        pad_y = (y2 - y1) * guard_ratio
                        left = max(0, int(np.floor(x1 - pad_x)))
                        top = max(0, int(np.floor(y1 - pad_y)))
                        right = min(width, int(np.ceil(x2 + pad_x)))
                        bottom = min(height, int(np.ceil(y2 + pad_y)))
                        if right > left and bottom > top:
                            person_guard[top:bottom, left:right] = 255

                    if previous_gray is None or previous_gray.shape != current_gray.shape:
                        low_motion = np.zeros(rgb.shape[:2], dtype=bool)
                    else:
                        frame_delta = cv2.absdiff(current_gray, previous_gray)
                        low_motion = frame_delta <= max(
                            0, int(config.background_motion_threshold)
                        )
                    safe = (aligned_mask == 0) & (person_guard == 0) & low_motion
                    background_safe_count[~safe] = 0
                    incrementable = safe & (
                        background_safe_count < np.iinfo(np.uint16).max
                    )
                    background_safe_count[incrementable] += 1
                    confirmed = background_safe_count >= max(
                        1, int(config.background_confirm_frames)
                    )
                    if np.any(confirmed):
                        cv2.accumulateWeighted(
                            rgb,
                            background_model,
                            max(0.0, min(1.0, float(config.background_update_alpha))),
                            mask=confirmed.astype(np.uint8) * 255,
                        )
                        background = np.clip(background_model, 0, 255).astype(np.uint8)
                previous_gray = current_gray
                thermal_view = raw_thermal_view if need_thermal_view else None
                color_view = rgb_raw if need_color_view else None
                # Matches warp.py's normal behavior before an optional empty
                # background is captured: masked pixels are blacked out.
                stick_view = core.stickfigure(
                    rgb,
                    aligned_mask,
                    detections,
                    background=background,
                    kpt_conf=config.keypoint_confidence,
                    lower_body_kpt_conf=config.lower_body_keypoint_confidence,
                )
                if recording_requested and not recorder.active:
                    recording_paths = recorder.start(
                        stick_view,
                        rgb_raw,
                        raw_thermal_view,
                        {
                            "model": str(config.model),
                            "pose_confidence_threshold": config.confidence,
                            "pose_iou_threshold": config.iou,
                            "keypoint_display_threshold": config.keypoint_confidence,
                            "pose_inference_interval": inference_interval,
                            "keypoint_order": [
                                "nose", "left_eye", "right_eye", "left_ear",
                                "right_ear", "left_shoulder", "right_shoulder",
                                "left_elbow", "right_elbow", "left_wrist",
                                "right_wrist", "left_hip", "right_hip",
                                "left_knee", "right_knee", "left_ankle",
                                "right_ankle",
                            ],
                            "rgb_resolution": list(rgb_raw.shape[1::-1]),
                            "thermal_resolution": list(thermal_raw16.shape[::-1]),
                        },
                    )
                    recording_started_at = time.monotonic()
                    self.recording_changed.emit(True, recording_paths)

                if recorder.active:
                    recording_limit_reached = (
                        recording_started_at is not None
                        and time.monotonic() - recording_started_at
                        >= config.recording_max_seconds
                    )
                    if recording_limit_reached:
                        # Clear the request as well as stopping the recorder;
                        # otherwise the next pipeline frame would start a new
                        # recording session immediately.
                        self.recording_requested.clear()
                        recording_paths = recorder.stop()
                        recording_started_at = None
                        self.recording_changed.emit(False, recording_paths)
                    elif recording_requested:
                        recorder.write(
                            stick_view,
                            rgb_raw,
                            raw_thermal_view,
                            raw_detections,
                            thermal_time,
                            rgb_time,
                            inference_ms=infer_ms,
                            inference_performed=inference_performed,
                        )
                    else:
                        recording_paths = recorder.stop()
                        recording_started_at = None
                        self.recording_changed.emit(False, recording_paths)
                now = time.perf_counter()
                frame_times.append(now)
                fps = core.fps_from_times(frame_times, now)
                if infer_ms is not None:
                    inference_times.append(infer_ms)

                self.frames_ready.emit(stick_view, thermal_view, color_view)
                self.stats_ready.emit(
                    {
                        "fps": fps,
                        "npu_ms": (
                            sum(inference_times) / len(inference_times)
                            if inference_times
                            else 0.0
                        ),
                        "detections": len(detections),
                        "temperature_min": float(np.nanmin(temp_c)),
                        "temperature_max": float(np.nanmax(temp_c)),
                        "threshold": threshold,
                        "pair_delta_ms": (thermal_time - rgb_time) * 1000.0,
                    }
                )

        except Exception as exc:
            if not self.stop_event.is_set():
                self.failed.emit(str(exc))
        finally:
            self.state_changed.emit("Shutting down…")
            if recorder.active:
                recording_paths = recorder.stop()
                self.recording_changed.emit(False, recording_paths)
            if thermal_reader:
                thermal_reader.stop()
            if rgb_reader:
                rgb_reader.stop()
            if thermal_reader:
                thermal_reader.thread.join(timeout=2)
            if rgb_reader:
                rgb_reader.thread.join(timeout=2)
            if thermal_cap:
                thermal_cap.release()
            if rgb_cap:
                rgb_cap.release()
            if rknn:
                rknn.release()
            self.state_changed.emit("Stopped")
