"""Configuration and background processing worker for the thermal pose UI."""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

UI_DIR = Path(__file__).resolve().parent
RKNN_DIR = UI_DIR.parent / "rknn"
if str(RKNN_DIR) not in sys.path:
    sys.path.insert(0, str(RKNN_DIR))

import warp as core  # noqa: E402
from thermal_mask import open_thermal_camera, parse_temperature, temp_to_display  # noqa: E402
from video_recorder import MultiVideoRecorder  # noqa: E402


@dataclass(frozen=True)
class PipelineConfig:
    # Parameters below are program-only settings by design.
    model: Path = UI_DIR / "model/yolov8n-pose.rknn"
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
    thermal_scale: float = 0.740
    thermal_x: float = -5.0
    thermal_y: float = 15.0
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
    keypoint_confidence: float = 0.4
    mask_alpha: float = 0.45
    mask_percentile: float | None = None
    warped_mask_dilate_px: int = 15
    recording_dir: Path = UI_DIR / "videos"
    recording_fourcc: str = "mp4v"


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
        self.requested_recording_modes = ("stickfigure",)
        self.thermal_view_requested = threading.Event()
        self.color_view_requested = threading.Event()

    def request_stop(self):
        self.stop_event.set()

    def request_background_capture(self):
        self.capture_background_event.set()

    def request_view(self, mode, enabled):
        event = {
            "thermal_mask": self.thermal_view_requested,
            "color_mode": self.color_view_requested,
        }[mode]
        if enabled:
            event.set()
        else:
            event.clear()

    def request_recording(self, enabled, modes=None):
        if enabled:
            with self.recording_lock:
                self.requested_recording_modes = tuple(modes or ("stickfigure",))
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
        recorder = MultiVideoRecorder(
            config.recording_dir, config.thermal_fps, config.recording_fourcc
        )

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
            parse_temperature(thermal_probe)
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
            detections = []
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
                temp_c = parse_temperature(thermal_raw)
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
                aligned_thermal = (
                    core.align(
                        raw_thermal_view,
                        size,
                        matrix,
                        cv2.INTER_LINEAR,
                        config.thermal_flip_during_alignment,
                    )
                    if need_thermal_view
                    else None
                )
                if roi:
                    rgb = core.crop(rgb, roi)
                    aligned_mask = core.crop(aligned_mask, roi)
                    if aligned_thermal is not None:
                        aligned_thermal = core.crop(aligned_thermal, roi)
                if self.capture_background_event.is_set():
                    background = rgb.copy()
                    self.capture_background_event.clear()
                    self.background_captured.emit()

                # Keep the captured image only under the current foreground.
                # Once a person moves away and the thermal mask clears, learn
                # the newly exposed pixels from the live RGB frame.  This makes
                # the transparency background follow scene changes instead of
                # permanently showing the original capture.
                if background is not None and background.shape == rgb.shape:
                    background[aligned_mask == 0] = rgb[aligned_mask == 0]
                inference_interval = max(1, int(config.pose_inference_interval))
                infer_ms = None
                if processed_frame_index % inference_interval == 0:
                    detections, infer_ms = core.run_pose(rknn, rgb, pose_args)
                processed_frame_index += 1
                thermal_view = (
                    core.overlay(
                        aligned_thermal,
                        aligned_mask,
                        alpha=config.mask_alpha,
                        contour=True,
                    )
                    if need_thermal_view
                    else None
                )
                color_view = (
                    core.overlay(rgb, aligned_mask, alpha=config.mask_alpha)
                    if need_color_view
                    else None
                )
                # Matches warp.py's normal behavior before an optional empty
                # background is captured: masked pixels are blacked out.
                stick_view = core.stickfigure(
                    rgb,
                    aligned_mask,
                    detections,
                    background=background,
                    kpt_conf=config.keypoint_confidence,
                )
                recordable_frames = {
                    "stickfigure": stick_view,
                    "raw_thermal": raw_thermal_view,
                    "raw_rgb": rgb_raw,
                    "thermal_mask": thermal_view,
                    "color_mode": color_view,
                }
                if recording_requested and not recorder.active:
                    recording_paths = recorder.start(
                        recording_modes, recordable_frames
                    )
                    self.recording_changed.emit(True, recording_paths)

                if recorder.active:
                    if recording_requested:
                        recorder.write(recordable_frames)
                    else:
                        recording_paths = recorder.stop()
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
