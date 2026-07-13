#!/usr/bin/env python3
import argparse
import os
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from calibration_common import apply_calibration_mask, estimate_calibration, smooth_calibration
from thermal_common import (
    make_full_temperature_mask,
    open_thermal_camera,
    overlay_mask,
    parse_temperature,
    put_text_top_right,
    temp_to_display,
    thermal_visible_to_bgr,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Jetson thermal mask + YOLO pose calibration."
    )
    parser.add_argument(
        "--model",
        default="model/yolo26n-pose.engine",
        help="Path to an existing YOLO pose TensorRT .engine model.",
    )
    parser.add_argument("--thermal-dev", default="/dev/video0", help="Thermal camera device")
    parser.add_argument("--rgb-dev", default="/dev/video2", help="RGB camera device")
    parser.add_argument("--rgb-fourcc", default="MJPG")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, cuda:0, etc.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--kpt-conf", type=float, default=0.4)
    parser.add_argument("--mask-min-temp", type=float, default=24.0)
    parser.add_argument("--mask-max-temp", type=float, default=42.0)
    parser.add_argument("--mask-percentile", type=float, default=None)
    parser.add_argument("--mask-alpha", type=float, default=0.45)
    parser.add_argument(
        "--warped-mask-dilate-px",
        type=int,
        default=0,
        help="Dilate the thermal mask after warping onto RGB by this many pixels.",
    )
    parser.add_argument("--calibration-mode", choices=("affine", "homography"), default="affine")
    parser.add_argument("--update-every", type=int, default=20)
    parser.add_argument("--smooth-alpha", type=float, default=0.25)
    parser.add_argument("--ransac-thresh", type=float, default=6.0)
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument(
        "--thermal-rgb-offset-ms",
        type=float,
        default=50.0,
        help=(
            "Temporal offset for calibration/overlay. Positive means thermal trails RGB, "
            "so current thermal is paired with an older RGB frame."
        ),
    )
    parser.add_argument("--timing", action="store_true", help="Print per-stage timing and frame brightness")
    return parser.parse_args()


def resolve_device(device):
    return 0 if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)


def resolve_engine(args):
    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = (Path.cwd() / model_path).resolve()

    if model_path.suffix != ".engine":
        raise ValueError("--model must point to an existing .engine file")
    if not model_path.exists():
        raise FileNotFoundError(f"TensorRT engine not found: {model_path}")
    return model_path


def open_rgb_camera(dev, fourcc):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened() and dev.startswith("/dev/video"):
        cap = cv2.VideoCapture(int(dev.replace("/dev/video", "")))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RGB camera {dev}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def predict_pose(model, frame, args, device):
    results = model.predict(
        source=frame,
        imgsz=args.img_size,
        conf=args.conf,
        iou=args.iou,
        device=device,
        verbose=False,
    )
    return results[0] if results else None


def detections_from_result(result, image_shape):
    if result is None or result.boxes is None or result.keypoints is None:
        return []

    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    kpt_xy = result.keypoints.xy.detach().cpu().numpy().astype(np.float32)
    if result.keypoints.conf is None:
        kpt_conf = np.ones(kpt_xy.shape[:2], dtype=np.float32)
    else:
        kpt_conf = result.keypoints.conf.detach().cpu().numpy().astype(np.float32)

    h, w = image_shape[:2]
    detections = []
    for box, score, xy, conf in zip(boxes, scores, kpt_xy, kpt_conf):
        x1, y1, x2, y2 = box
        x1 = np.clip(x1, 0, w - 1)
        y1 = np.clip(y1, 0, h - 1)
        x2 = np.clip(x2, 0, w - 1)
        y2 = np.clip(y2, 0, h - 1)
        if x2 <= x1 or y2 <= y1:
            continue

        xy[:, 0] = np.clip(xy[:, 0], 0, w - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)
        kpts = np.concatenate((xy, conf[:, None]), axis=1).astype(np.float32)
        detections.append((np.array([x1, y1, x2, y2], dtype=np.float32), float(score), kpts))
    return detections


def select_primary_person(detections):
    if not detections:
        return None

    def rank(det):
        box, score, _ = det
        x1, y1, x2, y2 = box
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return score * max(1.0, area)

    return max(detections, key=rank)


def fps_from_times(times, now):
    while times and now - times[0] > 3.0:
        times.popleft()
    if len(times) < 2:
        return 0.0
    return (len(times) - 1) / max(1e-6, times[-1] - times[0])


def select_frame_by_time(samples, target_time):
    if not samples:
        return None
    return min(samples, key=lambda item: abs(item[0] - target_time))


def dilate_mask(mask, radius_px):
    radius_px = max(0, int(radius_px))
    if radius_px == 0:
        return mask
    kernel_size = radius_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)


def show_labeled(name, frame, text):
    cv2.imshow(name, put_text_top_right(frame, text))


def main():
    args = parse_args()
    device = resolve_device(args.device)
    engine_path = resolve_engine(args)
    update_every = max(1, args.update_every)
    smooth_alpha = min(1.0, max(0.0, args.smooth_alpha))
    thermal_rgb_offset_s = args.thermal_rgb_offset_ms / 1000.0

    model = YOLO(str(engine_path))
    thermal_cap = open_thermal_camera(args.thermal_dev)
    rgb_cap = open_rgb_camera(args.rgb_dev, args.rgb_fourcc)
    rgb_width = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rgb_height = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    calibration = None
    frame_count = 0
    frame_times = deque()
    rgb_frame_buffer = deque(maxlen=120)
    last_success_frame = None
    last_success_points = []
    last_success_inliers = 0

    print(f"Model: {engine_path}")
    print(f"Device: {device}")
    print(f"Thermal camera: {args.thermal_dev}")
    print(f"RGB camera: {args.rgb_dev} {rgb_width}x{rgb_height} {args.rgb_fourcc}")
    print(f"Calibration mode: {args.calibration_mode}")
    print(f"Thermal/RGB offset: {args.thermal_rgb_offset_ms:+.1f} ms")
    print(f"Warped mask dilation: {max(0, args.warped_mask_dilate_px)} px")
    print("Windows: thermal_masked, thermal_yolo, rgb_yolo, rgb_calibration_masked")
    print("Press q in any OpenCV window to quit.")

    try:
        while True:
            t0 = time.perf_counter()
            ok_t, thermal_frame = thermal_cap.read()
            thermal_read_done = time.perf_counter()
            ok_r, rgb_frame = rgb_cap.read()
            rgb_read_done = time.perf_counter()
            t_read = rgb_read_done
            if not ok_t or not ok_r:
                print(f"read failed: thermal={ok_t} rgb={ok_r}")
                continue

            rgb_frame_buffer.append((rgb_read_done, rgb_frame.copy()))
            rgb_target_time = thermal_read_done - thermal_rgb_offset_s
            selected_rgb = select_frame_by_time(rgb_frame_buffer, rgb_target_time)
            rgb_pair_delta_ms = 0.0
            if selected_rgb is not None:
                selected_rgb_time, rgb_frame = selected_rgb
                rgb_pair_delta_ms = (thermal_read_done - selected_rgb_time) * 1000.0

            temp_c = parse_temperature(thermal_frame)
            thermal_yolo_frame = thermal_visible_to_bgr(thermal_frame)
            t_parse = time.perf_counter()

            thermal_result, rgb_result = (
                predict_pose(model, thermal_yolo_frame, args, device),
                predict_pose(model, rgb_frame, args, device),
            )
            t_yolo = time.perf_counter()

            thermal_dets, rgb_dets = (
                detections_from_result(thermal_result, thermal_yolo_frame.shape),
                detections_from_result(rgb_result, rgb_frame.shape),
            )
            thermal_mask = make_full_temperature_mask(
                temp_c,
                thermal_dets,
                args.mask_min_temp,
                args.mask_max_temp,
                args.mask_percentile,
            )
            t_mask = time.perf_counter()
            thermal_primary = select_primary_person(thermal_dets)
            rgb_primary = select_primary_person(rgb_dets)

            did_try_update = did_update = False
            current_points = []
            current_inliers = 0
            if frame_count % update_every == 0 and thermal_primary is not None and rgb_primary is not None:
                did_try_update = True
                new_calibration, inliers, current_points = estimate_calibration(
                    thermal_primary[2],
                    rgb_primary[2],
                    mode=args.calibration_mode,
                    min_conf=args.kpt_conf,
                    ransac_thresh=args.ransac_thresh,
                )
                if new_calibration is not None:
                    calibration = smooth_calibration(calibration, new_calibration, smooth_alpha)
                    current_inliers = int(np.count_nonzero(inliers)) if inliers is not None else len(current_points)
                    last_success_frame = frame_count + 1
                    last_success_points = current_points
                    last_success_inliers = current_inliers
                    did_update = True
            t_calib = time.perf_counter()

            rgb_h, rgb_w = rgb_frame.shape[:2]
            calibration_mask = apply_calibration_mask(
                thermal_mask,
                calibration,
                (rgb_w, rgb_h),
                mode=args.calibration_mode,
            )
            calibration_mask = dilate_mask(calibration_mask, args.warped_mask_dilate_px)

            thermal_masked = overlay_mask(
                temp_to_display(temp_c),
                thermal_mask,
                alpha=args.mask_alpha,
                draw_contour=True,
            )
            thermal_yolo = thermal_result.plot() if thermal_result is not None else thermal_yolo_frame
            rgb_yolo = rgb_result.plot() if rgb_result is not None else rgb_frame
            rgb_calibrated = overlay_mask(
                rgb_frame,
                calibration_mask,
                color=(0, 255, 255),
                alpha=args.mask_alpha,
            )

            frame_count += 1
            now = time.perf_counter()
            frame_times.append(now)
            fps = fps_from_times(frame_times, now)
            fps_text = f"{fps:.1f} fps"

            show_labeled("thermal_masked", thermal_masked, fps_text)
            show_labeled("thermal_yolo", thermal_yolo, f"{len(thermal_dets)} det {fps_text}")
            show_labeled("rgb_yolo", rgb_yolo, f"{len(rgb_dets)} det {fps_text}")
            show_labeled("rgb_calibration_masked", rgb_calibrated, fps_text)
            t_display = time.perf_counter()

            if frame_count % max(1, args.print_every) == 0:
                state = "ok" if calibration is not None else "waiting"
                if did_update:
                    update = "updated"
                elif did_try_update:
                    update = "try_failed"
                else:
                    update = "held"
                dt_ms = (now - t0) * 1000.0
                print(
                    f"frame={frame_count} fps={fps:.1f} total={dt_ms:.1f}ms "
                    f"thermal_det={len(thermal_dets)} rgb_det={len(rgb_dets)} "
                    f"offset_target={args.thermal_rgb_offset_ms:+.1f}ms "
                    f"paired_th_minus_rgb={rgb_pair_delta_ms:+.1f}ms "
                    f"calibration={state} this_frame={update} "
                    f"try_points={current_points} try_inliers={current_inliers} "
                    f"last_success_frame={last_success_frame or 'none'} "
                    f"last_success_points={last_success_points} "
                    f"last_success_inliers={last_success_inliers}"
                )
                if args.timing:
                    print(
                        f"timing read={(t_read - t0) * 1000.0:.1f}ms "
                        f"parse={(t_parse - t_read) * 1000.0:.1f}ms "
                        f"yolo={(t_yolo - t_parse) * 1000.0:.1f}ms "
                        f"mask={(t_mask - t_yolo) * 1000.0:.1f}ms "
                        f"calib={(t_calib - t_mask) * 1000.0:.1f}ms "
                        f"display={(t_display - t_calib) * 1000.0:.1f}ms "
                        f"thermal_yolo_mean={float(np.mean(thermal_yolo_frame)):.1f} "
                        f"rgb_mean={float(np.mean(rgb_frame)):.1f} "
                        f"temp={float(np.min(temp_c)):.1f}/{float(np.mean(temp_c)):.1f}/{float(np.max(temp_c)):.1f}C "
                        f"mask_hit={100.0 * float(np.count_nonzero(thermal_mask)) / thermal_mask.size:.1f}%"
                    )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        thermal_cap.release()
        rgb_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
