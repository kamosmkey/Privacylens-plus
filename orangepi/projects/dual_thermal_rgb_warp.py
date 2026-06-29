#!/usr/bin/env python3
import argparse
import time
from collections import deque

import cv2
import numpy as np

from pose_common import load_rknn, open_rgb_camera, run_pose, select_primary_person
from thermal_common import (
    open_thermal_camera,
    parse_temperature,
    temp_to_bgr,
    temp_to_display,
    make_full_temperature_mask,
    overlay_mask,
)
from warp_common import estimate_transform, smooth_transform, warp_mask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Warp thermal temperature mask into an RGB camera view using YOLOv8-pose keypoints."
    )
    parser.add_argument("--model", required=True, help="Path to yolov8n-pose.rknn")
    parser.add_argument("--thermal-dev", default="/dev/video2", help="Thermal camera device")
    parser.add_argument("--rgb-dev", default="/dev/video0", help="RGB camera device")
    parser.add_argument("--rgb-width", type=int, default=640)
    parser.add_argument("--rgb-height", type=int, default=480)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25, help="Person detection threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--kpt-conf", type=float, default=0.4, help="Keypoint confidence used for warp")
    parser.add_argument("--mask-min-temp", type=float, default=24.0)
    parser.add_argument("--mask-max-temp", type=float, default=42.0)
    parser.add_argument("--mask-percentile", type=float, default=None)
    parser.add_argument("--mask-alpha", type=float, default=0.45)
    parser.add_argument("--warp-mode", choices=("affine", "homography"), default="affine")
    parser.add_argument("--update-every", type=int, default=30, help="Update transform every N frames")
    parser.add_argument("--smooth-alpha", type=float, default=0.25, help="Transform smoothing factor")
    parser.add_argument("--ransac-thresh", type=float, default=6.0)
    parser.add_argument("--print-every", type=int, default=30)
    return parser.parse_args()


def fps_from_times(times, now):
    while times and now - times[0] > 3.0:
        times.popleft()
    if len(times) < 2:
        return 0.0
    return (len(times) - 1) / max(1e-6, times[-1] - times[0])


def main():
    args = parse_args()
    update_every = max(1, args.update_every)
    smooth_alpha = min(1.0, max(0.0, args.smooth_alpha))

    rknn = load_rknn(args.model)
    thermal_cap = open_thermal_camera(args.thermal_dev)
    rgb_cap = open_rgb_camera(args.rgb_dev, args.rgb_width, args.rgb_height)

    transform = None
    frame_count = 0
    frame_times = deque()
    last_update_frame = None
    last_update_points = []
    last_update_inliers = 0

    print(f"Model: {args.model}")
    print(f"Thermal camera: {args.thermal_dev}")
    print(f"RGB camera: {args.rgb_dev} {args.rgb_width}x{args.rgb_height}")
    print(f"Warp mode: {args.warp_mode}")
    print("Windows: thermal_masked, rgb_raw, rgb_overlay")
    print("Press q in any OpenCV window to quit.")

    try:
        while True:
            t0 = time.perf_counter()
            ok_t, thermal_frame = thermal_cap.read()
            ok_r, rgb_frame = rgb_cap.read()
            if not ok_t or not ok_r:
                print(f"read failed: thermal={ok_t} rgb={ok_r}")
                continue

            temp_c = parse_temperature(thermal_frame)
            thermal_model_bgr = temp_to_bgr(temp_c)
            thermal_dets = run_pose(rknn, thermal_model_bgr, args.img_size, args.conf, args.iou)

            rgb_dets = run_pose(rknn, rgb_frame, args.img_size, args.conf, args.iou)

            thermal_mask = make_full_temperature_mask(
                temp_c,
                thermal_dets,
                args.mask_min_temp,
                args.mask_max_temp,
                args.mask_percentile,
            )

            thermal_primary = select_primary_person(thermal_dets)
            rgb_primary = select_primary_person(rgb_dets)
            did_try_update = False
            did_update = False
            current_points = []
            current_inliers = 0

            if (
                frame_count % update_every == 0 and
                thermal_primary is not None and
                rgb_primary is not None
            ):
                did_try_update = True
                new_transform, inliers, used_points = estimate_transform(
                    thermal_primary[2],
                    rgb_primary[2],
                    mode=args.warp_mode,
                    min_conf=args.kpt_conf,
                    ransac_thresh=args.ransac_thresh,
                )
                current_points = used_points
                if new_transform is not None:
                    transform = smooth_transform(transform, new_transform, smooth_alpha)
                    current_inliers = int(np.count_nonzero(inliers)) if inliers is not None else len(used_points)
                    last_update_frame = frame_count + 1
                    last_update_points = used_points
                    last_update_inliers = current_inliers
                    did_update = True

            rgb_h, rgb_w = rgb_frame.shape[:2]
            rgb_mask = warp_mask(thermal_mask, transform, (rgb_w, rgb_h), mode=args.warp_mode)

            thermal_display = overlay_mask(
                temp_to_display(temp_c),
                thermal_mask,
                alpha=args.mask_alpha,
                draw_contour=True,
            )
            rgb_overlay = overlay_mask(
                rgb_frame,
                rgb_mask,
                color=(0, 255, 255),
                alpha=args.mask_alpha,
            )

            cv2.imshow("thermal_masked", thermal_display)
            cv2.imshow("rgb_raw", rgb_frame)
            cv2.imshow("rgb_overlay", rgb_overlay)

            frame_count += 1
            now = time.perf_counter()
            frame_times.append(now)

            if frame_count % max(1, args.print_every) == 0:
                fps = fps_from_times(frame_times, now)
                state = "ok" if transform is not None else "waiting"
                if did_update:
                    update = "updated"
                elif did_try_update:
                    update = "try_failed"
                else:
                    update = "held"
                last_success = str(last_update_frame) if last_update_frame is not None else "none"
                dt_ms = (now - t0) * 1000.0
                print(
                    f"frame={frame_count} fps={fps:.1f} total={dt_ms:.1f}ms "
                    f"thermal_det={len(thermal_dets)} rgb_det={len(rgb_dets)} "
                    f"transform={state} this_frame={update} "
                    f"try_points={current_points} try_inliers={current_inliers} "
                    f"last_success_frame={last_success} "
                    f"last_success_points={last_update_points} "
                    f"last_success_inliers={last_update_inliers}"
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        thermal_cap.release()
        rgb_cap.release()
        cv2.destroyAllWindows()
        rknn.release()


if __name__ == "__main__":
    main()
