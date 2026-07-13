#!/usr/bin/env python3
import argparse
import time

import cv2
import numpy as np

from thermal_mask_utils import (
    MASK_COLORS,
    apply_mask_morphology,
    apply_mask_x_stretch,
    apply_translation_to_mask,
    compensation_shift,
    load_homography,
    make_temperature_mask,
    mask_bbox_width,
    mask_bbox_x_center,
    open_rgb_camera,
    open_thermal_camera,
    overlay_mask,
    parse_temperature,
    put_text,
    put_text_top_right,
    size,
    temp_to_display,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rgb-dev", default="/dev/video0")
    p.add_argument("--rgb-size", default="640x480")
    p.add_argument("--thermal-dev", default="/dev/video2")
    p.add_argument("--thermal-size", default="256x384")
    p.add_argument("--min-temp", type=float, default=24.0)
    p.add_argument("--max-temp", type=float, default=42.0)
    p.add_argument("--alpha", type=float, default=0.55)
    p.add_argument("--mask-color", choices=MASK_COLORS.keys(), default=None)
    p.add_argument(
        "--homography",
        default=None,
        help="optional .npz from fixed_calibration containing H_thermal_to_rgb",
    )
    p.add_argument(
        "--center-comp-x",
        type=float,
        default=0.0,
        help="x translation gain for mask-center compensation; 0 disables it",
    )
    p.add_argument(
        "--center-comp-y",
        type=float,
        default=0.0,
        help="y translation gain for mask-center compensation; 0 disables it",
    )
    p.add_argument(
        "--center-comp-space",
        choices=("rgb", "thermal"),
        default="rgb",
        help="compute center compensation from warped RGB mask or original thermal mask",
    )
    p.add_argument(
        "--center-comp-side",
        choices=("both", "left", "right"),
        default="both",
        help="apply center compensation only when the mask center is on this side",
    )
    p.add_argument(
        "--rgb-mask-dilate",
        type=int,
        default=0,
        help="dilate final warped RGB mask with this kernel size; 0 disables it",
    )
    p.add_argument(
        "--rgb-mask-erode",
        type=int,
        default=0,
        help="erode final warped RGB mask with this kernel size; 0 disables it",
    )
    p.add_argument(
        "--rgb-mask-distance-alpha",
        "--rgb-mask-x-stretch",
        dest="rgb_mask_distance_alpha",
        type=float,
        default=0.0,
        help="x-stretch term: scale = 1 + alpha*d",
    )
    p.add_argument(
        "--rgb-mask-distance-side",
        choices=("both", "left", "right"),
        default="both",
        help="deprecated; stretch is hard-coded to left-trigger/right-expand",
    )
    p.add_argument("--print-every", type=int, default=30)
    args = p.parse_args()
    if args.rgb_mask_dilate < 0 or args.rgb_mask_erode < 0:
        raise RuntimeError("--rgb-mask-dilate and --rgb-mask-erode must be >= 0")

    cap = open_thermal_camera(args.thermal_dev, size(args.thermal_size))
    rgb_cap = None
    H_thermal_to_rgb = None
    if args.homography is not None:
        if args.mask_color is None:
            raise RuntimeError("--homography requires --mask-color so a thermal mask is produced")
        H_thermal_to_rgb = load_homography(args.homography)
        rgb_cap = open_rgb_camera(args.rgb_dev, size(args.rgb_size))
        print(f"loaded homography: {args.homography}")
        print(f"window: rgb_thermal_masked. warped thermal mask on RGB. press q to quit.")

    if args.mask_color is None:
        print("window: thermal_processed. no mask overlay. press q to quit.")
    else:
        print(f"thermal mask: {args.min_temp}-{args.max_temp}C")
        print(f"window: thermal_masked. {args.mask_color} = masked pixels. press q to quit.")

    try:
        frame_id = 0
        warp_fps = 0.0
        last_warp_ts = None
        while True:
            ok, frame = cap.read()
            if not ok:
                print("thermal read failed")
                continue

            temp_c = parse_temperature(frame)
            display = temp_to_display(temp_c)
            t_min = float(np.min(temp_c))
            t_mean = float(np.mean(temp_c))
            t_max = float(np.max(temp_c))

            if args.mask_color is None:
                label = f"processed temp {t_min:.1f}/{t_mean:.1f}/{t_max:.1f}C"
                cv2.imshow("thermal_processed", put_text(display, label))
            else:
                mask = make_temperature_mask(temp_c, args.min_temp, args.max_temp)
                masked = overlay_mask(display, mask, MASK_COLORS[args.mask_color], args.alpha)
                hit_pct = 100.0 * float(np.count_nonzero(mask)) / mask.size
                label = (
                    f"{args.mask_color} mask {args.min_temp:.1f}-{args.max_temp:.1f}C "
                    f"hit {hit_pct:.1f}% temp {t_min:.1f}/{t_mean:.1f}/{t_max:.1f}C"
                )
                cv2.imshow("thermal_masked", put_text(masked, label))

                if rgb_cap is not None:
                    ok_rgb, rgb_frame = rgb_cap.read()
                    if not ok_rgb:
                        print("rgb read failed")
                    else:
                        rgb_h, rgb_w = rgb_frame.shape[:2]
                        rgb_mask = cv2.warpPerspective(
                            mask,
                            H_thermal_to_rgb,
                            (rgb_w, rgb_h),
                            flags=cv2.INTER_NEAREST,
                        )
                        if args.center_comp_space == "thermal":
                            shift_x, shift_y = compensation_shift(
                                mask,
                                mask.shape[1],
                                mask.shape[0],
                                args.center_comp_x,
                                args.center_comp_y,
                                args.center_comp_side,
                            )
                        else:
                            shift_x, shift_y = compensation_shift(
                                rgb_mask,
                                rgb_w,
                                rgb_h,
                                args.center_comp_x,
                                args.center_comp_y,
                                args.center_comp_side,
                            )
                        rgb_mask = apply_translation_to_mask(rgb_mask, shift_x, shift_y)
                        rgb_mask_width_before_stretch = mask_bbox_width(rgb_mask)
                        rgb_mask_cx_before_stretch = mask_bbox_x_center(rgb_mask)
                        if rgb_mask_cx_before_stretch is None:
                            rgb_mask_dx_before_stretch = 0.0
                        else:
                            rgb_mask_dx_before_stretch = (
                                rgb_mask_cx_before_stretch - rgb_w / 2.0
                            ) / (rgb_w / 2.0)
                        rgb_mask = apply_mask_x_stretch(
                            rgb_mask,
                            args.rgb_mask_distance_alpha,
                            args.rgb_mask_distance_side,
                        )
                        rgb_mask_width_after_stretch = mask_bbox_width(rgb_mask)
                        rgb_mask = apply_mask_morphology(
                            rgb_mask,
                            args.rgb_mask_dilate,
                            args.rgb_mask_erode,
                        )
                        rgb_masked = overlay_mask(
                            rgb_frame,
                            rgb_mask,
                            MASK_COLORS[args.mask_color],
                            args.alpha,
                        )
                        rgb_hit_pct = 100.0 * float(np.count_nonzero(rgb_mask)) / rgb_mask.size
                        rgb_label = (
                            f"warped thermal {args.mask_color} mask "
                            f"{args.min_temp:.1f}-{args.max_temp:.1f}C hit {rgb_hit_pct:.1f}% "
                            f"w {rgb_mask_width_before_stretch}->{rgb_mask_width_after_stretch}px "
                            f"dx {rgb_mask_dx_before_stretch:+.2f} "
                            f"shift {shift_x:.1f},{shift_y:.1f}px "
                            f"xstretch {args.rgb_mask_distance_alpha:.3f}/left-trigger/right-expand "
                            f"morph d{args.rgb_mask_dilate}/e{args.rgb_mask_erode}"
                        )
                        now = time.perf_counter()
                        if last_warp_ts is not None:
                            dt = now - last_warp_ts
                            if dt > 0:
                                instant_fps = 1.0 / dt
                                if warp_fps <= 0:
                                    warp_fps = instant_fps
                                else:
                                    warp_fps = 0.9 * warp_fps + 0.1 * instant_fps
                        last_warp_ts = now
                        rgb_masked = put_text(rgb_masked, rgb_label)
                        rgb_masked = put_text_top_right(rgb_masked, f"warp {warp_fps:.1f} fps")
                        cv2.imshow("rgb_thermal_masked", rgb_masked)

            frame_id += 1
            if frame_id % args.print_every == 0:
                print(label)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if rgb_cap is not None:
            rgb_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
