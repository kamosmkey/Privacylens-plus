#!/home/orangepi/projects/rknn-env/bin/python
"""Render dense optical flow on the latest recorded raw RGB video.

The output overlays a motion heatmap and direction arrows on the video.  By
default, the latest session_*/raw_rgb/*.mp4 is used and the result is written
under that session's optical_flow/ directory.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/orangepi-matplotlib-cache")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parent


def latest_rgb_video() -> Path:
    videos = list(ROOT.glob("session_*/raw_rgb/*.mp4"))
    if not videos:
        raise SystemExit("No session_*/raw_rgb/*.mp4 found")
    return max(videos, key=lambda path: path.stat().st_mtime)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path, default=latest_rgb_video())
    parser.add_argument(
        "--sample-fps", type=float, default=8.0,
        help="optical-flow/output frames per second (default: 8)",
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="output and processing width (default: 640)",
    )
    parser.add_argument(
        "--motion-threshold", type=float, default=0.8,
        help="minimum displacement shown as motion, in output pixels (default: 0.8)",
    )
    parser.add_argument(
        "--heat-max", type=float, default=6.0,
        help="displacement mapped to the hottest colour, in output pixels (default: 6)",
    )
    parser.add_argument(
        "--arrow-step", type=int, default=32,
        help="spacing between direction arrows in output pixels (default: 32)",
    )
    parser.add_argument(
        "--no-camera-compensation", action="store_true",
        help="do not subtract median whole-frame motion",
    )
    parser.add_argument(
        "--no-text", action="store_true",
        help="do not draw the information panel in the top-left corner",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=0.0,
        help="process only the first N seconds; 0 means the whole video",
    )
    parser.add_argument(
        "--manual", nargs="+", default=[], metavar="TIME",
        help=(
            "ground-truth marker times in minutes, MM:SS, or HH:MM:SS "
            "(example: --manual 0:01 0:26 1:48)"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--plot-output", type=Path,
        help="time-series plot path (default: next to the output video)",
    )
    return parser.parse_args()


def parse_manual_times(values):
    elapsed_minutes = []
    for value in values:
        parts = value.split(":")
        try:
            if len(parts) == 1:
                minute = float(parts[0])
            elif len(parts) == 2:
                minutes, seconds = (float(part) for part in parts)
                if not 0 <= seconds < 60:
                    raise ValueError
                minute = minutes + seconds / 60.0
            elif len(parts) == 3:
                hours, minutes, seconds = (float(part) for part in parts)
                if not 0 <= minutes < 60 or not 0 <= seconds < 60:
                    raise ValueError
                minute = hours * 60.0 + minutes + seconds / 60.0
            else:
                raise ValueError
        except ValueError as exc:
            raise ValueError(
                f"Invalid manual time {value!r}; use minutes, MM:SS, or HH:MM:SS"
            ) from exc
        if not math.isfinite(minute) or minute < 0:
            raise ValueError(f"Manual time must be non-negative: {value!r}")
        elapsed_minutes.append(minute)
    return elapsed_minutes


def save_time_series(path, times, mean_motion, manual_times):
    fig, axis = plt.subplots(figsize=(12, 4))
    axis.plot(times, mean_motion, color="tab:blue", linewidth=0.8,
              rasterized=True)
    axis.set(
        xlabel="Elapsed time (minutes)",
        ylabel="Mean flow (pixels)",
        title="Mean optical flow over time",
    )
    for index, minute in enumerate(manual_times):
        axis.axvline(
            minute, color="tab:red", linestyle="--", linewidth=1.1,
            alpha=0.85,
            label="Ground truth" if index == 0 else None,
        )
    axis.grid(alpha=0.25)
    if manual_times:
        axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def resize(frame: np.ndarray, width: int) -> np.ndarray:
    height = max(2, round(frame.shape[0] * width / frame.shape[1]))
    # H.264/mp4 players are happiest with even dimensions.
    height += height % 2
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def prepared_gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def draw_arrow_field(
    image: np.ndarray,
    flow: np.ndarray,
    magnitude: np.ndarray,
    threshold: float,
    step: int,
) -> None:
    height, width = magnitude.shape
    radius = max(2, step // 5)
    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            y1, y2 = max(0, y - radius), min(height, y + radius + 1)
            x1, x2 = max(0, x - radius), min(width, x + radius + 1)
            local_mag = magnitude[y1:y2, x1:x2]
            if float(np.median(local_mag)) < threshold:
                continue
            dx = float(np.median(flow[y1:y2, x1:x2, 0]))
            dy = float(np.median(flow[y1:y2, x1:x2, 1]))
            length = float(np.hypot(dx, dy))
            if length < threshold:
                continue
            # Magnify short vectors, but cap long ones so the field stays legible.
            display_length = min(22.0, max(8.0, length * 3.0))
            end = (
                round(x + dx * display_length / length),
                round(y + dy * display_length / length),
            )
            cv2.arrowedLine(image, (x, y), end, (255, 255, 255), 2,
                            cv2.LINE_AA, tipLength=0.35)
            cv2.circle(image, (x, y), 2, (20, 20, 20), -1, cv2.LINE_AA)


def annotate(image, elapsed, mean_motion, moving_ratio, compensated):
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (430, 91), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, image, 0.42, 0, image)
    minutes, seconds = divmod(elapsed, 60.0)
    lines = (
        f"Dense optical flow   {int(minutes):02d}:{seconds:05.2f}",
        f"Mean motion: {mean_motion:.2f} px   Moving area: {moving_ratio * 100:.1f}%",
        "Heat = speed   Arrows = direction" + ("   Camera motion removed" if compensated else ""),
    )
    for index, line in enumerate(lines):
        cv2.putText(image, line, (20, 34 + index * 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)


def main():
    args = arguments()
    try:
        manual_times = parse_manual_times(args.manual)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    video = args.video.expanduser().resolve()
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")
    if args.sample_fps <= 0 or args.width <= 0 or args.motion_threshold < 0:
        raise SystemExit("sample-fps and width must be positive; threshold must be nonnegative")
    if args.heat_max <= 0 or args.arrow_step <= 0 or args.max_seconds < 0:
        raise SystemExit("heat-max and arrow-step must be positive; max-seconds must be nonnegative")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 1.0
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, round(source_fps / args.sample_fps))
    output_fps = source_fps / stride

    ok, first = capture.read()
    if not ok:
        capture.release()
        raise SystemExit("Video contains no decodable frames")
    display = resize(first, args.width)
    height, width = display.shape[:2]
    output = args.output or video.parent.parent / "optical_flow" / "optical_flow_overlay.mp4"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_output = args.plot_output or output.with_name("optical_flow_time_series.png")
    plot_output = plot_output.expanduser().resolve()
    plot_output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"Cannot create output video: {output}")

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseSpatialPropagation(True)
    previous = prepared_gray(display)
    frame_index = 1
    rendered = 0
    time_series_minutes = []
    mean_motion_series = []
    moving_ratio_series = []
    print(f"Input: {video}")
    print(f"Source: {source_frames} frames @ {source_fps:g} fps")
    print(f"Output: {output_fps:g} fps, {width}x{height}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            current_index = frame_index
            frame_index += 1
            elapsed = current_index / source_fps
            if args.max_seconds and elapsed > args.max_seconds:
                break
            if current_index % stride:
                continue

            display = resize(frame, args.width)
            gray = prepared_gray(display)
            flow = dis.calc(previous, gray, None)
            previous = gray
            if not args.no_camera_compensation:
                flow[..., 0] -= np.median(flow[..., 0])
                flow[..., 1] -= np.median(flow[..., 1])

            magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
            moving = magnitude >= args.motion_threshold
            mean_motion = float(np.mean(magnitude))
            moving_ratio = float(np.mean(moving))
            time_series_minutes.append(elapsed / 60.0)
            mean_motion_series.append(mean_motion)
            moving_ratio_series.append(moving_ratio)
            heat_values = np.clip(magnitude * 255.0 / args.heat_max, 0, 255).astype(np.uint8)
            heat = cv2.applyColorMap(heat_values, cv2.COLORMAP_TURBO)
            alpha = np.where(moving, np.clip(magnitude / args.heat_max, 0.22, 0.72), 0.0)
            alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 1.2)[..., None]
            visual = np.clip(
                display.astype(np.float32) * (1.0 - alpha)
                + heat.astype(np.float32) * alpha,
                0,
                255,
            ).astype(np.uint8)
            draw_arrow_field(
                visual, flow, magnitude, args.motion_threshold, args.arrow_step
            )
            if not args.no_text:
                annotate(
                    visual,
                    elapsed,
                    mean_motion,
                    moving_ratio,
                    not args.no_camera_compensation,
                )
            writer.write(visual)
            rendered += 1
            if rendered % 250 == 0:
                print(f"Rendered {rendered} frames ({elapsed / 60:.1f} min)")
    finally:
        capture.release()
        writer.release()

    if rendered == 0:
        output.unlink(missing_ok=True)
        raise SystemExit("Not enough sampled frames to calculate optical flow")
    save_time_series(
        plot_output,
        time_series_minutes,
        mean_motion_series,
        manual_times,
    )
    print(f"Rendered frames: {rendered}")
    print(f"Saved: {output}")
    print(f"Time-series plot: {plot_output}")


if __name__ == "__main__":
    main()
