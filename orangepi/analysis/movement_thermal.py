#!/home/orangepi/projects/rknn-env/bin/python
"""Create a movement timeline from a thermal video using frame differences."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/orangepi-matplotlib-cache")
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator


ROOT = Path(__file__).resolve().parent


def format_elapsed_time(minutes: float, _position: int) -> str:
    total_seconds = max(0, round(minutes * 60))
    whole_minutes, seconds = divmod(total_seconds, 60)
    return f"{whole_minutes}:{seconds:02d}"


def parse_timestamp(value: str) -> float:
    """Parse ground-truth time as seconds or MM:SS / HH:MM:SS."""
    try:
        parts = [float(part) for part in value.split(":")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid timestamp: {value}") from exc
    if not 1 <= len(parts) <= 3 or any(part < 0 for part in parts):
        raise argparse.ArgumentTypeError(f"invalid timestamp: {value}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60.0 + part
    return seconds


def add_ground_truth(ax, timestamps: list[float]) -> None:
    for index, seconds in enumerate(timestamps, 1):
        minutes = seconds / 60.0
        ax.axvline(minutes, color="#238b45", linestyle="--", linewidth=1.3,
                   alpha=0.9, label="Ground truth" if index == 1 else None)
        ax.annotate(f"GT {index}", xy=(minutes, 1), xycoords=("data", "axes fraction"),
                    xytext=(3, -4), textcoords="offset points", rotation=90,
                    va="top", ha="left", color="#006d2c", fontsize=8)


def latest_video() -> Path:
    files = list(ROOT.glob("session_*/raw_thermal/*.mp4"))
    if not files:
        raise SystemExit("No session_*/raw_thermal/*.mp4 found")
    return max(files, key=lambda path: path.parent.parent.name)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path, default=latest_video())
    parser.add_argument("--sample-fps", type=float, default=2.0,
                        help="frames analyzed per second (default: 2)")
    parser.add_argument("--smooth-seconds", type=float, default=2.0)
    parser.add_argument(
        "--ground-truth", nargs="*", type=parse_timestamp, default=[], metavar="TIME",
        help="movement times as seconds, MM:SS, or HH:MM:SS",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = arguments()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
    stride = max(1, round(fps / args.sample_fps))
    effective_fps = fps / stride
    times, motions = [], []
    previous = None
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % stride:
            frame_index += 1
            continue
        # The stored thermal video is an Inferno colour preview, not temperature data.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
        if previous is not None:
            signed = gray - previous
            # Especially important because every thermal frame was percentile-normalized.
            signed -= np.median(signed)
            motions.append(100.0 * float(np.mean(np.abs(signed))) / 255.0)
            times.append(frame_index / fps)
        previous = gray
        frame_index += 1
    capture.release()
    if not motions:
        raise SystemExit("Not enough video frames to plot")

    times, motions = np.asarray(times), np.asarray(motions)
    window = max(1, round(args.smooth_seconds * effective_fps))
    smooth = np.convolve(motions, np.ones(window) / window, mode="same")
    output = args.output or args.video.parent.parent / "movement_thermal.png"
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(times / 60, smooth, color="#6a51a3", linewidth=1.5)
    ax.set(title="Movement Timeline — Thermal Frame Difference", xlabel="Elapsed time (min:sec)", ylabel="Mean absolute frame difference (% intensity)")
    ax.xaxis.set_major_locator(MultipleLocator(15 / 60))
    ax.xaxis.set_major_formatter(FuncFormatter(format_elapsed_time))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    add_ground_truth(ax, args.ground_truth)
    if args.ground_truth:
        ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
