#!/home/orangepi/projects/rknn-env/bin/python
"""Create a movement timeline from pose_metadata.jsonl."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/orangepi-matplotlib-cache")
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


def latest_metadata() -> Path:
    files = list(ROOT.glob("session_*/pose_metadata.jsonl"))
    if not files:
        raise SystemExit("No session_*/pose_metadata.jsonl found")
    return max(files, key=lambda path: path.parent.name)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", nargs="?", type=Path, default=latest_metadata())
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--smooth-seconds", type=float, default=2.0)
    parser.add_argument(
        "--ground-truth", nargs="*", type=parse_timestamp, default=[], metavar="TIME",
        help="movement times as seconds, MM:SS, or HH:MM:SS",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    valid = np.isfinite(values).astype(float)
    filled = np.nan_to_num(values, nan=0.0)
    kernel = np.ones(window)
    totals = np.convolve(filled, kernel, mode="same")
    counts = np.convolve(valid, kernel, mode="same")
    return np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)


def main():
    args = arguments()
    times, motions = [], []
    previous_points = previous_time = None

    with args.metadata.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            elapsed = float(record["elapsed_seconds"])
            people = record.get("people") or []
            if not people:
                times.append(elapsed)
                motions.append(np.nan)
                previous_points = previous_time = None
                continue

            person = max(people, key=lambda p: float(p.get("person_confidence", 0)))
            raw = np.asarray(person.get("keypoints_xy_confidence", []), dtype=float)
            points = raw if raw.ndim == 2 and raw.shape[1] >= 3 else np.empty((0, 3))
            motion = np.nan
            if previous_points is not None and len(points) == len(previous_points):
                valid = (points[:, 2] >= args.confidence) & (previous_points[:, 2] >= args.confidence)
                dt = elapsed - previous_time
                if valid.any() and 0 < dt < 2.0:
                    displacement = np.linalg.norm(points[valid, :2] - previous_points[valid, :2], axis=1)
                    bbox = np.asarray(person.get("bbox_xyxy", []), dtype=float)
                    diagonal = np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]) if len(bbox) == 4 else 1.0
                    motion = 100.0 * float(np.median(displacement)) / max(diagonal, 1.0)
            times.append(elapsed)
            motions.append(motion)
            previous_points, previous_time = points, elapsed

    times = np.asarray(times)
    motions = np.asarray(motions)
    if len(times) < 2:
        raise SystemExit("Not enough metadata records to plot")
    sample_period = np.nanmedian(np.diff(times))
    window = max(1, round(args.smooth_seconds / sample_period))
    smooth = moving_average(motions, window)
    output = args.output or args.metadata.with_name("movement_keypoints.png")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(times / 60, smooth, color="#08519c", linewidth=1.5)
    ax.set(title="Movement Timeline — Pose Keypoints", xlabel="Elapsed time (min:sec)", ylabel="Movement (% of person-box diagonal)")
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
