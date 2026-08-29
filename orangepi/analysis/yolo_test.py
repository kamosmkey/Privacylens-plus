#!/home/orangepi/projects/rknn-env/bin/python
"""Analyze overnight YOLO-Pose metadata without computing movement metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/orangepi-matplotlib-cache")

try:
    import matplotlib
except ImportError as exc:
    raise SystemExit(
        "Matplotlib is required. Run this script with: "
        "/home/orangepi/projects/rknn-env/bin/python yolo_test.py"
    ) from exc

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


DEFAULT_SESSION = Path("/mnt/ssd/videos/session_20260825_045129_059393")
DEFAULT_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
CONFIDENCE_BINS = 1000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "session",
        nargs="?",
        type=Path,
        default=DEFAULT_SESSION,
        help=f"recording session directory (default: {DEFAULT_SESSION})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="minimum confidence for a valid keypoint (default: 0.5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: SESSION/yolo_analysis)",
    )
    parser.add_argument(
        "--manual",
        nargs="+",
        default=[],
        metavar="TIME",
        help=(
            "draw manual vertical markers on the valid-keypoints timeline; "
            "accepts minutes or MM:SS times (example: --manual 0:01 0:26 1:48)"
        ),
    )
    return parser.parse_args()


def parse_manual_annotations(values):
    annotations = []
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
            raise ValueError(
                f"Manual time must be non-negative: {value!r}"
            )
        annotations.append(minute)
    return annotations


def histogram_quantile(histogram, quantile):
    total = int(histogram.sum())
    if total == 0:
        return math.nan
    target = quantile * max(0, total - 1)
    cumulative = np.cumsum(histogram)
    index = int(np.searchsorted(cumulative, target + 1, side="left"))
    return min(index, CONFIDENCE_BINS) / CONFIDENCE_BINS


def safe_rate(numerator, denominator):
    return numerator / denominator if denominator else math.nan


def load_keypoint_names(session):
    info_path = session / "session.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        names = info.get("keypoint_order")
        if isinstance(names, list) and len(names) == 17:
            return tuple(str(name) for name in names)
    return DEFAULT_KEYPOINT_NAMES


def analyze(metadata_path, keypoint_names, threshold):
    keypoint_count = len(keypoint_names)
    inference_frames = detected_frames = total_records = 0
    person_confidences = []
    inference_times = []
    frame_elapsed = []
    frame_person_detected = []
    frame_valid_keypoints = []
    person_count_histogram = defaultdict(int)

    kp_histograms = np.zeros(
        (keypoint_count, CONFIDENCE_BINS + 1), dtype=np.int64
    )
    kp_sums = np.zeros(keypoint_count, dtype=np.float64)
    kp_observations = np.zeros(keypoint_count, dtype=np.int64)
    kp_valid = np.zeros(keypoint_count, dtype=np.int64)
    valid_keypoint_histogram = np.zeros(keypoint_count + 1, dtype=np.int64)

    hourly_stats = defaultdict(
        lambda: {
            "inference": 0,
            "detected": 0,
            "person_conf_sum": 0.0,
            "person_conf_count": 0,
            "valid_kp_sum": 0,
            "kp_sum": np.zeros(keypoint_count, dtype=np.float64),
            "kp_count": np.zeros(keypoint_count, dtype=np.int64),
        }
    )

    first_elapsed = last_elapsed = None
    current_miss_start = None
    longest_miss = (0.0, None, None)
    previous_inference_elapsed = None

    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {metadata_path}:{line_number}: {exc}"
                ) from exc

            total_records += 1
            elapsed = float(record.get("elapsed_seconds", 0.0))
            first_elapsed = elapsed if first_elapsed is None else first_elapsed
            last_elapsed = elapsed
            if not record.get("inference_performed", True):
                continue

            inference_frames += 1
            frame_elapsed.append(elapsed)
            hour = max(0, int(elapsed // 3600))
            bucket = hourly_stats[hour]
            bucket["inference"] += 1
            infer_ms = record.get("inference_ms")
            if infer_ms is not None:
                inference_times.append(float(infer_ms))

            people = record.get("people") or []
            person_count_histogram[len(people)] += 1
            valid_keypoint_count = 0
            if people:
                detected_frames += 1
                bucket["detected"] += 1
                person = max(
                    people, key=lambda item: float(item.get("person_confidence", 0.0))
                )
                person_confidence = float(person.get("person_confidence", 0.0))
                person_confidences.append(person_confidence)
                bucket["person_conf_sum"] += person_confidence
                bucket["person_conf_count"] += 1

                points = person.get("keypoints_xy_confidence") or []
                confidences = np.asarray(
                    [
                        float(points[index][2])
                        if index < len(points) and len(points[index]) >= 3
                        else math.nan
                        for index in range(keypoint_count)
                    ],
                    dtype=np.float64,
                )
                finite = np.isfinite(confidences)
                clipped = np.clip(confidences[finite], 0.0, 1.0)
                finite_indices = np.flatnonzero(finite)
                bin_indices = np.rint(clipped * CONFIDENCE_BINS).astype(int)
                np.add.at(kp_histograms, (finite_indices, bin_indices), 1)
                kp_sums[finite] += confidences[finite]
                kp_observations[finite] += 1
                valid = finite & (confidences >= threshold)
                kp_valid[valid] += 1
                valid_keypoint_count = int(valid.sum())
                bucket["valid_kp_sum"] += valid_keypoint_count
                bucket["kp_sum"][finite] += confidences[finite]
                bucket["kp_count"][finite] += 1

                if current_miss_start is not None:
                    miss_end = previous_inference_elapsed or elapsed
                    duration = max(0.0, miss_end - current_miss_start)
                    if duration > longest_miss[0]:
                        longest_miss = (duration, current_miss_start, miss_end)
                    current_miss_start = None
            elif current_miss_start is None:
                current_miss_start = elapsed

            valid_keypoint_histogram[valid_keypoint_count] += 1
            frame_person_detected.append(bool(people))
            frame_valid_keypoints.append(valid_keypoint_count)
            previous_inference_elapsed = elapsed

    if current_miss_start is not None and previous_inference_elapsed is not None:
        duration = max(0.0, previous_inference_elapsed - current_miss_start)
        if duration > longest_miss[0]:
            longest_miss = (duration, current_miss_start, previous_inference_elapsed)

    return {
        "total_records": total_records,
        "inference_frames": inference_frames,
        "detected_frames": detected_frames,
        "first_elapsed": first_elapsed,
        "last_elapsed": last_elapsed,
        "person_confidences": np.asarray(person_confidences, dtype=np.float32),
        "inference_times": np.asarray(inference_times, dtype=np.float32),
        "frame_elapsed": np.asarray(frame_elapsed, dtype=np.float64),
        "frame_person_detected": np.asarray(frame_person_detected, dtype=np.uint8),
        "frame_valid_keypoints": np.asarray(frame_valid_keypoints, dtype=np.uint8),
        "person_count_histogram": dict(person_count_histogram),
        "kp_histograms": kp_histograms,
        "kp_sums": kp_sums,
        "kp_observations": kp_observations,
        "kp_valid": kp_valid,
        "valid_keypoint_histogram": valid_keypoint_histogram,
        "hourly_stats": dict(hourly_stats),
        "longest_miss": longest_miss,
    }


def keypoint_rows(results, names):
    rows = []
    inference_frames = results["inference_frames"]
    for index, name in enumerate(names):
        observations = int(results["kp_observations"][index])
        histogram = results["kp_histograms"][index]
        rows.append(
            {
                "keypoint": name,
                "observations_when_person_detected": observations,
                "mean_confidence_when_observed": safe_rate(
                    results["kp_sums"][index], observations
                ),
                "q25_confidence_when_observed": histogram_quantile(histogram, 0.25),
                "median_confidence_when_observed": histogram_quantile(histogram, 0.50),
                "q75_confidence_when_observed": histogram_quantile(histogram, 0.75),
                "valid_frames": int(results["kp_valid"][index]),
                "coverage_over_inference_frames": safe_rate(
                    results["kp_valid"][index], inference_frames
                ),
            }
        )
    return rows


def write_outputs(output_dir, results, names, threshold, manual_annotations):
    output_dir.mkdir(parents=True, exist_ok=True)
    kp_rows = keypoint_rows(results, names)

    duration = max(
        0.0,
        (results["last_elapsed"] or 0.0) - (results["first_elapsed"] or 0.0),
    )
    inference_ms = results["inference_times"]
    longest_miss_seconds = results["longest_miss"][0]
    detection_coverage = safe_rate(
        results["detected_frames"], results["inference_frames"]
    )
    mean_valid_keypoints = safe_rate(
        results["kp_valid"].sum(), results["inference_frames"]
    )
    effective_fps = safe_rate(results["inference_frames"], duration)

    print("YOLO-Pose analysis summary")
    print(f"  Recording duration: {duration / 3600:.3f} hours")
    print(f"  Inference frames: {results['inference_frames']:,}")
    print(f"  Effective inference rate: {effective_fps:.3f} frames/s")
    print(f"  Frames with a detected person: {results['detected_frames']:,}")
    print(f"  Person-detection coverage: {detection_coverage:.2%}")
    print(f"  Keypoint confidence threshold: {threshold:g}")
    print(
        f"  Mean valid keypoints per inference frame: "
        f"{mean_valid_keypoints:.3f} / {len(names)}"
    )
    print(
        f"  Longest continuous no-detection interval: "
        f"{longest_miss_seconds / 60:.2f} minutes"
    )
    if inference_ms.size:
        print(f"  Mean inference time: {float(np.mean(inference_ms)):.3f} ms")
        print(
            f"  95th-percentile inference time: "
            f"{float(np.percentile(inference_ms, 95)):.3f} ms"
        )
    print("  Note: coverage measures model output, not ground-truth accuracy.")

    plot_outputs(
        output_dir, results, names, kp_rows, threshold, manual_annotations
    )


def plot_outputs(
    output_dir, results, names, kp_rows, threshold, manual_annotations
):
    elapsed_minutes = results["frame_elapsed"] / 60.0
    detection = results["frame_person_detected"]
    valid_counts = results["frame_valid_keypoints"]

    # Aggregate the binary per-frame detections into one-second coverage rates.
    # Only inference frames contribute to the denominator.
    second_index = np.floor(results["frame_elapsed"]).astype(np.int64)
    first_second = int(second_index.min())
    relative_second = second_index - first_second
    second_frame_counts = np.bincount(relative_second)
    second_detected_counts = np.bincount(relative_second, weights=detection)
    second_coverage = np.divide(
        second_detected_counts,
        second_frame_counts,
        out=np.full(second_frame_counts.shape, np.nan, dtype=np.float64),
        where=second_frame_counts > 0,
    )
    second_minutes = (np.arange(len(second_coverage)) + first_second + 0.5) / 60.0

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(second_minutes, second_coverage * 100, linewidth=0.7, rasterized=True)
    ax.set(
        xlabel="Elapsed time (minutes)",
        ylabel="Detection coverage per second (%)",
        ylim=(-2, 102),
        title="Per-second person-detection coverage",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "person_detection_timeline.png", dpi=160)
    plt.close(fig)

    positions = np.arange(len(names))
    coverage = np.asarray([row["coverage_over_inference_frames"] for row in kp_rows])
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(positions, coverage * 100)
    ax.set_xticks(positions, names, rotation=55, ha="right")
    ax.set(
        ylabel="Coverage over all inference frames (%)",
        ylim=(0, 102),
        title=f"Keypoint coverage at confidence >= {threshold:g}",
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "keypoint_coverage.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(elapsed_minutes, valid_counts, linewidth=0.35, rasterized=True)
    for minute in manual_annotations:
        ax.axvline(
            minute, color="tab:red", linewidth=1.2, linestyle="--", alpha=0.9
        )
    ax.set(
        xlabel="Elapsed time (minutes)",
        ylabel="Valid keypoints in frame",
        ylim=(-0.2, len(names) + 0.2),
        title=f"Valid keypoints per inference frame (confidence >= {threshold:g})",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "valid_keypoints_timeline.png", dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise SystemExit("--threshold must be between 0 and 1")
    session = args.session.expanduser().resolve()
    metadata_path = session / "pose_metadata.jsonl"
    if not metadata_path.is_file():
        raise SystemExit(f"Metadata file not found: {metadata_path}")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else session / "yolo_analysis"
    )
    names = load_keypoint_names(session)
    try:
        manual_annotations = parse_manual_annotations(args.manual)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    results = analyze(metadata_path, names, args.threshold)
    if not results["inference_frames"]:
        raise SystemExit("No inference frames found in metadata")
    write_outputs(
        output_dir, results, names, args.threshold, manual_annotations
    )
    print(f"Analysis written to: {output_dir}")


if __name__ == "__main__":
    main()
