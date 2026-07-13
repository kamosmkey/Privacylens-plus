#!/usr/bin/env python3
import argparse
import csv
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from thermal_common import open_thermal_camera, parse_temperature, temp_to_display


@dataclass
class Sample:
    idx: int
    read_start: float
    read_done: float
    ok: bool
    frame: object


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure RGB/thermal capture timing and estimate visual lag without YOLO/warp. "
            "Move a warm person/object left-right in view for the lag estimate."
        )
    )
    parser.add_argument("--thermal-dev", default="/dev/video0")
    parser.add_argument("--rgb-dev", default="/dev/video2")
    parser.add_argument("--rgb-width", type=int, default=640)
    parser.add_argument("--rgb-height", type=int, default=480)
    parser.add_argument("--rgb-fourcc", default="MJPG")
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds to run; 0 means until q")
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument("--csv", default=None, help="Optional path for timing/centroid rows")
    parser.add_argument(
        "--mode",
        choices=("sequential", "threaded"),
        default="sequential",
        help="sequential mimics the main script read loop; threaded keeps only latest frames",
    )
    parser.add_argument(
        "--read-order",
        choices=("thermal-first", "rgb-first"),
        default="thermal-first",
        help="Only used in sequential mode",
    )
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--thermal-min-temp", type=float, default=24.0)
    parser.add_argument("--thermal-max-temp", type=float, default=42.0)
    parser.add_argument(
        "--thermal-percentile",
        type=float,
        default=90.0,
        help="Use max(min_temp, percentile(temp)) as thermal mask lower bound",
    )
    parser.add_argument("--min-area", type=float, default=25.0, help="Minimum contour area for centroid")
    parser.add_argument("--lag-window", type=int, default=180, help="Samples used for lag correlation")
    parser.add_argument("--max-lag", type=int, default=45, help="Max lag in samples for correlation")
    return parser.parse_args()


def open_rgb_camera(dev, width, height, fourcc):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened() and dev.startswith("/dev/video"):
        cap = cv2.VideoCapture(int(dev.replace("/dev/video", "")))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RGB camera {dev}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class CameraReader:
    def __init__(self, name, cap):
        self.name = name
        self.cap = cap
        self.lock = threading.Lock()
        self.latest = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"{name}-reader", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopped.set()
        self.thread.join(timeout=2.0)

    def get_latest(self):
        with self.lock:
            return self.latest

    def _run(self):
        idx = 0
        while not self.stopped.is_set():
            read_start = time.perf_counter()
            ok, frame = self.cap.read()
            read_done = time.perf_counter()
            idx += 1
            with self.lock:
                self.latest = Sample(idx, read_start, read_done, ok, frame)


def read_sample(cap, idx):
    read_start = time.perf_counter()
    ok, frame = cap.read()
    read_done = time.perf_counter()
    return Sample(idx, read_start, read_done, ok, frame)


def largest_centroid(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    if not contours:
        return None, 0.0

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    m = cv2.moments(contour)
    if abs(m["m00"]) < 1e-6:
        return None, area
    return (float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])), area


def thermal_centroid(frame, args):
    t0 = time.perf_counter()
    temp_c = parse_temperature(frame)
    valid = np.isfinite(temp_c)
    lower = args.thermal_min_temp
    if args.thermal_percentile is not None and np.any(valid):
        lower = max(lower, float(np.percentile(temp_c[valid], args.thermal_percentile)))
    upper = max(args.thermal_max_temp, lower + 0.1)
    mask = np.zeros(temp_c.shape, dtype=np.uint8)
    mask[(temp_c >= lower) & (temp_c <= upper) & valid] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    centroid, area = largest_centroid(mask, args.min_area)
    display = temp_to_display(temp_c)
    t1 = time.perf_counter()
    return centroid, area, mask, display, (t1 - t0)


def rgb_motion_centroid(frame, previous_gray, args):
    t0 = time.perf_counter()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if previous_gray is None:
        return None, 0.0, np.zeros(gray.shape, dtype=np.uint8), gray, 0.0, (time.perf_counter() - t0)

    diff = cv2.absdiff(gray, previous_gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, mask = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    centroid, area = largest_centroid(mask, args.min_area)
    t1 = time.perf_counter()
    return centroid, area, mask, gray, float(np.mean(diff)), (t1 - t0)


def normalized_x(centroid, width):
    if centroid is None or width <= 0:
        return None
    return (centroid[0] / float(width)) - 0.5


def estimate_lag(records, max_lag):
    xs = [(r["rgb_x"], r["thermal_x"]) for r in records if r["rgb_x"] is not None and r["thermal_x"] is not None]
    if len(xs) < max(20, max_lag * 2):
        return None

    rgb = np.asarray([x[0] for x in xs], dtype=np.float32)
    thermal = np.asarray([x[1] for x in xs], dtype=np.float32)
    rgb -= np.mean(rgb)
    thermal -= np.mean(thermal)
    if np.std(rgb) < 1e-4 or np.std(thermal) < 1e-4:
        return None

    best = None
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            a = rgb[:-lag]
            b = thermal[lag:]
        elif lag < 0:
            a = rgb[-lag:]
            b = thermal[:lag]
        else:
            a = rgb
            b = thermal
        if len(a) < 10:
            continue
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-6:
            continue
        corr = float(np.dot(a, b) / denom)
        if best is None or corr > best["corr"]:
            best = {"lag_samples": lag, "corr": corr, "pairs": len(a)}
    return best


def fps_from_times(times):
    if len(times) < 2:
        return 0.0
    return (len(times) - 1) / max(1e-6, times[-1] - times[0])


def percentile_summary(rows, key, signed=False):
    values = [r[key] for r in rows if r.get(key) is not None]
    if not values:
        return f"{key}: n/a"
    arr = np.asarray(values, dtype=np.float32)
    sign = "+" if signed else ""
    return (
        f"{key}: "
        f"p50={np.percentile(arr, 50):{sign}.1f}ms "
        f"p90={np.percentile(arr, 90):{sign}.1f}ms "
        f"p99={np.percentile(arr, 99):{sign}.1f}ms "
        f"min/max={np.min(arr):{sign}.1f}/{np.max(arr):{sign}.1f}ms"
    )


def draw_centroid(image, centroid, color, label):
    out = image.copy()
    if centroid is not None:
        x, y = int(round(centroid[0])), int(round(centroid[1]))
        cv2.drawMarker(out, (x, y), color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        cv2.putText(out, label, (x + 8, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


def put_lines(image, lines):
    out = image.copy()
    y = 22
    for line in lines:
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        y += 22
    return out


def main():
    args = parse_args()
    thermal_cap = open_thermal_camera(args.thermal_dev)
    rgb_cap = open_rgb_camera(args.rgb_dev, args.rgb_width, args.rgb_height, args.rgb_fourcc)

    print(f"Thermal: {args.thermal_dev}")
    print(f"RGB: {args.rgb_dev} {args.rgb_width}x{args.rgb_height} {args.rgb_fourcc}")
    print(f"Mode: {args.mode} read_order={args.read_order}")
    print("Move a warm person/object left-right. Positive lag means thermal trails RGB.")
    if not args.no_display:
        print("Windows: lag_rgb_motion, lag_thermal_hot. Press q to quit.")

    rows = []
    recent = deque(maxlen=max(args.lag_window, args.max_lag * 3))
    frame_times = deque()
    prev_rgb_gray = None
    start = time.perf_counter()
    idx = 0
    last_loop_t = None
    last_threaded_pair = None
    thermal_reader = None
    rgb_reader = None

    csv_file = open(args.csv, "w", newline="") if args.csv else None
    writer = None
    if csv_file is not None:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "idx",
                "loop_t",
                "thermal_idx",
                "rgb_idx",
                "thermal_read_ms",
                "rgb_read_ms",
                "read_done_delta_ms",
                "thermal_age_ms",
                "rgb_age_ms",
                "loop_period_ms",
                "loop_total_ms",
                "thermal_parse_mask_ms",
                "rgb_motion_ms",
                "thermal_x",
                "thermal_y",
                "thermal_area",
                "rgb_x",
                "rgb_y",
                "rgb_area",
                "rgb_motion_mean",
                "lag_samples",
                "lag_ms",
                "lag_corr",
            ],
        )
        writer.writeheader()

    try:
        if args.mode == "threaded":
            thermal_reader = CameraReader("thermal", thermal_cap)
            rgb_reader = CameraReader("rgb", rgb_cap)
            thermal_reader.start()
            rgb_reader.start()
            time.sleep(0.2)

        while True:
            loop_t = time.perf_counter()
            idx += 1

            if args.mode == "threaded":
                thermal_sample = thermal_reader.get_latest()
                rgb_sample = rgb_reader.get_latest()
                if thermal_sample is None or rgb_sample is None:
                    time.sleep(0.005)
                    continue
                pair = (thermal_sample.idx, rgb_sample.idx)
                if pair == last_threaded_pair:
                    time.sleep(0.001)
                    continue
                if last_threaded_pair is not None and (
                    pair[0] == last_threaded_pair[0] or pair[1] == last_threaded_pair[1]
                ):
                    time.sleep(0.001)
                    continue
                last_threaded_pair = pair
            elif args.read_order == "thermal-first":
                thermal_sample = read_sample(thermal_cap, idx)
                rgb_sample = read_sample(rgb_cap, idx)
            else:
                rgb_sample = read_sample(rgb_cap, idx)
                thermal_sample = read_sample(thermal_cap, idx)

            if not thermal_sample.ok or not rgb_sample.ok:
                print(f"read failed: thermal={thermal_sample.ok} rgb={rgb_sample.ok}")
                continue

            thermal_c, thermal_area, thermal_mask, thermal_display, thermal_process_s = thermal_centroid(
                thermal_sample.frame,
                args,
            )
            rgb_c, rgb_area, rgb_mask, rgb_gray, rgb_motion_mean, rgb_process_s = rgb_motion_centroid(
                rgb_sample.frame,
                prev_rgb_gray,
                args,
            )
            prev_rgb_gray = rgb_gray

            rgb_x = normalized_x(rgb_c, rgb_sample.frame.shape[1])
            thermal_x = normalized_x(thermal_c, thermal_mask.shape[1])
            now = time.perf_counter()
            frame_times.append(now)
            while frame_times and now - frame_times[0] > 3.0:
                frame_times.popleft()

            record = {
                "rgb_x": rgb_x,
                "thermal_x": thermal_x,
            }
            recent.append(record)
            lag = estimate_lag(recent, args.max_lag)
            fps = fps_from_times(frame_times)
            lag_samples = lag["lag_samples"] if lag else None
            lag_ms = (lag_samples / fps * 1000.0) if lag and fps > 0 else None
            lag_corr = lag["corr"] if lag else None
            loop_period_ms = ((loop_t - last_loop_t) * 1000.0) if last_loop_t is not None else None
            last_loop_t = loop_t

            row = {
                "idx": idx,
                "loop_t": loop_t - start,
                "thermal_idx": thermal_sample.idx,
                "rgb_idx": rgb_sample.idx,
                "thermal_read_ms": (thermal_sample.read_done - thermal_sample.read_start) * 1000.0,
                "rgb_read_ms": (rgb_sample.read_done - rgb_sample.read_start) * 1000.0,
                "read_done_delta_ms": (thermal_sample.read_done - rgb_sample.read_done) * 1000.0,
                "thermal_age_ms": (now - thermal_sample.read_done) * 1000.0,
                "rgb_age_ms": (now - rgb_sample.read_done) * 1000.0,
                "loop_period_ms": loop_period_ms,
                "loop_total_ms": (now - loop_t) * 1000.0,
                "thermal_parse_mask_ms": thermal_process_s * 1000.0,
                "rgb_motion_ms": rgb_process_s * 1000.0,
                "thermal_x": thermal_c[0] if thermal_c else None,
                "thermal_y": thermal_c[1] if thermal_c else None,
                "thermal_area": thermal_area,
                "rgb_x": rgb_c[0] if rgb_c else None,
                "rgb_y": rgb_c[1] if rgb_c else None,
                "rgb_area": rgb_area,
                "rgb_motion_mean": rgb_motion_mean,
                "lag_samples": lag_samples,
                "lag_ms": lag_ms,
                "lag_corr": lag_corr,
            }
            rows.append(row)
            if writer is not None:
                writer.writerow(row)

            if idx % max(1, args.print_every) == 0:
                lag_text = "lag=n/a"
                if lag is not None:
                    lag_text = f"lag={lag_samples:+d} samples {lag_ms:+.1f}ms corr={lag_corr:.2f}"
                print(
                    f"frame={idx} fps={fps:.1f} {lag_text} "
                    f"read_delta(th-rgb)={row['read_done_delta_ms']:+.1f}ms "
                    f"read_ms(th/rgb)={row['thermal_read_ms']:.1f}/{row['rgb_read_ms']:.1f} "
                    f"proc_ms(th/rgb)={row['thermal_parse_mask_ms']:.1f}/{row['rgb_motion_ms']:.1f} "
                    f"area(th/rgb)={thermal_area:.0f}/{rgb_area:.0f}"
                )

            if not args.no_display:
                rgb_view = draw_centroid(rgb_sample.frame, rgb_c, (0, 255, 255), "rgb motion")
                thermal_view = draw_centroid(thermal_display, thermal_c, (0, 255, 255), "thermal hot")
                rgb_mask_bgr = cv2.cvtColor(rgb_mask, cv2.COLOR_GRAY2BGR)
                thermal_mask_bgr = cv2.cvtColor(thermal_mask, cv2.COLOR_GRAY2BGR)
                rgb_view = cv2.addWeighted(rgb_view, 0.78, rgb_mask_bgr, 0.22, 0)
                thermal_view = cv2.addWeighted(thermal_view, 0.78, thermal_mask_bgr, 0.22, 0)

                lag_line = "lag: collecting"
                if lag is not None:
                    lag_line = f"lag thermal-vs-rgb: {lag_samples:+d} samples {lag_ms:+.0f} ms corr {lag_corr:.2f}"
                common_lines = [
                    f"{fps:.1f} fps",
                    lag_line,
                    f"read th-rgb {row['read_done_delta_ms']:+.1f} ms",
                ]
                cv2.imshow("lag_rgb_motion", put_lines(rgb_view, common_lines))
                cv2.imshow("lag_thermal_hot", put_lines(thermal_view, common_lines))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.duration > 0 and now - start >= args.duration:
                break
    finally:
        if thermal_reader is not None:
            thermal_reader.stop()
        if rgb_reader is not None:
            rgb_reader.stop()
        thermal_cap.release()
        rgb_cap.release()
        if csv_file is not None:
            csv_file.close()
        cv2.destroyAllWindows()

    valid_lags = [r for r in rows if r["lag_samples"] is not None and r["lag_ms"] is not None]
    if valid_lags:
        tail = valid_lags[-max(10, min(60, len(valid_lags))):]
        lag_ms = np.asarray([r["lag_ms"] for r in tail], dtype=np.float32)
        corr = np.asarray([r["lag_corr"] for r in tail], dtype=np.float32)
        print(
            "Summary: "
            f"median_lag={float(np.median(lag_ms)):+.1f}ms "
            f"mean_lag={float(np.mean(lag_ms)):+.1f}ms "
            f"median_corr={float(np.median(corr)):.2f}"
        )
    else:
        print("Summary: not enough paired motion to estimate lag. Try larger left-right motion or longer duration.")

    print("Latency distribution:")
    print("  " + percentile_summary(rows, "thermal_read_ms"))
    print("  " + percentile_summary(rows, "rgb_read_ms"))
    print("  " + percentile_summary(rows, "read_done_delta_ms", signed=True))
    print("  " + percentile_summary(rows, "thermal_age_ms"))
    print("  " + percentile_summary(rows, "rgb_age_ms"))
    print("  " + percentile_summary(rows, "thermal_parse_mask_ms"))
    print("  " + percentile_summary(rows, "rgb_motion_ms"))
    print("  " + percentile_summary(rows, "loop_total_ms"))
    print("  " + percentile_summary(rows, "loop_period_ms"))
    print("  " + percentile_summary(valid_lags, "lag_ms", signed=True))


if __name__ == "__main__":
    main()
