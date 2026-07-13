#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark YOLO on an RGB camera.")
    parser.add_argument("--model", default=str(PROJECT_DIR / "yolo26n.engine"))
    parser.add_argument("--dev", default="/dev/video2")
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="0")
    parser.add_argument("--precision", default="fp16", choices=("fp32", "fp16"))
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--verbose-export", action="store_true")
    parser.add_argument("--use-dla", action="store_true")
    parser.add_argument("--dla-core", type=int, default=0, choices=(0, 1))
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def open_camera(args):
    cap = cv2.VideoCapture(args.dev, cv2.CAP_V4L2)
    if not cap.isOpened() and args.dev.startswith("/dev/video"):
        cap = cv2.VideoCapture(int(args.dev.replace("/dev/video", "")))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {args.dev}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    if args.camera_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def ensure_pt_model(pt_path):
    if pt_path.exists():
        return pt_path

    pt_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"PT model not found, downloading: {pt_path.name}")
    downloaded = Path(attempt_download_asset(pt_path))

    if downloaded.exists() and downloaded.resolve() != pt_path.resolve():
        pt_path.write_bytes(downloaded.read_bytes())

    if not pt_path.exists():
        YOLO(pt_path.name)
        downloaded = Path(pt_path.name)
        if downloaded.exists():
            pt_path.write_bytes(downloaded.read_bytes())

    if not pt_path.exists():
        raise FileNotFoundError(f"Could not find or download PT model: {pt_path}")

    return pt_path


def export_engine(pt_path, args):
    pt_path = ensure_pt_model(pt_path)
    print(f"Exporting TensorRT engine from {pt_path}")
    export_args = {
        "format": "engine",
        "dynamic": args.dynamic,
        "simplify": args.simplify,
        "verbose": args.verbose_export,
    }
    if not args.dynamic:
        export_args["imgsz"] = args.imgsz

    exported = YOLO(str(pt_path)).export(**export_args)
    return Path(exported)


def resolve_model(args):
    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = (Path.cwd() / model_path).resolve()

    if model_path.suffix == ".engine":
        if model_path.exists():
            return model_path
        return export_engine(model_path.with_suffix(".pt"), args)

    if model_path.suffix == ".pt":
        engine_path = model_path.with_suffix(".engine")
        if engine_path.exists():
            print(f"Using existing engine: {engine_path}")
            return engine_path
        return export_engine(model_path, args)

    raise ValueError("--model must point to a .engine or .pt file")


def main():
    args = parse_args()
    model_path = resolve_model(args)
    model = YOLO(str(model_path))
    cap = open_camera(args)

    camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Model: {model_path}")
    print(f"Camera: {args.dev} {camera_width}x{camera_height} {args.fourcc} actual_fps={camera_fps:.2f}")
    print(f"Benchmark: {args.duration:.1f}s after {args.warmup_frames} warmup frames")

    frame_count = 0
    measured_count = 0
    measured_start = None
    measured_end = None
    inference_times = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                continue

            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )
            result = results[0] if results else None
            inference_ms = float(result.speed.get("inference", 0.0)) if result is not None else 0.0

            frame_count += 1
            if frame_count > args.warmup_frames:
                if measured_start is None:
                    measured_start = time.perf_counter()
                inference_times.append(inference_ms)
                measured_count += 1
                measured_end = time.perf_counter()

            if args.show:
                view = result.plot() if result is not None else frame
                cv2.imshow("YOLO RGB", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if measured_start is not None and time.perf_counter() - measured_start >= args.duration:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not inference_times or measured_start is None or measured_end is None:
        print("No measured frames.")
        return

    measured_elapsed = measured_end - measured_start
    average_inference = sum(inference_times) / len(inference_times)
    pipeline_fps = measured_count / measured_elapsed if measured_elapsed > 0 else 0.0
    inference_fps = 1000.0 / average_inference if average_inference > 0 else 0.0

    print("\nSummary")
    print(f"frames_total={frame_count}")
    print(f"frames_measured={measured_count}")
    print(f"measured_elapsed={measured_elapsed:.2f} s")
    print(f"average_inference={average_inference:.2f} ms")
    print(f"inference_fps={inference_fps:.2f}")
    print(f"pipeline_fps={pipeline_fps:.2f}")


if __name__ == "__main__":
    main()
