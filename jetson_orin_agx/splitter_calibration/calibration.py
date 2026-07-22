#!/usr/bin/env python3
"""Capture checkerboard images, calibrate a camera, and preview dewarping.

Examples:
  python calibration.py capture --camera 0 --cols 9 --rows 6
  python calibration.py calibrate --cols 9 --rows 6 --square-size 25
  python calibration.py preview --camera 0 --balance 0.2

``cols`` and ``rows`` are the number of INNER corners, not squares.  The square
size can be in millimetres (or any consistent unit); it does not affect image
undistortion, but it makes estimated translations use that unit.
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import cv2
import numpy as np


DEFAULT_IMAGE_DIR = Path(__file__).resolve().parent / "calibration_images"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "camera_calibration.npz"


def camera_source(value: str):
    """Accept a V4L2 index ("0") or a device/video path."""
    return int(value) if value.isdigit() else value


def open_camera(source, width: int, height: int, fps: int,
                fourcc: str) -> cv2.VideoCapture:
    backend = (cv2.CAP_V4L2
               if isinstance(source, str) and source.startswith("/dev/video")
               else cv2.CAP_ANY)
    cap = cv2.VideoCapture(source, backend)
    if fourcc:
        if len(fourcc) != 4:
            raise RuntimeError("--fourcc must contain exactly 4 characters")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {source}")
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"Camera opened but returned no frame: {source}")
    raw_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_fourcc = "".join(
        chr((raw_fourcc >> (8 * index)) & 0xFF) for index in range(4)
    )
    print(f"Camera output: {frame.shape[1]}x{frame.shape[0]}, "
          f"fourcc={actual_fourcc!r}")
    return cap


def find_corners(gray: np.ndarray, pattern: tuple[int, int],
                 exhaustive: bool = True):
    # SB is substantially more reliable near the image edges. Fall back for
    # older OpenCV builds.
    if exhaustive and hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags)
        if found:
            return True, corners.astype(np.float32)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if not exhaustive:
        flags |= cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if found:
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def capture_images(args) -> None:
    image_dir = Path(args.images)
    image_dir.mkdir(parents=True, exist_ok=True)
    cap = open_camera(camera_source(args.camera), args.width, args.height,
                      args.fps, args.fourcc)
    pattern = (args.cols, args.rows)
    saved = len(list(image_dir.glob("calib_*.png")))

    print("SPACE = save (only when corners are found), Q/ESC = finish")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read frame", file=sys.stderr)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = find_corners(gray, pattern, exhaustive=False)
            display = frame.copy()
            if found:
                cv2.drawChessboardCorners(display, pattern, corners, found)

            status = f"saved {saved}/{args.target} | corners: {'YES' if found else 'NO'}"
            color = (0, 220, 0) if found else (0, 0, 255)
            cv2.putText(display, status, (20, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, color, 2, cv2.LINE_AA)
            cv2.imshow("Calibration capture", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if not found:
                    print("Not saved: all inner corners must be visible.")
                    continue
                filename = image_dir / f"calib_{int(time.time() * 1000)}.png"
                if not cv2.imwrite(str(filename), frame):
                    raise RuntimeError(f"Failed to save {filename}")
                saved += 1
                print(f"Saved {filename.name} ({saved}/{args.target})")
                if saved >= args.target:
                    print("Target reached; press Q when finished, or capture more views.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def checkerboard_object_points(cols: int, rows: int, square_size: float):
    points = np.zeros((rows * cols, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points *= square_size
    return points


def load_observations(image_dir: Path, pattern, square_size: float):
    paths = sorted(
        set(glob.glob(str(image_dir / "*.png")) +
            glob.glob(str(image_dir / "*.jpg")) +
            glob.glob(str(image_dir / "*.jpeg")))
    )
    if not paths:
        raise RuntimeError(f"No PNG/JPG images found in {image_dir}")

    object_template = checkerboard_object_points(*pattern, square_size)
    object_points, image_points, used = [], [], []
    image_size = None
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            print(f"Skipping unreadable image: {path}")
            continue
        size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = size
        if size != image_size:
            print(f"Skipping {path}: size {size}, expected {image_size}")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = find_corners(gray, pattern)
        if found:
            object_points.append(object_template.copy())
            image_points.append(corners)
            used.append(path)
        else:
            print(f"Skipping (corners not found): {path}")
    if len(used) < 10:
        raise RuntimeError(f"Only {len(used)} valid views; capture at least 10 (20-30 recommended).")
    return object_points, image_points, image_size, used


def calibrate(args) -> None:
    pattern = (args.cols, args.rows)
    obj, img, image_size, used = load_observations(
        Path(args.images), pattern, args.square_size
    )
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-7,
    )

    if args.model == "fisheye":
        obj_f = [p.reshape(1, -1, 3).astype(np.float64) for p in obj]
        img_f = [p.reshape(1, -1, 2).astype(np.float64) for p in img]
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            obj_f, img_f, image_size, K, D, None, None, flags, criteria
        )
    else:
        flags = cv2.CALIB_RATIONAL_MODEL if args.rational else 0
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            obj, img, image_size, None, None, flags=flags, criteria=criteria
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        K=K,
        D=D,
        image_size=np.asarray(image_size, dtype=np.int32),
        model=np.asarray(args.model),
        cols=np.asarray(args.cols),
        rows=np.asarray(args.rows),
        square_size=np.asarray(args.square_size),
        rms=np.asarray(rms),
    )
    print(f"Used {len(used)} valid images")
    print(f"RMS reprojection error: {rms:.4f} pixels")
    print(f"Saved calibration: {output}")
    if rms > 1.0:
        print("Warning: RMS > 1 px. Retake blurry/repetitive views and cover all edges.")


def make_maps(K, D, calibrated_size, output_size, model: str, balance: float):
    cw, ch = calibrated_size
    ow, oh = output_size
    if (cw, ch) != (ow, oh):
        if abs(cw / ch - ow / oh) > 1e-3:
            raise RuntimeError(
                f"Calibration is {cw}x{ch}, camera is {ow}x{oh}; aspect ratios differ. "
                "Calibrate at the exact runtime resolution."
            )
        K = K.copy()
        K[0, :] *= ow / cw
        K[1, :] *= oh / ch

    if model == "fisheye":
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, output_size, np.eye(3), balance=balance
        )
        return cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, output_size, cv2.CV_16SC2
        )
    new_K, _ = cv2.getOptimalNewCameraMatrix(
        K, D, output_size, balance, output_size
    )
    return cv2.initUndistortRectifyMap(
        K, D, None, new_K, output_size, cv2.CV_16SC2
    )


def preview(args) -> None:
    data = np.load(args.calibration)
    K, D = data["K"], data["D"]
    calibrated_size = tuple(int(v) for v in data["image_size"])
    model = str(data["model"].item())
    cap = open_camera(camera_source(args.camera), args.width, args.height,
                      args.fps, args.fourcc)
    maps = None
    print(f"Loaded {model} calibration; Q/ESC = quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            size = (frame.shape[1], frame.shape[0])
            if maps is None:
                maps = make_maps(K, D, calibrated_size, size, model, args.balance)
            flat = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
            comparison = np.hstack((frame, flat))
            cv2.putText(comparison, "original", (20, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(comparison, "flat", (frame.shape[1] + 20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Original | Flat", comparison)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def add_camera_args(parser):
    parser.add_argument("--camera", default="0", help="camera index or /dev/video path")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=0)
    parser.add_argument("--fourcc", default="YUYV",
                        help="V4L2 pixel format (default: YUYV)")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("capture", help="interactively capture checkerboard images")
    add_camera_args(capture)
    capture.add_argument("--images", default=str(DEFAULT_IMAGE_DIR))
    capture.add_argument("--cols", type=int, default=9, help="horizontal inner corners")
    capture.add_argument("--rows", type=int, default=6, help="vertical inner corners")
    capture.add_argument("--target", type=int, default=25)
    capture.set_defaults(func=capture_images)

    cal = sub.add_parser("calibrate", help="calculate and save lens parameters")
    cal.add_argument("--images", default=str(DEFAULT_IMAGE_DIR))
    cal.add_argument("--output", default=str(DEFAULT_OUTPUT))
    cal.add_argument("--cols", type=int, default=9, help="horizontal inner corners")
    cal.add_argument("--rows", type=int, default=6, help="vertical inner corners")
    cal.add_argument("--square-size", type=float, default=25.0,
                     help="one square side length; mm recommended")
    cal.add_argument("--model", choices=("standard", "fisheye"), default="standard")
    cal.add_argument("--rational", action="store_true",
                     help="use extra distortion coefficients for strong standard lenses")
    cal.set_defaults(func=calibrate)

    view = sub.add_parser("preview", help="show original and flat camera views")
    add_camera_args(view)
    view.add_argument("--calibration", default=str(DEFAULT_OUTPUT))
    view.add_argument("--balance", type=float, default=0.2,
                      help="0=crop more, 1=retain more field of view")
    view.set_defaults(func=preview)
    return parser


def main():
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (RuntimeError, cv2.error, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
