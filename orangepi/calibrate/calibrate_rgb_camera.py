#!/usr/bin/env python3
"""Calibrate the RGB camera from chessboard images."""

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", help="Image glob, for example 'captures/*.png'")
    parser.add_argument("--columns", type=int, default=9, help="Inner corner columns")
    parser.add_argument("--rows", type=int, default=6, help="Inner corner rows")
    parser.add_argument("--square-size", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "ui/calibration_standard.npz",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paths = sorted(glob.glob(args.images))
    if not paths:
        raise SystemExit(f"No images match: {args.images}")

    pattern = (args.columns, args.rows)
    object_template = np.zeros((args.columns * args.rows, 3), np.float32)
    object_template[:, :2] = np.mgrid[0 : args.columns, 0 : args.rows].T.reshape(-1, 2)
    object_template *= args.square_size
    object_points = []
    image_points = []
    image_size = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

    for path in paths:
        image = cv2.imread(path)
        if image is None:
            print(f"SKIP unreadable: {path}")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        if image_size is not None and size != image_size:
            print(f"SKIP different size: {path}")
            continue
        found, corners = cv2.findChessboardCorners(gray, pattern)
        if not found:
            print(f"SKIP no chessboard: {path}")
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(object_template.copy())
        image_points.append(corners)
        image_size = size
        print(f"USE  {path}")

    if len(object_points) < 8:
        raise SystemExit(f"Only {len(object_points)} valid views; at least 8 are required")

    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        K=matrix,
        D=distortion,
        image_size=np.asarray(image_size),
        rms=np.asarray(rms),
    )
    print(f"Saved {args.output} ({len(object_points)} views, RMS={rms:.6f})")


if __name__ == "__main__":
    main()
