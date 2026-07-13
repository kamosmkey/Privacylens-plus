#!/usr/bin/env python3
import cv2
import numpy as np


UPPER_BODY_POINTS = (5, 6, 0, 7, 8, 9, 10)
LOWER_TORSO_POINTS = (11, 12)
LOWER_BODY_POINTS = (13, 14, 15, 16)
DEFAULT_KEYPOINT_ORDER = UPPER_BODY_POINTS + LOWER_TORSO_POINTS + LOWER_BODY_POINTS


def matched_keypoints(thermal_kpts, rgb_kpts, min_conf=0.4, indices=DEFAULT_KEYPOINT_ORDER):
    thermal_pts = []
    rgb_pts = []
    used = []

    for idx in indices:
        tx, ty, tc = thermal_kpts[idx]
        rx, ry, rc = rgb_kpts[idx]
        if tc >= min_conf and rc >= min_conf:
            thermal_pts.append((tx, ty))
            rgb_pts.append((rx, ry))
            used.append(idx)

    if not thermal_pts:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty, used

    return np.asarray(thermal_pts, dtype=np.float32), np.asarray(rgb_pts, dtype=np.float32), used


def estimate_calibration(thermal_kpts, rgb_kpts, mode="affine", min_conf=0.4, ransac_thresh=6.0):
    src, dst, used = matched_keypoints(thermal_kpts, rgb_kpts, min_conf=min_conf)

    if mode == "homography":
        if len(src) < 4:
            return None, None, used
        matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
        if matrix is None:
            return None, None, used
        return matrix.astype(np.float32), inliers, used

    if mode == "affine":
        if len(src) < 3:
            return None, None, used
        matrix, inliers = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=2000,
            confidence=0.98,
            refineIters=10,
        )
        if matrix is None:
            return None, None, used
        return matrix.astype(np.float32), inliers, used

    raise ValueError(f"Unknown calibration mode: {mode}")


def smooth_calibration(old_matrix, new_matrix, alpha=0.25):
    if new_matrix is None:
        return old_matrix
    if old_matrix is None or old_matrix.shape != new_matrix.shape:
        return new_matrix
    return (old_matrix * (1.0 - alpha) + new_matrix * alpha).astype(np.float32)


def apply_calibration_mask(mask, matrix, out_size, mode="affine"):
    if matrix is None:
        return np.zeros((out_size[1], out_size[0]), dtype=np.uint8)

    if mode == "homography":
        return cv2.warpPerspective(mask, matrix, out_size, flags=cv2.INTER_NEAREST)

    return cv2.warpAffine(mask, matrix, out_size, flags=cv2.INTER_NEAREST)
