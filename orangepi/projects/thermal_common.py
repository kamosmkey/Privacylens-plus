#!/usr/bin/env python3
import cv2
import numpy as np


THERMAL_W = 256
THERMAL_H_FULL = 384
THERMAL_H = 192


def open_thermal_camera(dev):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, THERMAL_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, THERMAL_H_FULL)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open thermal camera {dev}")

    return cap


def parse_temperature(frame):
    arr = np.asarray(frame)

    if arr.shape == (1, THERMAL_H_FULL, THERMAL_W, 2):
        arr = arr[0]
    elif arr.shape == (THERMAL_H_FULL, THERMAL_W, 2):
        pass
    else:
        data = arr.reshape(-1).view(np.uint8)
        expected = THERMAL_W * THERMAL_H_FULL * 2
        if data.size != expected:
            raise RuntimeError(f"Unexpected thermal frame shape: {arr.shape}, bytes={data.size}")
        arr = data.reshape(THERMAL_H_FULL, THERMAL_W, 2)

    bottom = arr[THERMAL_H:THERMAL_H_FULL, :, :].astype(np.uint16)
    raw16 = bottom[:, :, 0] + (bottom[:, :, 1] << 8)
    return raw16.astype(np.float32) / 64.0 - 273.15


def temp_to_bgr(temp_c):
    lo = np.percentile(temp_c, 1)
    hi = np.percentile(temp_c, 99)
    if hi <= lo:
        hi = lo + 1.0

    gray = np.clip((temp_c - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def temp_to_display(temp_c):
    lo = np.percentile(temp_c, 1)
    hi = np.percentile(temp_c, 99)
    if hi <= lo:
        hi = lo + 1.0

    gray = np.clip((temp_c - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def clamp_box(box, width=THERMAL_W, height=THERMAL_H):
    x1, y1, x2, y2 = box.astype(int)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def extract_bbox_temperature_mask(temp_c, box, min_temp, max_temp, percentile=None):
    x1, y1, x2, y2 = clamp_box(box, temp_c.shape[1], temp_c.shape[0])
    roi = temp_c[y1:y2, x1:x2]
    if roi.size == 0:
        return None, (x1, y1, x2, y2)

    valid = np.isfinite(roi)
    if not np.any(valid):
        return None, (x1, y1, x2, y2)

    threshold = float(min_temp)
    if percentile is not None:
        threshold = max(threshold, float(np.percentile(roi[valid], percentile)))

    upper = float(max_temp)
    if upper <= threshold:
        upper = threshold + 0.1

    mask = np.zeros(roi.shape, dtype=np.uint8)
    mask[(roi >= threshold) & (roi <= upper) & valid] = 255

    h, w = mask.shape
    if h < 3 or w < 3:
        return mask, (x1, y1, x2, y2)

    kernel_size = 3 if min(h, w) < 24 else 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(8.0, h * w * 0.015)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

    clean = np.zeros_like(mask)
    if contours:
        cv2.drawContours(clean, contours, -1, 255, thickness=cv2.FILLED)

    return clean, (x1, y1, x2, y2)


def make_full_temperature_mask(temp_c, detections, min_temp, max_temp, percentile=None):
    full = np.zeros(temp_c.shape, dtype=np.uint8)

    for box, _, _ in detections:
        local_mask, (x1, y1, x2, y2) = extract_bbox_temperature_mask(
            temp_c,
            box,
            min_temp,
            max_temp,
            percentile,
        )
        if local_mask is None:
            continue
        full[y1:y2, x1:x2] = np.maximum(full[y1:y2, x1:x2], local_mask)

    return full


def overlay_mask(display, mask, color=(255, 255, 0), alpha=0.45, draw_contour=True):
    if mask is None or display.size == 0:
        return display

    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return display

    out = display.copy()
    color_arr = np.array(color, dtype=np.float32)
    blend = out.astype(np.float32)
    blend[mask_bool] = blend[mask_bool] * (1.0 - alpha) + color_arr * alpha
    out[:] = np.clip(blend, 0, 255).astype(np.uint8)

    if draw_contour:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 1, cv2.LINE_AA)

    return out
