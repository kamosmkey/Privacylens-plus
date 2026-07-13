import cv2
import numpy as np


THERMAL_W = 256
THERMAL_H_FULL = 384
THERMAL_H = 192


def size(text):
    w, h = text.split("x")
    return int(w), int(h)


def open_thermal_camera(dev, wh):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wh[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, wh[1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {dev}")
    return cap


def open_rgb_camera(dev, wh):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wh[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, wh[1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {dev}")
    return cap


def load_homography(path):
    data = np.load(path)
    if "H_thermal_to_rgb" not in data:
        raise RuntimeError(f"{path} does not contain H_thermal_to_rgb")
    H = data["H_thermal_to_rgb"].astype(np.float64)
    if H.shape != (3, 3):
        raise RuntimeError(f"expected 3x3 H_thermal_to_rgb, got {H.shape}")
    return H


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
            raise RuntimeError(f"unexpected thermal frame shape: {arr.shape}, bytes={data.size}")
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


def make_temperature_mask(temp_c, min_temp, max_temp):
    mask = ((temp_c >= min_temp) & (temp_c <= max_temp)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


MASK_COLORS = {
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "magenta": (255, 0, 255),
    "cyan": (255, 255, 0),
}


def overlay_mask(image, mask, color_bgr, alpha=0.55):
    color = np.zeros_like(image)
    color[:, :] = color_bgr
    blended = cv2.addWeighted(image, 1.0, color, alpha, 0.0)
    return np.where(mask[:, :, None] > 0, blended, image)


def mask_center(mask):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def mask_bbox_width(mask):
    xs = np.nonzero(mask)[1]
    if xs.size == 0:
        return 0
    return int(xs.max() - xs.min() + 1)


def mask_bbox_x_center(mask):
    xs = np.nonzero(mask)[1]
    if xs.size == 0:
        return None
    return 0.5 * (float(xs.min()) + float(xs.max()))


def compensation_shift(mask, width, height, comp_x, comp_y, side):
    center = mask_center(mask)
    if center is None:
        return 0.0, 0.0

    cx, cy = center
    dx = cx - width / 2.0
    dy = cy - height / 2.0
    if side == "left" and dx >= 0:
        return 0.0, 0.0
    if side == "right" and dx <= 0:
        return 0.0, 0.0

    return comp_x * dx, comp_y * dy


def apply_translation_to_mask(mask, shift_x, shift_y):
    if abs(shift_x) < 1e-6 and abs(shift_y) < 1e-6:
        return mask
    h, w = mask.shape[:2]
    transform = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)
    return cv2.warpAffine(
        mask,
        transform,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def apply_mask_morphology(mask, dilate_size=0, erode_size=0):
    out = mask
    if dilate_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        out = cv2.dilate(out, kernel, iterations=1)
    if erode_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
        out = cv2.erode(out, kernel, iterations=1)
    return out


def apply_mask_x_stretch(mask, alpha=0.0, side="both"):
    if abs(alpha) < 1e-9:
        return mask

    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return mask

    h, w = mask.shape[:2]
    x_min = float(xs.min())
    x_max = float(xs.max())
    mask_cx = 0.5 * (x_min + x_max)
    image_cx = w / 2.0
    center_tolerance = 0.02
    dx_norm = (mask_cx - image_cx) / image_cx

    if dx_norm >= -center_tolerance:
        return mask
    anchor_x = x_min
    distance_ratio = min(1.0, -dx_norm)

    scale = 1.0 + alpha * distance_ratio
    if scale <= 0:
        raise RuntimeError("rgb mask x-stretch scale must be greater than 0 for this mask position")

    transform = np.array(
        [[scale, 0.0, (1.0 - scale) * anchor_x], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        mask,
        transform,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def put_text(image, text):
    out = image.copy()
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def put_text_top_right(image, text):
    out = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(8, out.shape[1] - text_w - 8)
    y = 24
    cv2.rectangle(out, (x - 4, y - text_h - 4), (x + text_w + 4, y + baseline + 4), (0, 0, 0), -1)
    cv2.putText(out, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out
