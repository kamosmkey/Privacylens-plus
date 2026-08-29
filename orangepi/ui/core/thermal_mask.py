#!/usr/bin/env python3
import argparse
import time
from collections import defaultdict, deque

import cv2
import numpy as np

from rknnlite.api import RKNNLite


W = 256
H_FULL = 384
H_TEMP = 192

COCO_SKELETON = (
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to a supported pose RKNN model")
    parser.add_argument("--dev", default="/dev/video0")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--kpt-conf", type=float, default=0.0, help="Keypoint draw threshold")
    parser.add_argument("--print-every", type=int, default=30, help="Print latency every N frames")
    parser.add_argument("--mask-min-temp", type=float, default=24.0, help="Minimum Celsius value for bbox mask")
    parser.add_argument("--mask-max-temp", type=float, default=42.0, help="Maximum Celsius value for bbox mask")
    parser.add_argument("--mask-percentile", type=float, default=None, help="Optional local percentile lower bound for bbox mask")
    parser.add_argument("--mask-alpha", type=float, default=0.45, help="Mask overlay alpha")
    parser.add_argument("--no-mask", action="store_true", help="Disable bbox temperature mask overlay")
    return parser.parse_args()


def open_thermal_camera(dev):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H_FULL)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))

    # 关键：不要让 OpenCV 自动转 BGR，否则会丢 thermal raw 的高 8 bits
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {dev}")

    return cap


def parse_raw_temperature(frame):
    arr = np.asarray(frame)

    if arr.shape == (1, H_FULL, W, 2):
        arr = arr[0]
    elif arr.shape == (H_FULL, W, 2):
        pass
    else:
        data = arr.reshape(-1).view(np.uint8)
        expected = W * H_FULL * 2
        if data.size != expected:
            raise RuntimeError(f"Unexpected frame shape: {arr.shape}, bytes={data.size}")
        arr = data.reshape(H_FULL, W, 2)

    # 下半部分 256x192 是温度 raw data
    bottom = arr[192:384, :, :].astype(np.uint16)

    # little-endian uint16
    raw16 = bottom[:, :, 0] + (bottom[:, :, 1] << 8)

    return raw16


def parse_temperature(frame):
    raw16 = parse_raw_temperature(frame)

    # P2Pro-like 温度公式
    temp_c = raw16.astype(np.float32) / 64.0 - 273.15

    return temp_c


def temp_to_bgr(temp_c):
    lo = np.percentile(temp_c, 1)
    hi = np.percentile(temp_c, 99)
    if hi <= lo:
        hi = lo + 1.0

    gray = np.clip((temp_c - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return bgr


def temp_to_display(temp_c):
    lo = np.percentile(temp_c, 1)
    hi = np.percentile(temp_c, 99)
    if hi <= lo:
        hi = lo + 1.0

    gray = np.clip((temp_c - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def letterbox(image, new_shape=640, color=(114, 114, 114)):
    src_h, src_w = image.shape[:2]
    scale = min(new_shape / src_w, new_shape / src_h)

    resized_w = int(round(src_w * scale))
    resized_h = int(round(src_h * scale))

    pad_w = new_shape - resized_w
    pad_h = new_shape - resized_h

    pad_left = pad_w // 2
    pad_top = pad_h // 2

    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    canvas[pad_top:pad_top + resized_h, pad_left:pad_left + resized_w] = resized

    return canvas, scale, pad_left, pad_top


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)
        inter = inter_w * inter_h

        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / union

        order = order[1:][iou <= iou_thres]

    return keep


def make_anchor_points(img_size, shapes):
    points = []
    strides = []

    for h, w in shapes:
        stride = img_size / h
        grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        anchor = np.stack(
            (grid_x.reshape(-1) + 0.5, grid_y.reshape(-1) + 0.5),
            axis=1,
        )
        points.append(anchor)
        strides.append(np.full((h * w, 1), stride, dtype=np.float32))

    return (
        np.concatenate(points, axis=0).astype(np.float32),
        np.concatenate(strides, axis=0),
    )


def decode_keypoints(kpt_head, anchor_points, strides, img_size, mode):
    kpts = kpt_head[0].transpose(2, 0, 1).reshape(-1, 17, 3).astype(np.float32)

    if mode == "raw":
        kpts[:, :, 0] = (kpts[:, :, 0] * 2.0 + (anchor_points[:, 0:1] - 0.5)) * strides
        kpts[:, :, 1] = (kpts[:, :, 1] * 2.0 + (anchor_points[:, 1:2] - 0.5)) * strides
    elif mode == "decoded":
        if np.nanmax(kpts[:, :, :2]) <= 2.0:
            kpts[:, :, :2] *= img_size
    else:
        raise ValueError(f"Unknown keypoint mode: {mode}")

    if kpts[:, :, 2].min() < 0.0 or kpts[:, :, 2].max() > 1.0:
        kpts[:, :, 2] = sigmoid(kpts[:, :, 2])

    return kpts


def kpt_fit_score(boxes, scores, kpts, conf_thres, img_size):
    candidates = np.where(scores >= conf_thres)[0]
    if candidates.size == 0:
        candidates = scores.argsort()[-10:]
    else:
        candidates = candidates[np.argsort(scores[candidates])[-20:]]

    total = 0
    for i in candidates:
        x1, y1, x2, y2 = boxes[i]
        bw = x2 - x1
        bh = y2 - y1
        margin = max(bw, bh) * 0.25
        xs = kpts[i, :, 0]
        ys = kpts[i, :, 1]
        inside_image = (xs >= 0) & (xs <= img_size) & (ys >= 0) & (ys <= img_size)
        inside_box = (
            (xs >= x1 - margin) & (xs <= x2 + margin) &
            (ys >= y1 - margin) & (ys <= y2 + margin)
        )
        total += int(np.count_nonzero(inside_image & inside_box))

    return total


def choose_keypoint_mode(kpt_head, anchor_points, strides, boxes, scores, conf_thres, img_size):
    raw_kpts = decode_keypoints(kpt_head, anchor_points, strides, img_size, "raw")
    decoded_kpts = decode_keypoints(kpt_head, anchor_points, strides, img_size, "decoded")
    raw_score = kpt_fit_score(boxes, scores, raw_kpts, conf_thres, img_size)
    decoded_score = kpt_fit_score(boxes, scores, decoded_kpts, conf_thres, img_size)
    if decoded_score > raw_score:
        return decoded_kpts
    return raw_kpts


def decode_yolov8_pose(outputs, img_size, conf_thres):
    """
    解 yolov8n-pose.rknn 输出：
      3 个 [1,65,H,W] bbox head
      1 个 [1,17,3,8400] keypoint head
    """
    box_heads = []
    kpt_head = None

    for out in outputs:
        arr = np.asarray(out)
        if arr.ndim == 4 and arr.shape[1] == 65:
            box_heads.append(arr)
        elif arr.ndim == 4 and arr.shape[1:3] == (17, 3):
            kpt_head = arr

    if len(box_heads) != 3 or kpt_head is None:
        shapes = [tuple(np.asarray(out).shape) for out in outputs]
        raise RuntimeError(f"Unsupported RKNN output shape: {shapes}")

    box_heads.sort(key=lambda x: x.shape[2], reverse=True)

    shapes = [(head.shape[2], head.shape[3]) for head in box_heads]
    anchor_points, strides = make_anchor_points(img_size, shapes)

    projection = np.arange(16, dtype=np.float32)

    all_distances = []
    all_scores = []

    for head in box_heads:
        pred = head[0].transpose(1, 2, 0).reshape(-1, 65).astype(np.float32)

        dfl = pred[:, :64].reshape(-1, 4, 16)
        distances = (softmax(dfl, axis=2) * projection).sum(axis=2)

        cls = pred[:, 64]
        if cls.min() < 0.0 or cls.max() > 1.0:
            cls = sigmoid(cls)

        all_distances.append(distances)
        all_scores.append(cls)

    distances = np.concatenate(all_distances, axis=0)
    scores = np.concatenate(all_scores, axis=0)

    xy1 = (anchor_points - distances[:, 0:2]) * strides
    xy2 = (anchor_points + distances[:, 2:4]) * strides
    boxes = np.concatenate((xy1, xy2), axis=1)

    if kpt_head.shape[-1] != boxes.shape[0]:
        raise RuntimeError(f"Keypoint count {kpt_head.shape[-1]} does not match box count {boxes.shape[0]}")

    kpts = choose_keypoint_mode(kpt_head, anchor_points, strides, boxes, scores, conf_thres, img_size)

    return boxes, scores, kpts


def decode_yolo26_pose(outputs, img_size):
    """Decode the standard non-end-to-end YOLO26 pose output.

    The exported single-class COCO pose model returns one tensor containing
    ``xywh + class score + 17 * (x, y, visibility)`` for every anchor.
    Accepted layouts are [1, 56, N] and [1, N, 56].
    """
    if len(outputs) != 1:
        shapes = [tuple(np.asarray(out).shape) for out in outputs]
        raise RuntimeError(f"Unsupported YOLO26 RKNN output shape: {shapes}")

    output = np.asarray(outputs[0])
    if output.ndim != 3 or output.shape[0] != 1:
        raise RuntimeError(
            f"Unsupported YOLO26 RKNN output shape: {tuple(output.shape)}"
        )
    if output.shape[1] == 56:
        prediction = output[0].T.astype(np.float32)
    elif output.shape[2] == 56:
        prediction = output[0].astype(np.float32)
    else:
        raise RuntimeError(
            f"Unsupported YOLO26 RKNN output shape: {tuple(output.shape)}"
        )

    xywh = prediction[:, :4]
    scores = prediction[:, 4]
    if scores.min() < 0.0 or scores.max() > 1.0:
        scores = sigmoid(scores)

    boxes = np.empty_like(xywh)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0

    kpts = prediction[:, 5:].reshape(-1, 17, 3)
    if np.nanmax(kpts[:, :, :2]) <= 2.0:
        kpts[:, :, :2] *= img_size
    if kpts[:, :, 2].min() < 0.0 or kpts[:, :, 2].max() > 1.0:
        kpts[:, :, 2] = sigmoid(kpts[:, :, 2])
    return boxes, scores, kpts


def decode_yolo_pose(outputs, img_size, conf_thres):
    """Dispatch between the supported YOLOv8 and YOLO26 pose heads."""
    shapes = [tuple(np.asarray(out).shape) for out in outputs]
    if len(shapes) == 1 and len(shapes[0]) == 3 and 56 in shapes[0][1:]:
        return decode_yolo26_pose(outputs, img_size)
    return decode_yolov8_pose(outputs, img_size, conf_thres)


def postprocess_pose(outputs, img_size, scale, pad_left, pad_top, conf_thres, iou_thres):
    boxes, scores, kpts = decode_yolo_pose(outputs, img_size, conf_thres)

    mask = scores >= conf_thres
    boxes = boxes[mask]
    scores = scores[mask]
    kpts = kpts[mask]

    if len(scores) == 0:
        return []

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale
    kpts[:, :, 0] = (kpts[:, :, 0] - pad_left) / scale
    kpts[:, :, 1] = (kpts[:, :, 1] - pad_top) / scale

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H_TEMP - 1)
    kpts[:, :, 0] = kpts[:, :, 0].clip(0, W - 1)
    kpts[:, :, 1] = kpts[:, :, 1].clip(0, H_TEMP - 1)

    keep = nms(boxes, scores, iou_thres)

    detections = []
    for i in keep:
        box = boxes[i].astype(int)
        score = float(scores[i])
        keypoints = kpts[i]

        x1, y1, x2, y2 = box
        if x2 > x1 and y2 > y1:
            detections.append((box, score, keypoints))

    return detections


def bbox_temp(temp_c, box):
    x1, y1, x2, y2 = box
    roi = temp_c[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    mean_t = float(np.mean(roi))
    max_t = float(np.max(roi))
    min_t = float(np.min(roi))

    return mean_t, max_t, min_t


def extract_bbox_temperature_mask(temp_c, box, min_temp, max_temp, percentile=None):
    x1, y1, x2, y2 = box
    roi = temp_c[y1:y2, x1:x2]

    if roi.size == 0:
        return None, []

    valid = np.isfinite(roi)
    if not np.any(valid):
        return None, []

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
        return mask, []

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

    return clean, contours


def draw_temperature_mask(display, temp_c, box, min_temp, max_temp, percentile, alpha):
    x1, y1, x2, y2 = box
    mask, contours = extract_bbox_temperature_mask(temp_c, box, min_temp, max_temp, percentile)
    if mask is None or not contours:
        return

    color = np.array((255, 255, 0), dtype=np.float32)
    roi = display[y1:y2, x1:x2]
    mask_bool = mask.astype(bool)
    if roi.size == 0 or not np.any(mask_bool):
        return

    blend = roi.astype(np.float32)
    blend[mask_bool] = blend[mask_bool] * (1.0 - alpha) + color * alpha
    roi[:] = np.clip(blend, 0, 255).astype(np.uint8)
    cv2.drawContours(roi, contours, -1, (255, 255, 0), 1, cv2.LINE_AA)


def draw_keypoints(display, kpts, kpt_conf):
    for p1, p2 in COCO_SKELETON:
        x1, y1, c1 = kpts[p1]
        x2, y2, c2 = kpts[p2]
        if c1 >= kpt_conf and c2 >= kpt_conf:
            cv2.line(
                display,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (255, 230, 80),
                1,
                cv2.LINE_AA,
            )

    for x, y, conf in kpts:
        if conf >= kpt_conf:
            cv2.circle(display, (int(x), int(y)), 2, (40, 80, 255), -1, cv2.LINE_AA)


def draw_results(
    display,
    temp_c,
    detections,
    kpt_conf,
    mask_min_temp,
    mask_max_temp,
    mask_percentile,
    mask_alpha,
    show_mask,
):
    for box, score, kpts in detections:
        stats = bbox_temp(temp_c, box)
        if stats is None:
            continue

        mean_t, max_t, min_t = stats
        x1, y1, x2, y2 = box

        if show_mask:
            draw_temperature_mask(
                display,
                temp_c,
                box,
                mask_min_temp,
                mask_max_temp,
                mask_percentile,
                mask_alpha,
            )

        cv2.rectangle(display, (x1, y1), (x2, y2), (40, 220, 40), 1)
        draw_keypoints(display, kpts, kpt_conf)

        text1 = f"person {score:.2f}"
        text2 = f"mean {mean_t:.1f}C max {max_t:.1f}C"

        y_text = max(14, y1 - 18)

        cv2.putText(
            display,
            text1,
            (x1, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (40, 220, 40),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            text2,
            (x1, y_text + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (40, 220, 40),
            1,
            cv2.LINE_AA,
        )


def load_rknn(model_path):
    rknn = RKNNLite()

    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {model_path}, ret={ret}")

    try:
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    except TypeError:
        ret = rknn.init_runtime()

    if ret != 0:
        raise RuntimeError(f"init_runtime failed, ret={ret}")

    return rknn


def avg_ms(history, name):
    values = history[name]
    return sum(values) / len(values) if values else 0.0


def real_fps(frame_times, now):
    while frame_times and now - frame_times[0] > 3.0:
        frame_times.popleft()
    if len(frame_times) < 2:
        return 0.0
    return (len(frame_times) - 1) / max(1e-6, frame_times[-1] - frame_times[0])


def main():
    args = parse_args()

    rknn = load_rknn(args.model)
    cap = open_thermal_camera(args.dev)
    history = defaultdict(lambda: deque(maxlen=60))
    frame_times = deque()
    frame_count = 0

    print(f"Model: {args.model}")
    print(f"Camera: {args.dev}")
    print("Press q in display window to quit.")

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            t1 = time.perf_counter()
            if not ok:
                continue

            temp_c = parse_temperature(frame)

            model_img = temp_to_bgr(temp_c)
            input_img, scale, pad_left, pad_top = letterbox(model_img, args.img_size)

            input_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            input_tensor = np.expand_dims(input_rgb, axis=0)
            t2 = time.perf_counter()

            outputs = rknn.inference(inputs=[input_tensor])
            t3 = time.perf_counter()

            detections = postprocess_pose(
                outputs,
                args.img_size,
                scale,
                pad_left,
                pad_top,
                args.conf,
                args.iou,
            )
            t4 = time.perf_counter()

            display = temp_to_display(temp_c)
            draw_results(
                display,
                temp_c,
                detections,
                args.kpt_conf,
                args.mask_min_temp,
                args.mask_max_temp,
                args.mask_percentile,
                args.mask_alpha,
                not args.no_mask,
            )
            t5 = time.perf_counter()

            metrics = {
                "capture": (t1 - t0) * 1000.0,
                "pre": (t2 - t1) * 1000.0,
                "infer": (t3 - t2) * 1000.0,
                "post": (t4 - t3) * 1000.0,
                "draw": (t5 - t4) * 1000.0,
                "total": (t5 - t0) * 1000.0,
            }
            for name, value in metrics.items():
                history[name].append(value)

            frame_count += 1
            now = time.perf_counter()
            frame_times.append(now)
            fps = real_fps(frame_times, now)

            if frame_count % max(1, args.print_every) == 0:
                print(
                    f"frame={frame_count} det={len(detections)} "
                    f"capture={avg_ms(history, 'capture'):.1f}ms "
                    f"pre={avg_ms(history, 'pre'):.1f}ms "
                    f"infer={avg_ms(history, 'infer'):.1f}ms "
                    f"post={avg_ms(history, 'post'):.1f}ms "
                    f"draw={avg_ms(history, 'draw'):.1f}ms "
                    f"total={avg_ms(history, 'total'):.1f}ms "
                    f"fps={fps:.1f}"
                )

            cv2.imshow("thermal bbox temperature", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if frame_count > 0:
            final_fps = real_fps(frame_times, time.perf_counter())
            print(
                f"final frames={frame_count} "
                f"capture={avg_ms(history, 'capture'):.1f}ms "
                f"pre={avg_ms(history, 'pre'):.1f}ms "
                f"infer={avg_ms(history, 'infer'):.1f}ms "
                f"post={avg_ms(history, 'post'):.1f}ms "
                f"draw={avg_ms(history, 'draw'):.1f}ms "
                f"total={avg_ms(history, 'total'):.1f}ms "
                f"fps={final_fps:.1f}"
            )
        cap.release()
        cv2.destroyAllWindows()
        rknn.release()


if __name__ == "__main__":
    main()
