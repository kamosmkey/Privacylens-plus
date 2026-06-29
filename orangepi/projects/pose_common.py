#!/usr/bin/env python3
import cv2
import numpy as np

try:
    from rknnlite.api import RKNNLite
except ImportError as exc:
    RKNNLite = None
    RKNN_IMPORT_ERROR = exc
else:
    RKNN_IMPORT_ERROR = None


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
        order = order[1:][inter / union <= iou_thres]

    return keep


def make_anchor_points(img_size, shapes):
    points = []
    strides = []
    for h, w in shapes:
        stride = img_size / h
        grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        anchor = np.stack((grid_x.reshape(-1) + 0.5, grid_y.reshape(-1) + 0.5), axis=1)
        points.append(anchor)
        strides.append(np.full((h * w, 1), stride, dtype=np.float32))
    return np.concatenate(points, axis=0).astype(np.float32), np.concatenate(strides, axis=0)


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


def postprocess_pose(outputs, original_shape, img_size, scale, pad_left, pad_top, conf_thres, iou_thres):
    boxes, scores, kpts = decode_yolov8_pose(outputs, img_size, conf_thres)
    keep_mask = scores >= conf_thres
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    kpts = kpts[keep_mask]

    if len(scores) == 0:
        return []

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale
    kpts[:, :, 0] = (kpts[:, :, 0] - pad_left) / scale
    kpts[:, :, 1] = (kpts[:, :, 1] - pad_top) / scale

    h, w = original_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
    kpts[:, :, 0] = kpts[:, :, 0].clip(0, w - 1)
    kpts[:, :, 1] = kpts[:, :, 1].clip(0, h - 1)

    keep = nms(boxes, scores, iou_thres)
    detections = []
    for i in keep:
        box = boxes[i].astype(np.float32)
        x1, y1, x2, y2 = box
        if x2 > x1 and y2 > y1:
            detections.append((box, float(scores[i]), kpts[i].astype(np.float32)))
    return detections


def run_pose(rknn, bgr_image, img_size, conf_thres, iou_thres):
    input_img, scale, pad_left, pad_top = letterbox(bgr_image, img_size)
    input_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(input_rgb, axis=0)
    outputs = rknn.inference(inputs=[input_tensor])
    return postprocess_pose(
        outputs,
        bgr_image.shape,
        img_size,
        scale,
        pad_left,
        pad_top,
        conf_thres,
        iou_thres,
    )


def load_rknn(model_path):
    if RKNNLite is None:
        raise RuntimeError("Cannot import RKNNLite. Install rknn-toolkit-lite2 first.") from RKNN_IMPORT_ERROR

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


def open_rgb_camera(dev, width, height):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened() and dev.startswith("/dev/video"):
        cap = cv2.VideoCapture(int(dev.replace("/dev/video", "")))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RGB camera {dev}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def select_primary_person(detections):
    if not detections:
        return None

    def rank(det):
        box, score, _ = det
        x1, y1, x2, y2 = box
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return score * max(1.0, area)

    return max(detections, key=rank)
