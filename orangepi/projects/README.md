# Thermal to RGB Mask Warp v1

This first version runs YOLOv8-pose on both cameras, builds a temperature mask in the thermal image, estimates a 2D transform from matching body keypoints, and warps the thermal mask into the RGB image coordinate system.

## Files

- `dual_thermal_rgb_warp.py`: main program.
- `pose_common.py`: RKNN YOLOv8-pose decode, inference helper, RGB camera helper.
- `thermal_common.py`: thermal camera parsing, temperature display, bbox temperature mask.
- `warp_common.py`: keypoint matching, affine/homography estimation, mask warp.

## Run

From this folder:

```bash
cd /home/orangepi/projects/Projects
python3 dual_thermal_rgb_warp.py --model /path/to/yolov8n-pose.rknn
```

Common camera setup:

```bash
python3 dual_thermal_rgb_warp.py \
  --model /home/orangepi/projects/rknn_yolov8_pose_demo/yolov8n-pose.rknn \
  --thermal-dev /dev/video0 \
  --rgb-dev /dev/video2 \
  --rgb-width 640 \
  --rgb-height 480
```

If the mask is too large or too small, tune:

```bash
--mask-min-temp 24 --mask-max-temp 42 --mask-percentile 60
```

If the warp is unstable, tune:

```bash
--kpt-conf 0.5 --update-every 15 --smooth-alpha 0.15
```

For a more flexible but less stable transform:

```bash
--warp-mode homography
```

## Windows

- `thermal_masked`: processed thermal image with thermal mask only. No boxes, labels, or keypoints.
- `rgb_raw`: original RGB camera frame.
- `rgb_overlay`: RGB camera frame with the warped thermal mask overlay.

Press `q` to quit.

## Keypoints Used For Warp

The code tries these COCO points in this order:

```text
5 left_shoulder
6 right_shoulder
11 left_hip
12 right_hip
7 left_elbow
8 right_elbow
13 left_knee
14 right_knee
0 nose
9 left_wrist
10 right_wrist
15 left_ankle
16 right_ankle
```

The first four torso points are the most important. A transform update is skipped if there are not enough confident matching keypoints.
