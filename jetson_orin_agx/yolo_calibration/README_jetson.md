# Jetson Thermal Mask + YOLO Calibration

This version runs YOLO pose on the thermal visual frame and RGB frame. The
thermal mask is generated inside the detected thermal YOLO boxes using the
thermal temperature thresholds. Matching keypoints are used to estimate a live
calibration matrix, and the thermal mask is projected onto the RGB image.

Run from the repository root:

```bash
privacylens/bin/python projects/jetson_thermal_rgb_warp.py \
  --model projects/model/yolo26n-pose.pt \
  --thermal-dev /dev/video0 \
  --rgb-dev /dev/video2 \
  --rgb-width 640 \
  --rgb-height 480 \
  --device 0 \
  --quantize fp16
```

Useful tuning flags:

```bash
--mask-min-temp 24 --mask-max-temp 42 --mask-percentile 60
--conf 0.25 --kpt-conf 0.4
--calibration-mode affine
--update-every 1 --smooth-alpha 0.25
```

Windows:

- `thermal_masked`: baseline-style thermal temperature mask.
- `thermal_yolo`: thermal visual frame with YOLO bbox/keypoints.
- `rgb_yolo`: RGB frame with YOLO bbox/keypoints.
- `rgb_calibration_masked`: RGB frame with the projected thermal mask.

Calibration keypoints prefer upper-body points first: shoulders, nose, elbows,
wrists, then hips, knees, and ankles.

Each window shows FPS in the top-right corner. Press `q` to quit.
