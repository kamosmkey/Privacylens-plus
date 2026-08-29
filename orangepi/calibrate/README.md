# Camera Calibration Tools

This directory owns both calibration workflows. By default, each tool writes
its result directly into the self-contained `ui/` application.

## 1. RGB camera intrinsic calibration

Capture at least 8 sharp chessboard images at varied positions and angles. For
a board with 9 by 6 inner corners, run:

```bash
python3 calibrate_rgb_camera.py 'captures/*.png' --columns 9 --rows 6
```

The default output is `../ui/calibration_standard.npz`.

## 2. Thermal-to-RGB alignment

```bash
python3 thermal_alignment.py --thermal-dev /dev/video0 --rgb-dev /dev/video2
```

Controls:

- Arrow keys: translate the thermal image.
- `+` / `-`: change thermal scale.
- `S`: save to `../ui/thermal_alignment.json`.
- `Q` or Escape: quit.

Restart the UI after replacing either calibration file.
