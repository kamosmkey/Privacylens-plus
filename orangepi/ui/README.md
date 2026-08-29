# Thermal Pose Monitor UI

This directory is the self-contained user application. It contains the UI,
camera pipeline, RKNN inference code, pose processing, recording code, model,
camera calibration, thermal alignment, and UI assets. It does not import code
from the top-level `rknn/` directory.

## Directory layout

```text
ui/
├── app.py                         # Application entry point
├── run.sh                         # Recommended launcher
├── thermal_pose_ui.py             # PyQt5 window and user interaction
├── pipeline_worker.py             # Cameras, alignment, NPU inference and masking
├── pose_smoother.py               # Temporal pose smoothing
├── video_panel.py                 # Video display widget
├── video_recorder.py              # H.264 videos and pose JSONL recording
├── temperature.py                 # Orange Pi temperature monitoring
├── core/
│   ├── warp.py                    # Runtime alignment and pose helpers
│   └── thermal_mask.py            # Thermal parsing and YOLO pose decoding
├── assets/                        # Qt icons
├── model/                         # RKNN model and source-model artifacts
├── calibration_standard.npz       # RGB intrinsic calibration
└── thermal_alignment.json         # Thermal-to-RGB scale and translation
```

## Start the UI

From the project root:

```bash
cd /home/orangepi/projects
./ui/run.sh
```

`run.sh` uses `/home/orangepi/projects/rknn-env/bin/python` by default. To use
another compatible environment:

```bash
UI_PYTHON=/path/to/python ./ui/run.sh
```

Equivalent direct invocation:

```bash
cd /home/orangepi/projects/ui
../rknn-env/bin/python app.py
```

The application currently has no command-line configuration arguments. Its
runtime settings are defined by `PipelineConfig` in `pipeline_worker.py`, while
the two calibration results are loaded from files in this directory.

## Call flow

```text
run.sh
  -> app.py
     -> thermal_pose_ui.main()
        -> MainWindow
           -> PipelineWorker
              -> ui/core/warp.py
              -> ui/core/thermal_mask.py
              -> RKNNLite / RK3588 NPU
```

`PipelineWorker` opens both cameras in background capture threads, pairs RGB
and thermal frames by timestamp, undistorts RGB, aligns the thermal mask, runs
pose inference, builds the privacy/stick-figure output, and optionally records
the synchronized results.

## Required files

The default configuration requires:

```text
model/yolo26n-pose_rknn_model/best.sanitized-rk3588.rknn
calibration_standard.npz
thermal_alignment.json
```

`calibration_standard.npz` is produced by
`../calibrate/calibrate_rgb_camera.py`. `thermal_alignment.json` is produced by
`../calibrate/thermal_alignment.py`. Restart the UI after replacing either
file.

## Pipeline parameters

The following values are in `PipelineConfig` in `pipeline_worker.py`.

### Model and calibration

| Parameter | Default | Meaning |
| --- | --- | --- |
| `model` | `model/yolo26n-pose_rknn_model/best.sanitized-rk3588.rknn` | RK3588 RKNN pose model |
| `calibration` | `calibration_standard.npz` | RGB intrinsic/distortion calibration |
| `use_undistort` | `True` | Apply RGB lens-undistortion maps |
| `rgb_balance` | `0.0` | Undistortion crop/field-of-view balance |

### Camera capture

| Parameter | Default | Meaning |
| --- | --- | --- |
| `thermal_dev` | `/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0` | Stable V4L2 path for the thermal camera |
| `thermal_fps` | `25.0` | Requested thermal capture rate |
| `rgb_dev` | `/dev/v4l/by-id/usb-HD_USB_Camera_HD_USB_Camera-video-index0` | Stable V4L2 path for the RGB camera |
| `rgb_width` | `640` | Requested RGB width |
| `rgb_height` | `480` | Requested RGB height |
| `rgb_fps` | `30.0` | Requested RGB capture rate |
| `rgb_fourcc` | `YUYV` | Requested RGB V4L2 pixel format |
| `thermal_rgb_offset_ms` | `50.0` | Timestamp offset used when pairing RGB with thermal frames |

List available camera paths with:

```bash
v4l2-ctl --list-devices
ls -l /dev/v4l/by-id/
```

Prefer `/dev/v4l/by-id/...` over `/dev/videoN`, because numeric device IDs can
change after a reboot or reconnect.

### Thermal alignment

The initial values are loaded from `thermal_alignment.json`:

```json
{
  "scale": 0.74,
  "x": -5.0,
  "y": 15.0
}
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `thermal_scale` | JSON `scale` | Thermal image scale about the RGB image center |
| `thermal_x` | JSON `x` | Horizontal translation in RGB pixels |
| `thermal_y` | JSON `y` | Vertical translation in RGB pixels |
| `mirror_thermal_on_read` | `False` | Mirror decoded thermal input before processing |
| `thermal_flip_during_alignment` | `False` | Mirror thermal data during affine alignment |
| `crop_to_thermal` | `True` | Crop outputs to the valid aligned-thermal region |

The UI exposes live `THERMAL X` and `THERMAL Y` controls. These adjustments are
temporary and are not written to disk. Use the calibration tool and press `S`
to permanently update `thermal_alignment.json`.

### RKNN pose inference

| Parameter | Default | Meaning |
| --- | --- | --- |
| `img_size` | `640` | Square model input size |
| `pose_inference_interval` | `2` | Run NPU inference once every N processed frames |
| `confidence` | `0.25` | Minimum person detection confidence |
| `iou` | `0.45` | NMS intersection-over-union threshold |
| `keypoint_confidence` | `0.2` | Minimum upper-body keypoint drawing confidence |
| `lower_body_keypoint_confidence` | `0.7` | Minimum lower-body keypoint drawing confidence |

Set `pose_inference_interval` to `1` for inference on every frame. Higher values
reduce NPU load but reuse the latest smoothed pose between inference frames.

### Thermal mask and background

| Parameter | Default | Meaning |
| --- | --- | --- |
| `mask_alpha` | `0.45` | Thermal-mask overlay opacity |
| `mask_percentile` | `None` | Optional percentile-based lower threshold |
| `warped_mask_dilate_px` | `15` | Expand the aligned mask to reduce edge leakage |
| `background_confirm_frames` | `20` | Safe frames required before background learning |
| `background_update_alpha` | `0.01` | Background-model update rate |
| `background_motion_threshold` | `10` | Pixel-motion threshold for safe background updates |
| `background_person_guard_ratio` | `0.20` | Padding around detected people excluded from learning |

The minimum and maximum mask temperatures are controlled at runtime by the
`Temperature range` fields. Their initial values are 26 °C and 36 °C.

### Recording

| Parameter | Default | Meaning |
| --- | --- | --- |
| `recording_dir` | `/mnt/ssd/videos` | Root output directory |
| `recording_fourcc` | `libx264` | PyAV/FFmpeg video encoder |
| `recording_max_seconds` | `28800` | Maximum session duration (8 hours) |

Each recording creates:

```text
session_YYYYMMDD_HHMMSS_microseconds/
├── stickfigure/stickfigure_*.mp4
├── raw_rgb/raw_rgb_*.mp4
├── raw_thermal/raw_thermal_*.mp4
├── pose_metadata.jsonl
└── session.json
```

The thermal video is an 8-bit Inferno visualization, not absolute-temperature
data. Pose metadata contains bounding boxes, confidence values, keypoints,
timestamps, frame-pair delta, and inference timing.

## UI controls

| Control | Action |
| --- | --- |
| `Start` | Open cameras, load RKNN and start processing |
| `Close` | Stop workers and close the application |
| `Capture Background` | Capture the current empty scene for privacy replacement |
| `Record` | Start or stop synchronized research recording |
| `Stick Figure` | Show or hide the processed privacy output |
| `Raw Thermal` | Show or hide the thermal visualization |
| `Raw RGB` | Show or hide the RGB camera |
| Temperature fields | Change the live thermal-mask range |
| `THERMAL X/Y` | Temporarily adjust live thermal alignment |
| `Esc` | Leave full-screen mode |

Capture the background only when no person is visible. The background model
then updates slowly only in areas considered safe.

## Board-temperature protection

The UI reads `/sys/class/thermal`. If the hottest reported zone remains above
70 °C for five minutes, processing stops and the application closes.

## Code simplification status

The application structure is separated into UI, worker, display, recording,
smoothing, and runtime-core modules. The previous external `rknn/` dependency
has been removed from the call path. Unused selectable recording modes and the
unused generic multi-recorder have also been removed; the application now keeps
only its actual research-recording path.

The files under `core/` still contain some legacy standalone CLI/test code from
the earlier prototype. It is not called by the UI. It can be removed, but the
RKNN decoder and post-processing helpers must be retained because supported
YOLO output formats share them.

## Troubleshooting

- **Python environment not found:** set `UI_PYTHON=/path/to/python`.
- **Camera cannot open:** check device paths, permissions, and whether another
  process is using the device.
- **RKNN model cannot load:** verify the RKNNLite wheel, NPU runtime/driver, and
  model compatibility.
- **Recording fails:** confirm PyAV and `libx264` are installed and that
  `/mnt/ssd/videos` is writable.
- **Images do not align:** rerun both tools in `../calibrate/`, save their
  outputs, and restart the UI.
