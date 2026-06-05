# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A Python package named `kps-analyze-demo` (Python >=3.10), managed by uv. Real-time 2D/3D human pose estimation pipeline with WebSocket video streaming to browser.

## Commands

```bash
# Default: mock mode using pre-recorded video + cached keypoints
uv run main.py

# Real camera mode (requires Qualcomm DSP hardware)
uv run main.py --mode camera --camera 0 --width 640 --height 480 --fps 30

# Run with shell launcher (sets up aidlite SDK path)
bash run.sh
```

`main.py` serves on `0.0.0.0:28888`. Open `http://localhost:28888` to see the video stream with skeleton overlay.

## Architecture

### Core pipeline

```
Camera/Video → 2D Pose Extract → (COCO→H36M convert) → 3D Reconstruct → Judge → Render → WebSocket → Browser
```

The pipeline is orchestrated by `FrameAnalyzer` ([core/analyzer.py](core/analyzer.py)), which accepts swappable 2D extractor and 3D reconstructor via dependency injection.

### Key abstractions (in `core/`)

| Interface | File | Purpose |
| --- | --- | --- |
| `IRgbVideoSource` | `video_source/` | Frame source: camera or video file |
| `I2dPoseExtractor` | `kp2d_extractor.py` | 2D keypoints from BGR frame → `(17,3)` |
| `I3dPoseReconstructor` | `kp3d_reconstructor.py` | 3D lifting from 2D sequence → `(17,3)` |

Each abstraction has a **Mock** implementation (pre-recorded data, no hardware) and a **real** implementation (QNN/AID-Lite DSP inference). Mock mode is default.

### External model directories (outside `core/`)

- **`rtm-det-aidlite/`** — RTMDet (person detection) + RTMPose (2D keypoints) running on Qualcomm AID-Lite DSP
- **`mhformer-aidlite/`** — MHFormer temporal Transformer for 2D→3D lifting, 351-frame window, QNN DSP
- **`_unused/rtmpose/`** — Old ONNX-runtime CPU implementation, superseded. `onnxruntime` in `pyproject.toml` exists only for this unused code.

### Format conversion

`DataConverter` ([core/converter.py](core/converter.py)) maps COCO-17 ↔ H36M-17 keypoints. RTMPose outputs COCO-17; the rest of the pipeline uses H36M. 11 joints map directly, 6 are geometrically interpolated.

### Temporal buffering

MHFormer requires a 351-frame window. `FrameAnalyzer` keeps a `deque(maxlen=351)` of H36M keypoints and passes the full buffer each frame; the reconstructor returns only the latest frame's result (`frame_index=-1`).

### Lazy loading

Real extractors/reconstructors defer QNN model loading (and DSP resource allocation) to the first inference call, not construction time. This means `FrameAnalyzer.__init__` is cheap even with real hardware plugins.

### Frontend

Single-page app in [static/index.html](static/index.html): left panel `<img>` for JPEG frames, right panel for timestamped text messages. Both arrive over a single WebSocket (`/ws`) — binary for frames, text (JSON) for log messages.

### Linux camera troubleshooting

If `/dev/video0` fails with "Permission denied", the user likely isn't in the `video` group:

```bash
sudo usermod -aG video $USER   # re-login required
# Quick verify: sudo chmod 666 /dev/video0 then re-run
```
