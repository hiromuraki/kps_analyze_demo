# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A Python package named `kps-analyze-demo` (Python >=3.10), managed by uv. Real-time 2D/3D human pose estimation pipeline with WebSocket video streaming to browser.

## Commands

```bash
# Default: mock analyzer with mock video file
uv run main.py --analyzer-2d mock --analyzer-3d mock --camera -1

# Real camera with real analyzers (requires Qualcomm QCS8550 DSP + rtm_vision SDK)
uv run main.py --analyzer-2d rtmpose --analyzer-3d mhformer --camera 0 --width 640 --height 480 --fps 30

# Mixed: real 2D + mock 3D
uv run main.py --analyzer-2d rtmpose --analyzer-3d mock --camera 0

# Demo frontend (AR-style UI)
uv run example.py --analyzer-2d mock --analyzer-3d mock --camera -1
```

`main.py` starts on `0.0.0.0:2800`. Open `http://localhost:2800` in a browser.
`example.py` starts on `0.0.0.0:28001` (demo UI at `http://localhost:28001`).

## Architecture

### Core pipeline

```
Camera/Video → 2D Pose Extract (COCO17) → COCO→H36M convert → 3D Reconstruct → Judge → Render → WebSocket → Browser
```

`FrameAnalyzer` ([core/analyzer.py](core/analyzer.py)) orchestrates the pipeline via dependency injection. Returns `AnalysisResult` (dataclass): `rendered`, `kps_3d`, `violations`, `rep_counted`.

**Extension points** — four external functions in [core/rep_counter.py](core/rep_counter.py) must be implemented per exercise:
- `get_rep_feature_value(kps_3d, rule) -> float` — angle or distance
- `get_rep_ceiling(rule) -> float` / `get_rep_floor(rule) -> float`
- `get_rep_count_direction(rule) -> str` — `"down_up"` or `"up_down"`

### Key abstractions

| Interface | Location | Purpose |
|---|---|---|
| `IRgbVideoSource` | `core/video_source/` | Frame source: camera or video file. Mock loops a video file, real uses `cv2.VideoCapture` with DSHOW on Windows. |
| `I2dPoseExtractor` | `core/kp2d_extractor/` | 2D keypoints from BGR frame. Mock reads `.npz`, real wraps RTMDet+RTMPose QNN inference. |
| `I3dPoseReconstructor` | `core/kp3d_reconstructor/` | 3D lifting from 351-frame 2D history using MHFormer. Mock reads `.npz`. |

Mock mode avoids hardware dependency. Use `--analyzer-2d mock --analyzer-3d mock --camera -1` to enable it.

### Model directories

- `rtm-det-aidlite/` — RTMDet + RTMPose QNN/DSP inference (`rtm_vision` module)
- `mhformer-aidlite/` — MHFormer QNN/DSP (`qnn_reconstruct` module)
- `_unused/rtmpose/` — Old ONNX-runtime CPU RTMPose, superseded

### Format conversion

`DataConverter` ([core/converter.py](core/converter.py)) maps COCO-17 ↔ H36M-17. RTMPose outputs COCO-17 `(17,3)`; the rest of the pipeline uses H36M `(17,2)`. 11 joints map directly, 6 interpolated.

### Temporal buffer

MHFormer requires a 351-frame window. `FrameAnalyzer` maintains a `deque(maxlen=351)` of H36M `(17,2)` keypoints and passes the full stack each frame; returns `frame_index=-1` (latest frame).

### Frontend

Two frontends: [static/index.html](static/index.html) (debug/developer UI) and [static/example.html](static/example.html) (AR-style demo UI with blurred background, physiological load panel, workout tracking). Both over a single WebSocket (`/ws`) — binary (JPEG frames) and text (JSON: log/kps3d/stats/alert).

### REST API

| Endpoint | Method | Purpose |
|---|---|---|
| `/poses` | GET/POST | List and switch pose/rule sets |
| `/control/{start,pause,stop}` | POST | Training session lifecycle |
| `/stats/{training_id}` | GET | Stats for a session (`latest` = most recent) |
| `/history` | GET | All training history entries |

### Training state machine (FrameAnalyzer)

Three states (`running` | `paused` | `stopped`), accessed via `state` property:

- **running** — full pipeline: 2D+3D+judge+rep_count. `_active_frames` increments.
- **paused** — 2D+3D continues (skeleton still renders). judge+rep_count skipped. Stats use `_active_frames` only, so accuracy/density don't drift.
- **stopped** — same as paused functionally. Saves snapshot to `_stats_history`, sets `_frozen` dict so all stats properties return final values. `resume()` from stopped resets all counters + rep counter + generates new training_id.

### Rep counting

`RepCounter` ([core/analyzer.py](core/analyzer.py)) is a state machine tracking `UP ↔ DOWN` phases via threshold crossings on the rep feature value. Edge-triggered (not level-triggered) to avoid false counts. Tracks per-rep ROM and timestamps for fatigue estimation.

Rule files in `data/rules/` define thresholds, balance pairs, and `calories_per_rep`. The `rep_counting` block is optional.

### Violation debounce

Violations must persist for `_debounce_frames` (default 30 ≈ 1s at 30fps) before reporting. Same violation ID is reported only once per continuous segment (re-arms when the violation clears and re-enters). Per-frame debounce state tracked via `_v_first_seen` and `_v_reported`.

### Lazy loading

Real QNN models load on first inference call, not at construction. `FrameAnalyzer.__init__` is cheap even with hardware plugins.

### Linux camera troubleshooting

```bash
sudo usermod -aG video $USER   # re-login required
# Quick verify: sudo chmod 666 /dev/video0 then re-run
```
