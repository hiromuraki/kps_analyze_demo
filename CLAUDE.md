# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A Python package named `kps-analyze-demo` (Python >=3.10), managed by uv. Real-time 2D/3D human pose estimation pipeline with WebSocket video streaming to browsers. Multiple frontends: developer debug UI, AR-style demo UI, Rule Builder editor, and EMA/demotion visualization.

## Commands

```bash
# Mock mode (no hardware): mock analyzer with mock video file
uv run main.py --analyzer-2d mock --analyzer-3d mock --camera -1

# Mock with custom NPZ paths
uv run main.py --analyzer-2d mock --analyzer-3d mock --camera -1 \
    --mock-kp2d ./sample_data/example-1/2d_h36m_kps.npz \
    --mock-kp3d ./sample_data/example-1/3d_kps.npz

# Real camera + analyzers (Qualcomm QCS8550 DSP + rtm_vision SDK)
uv run main.py --analyzer-2d rtmpose --analyzer-3d mhformer --camera 0 --width 640 --height 480 --fps 30

# Rule Builder (mock mode + UI for creating rule JSON files)
bash run-rule-builder.sh

# Demo frontend (AR-style UI)
uv run example.py --analyzer-2d mock --analyzer-3d mock --camera -1
```

`main.py` on `0.0.0.0:2800`; `example.py` on `0.0.0.0:28001`.

### Frontend URLs

| URL | Purpose |
|---|---|
| `/` or `/static/index.html` | Debug/developer UI with 3D skeleton, stats, training history. Supports `?pose=<name>` query param for initial pose selection. |
| `/static/example.html` | AR-style demo with blurred video background, physiological load, workout tracking |
| `/static/rule-builder.html` | Visual rule builder: video + 2D/3D skeleton, SVG bone selector, frame-accurate rule capture, JSON export |
| `/static/ema.html` | EMA smoothing + motion debounce visualization (standalone demo, no server dependency) |

## Architecture

### Core pipeline

```
Camera/Video → 2D Pose Extract (COCO17) → COCO→H36M convert → 3D Reconstruct → Judge → RepCount → Render → WebSocket → Browser
```

### Key abstractions

| Interface | Location | Purpose |
|---|---|---|
| `IRgbVideoSource` | `core/video_source/` | Frame source. Mock loops a video file; real uses `cv2.VideoCapture` (DSHOW on Windows). Uses `--mock-kp2d` / `--mock-kp3d` for custom NPZ paths. |
| `I2dPoseExtractor` | `core/kp2d_extractor/` | 2D keypoints from BGR frame. Mock reads `.npz`; real wraps RTMDet+RTMPose QNN. |
| `I3dPoseReconstructor` | `core/kp3d_reconstructor/` | 3D lifting from 351-frame 2D history via MHFormer. Mock reads `.npz`. |

### FrameAnalyzer ([core/analyzer.py](core/analyzer.py))

Orchestrates pipeline via dependency injection. Key aspects:

- **`analyzer_id`**: UUID4 generated at construction, logged alongside RepCounter params.
- **State machine**: `running` → full pipeline; `paused` → 2D/3D continue, judge+rep_count skipped; `stopped` → snapshot to `_stats_history`, frozen dict with final values. `resume()` from stopped resets everything + new `training_id`.
- **Violation debounce**: 30-frame persistence before reporting; each violation ID reported once per continuous segment (re-arms on clear/re-enter). Tracked via `_v_first_seen` and `_v_reported`.
- **Returns** `AnalysisResult` dataclass: `rendered`, `kps_3d`, `violations`, `rep_counted: bool`, `motion: str` (descent/ascent/static_up/static_down).

### RepCounter ([core/rep_counter.py](core/rep_counter.py))

State machine for exercise rep counting, separate from FrameAnalyzer (imported directly).

**Motion detection** (two-layer filtering):
1. **EMA smoothing**: Raw feature value smoothed via `ema_alpha` (default 0.25). `smooth = α × raw + (1-α) × prev`.
2. **Direction debounce**: Smoothed delta crosses `delta_threshold` (±1.0 default) → candidate direction. Must persist `motion_debounce` frames (default 4) before switching output motion.

**RepMotion enum**: `DESCENT`, `ASCENT`, `STATIC_UP`, `STATIC_DOWN`. The UP/DOWN distinction on static tells `judge_pose` whether the user is at the top or bottom of the movement.

**Key parameters** (exposed as properties): `ema_alpha`, `motion_debounce`, `delta_threshold`. `delta_threshold` can be set per-exercise via `rep_counting.delta_threshold` in rule JSON.

**Extension points** — four external functions in [core/rep_counter.py](core/rep_counter.py):
- `get_rep_feature_value(kps_3d, rule) -> float`
- `get_rep_ceiling(rule) -> float` / `get_rep_floor(rule) -> float`
- `get_rep_count_direction(rule) -> str` — `"down_up"` or `"up_down"`

### Pose Judger ([core/pose_judger.py](core/pose_judger.py))

Signature: `judge_pose(kp2d, kp3d, rule, motion="") -> (violated_rule_ids, affected_keypoints)`

Key features:
- **Phase filtering**: Rules with `phase` field only trigger when `motion in phase` matches. Rules without `phase` (or `phase: ["all"]`) always trigger.
- **`keypoints_mode`**: Per-rule field (`"2d"` or `"3d"`) selects which coordinate source to use for angle computation.
- **`parts` structure**: Rules can group multiple angle checks under a single rule_no via `parts: [{p1,p2,p3,p4,min_value,max_value}, ...]`. The judger flattens these before processing.
- **`rule_set` vs `rule_list`**: New format uses `rule_set`; judger reads both for backward compatibility.
- **Rule types**: `"夹角"` (4-point segment angle) and `"水平夹角"` (2-point horizontal tilt, e.g., shoulder level).
- **Grouping**: Rules with identical segment keys `(p1,p2,p3,p4)` are grouped; ANY range match = posture OK; ALL fail = violation. Phase-filtered per group.

### Rule JSON format

New format (recommended):

```json
{
    "action_no": "04",
    "action_name": "哈克深蹲",
    "rule_set": [
        {
            "rule_no": "R1",
            "phase": ["static_down"],
            "keypoints_mode": "3d",
            "parts": [
                {"rule_type": "夹角", "p1": "左髋", "p2": "左膝", "p3": "左膝", "p4": "左脚踝",
                 "min_value": 90, "max_value": 115}
            ]
        }
    ],
    "rep_counting": {
        "type": "angle",
        "p1": "左髋", "p2": "左膝", "p3": "左膝", "p4": "左脚踝",
        "top_threshold": 160, "bottom_threshold": 115,
        "delta_threshold": 0.3,
        "count_on": "down_up"
    },
    "calories_per_rep": 0.5,
    "balance_pairs": [["左膝","右膝"], ["左髋","右髋"]]
}
```

Old format: `rule_list` (array of flat rule objects) instead of `rule_set`, no `phase`/`keypoints_mode`/`parts`. Both are supported.

`phase` is always a list: `["all"]`, `["static_down"]`, `["static_up", "descent"]`, etc.

### REST API

| Endpoint | Method | Purpose |
|---|---|---|
| `/poses` | GET/POST | List and switch pose/rule sets |
| `/control/{start,pause,stop}` | POST | Training session lifecycle |
| `/stats/{training_id}` | GET | Stats for a session (`latest` = most recent) |
| `/history` | GET | All training history entries |
| `/api/upload_npz` | POST | Multipart .npz upload → `{frames, bones_topology, num_frames}` (for Rule Builder) |

### WebSocket `/ws`

Binary: JPEG frames (cv2.imencode, quality=50). Text: JSON messages — `"stats"` (every 30 frames: state, accuracy, rom, calories, reps, etc.), `"kps3d"` (every frame: 3D keypoints + feature_value + motion), `"log"` (violations, rep counts), `"alert"` (demo UI).

### Rule Builder ([static/rule-builder.html](static/rule-builder.html))

Visual tool for creating rule JSON files:
- Upload video + 2D/3D NPZ → frame-synced playback via rAF loop
- SVG A-pose skeleton selector (click bones to toggle, color-coded palette)
- Adjacent joint angle display (floating panel, updates with selected bones)
- Frame-by-frame navigation (keyboard arrows, long-press step buttons)
- 2D skeleton overlay on video (pixel-coordinate 1:1); 3D skeleton in split-pane (OrbitControls, per-bone coloring based on selection)
- 3D pane layout adapts to video aspect ratio (horizontal=side-by-side, vertical=top-bottom) with draggable grip
- Tabbed right panel: "规则集" tab (rule list + capture button), "元数据" tab (action_name, calories_per_rep, export)
- Export: UUID4 `action_no`, sequential R1/R2 `rule_no`, `rule_set` format JSON

### index.html pose switching

Pose menu items navigate to `/static/index.html?pose=<name>` for full page refresh. On load, reads `?pose=`, POSTs to `/poses`, then opens WebSocket. This avoids sync issues in mock mode where FrameAnalyzer rebuild doesn't reset the video source. Query param stays in URL for visibility.

### Format conversion

`DataConverter.coco17_to_h36m()`: COCO-17 `(17,3)` → H36M `(17,2)` (drops confidence). 11 joints map directly, 6 interpolated.

### Temporal buffer

MHFormer needs 351-frame window. `FrameAnalyzer` maintains `deque(maxlen=351)` of H36M `(17,2)` keypoints; passes full stack with `frame_index=-1` (latest frame).

### Model directories

- `rtm-det-aidlite/` — RTMDet + RTMPose QNN/DSP inference (`rtm_vision` module)
- `mhformer-aidlite/` — MHFormer QNN/DSP (`qnn_reconstruct` module)
- `_unused/rtmpose/` — Old ONNX-runtime CPU RTMPose, superseded

### Lazy loading

Real QNN models load on first inference call, not at construction. `FrameAnalyzer.__init__` is cheap even with hardware plugins.
