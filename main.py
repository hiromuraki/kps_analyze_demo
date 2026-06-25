from __future__ import annotations
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from core.video_source import CameraRgbVideoSource, MockRgbVideoSource, IRgbVideoSource
from collections import deque
from datetime import datetime
import argparse
import asyncio
import json
import logging
import time
import cv2
from core import (
    AnalysisResult,
    FrameAnalyzer,
    Mock2dExtractor,
    Mock3dReconstructor,
    RTMPose2dPoseExtractor,
    MHFormer3dPoseReconstructor,
    get_rule_names,
    load_rule,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("main")

parser = argparse.ArgumentParser()
parser.add_argument("--analyzer-2d", choices=["mock", "rtmpose"], default="rtmpose")
parser.add_argument("--analyzer-3d", choices=["mock", "mhformer"], default="mhformer")
parser.add_argument("--camera", type=int, default=None, help="Camera device index")
parser.add_argument("--width", type=int, default=640, help="Camera capture width")
parser.add_argument("--height", type=int, default=480, help="Camera capture height")
parser.add_argument("--fps", type=float, default=30.0, help="Camera capture FPS")
parser.add_argument("--video-path", default="./sample_data/small/example.mp4")
args = parser.parse_args()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


def video_source_factory(camera_id: int | None) -> IRgbVideoSource:
    """根据 camera_id 创建视频源。None=自动探测, -1=mock视频文件, >=0=真实摄像头。"""
    if camera_id == -1:
        logger.info(f"Using mock video source with video file: {args.video_path}")
        return MockRgbVideoSource(args.video_path)

    if camera_id is None:
        cameras: list[int] = []
        for i in range(8 + 1):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append(i)
                cap.release()
        if cameras:
            logger.info(f"Available cameras: {cameras}")
            camera_id = cameras[0]
        else:
            logger.warning("No camera devices found, falling back to index 0")
            camera_id = 0

    logger.info(f"Opening camera {camera_id}: {args.width}x{args.height}@{args.fps:.0f}fps")
    video_source = CameraRgbVideoSource(camera_id=camera_id, width=args.width, height=args.height, fps=args.fps)
    video_source.flip_x = True
    return video_source


def frame_analyzer_factory(mode_2d: str, mode_3d: str, pose_type: str) -> FrameAnalyzer:
    pose_rule = load_rule(pose_type)

    kp2d = (
        RTMPose2dPoseExtractor() if mode_2d == "rtmpose"
        else Mock2dExtractor("./sample_data/small/example_2d_coco17_kps.npz")
    )
    kp3d = (
        MHFormer3dPoseReconstructor() if mode_3d == "mhformer"
        else Mock3dReconstructor("./sample_data/small/example_3d_kps.npz")
    )
    logger.info(f"FrameAnalyzer: 2D={mode_2d}, 3D={mode_3d}, pose={pose_type}")

    return FrameAnalyzer(
        kp2d_extractor=kp2d,
        kp3d_reconstructor=kp3d,
        pose_name=pose_type,
        pose_rule=pose_rule,
    )


AVAILABLE_POSES = get_rule_names()
selected_pose: str = AVAILABLE_POSES[0] if AVAILABLE_POSES else ""
_analyzer: FrameAnalyzer | None = None
_camera: IRgbVideoSource | None = None
_training_history: deque[dict] = deque(maxlen=64)  # 模块级持久化


@app.get("/poses")
async def get_poses():
    AVAILABLE_POSES = get_rule_names()
    return {"poses": AVAILABLE_POSES, "selected": selected_pose}


@app.post("/poses")
async def set_pose(data: dict):
    global selected_pose
    pose = data.get("pose", "")
    if pose not in AVAILABLE_POSES:
        return {"ok": False, "error": f"unknown pose: {pose}"}
    selected_pose = pose
    logger.info(f"Pose changed to: {selected_pose}")
    return {"ok": True, "selected": selected_pose}


@app.get("/history")
async def get_history():
    """返回最近的训练历史列表。"""
    return {"ok": True, "data": list(_training_history)}


@app.get("/stats/{training_id}")
async def get_stats(training_id: str):
    """获取训练统计。training_id='latest' 时返回最近一次。"""
    if training_id == "latest":
        if not _training_history:
            return {"ok": False, "error": "no training history"}
        return {"ok": True, "data": _training_history[-1]}
    for s in _training_history:
        if s["training_id"] == training_id:
            return {"ok": True, "data": s}
    return {"ok": False, "error": f"training_id '{training_id}' not found"}


@app.post("/control/{action}")
async def control(action: str):
    """训练控制：start / pause / stop。"""
    global _analyzer, _camera
    if _analyzer is None:
        return {"ok": False, "error": "no active analyzer"}
    if action == "start":
        _analyzer.resume()
    elif action == "pause":
        _analyzer.pause()
    elif action == "stop":
        _analyzer.stop()
        hist = _analyzer.stats_history
        if hist:
            _training_history.append(hist[-1])
            logger.info(f"Training saved to history (total: {len(_training_history)})")
    else:
        return {"ok": False, "error": f"unknown action: {action}"}
    return {"ok": True, "action": action}


@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info(f"WebSocket client connected from {ws.client}")

    # 构建视频源和帧分析器
    global _analyzer, _camera
    camera = video_source_factory(args.camera)
    if not camera.open():
        logger.error(f"Failed to open video source (camera={args.camera}, video_path={args.video_path})")
        await ws.close(code=1011, reason="Cannot open source")
        return
    _camera = camera
    _analyzer = frame_analyzer = frame_analyzer_factory(args.analyzer_2d, args.analyzer_3d, selected_pose)

    # 进入主循环
    logger.info(f"Streaming started: {camera.width}x{camera.height}@{camera.fps:.0f}fps")
    frame_count = 0
    wall_start = time.monotonic()
    total_capture_ms = 0.0
    total_analysis_ms = 0.0
    total_encode_ms = 0.0
    total_ws_video_ms = 0.0
    total_ws_3d_ms = 0.0

    try:
        frame_interval = 1.0 / camera.fps
        while camera.is_open():
            t_frame = time.monotonic()

            # 检测动作切换，重建 FrameAnalyzer
            if frame_analyzer.pose_name != selected_pose:
                frame_analyzer = frame_analyzer_factory(args.analyzer_2d, args.analyzer_3d, selected_pose)

            # (1) 捕获帧
            t = time.monotonic()
            frame = camera.get_frame()
            total_capture_ms += (time.monotonic() - t) * 1000

            # (2) 分析
            t = time.monotonic()
            result = frame_analyzer.analyze_frame(frame)
            total_analysis_ms += (time.monotonic() - t) * 1000

            if result.violations:
                msg = json.dumps(
                    {"type": "log", "ts": datetime.now().strftime("%H:%M:%S"), "text": ";".join(result.violations)}
                )
                await ws.send_text(msg)

            if result.rep_counted:
                rep_msg = json.dumps(
                    {"type": "log", "ts": datetime.now().strftime("%H:%M:%S"),
                     "text": f"rep:{frame_analyzer.rep_count}"}
                )
                await ws.send_text(rep_msg)

            # 每 30 帧推送统计数据
            if frame_count % 30 == 0:
                stats_msg = json.dumps({
                    "type": "stats",
                    "state": frame_analyzer.state,
                    "training_id": frame_analyzer.training_id,
                    "accuracy": round(frame_analyzer.accuracy, 3),
                    "rom": round(frame_analyzer.rom, 1),
                    "balance_score": round(frame_analyzer.balance_score, 1),
                    "density": round(frame_analyzer.density, 1),
                    "calories": round(frame_analyzer.calories, 1),
                    "total_reps": frame_analyzer.rep_count,
                    "fatigue_score": round(frame_analyzer.fatigue_score, 1),
                })
                await ws.send_text(stats_msg)

            # 首帧用于诊断
            if frame_count == 0:
                logger.info(f"First frame: shape={frame.shape}, mean_pixel={frame.mean():.1f}")
            frame_count += 1

            # (3) JPEG 编码
            t = time.monotonic()
            _, buffer = cv2.imencode(".jpg", result.rendered, [cv2.IMWRITE_JPEG_QUALITY, 50])
            total_encode_ms += (time.monotonic() - t) * 1000

            # (4) 发送视频帧
            t = time.monotonic()
            await ws.send_bytes(buffer.tobytes())
            total_ws_video_ms += (time.monotonic() - t) * 1000

            # (5) 发送 3D 骨骼数据
            t = time.monotonic()
            kps3d_msg = json.dumps({"type": "kps3d", "data": result.kps_3d.tolist()})
            await ws.send_text(kps3d_msg)
            total_ws_3d_ms += (time.monotonic() - t) * 1000

            await asyncio.sleep(max(0, frame_interval - (time.monotonic() - t_frame)))
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except RuntimeError as e:
        logger.error(f"Runtime error in streaming loop: {e}")
    except Exception:
        logger.exception("Unexpected error in streaming loop")
    finally:
        wall_elapsed = time.monotonic() - wall_start
        _camera = None
        camera.release()

        if frame_count == 0:
            logger.info("No frames processed")
            return

        avg_total   = (wall_elapsed / frame_count) * 1000
        avg_cap     = total_capture_ms / frame_count
        avg_ana     = total_analysis_ms / frame_count
        avg_enc     = total_encode_ms / frame_count
        avg_ws_vid  = total_ws_video_ms / frame_count
        avg_ws_3d   = total_ws_3d_ms / frame_count
        avg_other   = avg_total - (avg_cap + avg_ana + avg_enc + avg_ws_vid + avg_ws_3d)

        logger.info("=" * 60)
        logger.info("  Per‑frame Timing Summary")
        logger.info("=" * 60)
        logger.info(f"  Total wall time                {wall_elapsed:8.2f} s")
        logger.info(f"  Total frames                   {frame_count:8d}")
        logger.info(f"  Effective FPS                  {frame_count / wall_elapsed:8.1f}")
        logger.info("-" * 60)
        logger.info(f"  Avg capture        (get_frame)  {avg_cap:8.2f} ms")
        logger.info(f"  Avg analysis       (analyze)    {avg_ana:8.2f} ms")
        logger.info(f"  Avg JPEG encode    (imencode)   {avg_enc:8.2f} ms")
        logger.info(f"  Avg WS send        (video)      {avg_ws_vid:8.2f} ms")
        logger.info(f"  Avg WS send        (3D data)    {avg_ws_3d:8.2f} ms")
        logger.info(f"  Avg other          (sleep/etc)  {avg_other:8.2f} ms")
        logger.info("-" * 60)
        logger.info(f"  Avg per frame (total wall)      {avg_total:8.2f} ms")
        logger.info("=" * 60)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=2800)
