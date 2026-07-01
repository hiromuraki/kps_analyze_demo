"""
Demo 服务器 —— 与 main.py 相同的管线，默认入口为 example.html。
用法: python example.py --analyzer-2d mock --analyzer-3d mock --camera -1
"""
from __future__ import annotations
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from core.video_source import CameraRgbVideoSource, MockRgbVideoSource, IRgbVideoSource
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
logger = logging.getLogger("example")

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

AVAILABLE_POSES = get_rule_names()
selected_pose: str = AVAILABLE_POSES[0] if AVAILABLE_POSES else ""
_analyzer: FrameAnalyzer | None = None
_camera: IRgbVideoSource | None = None


def video_source_factory(camera_id: int | None) -> IRgbVideoSource:
    if camera_id == -1:
        logger.info(f"Using mock video: {args.video_path}")
        return MockRgbVideoSource(args.video_path)
    if camera_id is None:
        cameras: list[int] = []
        for i in range(8 + 1):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append(i)
                cap.release()
        camera_id = cameras[0] if cameras else 0
    logger.info(f"Opening camera {camera_id}: {args.width}x{args.height}@{args.fps:.0f}fps")
    vs = CameraRgbVideoSource(camera_id=camera_id, width=args.width, height=args.height, fps=args.fps)
    vs.flip_x = True
    return vs


def frame_analyzer_factory(mode_2d: str, mode_3d: str, pose_type: str) -> FrameAnalyzer:
    kp2d = (
        RTMPose2dPoseExtractor() if mode_2d == "rtmpose"
        else Mock2dExtractor("./sample_data/small/example_2d_coco17_kps.npz")
    )
    kp3d = (
        MHFormer3dPoseReconstructor() if mode_3d == "mhformer"
        else Mock3dReconstructor("./sample_data/small/example_3d_kps.npz")
    )
    return FrameAnalyzer(
        kp2d_extractor=kp2d,
        kp3d_reconstructor=kp3d,
        pose_name=pose_type,
        pose_rule=load_rule(pose_type),
    )


@app.get("/poses")
async def get_poses():
    return {"poses": get_rule_names(), "selected": selected_pose}


@app.post("/poses")
async def set_pose(data: dict):
    global selected_pose
    pose = data.get("pose", "")
    rules = get_rule_names()
    if pose not in rules:
        return {"ok": False, "error": f"unknown pose: {pose}"}
    selected_pose = pose
    logger.info(f"Pose → {selected_pose}")
    return {"ok": True, "selected": selected_pose}


@app.post("/control/{action}")
async def control(action: str):
    global _analyzer
    if _analyzer is None:
        return {"ok": False, "error": "no active analyzer"}
    if action == "start":
        _analyzer.resume()
    elif action == "pause":
        _analyzer.pause()
    elif action == "stop":
        _analyzer.stop()
    else:
        return {"ok": False, "error": f"unknown action: {action}"}
    return {"ok": True, "action": action}


@app.get("/")
async def root():
    return RedirectResponse(url="/static/example.html")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info(f"WS client connected from {ws.client}")

    global _analyzer, _camera
    camera = video_source_factory(args.camera)
    if not camera.open():
        await ws.close(code=1011, reason="Cannot open source")
        return
    _camera = camera
    _analyzer = fa = frame_analyzer_factory(args.analyzer_2d, args.analyzer_3d, selected_pose)

    logger.info(f"Streaming: {camera.width}x{camera.height}@{camera.fps:.0f}fps")
    fc = 0

    try:
        fi = 1.0 / camera.fps
        while camera.is_open():
            t0 = time.monotonic()

            if fa.pose_name != selected_pose:
                fa = _analyzer = frame_analyzer_factory(args.analyzer_2d, args.analyzer_3d, selected_pose)

            frame = camera.get_frame()
            result = fa.analyze_frame(frame)

            if result.violations:
                await ws.send_text(json.dumps({
                    "type": "alert",
                    "text": ";".join(result.violations),
                }))

            if result.rep_counted:
                await ws.send_text(json.dumps({
                    "type": "log",
                    "ts": datetime.now().strftime("%H:%M:%S"),
                    "text": f"rep:{fa.rep_count}",
                }))

            if fc % 30 == 0:
                await ws.send_text(json.dumps({
                    "type": "stats",
                    "state": fa.state,
                    "training_id": fa.training_id,
                    "pose_name": fa.pose_name,
                    "accuracy": round(fa.accuracy, 3),
                    "rom": round(fa.rom, 1),
                    "balance_score": round(fa.balance_score, 1),
                    "density": round(fa.density, 1),
                    "calories": round(fa.calories, 1),
                    "total_reps": fa.rep_count,
                    "fatigue_score": round(fa.fatigue_score, 1),
                }))

            fc += 1
            _, buf = cv2.imencode(".jpg", result.rendered, [cv2.IMWRITE_JPEG_QUALITY, 50])
            await ws.send_bytes(buf.tobytes())
            await ws.send_text(json.dumps({"type": "kps3d", "data": result.kps_3d.tolist()}))
            await asyncio.sleep(max(0, fi - (time.monotonic() - t0)))
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception:
        logger.exception("Streaming error")
    finally:
        _camera = None
        camera.release()
        logger.info(f"Streaming ended, {fc} frames")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=28001)
