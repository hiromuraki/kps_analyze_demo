from __future__ import annotations
from typing import Literal
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
    FrameAnalyzer,
    Mock2dExtractor,
    Mock3dReconstructor,
    RTMPose2dPoseExtractor,
    MHFormer3dPoseReconstructor,
    get_rule_names,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("main")

parser = argparse.ArgumentParser()
parser.add_argument("--analyzer", choices=["mock", "default"], default="default")
parser.add_argument("--camera", type=int, default=None, help="Camera device index")
parser.add_argument("--width", type=int, default=640, help="Camera capture width")
parser.add_argument("--height", type=int, default=480, help="Camera capture height")
parser.add_argument("--fps", type=float, default=30.0, help="Camera capture FPS")
parser.add_argument("--video-path", default="./sample_data/video_39.mp4")
args = parser.parse_args()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


def video_source_factory(camera_id: int) -> IRgbVideoSource:
    def probe_cameras(max_index: int = 8) -> list[int]:
        available = []
        for i in range(max_index + 1):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    if camera_id is None or camera_id < 0:
        cameras = probe_cameras()
        if cameras:
            logger.info(f"Available cameras: {cameras}")
            camera_id = cameras[0]
        else:
            logger.warning("No camera devices found, falling back to index 0")
            camera_id = 0

    if camera_id == -1:
        logger.info(f"Using mock video source with video file: {args.video_path}")
        return MockRgbVideoSource(args.video_path)
    elif camera_id >= 0:
        logger.info(f"Selected camera ID: {camera_id}")
        video_source = CameraRgbVideoSource(camera_id=camera_id, width=args.width, height=args.height, fps=args.fps)
        video_source.flip_x = True  # 前置摄像头通常需要水平翻转
        return video_source
    else:
        raise ValueError(f"Unsupported camera index: {camera_id} (should be -1 for mock or >=0 for real camera)")


def frame_analyzer_factory(mode: Literal["mock", "default"]) -> FrameAnalyzer:
    if mode == "mock":
        logger.info("Using Mock FrameAnalyzer with preloaded 2D keypoints and dummy 3D reconstructor")
        return FrameAnalyzer(
            kp2d_extractor=Mock2dExtractor(),
            kp3d_reconstructor=Mock3dReconstructor(),
        )
    elif mode == "default":
        logger.info("Using default FrameAnalyzer with RTMPose 2D extractor and MHFormer 3D reconstructor")
        return FrameAnalyzer(
            kp2d_extractor=RTMPose2dPoseExtractor(),
            kp3d_reconstructor=MHFormer3dPoseReconstructor(),
        )


AVAILABLE_POSES = get_rule_names()
selected_pose: str = AVAILABLE_POSES[0] if AVAILABLE_POSES else ""


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


@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info(f"WebSocket client connected from {ws.client}")

    # 构建视频源和帧分析器
    camera = video_source_factory(args.camera)
    if not camera.open():
        logger.error("Failed to open video source")
        await ws.close(code=1011, reason="Cannot open source")
        return
    frame_analyzer = frame_analyzer_factory(args.analyzer)

    # 进入主循环，捕获、分析并发送视频帧
    logger.info(f"Streaming started: {camera.width}x{camera.height}@{camera.fps:.0f}fps")
    frame_count = 0

    try:
        frame_interval = 1.0 / camera.fps
        while camera.is_open():
            t0 = time.monotonic()

            # 捕获当前帧并进行分析
            frame = camera.get_frame()
            rendered_frame, keypoints_3d, violated_rule_id_set = frame_analyzer.analyze_frame(frame, selected_pose)
            if violated_rule_id_set:
                msg = json.dumps(
                    {"type": "log", "ts": datetime.now().strftime("%H:%M:%S"), "text": ";".join(violated_rule_id_set)}
                )
                await ws.send_text(msg)

            # 首帧用于诊断
            if frame_count == 0:
                logger.info(f"First frame: shape={frame.shape}, mean_pixel={frame.mean():.1f}")
            frame_count += 1

            # 发送视频帧
            _, buffer = cv2.imencode(".jpg", rendered_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            await ws.send_bytes(buffer.tobytes())

            # 发送 3D 骨骼数据
            kps3d_msg = json.dumps({"type": "kps3d", "data": keypoints_3d.tolist()})
            await ws.send_text(kps3d_msg)

            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0, frame_interval - elapsed))
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except RuntimeError as e:
        logger.error(f"Runtime error in streaming loop: {e}")
    finally:
        logger.info(f"Streaming ended, {frame_count} frames sent")
        camera.release()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=2800)
