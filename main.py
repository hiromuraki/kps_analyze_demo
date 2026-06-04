from __future__ import annotations
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from core.video_source import CameraRgbVideoSource, MockRgbVideoSource, IRgbVideoSource
import argparse
import asyncio
import json
import logging
import random
import time
from datetime import datetime
import cv2
from core import (
    FrameAnalyzer,
    Mock2dExtractor,
    Mock3dReconstructor,
    RTMPose2dPoseExtractor,
    MHFormer3dPoseReconstructor,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("main")

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["camera", "mock"], default="mock")
parser.add_argument("--camera", type=int, default=None, help="Camera device index")
parser.add_argument("--width", type=int, default=640, help="Camera capture width")
parser.add_argument("--height", type=int, default=480, help="Camera capture height")
parser.add_argument("--fps", type=float, default=30.0, help="Camera capture FPS")
parser.add_argument("--video-path", default="./sample_data/video_39.mp4")
args = parser.parse_args()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
SELECTED_POSE_TYPE = "深蹲"
MOCK_MODE = False


def video_source_factory(mode: str) -> IRgbVideoSource:
    def probe_cameras(max_index: int = 8) -> list[int]:
        available = []
        for i in range(max_index + 1):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    camera_id = args.camera
    if camera_id is None and mode == "camera":
        cameras = probe_cameras()
        if cameras:
            logger.info(f"Available cameras: {cameras}")
            camera_id = cameras[0]
        else:
            logger.warning("No camera devices found, falling back to index 0")
            camera_id = 0

    if mode == "camera":
        logger.info(f"Selected camera ID: {camera_id}")
        video_source = CameraRgbVideoSource(camera_id=camera_id, width=args.width, height=args.height, fps=args.fps)
        video_source.flip_x = True  # 前置摄像头通常需要水平翻转
        return video_source
    elif mode == "mock":
        logger.info(f"Using mock video source with video file: {args.video_path}")
        return MockRgbVideoSource(args.video_path)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


async def broadcast_log(ws: WebSocket, stop_event: asyncio.Event):
    MSG_POOL = [
        "一切正常",
        "检测到运动",
        "骨骼追踪中...",
        "帧率稳定",
        "分析中...",
        "光线条件良好",
        "正在处理当前帧",
        "连接稳定",
        "模型已加载",
    ]

    while not stop_event.is_set():
        await asyncio.sleep(random.uniform(2, 5))
        msg = json.dumps(
            {
                "ts": datetime.now().strftime("%H:%M:%S"),
                "text": random.choice(MSG_POOL),
            }
        )
        try:
            await ws.send_text(msg)
        except Exception:
            break


@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info(f"WebSocket client connected from {ws.client}")
    camera = video_source_factory(args.mode)
    if not camera.open():
        logger.error("Failed to open video source")
        await ws.close(code=1011, reason="Cannot open source")
        return

    logger.info(f"Streaming started: {camera.width}x{camera.height}@{camera.fps:.0f}fps")
    frame_count = 0
    stop_event = asyncio.Event()
    log_task = asyncio.create_task(broadcast_log(ws, stop_event))

    if MOCK_MODE:
        fa = FrameAnalyzer(
            kp2d_extractor=RTMPose2dPoseExtractor(),
            kp3d_reconstructor=MHFormer3dPoseReconstructor(),
        )
    else:
        fa = FrameAnalyzer(
            kp2d_extractor=Mock2dExtractor(),
            kp3d_reconstructor=Mock3dReconstructor(),
        )

    try:
        frame_interval = 1.0 / camera.fps
        while camera.is_open():
            t0 = time.monotonic()
            frame = camera.get_frame()
            rendered_frame, violated_rule_id_set = fa.analyze_frame(frame, SELECTED_POSE_TYPE)
            if frame_count == 0:
                logger.info(f"First frame: shape={frame.shape}, mean_pixel={frame.mean():.1f}")
            frame_count += 1
            _, buffer = cv2.imencode(".jpg", rendered_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            await ws.send_bytes(buffer.tobytes())
            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0, frame_interval - elapsed))
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except RuntimeError as e:
        logger.error(f"Runtime error in streaming loop: {e}")
    finally:
        stop_event.set()
        log_task.cancel()
        try:
            await log_task
        except asyncio.CancelledError:
            pass
        logger.info(f"Streaming ended, {frame_count} frames sent")
        camera.release()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
