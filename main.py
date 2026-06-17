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
    load_rule,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("main")

parser = argparse.ArgumentParser()
parser.add_argument("--analyzer", choices=["mock", "default"], default="default")
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


def frame_analyzer_factory(mode: Literal["mock", "default"], pose_type: str) -> FrameAnalyzer:
    pose_rule = load_rule(pose_type)

    if mode == "mock":
        logger.info("Using Mock FrameAnalyzer with preloaded 2D keypoints and dummy 3D reconstructor")
        return FrameAnalyzer(
            kp2d_extractor=Mock2dExtractor("./sample_data/small/example_2d_h36m_kps.npz"),
            kp3d_reconstructor=Mock3dReconstructor("./sample_data/small/example_3d_kps.npz"),
            pose_name=pose_type,
            pose_rule=pose_rule,
        )
    elif mode == "default":
        logger.info("Using default FrameAnalyzer with RTMPose 2D extractor and MHFormer 3D reconstructor")
        return FrameAnalyzer(
            kp2d_extractor=RTMPose2dPoseExtractor(),
            kp3d_reconstructor=MHFormer3dPoseReconstructor(),
            pose_name=pose_type,
            pose_rule=pose_rule,
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
        logger.error(f"Failed to open video source (camera={args.camera}, video_path={args.video_path})")
        await ws.close(code=1011, reason="Cannot open source")
        return
    frame_analyzer = frame_analyzer_factory(args.analyzer, selected_pose)

    # 进入主循环，捕获、分析并发送视频帧
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
                frame_analyzer = frame_analyzer_factory(args.analyzer, selected_pose)

            # (1) 捕获帧
            t = time.monotonic()
            frame = camera.get_frame()
            total_capture_ms += (time.monotonic() - t) * 1000

            # (2) 分析
            t = time.monotonic()
            rendered_frame, keypoints_3d, violated_rule_id_set = frame_analyzer.analyze_frame(frame)
            total_analysis_ms += (time.monotonic() - t) * 1000

            if violated_rule_id_set:
                msg = json.dumps(
                    {"type": "log", "ts": datetime.now().strftime("%H:%M:%S"), "text": ";".join(violated_rule_id_set)}
                )
                await ws.send_text(msg)

            # 首帧用于诊断
            if frame_count == 0:
                logger.info(f"First frame: shape={frame.shape}, mean_pixel={frame.mean():.1f}")
            frame_count += 1

            # (3) JPEG 编码
            t = time.monotonic()
            _, buffer = cv2.imencode(".jpg", rendered_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            total_encode_ms += (time.monotonic() - t) * 1000

            # (4) 发送视频帧
            t = time.monotonic()
            await ws.send_bytes(buffer.tobytes())
            total_ws_video_ms += (time.monotonic() - t) * 1000

            # (5) 发送 3D 骨骼数据
            t = time.monotonic()
            kps3d_msg = json.dumps({"type": "kps3d", "data": keypoints_3d.tolist()})
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
