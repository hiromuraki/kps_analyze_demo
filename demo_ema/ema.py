"""
ema.py —— ema.html 的独立辅助服务。

提供三件事：

1. 把同目录下的 ema.html 作为首页返回。
2. 解析前端上传的 3D H36M 骨骼 ``.npz`` 文件（浏览器无法直接解析 numpy 的
   npz 容器格式），返回逐帧 ``[x, y, z]`` 数组。
3. 从视频文件**自动推导** 3D 骨骼：RTMPose 提 2D → MHFormer 升维，全程走本机
   QNN。实测约 85 ms/帧（RTMPose 67 ms + MHFormer 18 ms），2 分钟的视频需要
   两分钟左右，因此走后台线程 + 进度轮询，结果只留在内存，不落盘。

独立端口运行：
    bash run-ema.sh          # 等价于 uv run demo_ema/ema.py
    浏览器打开 http://localhost:28080/
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

# python 只把「脚本所在目录」加入 sys.path，而本文件在 demo_ema/ 下，
# 因此项目根需要显式注入才能 import core。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.converter import DataConverter  # noqa: E402
from core.kp2d_extractor.rtmpose import RTMPose2dPoseExtractor  # noqa: E402
from core.kp3d_reconstructor.mhformer import MHFormer3dPoseReconstructor  # noqa: E402

APP_DIR = Path(__file__).resolve().parent

app = FastAPI()


# 禁用缓存：ema.html 迭代频繁，浏览器启发式缓存曾导致旧版脚本继续运行
# （表现为「改了的交互不生效/旧 bug 复现」），强制每次重新获取。
_NO_CACHE = {"Cache-Control": "no-store"}


@app.get("/")
async def root():
    return FileResponse(APP_DIR / "ema.html", headers=_NO_CACHE)


@app.get("/ema.html")
async def ema_page():
    return FileResponse(APP_DIR / "ema.html", headers=_NO_CACHE)


def _decode_npz(contents: bytes) -> tuple[np.ndarray | None, str | None]:
    """解开 .npz 并取出关键点数组。

    兼容形状（与 main.py:/api/upload_npz 一致）：
    - (frames, 17, 2|3)
    - (1, frames, 17, 2|3)（带 batch 维，自动挤压）

    Returns:
        ``(kps, None)`` 成功；``(None, 错误信息)`` 失败。
    """
    try:
        data = np.load(BytesIO(contents))
    except Exception as e:
        return None, f"无法解析 NPZ 文件: {e}"

    key = "reconstruction" if "reconstruction" in data else list(data.keys())[0]
    kps = data[key]
    if kps.ndim == 4:
        kps = kps.squeeze(0)

    if kps.ndim != 3 or kps.shape[1] != 17 or kps.shape[2] not in (2, 3):
        return None, f"不支持的数据形状: {kps.shape}，期望 17 点骨骼 (frames, 17, 2|3)"
    return kps, None


def _looks_like_2d_pixels(kps: np.ndarray) -> bool:
    """判断关键点是「2D 像素」还是「3D 归一化」。

    两者的数组形状完全一样，只能靠量纲区分：2D 的 x/y 是几百量级的像素坐标、
    第三列是 [0,1] 置信度；H36M 3D 归一化坐标的 x/y 在 ±2 之内。
    只有 x/y 两列时必然是 2D。
    """
    if kps.shape[2] < 3:
        return True
    mx = float(np.abs(kps[:, :, 0]).max())
    my = float(np.abs(kps[:, :, 1]).max())
    mz = float(np.abs(kps[:, :, 2]).max())
    return max(mx, my) > 50 and mz < 1.5


@app.post("/api/parse_kp2d")
async def parse_kp2d(file: UploadFile = File(...)):
    """解析上传的 2D H36M 骨骼 .npz（像素坐标），返回逐帧 ``[x, y, conf]``。

    供**叠加层**使用：像素坐标与视频画面同源，可直接按画布尺寸描点，无需投影。
    关节顺序需为 H36M（与 ``/api/parse_kp3d`` 一致），传 COCO17 文件会导致
    腿部取到错误的关节。
    """
    kps, err = _decode_npz(await file.read())
    if err:
        return {"status": "error", "message": err}
    if not _looks_like_2d_pixels(kps):
        return {
            "status": "error",
            "message": "该文件看起来是 3D 骨骼（归一化坐标）。2D 通道请上传像素坐标的 2D 骨骼文件",
        }
    frames = kps.tolist()
    return {"status": "success", "frames": frames, "num_frames": len(frames)}


@app.post("/api/parse_kp3d")
async def parse_kp3d(file: UploadFile = File(...)):
    """解析上传的 3D H36M 骨骼 .npz（归一化坐标），返回逐帧 ``[x, y, z]``。

    供**特征值计算**使用。前端只取三点夹角，因此坐标系约定（是否做过
    ``to_world`` 旋转）不影响结果。
    """
    kps, err = _decode_npz(await file.read())
    if err:
        return {"status": "error", "message": err}
    if _looks_like_2d_pixels(kps):
        return {
            "status": "error",
            "message": "该文件看起来是 2D 骨骼（x/y 为像素坐标，第三列为置信度）。3D 通道请上传归一化坐标的 3D 骨骼文件",
        }
    frames = kps.tolist()
    return {"status": "success", "frames": frames, "num_frames": len(frames)}


# ---------------------------------------------------------------------------
# 3D 骨骼自动推导（单任务，内存态）
#
# 全局只保留一份任务状态，够用且无需任务队列。多个浏览器标签页共享同一份，
# 后发起的请求会在任务运行期间被拒绝。
# ---------------------------------------------------------------------------

_job_lock = threading.Lock()
_job: dict = {
    "state": "idle",  # idle | running | done | error
    "phase": "",  # "" | "2d" | "3d"
    "done": 0,
    "total": 0,
    "message": "",
    # 推导结果，完成后留给 /api/derive_status 交付。只留在内存，服务重启即失效：
    #   frames_2d: (T, 17, 2) H36M 约定、**像素**坐标 → 叠加层（与视频画面同源，能对齐）
    #   frames_3d: (T, 17, 3) H36M 约定、归一化坐标 → 特征值计算
    "frames_2d": None,
    "frames_3d": None,
}


def _derive_worker(video_path: str) -> None:
    """后台推导：RTMPose 逐帧提 2D → MHFormer 逐帧升维 3D。

    逐帧推进以便 /api/derive_status 能报出真实进度。异常统一写进 _job['message']
    并把 state 置为 error——线程里抛出的异常无法被 FastAPI 捕获，必须自行兜住。
    """
    extractor_2d: RTMPose2dPoseExtractor | None = None
    reconstructor_3d: MHFormer3dPoseReconstructor | None = None
    try:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError("无法打开视频文件（可能缺少解码器）")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _job.update(phase="2d", done=0, total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))

        extractor_2d = RTMPose2dPoseExtractor()
        kps_2d: list[np.ndarray] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            kps_2d.append(extractor_2d.extract(frame))
            _job["done"] = len(kps_2d)
        capture.release()

        if not kps_2d:
            raise RuntimeError("视频中没有可解码的帧")
        # CAP_PROP_FRAME_COUNT 对某些编码不准，以实际解出的帧数为准
        total = len(kps_2d)
        _job.update(phase="3d", done=0, total=total, message="正在重建 3D 骨骼…")

        # 转成 H36M 后顺手留用：它是像素坐标、与视频画面同源，是叠加层唯一能对齐
        # 画面的数据源，存在这里就不必为叠加层再单独跑一遍 RTMPose。
        sequence = DataConverter.coco17_to_h36m(np.stack(kps_2d))  # (T, 17, 2) 像素坐标
        _job["frames_2d"] = sequence.tolist()

        reconstructor_3d = MHFormer3dPoseReconstructor()
        result = np.empty((total, 17, 3), dtype=np.float32)
        for i in range(total):
            # 逐帧调用而非一次性传全量索引，这样每帧都能刷新进度
            result[i] = reconstructor_3d.reconstruct(sequence, [i], (width, height))[0]
            _job["done"] = i + 1

        _job["frames_3d"] = result.tolist()
        _job.update(state="done", phase="", message=f"完成，共 {total} 帧")
    except Exception as e:  # noqa: BLE001 —— 线程内异常必须自行兜住
        _job.update(state="error", phase="", message=f"{type(e).__name__}: {e}")
    finally:
        for model in (extractor_2d, reconstructor_3d):
            if model is not None:
                model.close()
        try:
            os.unlink(video_path)  # cv2 只能读路径，所以落过临时文件，用完即删
        except OSError:
            pass


@app.post("/api/derive_kp3d")
async def derive_kp3d(file: UploadFile = File(...)):
    """启动后台 3D 推导任务，立即返回。进度请轮询 /api/derive_status。"""
    contents = await file.read()

    with _job_lock:
        if _job["state"] == "running":
            return {"status": "error", "message": "已有推导任务正在进行，请稍候"}

        suffix = Path(file.filename or "").suffix or ".mp4"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as fp:
            fp.write(contents)

        _job.update(
            state="running",
            phase="2d",
            done=0,
            total=0,
            message="正在提取 2D 关键点…",
            frames_2d=None,
            frames_3d=None,
        )

    threading.Thread(target=_derive_worker, args=(tmp_path,), daemon=True).start()
    return {"status": "started"}


@app.get("/api/derive_status")
async def derive_status():
    """查询推导进度。

    ``state`` 为 ``done`` 时附带 ``frames_2d``（像素坐标，供叠加层）与
    ``frames_3d``（归一化坐标，供特征值计算），两者结构分别与
    ``/api/parse_kp2d``、``/api/parse_kp3d`` 一致，可直接喂给前端同一套逻辑。
    结果保留在内存中，重复查询会重复返回，直到下一次推导开始才被清空。
    """
    payload = {
        "state": _job["state"],
        "phase": _job["phase"],
        "done": _job["done"],
        "total": _job["total"],
        "message": _job["message"],
    }
    if _job["state"] == "done" and _job["frames_3d"] is not None:
        payload["status"] = "success"
        payload["frames_2d"] = _job["frames_2d"]
        payload["frames_3d"] = _job["frames_3d"]
        payload["num_frames"] = len(_job["frames_3d"])
    return payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=28080)
