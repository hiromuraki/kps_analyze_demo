"""
ema.py —— ema.html 的独立辅助服务。

只做两件事：
1. 把同目录下的 ema.html 作为首页返回。
2. 解析前端上传的 3D H36M 骨骼 .npz 文件（浏览器无法直接解析 numpy 的
   npz 容器格式，这是本文件存在的唯一原因），返回逐帧 [x, y, z] 数组供
   ema.html 做真实数据 EMA 演示。

不依赖 core/，不修改 main.py / example.py，独立端口运行：
    uv run static/ema.py
    浏览器打开 http://localhost:28080/
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

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


@app.post("/api/parse_kp3d")
async def parse_kp3d(file: UploadFile = File(...)):
    """解析上传的 3D H36M 骨骼 .npz 文件，返回逐帧 [x, y, z] 坐标数组。

    兼容形状（与 main.py:/api/upload_npz 一致的兼容口径）：
    - (frames, 17, 3)
    - (1, frames, 17, 3)（带 batch 维，自动挤压）
    """
    try:
        contents = await file.read()
        data = np.load(BytesIO(contents))
    except Exception as e:
        return {"status": "error", "message": f"无法解析 NPZ 文件: {e}"}

    key = "reconstruction" if "reconstruction" in data else list(data.keys())[0]
    kps = data[key]

    if kps.ndim == 4:
        kps = kps.squeeze(0)

    if kps.ndim != 3 or kps.shape[1] != 17 or kps.shape[2] != 3:
        return {
            "status": "error",
            "message": f"不支持的数据形状: {kps.shape}，期望 3D 骨骼 (frames, 17, 3)",
        }

    # 2D 文件（x/y 为像素坐标、第三列为置信度）与 3D 文件形状同为 (frames, 17, 3)，
    # 只能靠量纲区分：像素坐标是几百的量级，H36M 3D 归一化坐标在 ±2 内。
    mx = float(np.abs(kps[:, :, 0]).max())
    my = float(np.abs(kps[:, :, 1]).max())
    mz = float(np.abs(kps[:, :, 2]).max())
    if max(mx, my) > 50 and mz < 1.5:
        return {
            "status": "error",
            "message": "该文件看起来是 2D 骨骼（x/y 为像素坐标，第三列为置信度），请上传 3D 骨骼 (frames, 17, 3)",
        }

    frames = kps.tolist()
    return {"status": "success", "frames": frames, "num_frames": len(frames)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=28080)
