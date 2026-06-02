"""RTMPose 2D 人体关键点提取。"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from .constants import COCO_KP_NAMES
from .extractor import RTMPoseExtractor

logger = logging.getLogger("rtmpose")

# 模块级单例
_extractor: RTMPoseExtractor | None = None


def extract_2d_keypoint(
    frame: np.ndarray,
    model_path: str | Path | None = None,
) -> np.ndarray:
    """
    从 BGR 图像中提取 COCO-17 格式的 2D 人体关键点。

    首次调用时加载 ONNX 模型（惰性初始化），后续调用复用。

    Args:
        frame: BGR, shape=(H, W, 3), uint8。由 IRgbVideoSource.get_frame() 返回。
        model_path: RTMPose ONNX 模型路径。为 None 时使用环境变量
                    ``RTMPOSE_MODEL``。

    Returns:
        shape=(17, 3), dtype=float32。每行 [x, y, confidence]，
        坐标位于原始图像空间。关键点顺序见 ``COCO_KP_NAMES``。

    Raises:
        ValueError: 首次调用且 model_path 和 RTMPOSE_MODEL 均未设置时。
    """
    global _extractor
    if _extractor is None:
        path = model_path or os.environ.get("RTMPOSE_MODEL")
        if path is None:
            raise ValueError(
                "model_path is required on first call, or set RTMPOSE_MODEL env var"
            )
        _extractor = RTMPoseExtractor(path)
    return _extractor.extract(frame)


__all__ = ["extract_2d_keypoint", "RTMPoseExtractor", "COCO_KP_NAMES"]
