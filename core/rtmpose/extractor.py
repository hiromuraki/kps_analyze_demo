from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .constants import DEFAULT_INPUT_SIZE, MEAN, STD
from .utils import decode_simcc, preprocess

logger = logging.getLogger("rtmpose")


class RTMPoseExtractor:
    """
    RTMPose 2D 关键点提取器。

    使用 ONNX Runtime 加载 RTMPose 模型，对输入帧进行预处理、推理、
    SimCC 解码，返回 COCO-17 格式关键点。
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    ):
        self.input_size = input_size
        self._input_w, self._input_h = input_size
        logger.info(f"Loading RTMPose model from {model_path}")
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        logger.info(
            "  input : %s %s",
            self._session.get_inputs()[0].name,
            self._session.get_inputs()[0].shape,
        )
        for out in self._session.get_outputs():
            logger.info("  output: %s %s", out.name, out.shape)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self.extract(frame)

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """
        从 BGR 图像中提取 COCO-17 格式 2D 关键点。

        Args:
            frame: BGR, shape=(H, W, 3), uint8.

        Returns:
            np.ndarray, shape=(17, 3), 每行 [x, y, confidence],
            坐标位于原始图像空间。
        """
        h, w = frame.shape[:2]

        tensor = preprocess(frame, self._input_w, self._input_h, MEAN, STD)
        outputs = self._session.run(None, {self._session.get_inputs()[0].name: tensor})

        if len(outputs) >= 2:
            # SimCC 双头输出 → 解码
            kps = decode_simcc(outputs[0][0], outputs[1][0])
        else:
            # 模型已内置 decode head，直接 reshape
            kps = outputs[0].reshape(-1, 3)

        # 缩放回原始图像坐标
        kps[:, 0] *= w / self._input_w
        kps[:, 1] *= h / self._input_h
        return kps
