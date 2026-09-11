from __future__ import annotations
from pathlib import Path
from typing import Literal
from .interface import I2dPoseExtractor
import numpy as np
import logging

logger = logging.getLogger("kp2d_extractor")


class Mock2dExtractor(I2dPoseExtractor):
    """2D 提取器的 mock 实现：回放预录的 2D 关键点，不做任何推理。

    用于在无 QNN 硬件的环境中验证整条流水线。数据按帧顺序循环回放，
    播完自动从头开始。

    Args:
        preset_2d_coco17_kps_file: 预录 COCO17 关键点 ``.npz`` 路径。取文件中
            的第一个数组，支持 ``(Frames, 17, 3)`` 与 ``(1, Frames, 17, 3)``
            两种形状，后者会自动挤掉 batch 维。
    """

    def __init__(self, preset_2d_coco17_kps_file: Path | str):
        self._kps_npz = np.load(preset_2d_coco17_kps_file)
        raw = self._kps_npz[self._kps_npz.files[0]]
        # 兼容 (Frames, 17, 3) 和 (1, Frames, 17, 3) 两种 shape
        if raw.ndim == 4:
            raw = raw[0]
        self._kps_frames: np.ndarray = raw  # (Frames, 17, 3)
        self._kps_frame_count = self._kps_frames.shape[0]
        self._frame_index = 0
        logger.info(f"Loaded 2D keypoints: {self._kps_frames.shape}")

    @property
    def data_out(self) -> Literal["COCO17", "H36M"]:
        return "COCO17"

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """返回预录序列中的下一帧关键点。

        与接口契约的差异：``frame`` **被完全忽略**——本类不读取图像内容，
        只按内部计数器 ``_frame_index`` 逐帧推进并在末尾循环回绕。

        Returns:
            ``shape=(17, 3)``，内容为预录数据当前帧的原样回放。
        """
        kps = self._kps_frames[self._frame_index]
        self._frame_index = (self._frame_index + 1) % self._kps_frame_count
        return kps
