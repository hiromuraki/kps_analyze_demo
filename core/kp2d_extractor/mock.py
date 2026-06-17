from __future__ import annotations
from pathlib import Path
from typing import Literal
from .interface import I2dPoseExtractor
import numpy as np
import logging

logger = logging.getLogger("kp2d_extractor")


class Mock2dExtractor(I2dPoseExtractor):
    def __init__(self, preset_2d_h36m_kps_file: Path | str):
        self._kps_npz = np.load(preset_2d_h36m_kps_file)
        self._kps_frames: np.ndarray = self._kps_npz[self._kps_npz.files[0]]  # (Frames, 17, 3)
        self._kps_frame_count = self._kps_frames.shape[0]
        self._frame_index = 0
        logger.info(f"Loaded 2D keypoints: {self._kps_frames.shape}")

    @property
    def data_out(self) -> Literal["COCO17", "H36M"]:
        return "COCO17"

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """
        从输入帧中提取 2D 关键点。

        Args:
            frame: BGR 图像，shape=(H, W, 3)，dtype=uint8，值域 [0, 255]。

        Returns:
            关键点数组，shape=(17, 2)，每行 [x, y]。
        """

        kps = self._kps_frames[self._frame_index]
        self._frame_index = (self._frame_index + 1) % self._kps_frame_count
        return kps
