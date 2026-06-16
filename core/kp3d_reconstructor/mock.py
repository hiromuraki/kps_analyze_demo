from __future__ import annotations
from pathlib import Path
from .interface import I3dPoseReconstructor
import numpy as np
import logging

logger = logging.getLogger("kp3d_reconstructor")


class Mock3dReconstructor(I3dPoseReconstructor):
    def __init__(self, preset_3d_kps_file: Path | str):
        self._kps_npz = np.load(preset_3d_kps_file)
        self._kps_frames: np.ndarray = self._kps_npz[self._kps_npz.files[0]]  # (Frames, 17, 3)
        self._kps_frame_count = self._kps_frames.shape[0]
        self._frame_index = 0
        logger.info(f"Loaded 3D keypoints: {self._kps_frames.shape}")

    @property
    def data_out(self) -> str:
        return "h36m_3d"

    def reconstruct(self, kps2d_seq: np.ndarray, frame_index: int) -> np.ndarray:
        """
        从 2D 关键点序列重建 3D 骨骼点。

        Args:
            kps2d_seq: 2D 关键点序列, shape=(T, 17, 2)，每行 [x, y]。
                T 是提供给 MHFormer 的时间维度长度，最大取决于模型需求，不足自动补齐。

        Returns:
            返回第 frame_index 帧的重建结果
        """
        kps = self._kps_frames[self._frame_index]
        self._frame_index = (self._frame_index + 1) % self._kps_frame_count
        return kps
