from __future__ import annotations
from pathlib import Path
from typing import Literal
from .interface import I3dPoseReconstructor
import numpy as np
import logging

logger = logging.getLogger("kp3d_reconstructor")


class Mock3dReconstructor(I3dPoseReconstructor):
    """3D 重建器的 mock 实现：回放预录的 3D 关键点，不依赖 QNN 与 aidlite SDK。

    用于在无 QNN 硬件的环境中验证整条流水线。数据按帧顺序循环回放，
    播完自动从头开始。

    Args:
        preset_3d_kps_file: 预录 3D 关键点 ``.npz`` 路径。取文件中的第一个数组，
            形状须为 ``(Frames, 17, 3)``（H36M 约定）。
    """

    def __init__(self, preset_3d_kps_file: Path | str):
        self._kps_npz = np.load(preset_3d_kps_file)
        self._kps_frames: np.ndarray = self._kps_npz[self._kps_npz.files[0]]  # (Frames, 17, 3)
        self._kps_frame_count = self._kps_frames.shape[0]
        self._frame_index = 0
        logger.info(f"Loaded 3D keypoints: {self._kps_frames.shape}")

    @property
    def data_out(self) -> Literal["H36M_3D"]:
        return "H36M_3D"

    def reconstruct(
        self,
        kps2d_seq: np.ndarray, 
        frame_indices: list[int],
        frame_size: tuple[int, int]
    ) -> np.ndarray:
        """回放当前帧的预录 3D 坐标。

        与接口契约的差异（调用方需知晓）：

        - ``frame_indices`` 与 ``frame_size`` **均被忽略**。本类并不知道视频实际
          播放到第几帧，内部计数器 ``_frame_index`` 是它与循环播放保持对齐的
          唯一依据，因此不能解析索引；``frame_size`` 对回放也没有意义。
        - 返回的 ``N`` 片内容是**同一帧的副本**，仅为满足接口 ``(N, 17, 3)``
          的形状约定。想要不同帧的内容，需分多次调用。

        Returns:
            ``shape=(len(frame_indices), 17, 3)``，各片内容相同。
        """
        kps = self._kps_frames[self._frame_index]
        self._frame_index = (self._frame_index + 1) % self._kps_frame_count
        # 接口约定返回 (N, 17, 3)。frame_indices 不参与解析——本类不知道视频播到第几帧，
        # 内部计数器 _frame_index 才是与循环播放保持对齐的唯一依据；此处仅按请求数量复制当前帧。
        return np.tile(kps, (len(frame_indices), 1, 1))
