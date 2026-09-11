from __future__ import annotations
from typing import Literal
import numpy as np


class I3dPoseReconstructor:
    """3D 人体姿态重建器的抽象接口。

    实现类接收 H36M 格式的 2D 关键点序列，重建出同一序列的 3D 坐标。
    """

    @property
    def data_out(self) -> Literal["H36M_3D"]:
        """输出 3D 坐标的格式标识，恒为 ``"H36M_3D"``。

        关节顺序遵循 H36M 17 点约定（见 ``core.converter``），坐标为归一化
        空间坐标而非像素坐标。
        """
        raise NotImplementedError

    def reconstruct(
        self,
        kps2d_seq: np.ndarray, 
        frame_indices: list[int],
        frame_size: tuple[int, int]
    ) -> np.ndarray:
        """从 2D 关键点序列重建指定帧的 3D 坐标。

        Args:
            kps2d_seq: H36M 格式的 2D 关键点序列，``shape=(T, 17, 2)``，
                每行为 ``[x, y]`` 像素坐标。``T`` 为可用帧数。
            frame_indices: 需要重建的帧索引列表，支持负索引
                （``-1`` 表示最后一帧、``-2`` 表示倒数第二帧，依此类推）。
                由实现类负责把负索引解析为绝对索引。
            frame_size: ``kps2d_seq`` 的像素坐标**所在的图像尺寸** ``(width, height)``。
                必须填写真实尺寸：它直接参与 screen→NDC 归一化，填错会静默
                产生错误的 3D 结果，而不会报错。

        Returns:
            ``shape=(N, 17, 3)`` 的数组，``N == len(frame_indices)``，
            第 ``i`` 片对应 ``frame_indices[i]``。返回值恒为三阶，
            即使只请求一帧也不会降为 ``(17, 3)``。

        Raises:
            ValueError: ``kps2d_seq`` 形状不是 ``(T, 17, 2)``。
            IndexError: 某个索引（负索引解析后）超出 ``[0, T)`` 范围。
        """
        raise NotImplementedError
