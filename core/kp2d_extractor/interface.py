from __future__ import annotations
from typing import Literal
import numpy as np


class I2dPoseExtractor:
    """2D 人体姿态提取器的抽象接口。

    实现类从单帧 BGR 图像中提取 17 个人体关键点，同时通过 ``data_out``
    声明输出的关节编号约定，上层据此决定是否需要格式转换。
    """

    @property
    def data_out(self) -> Literal["COCO17", "H36M"]:
        """输出关键点所遵循的关节编号约定。

        - ``"COCO17"``：COCO 17 点约定，与 RTMPose 原生输出一致。
          上层需调用 ``DataConverter.coco17_to_h36m()`` 转换后再送入 3D 重建。
        - ``"H36M"``：H36M 17 点约定，可直接送入 3D 重建器，无需转换。
        """
        raise NotImplementedError

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """从单帧图像中提取人体关键点。

        Args:
            frame: BGR 图像，``shape=(H, W, 3)``，``dtype=uint8``，值域 ``[0, 255]``。

        Returns:
            ``shape=(17, 3)`` 的 ``float32`` 数组，每行为 ``[x, y, confidence]``：
            ``x``/``y`` 为像素坐标，``confidence`` 为 ``[0, 1]`` 区间的置信度。
            关节顺序由 ``data_out`` 决定。未检测到人体时返回全零数组。

        Note:
            多目标场景下返回得分最高者。需要全部检测结果时，由实现类自行扩展
            （例如 ``RTMPose2dPoseExtractor.extract`` 的关键字参数 ``return_all``）。
        """
        raise NotImplementedError
