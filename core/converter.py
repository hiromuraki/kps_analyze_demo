import numpy as np


class DataConverter:
    @classmethod
    def coco17_to_h36m(cls, kp_coco17: np.ndarray) -> np.ndarray:
        """
        将 COCO17 格式的 2D 关键点转换为 H36M 格式。

        Args:
            kp_coco17: shape=(17, 3)，每行 [x, y, conf]，对应 COCO17 的 17 个关键点。

        Returns:
            shape=(17, 2)，每行 [x, y]，对应 H36M 的 17 个关键点。
        """
        raise NotImplementedError
