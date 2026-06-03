import numpy as np


class MHFormer3DReconstructor:
    def __init__(self): ...

    def reconstruct(self, kps2d_seq: np.ndarray, frame_index: int) -> np.ndarray:
        """
        从 2D 关键点序列重建 3D 骨骼点。

        Args:
            kps2d_seq: 2D 关键点序列, shape=(T, 17, 2)，每行 [x, y]。
                T 是提供给 MHFormer 的时间维度长度，最大取决于模型需求，不足自动补齐。

        Returns:
            返回第 frame_index 帧的重建结果
        """
        raise NotImplementedError
