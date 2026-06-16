from __future__ import annotations
import numpy as np


class I3dPoseReconstructor:
    @property
    def data_out(self) -> str:
        raise NotImplementedError

    def reconstruct(self, kps2d_seq: np.ndarray, frame_index: int) -> np.ndarray:
        raise NotImplementedError
