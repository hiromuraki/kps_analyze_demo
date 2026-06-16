from __future__ import annotations
from typing import Literal
import numpy as np


class I2dPoseExtractor:
    @property
    def data_out(self) -> Literal["COCO17", "H36M"]:
        raise NotImplementedError

    def extract(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError
