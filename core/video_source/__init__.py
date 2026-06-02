import cv2
import numpy as np


def _apply_flip(frame: np.ndarray, flip_x: bool, flip_y: bool) -> np.ndarray:
    """按需翻转图像。flip_x=水平, flip_y=垂直。"""
    if flip_x and flip_y:
        return cv2.flip(frame, -1)
    if flip_x:
        return cv2.flip(frame, 1)
    if flip_y:
        return cv2.flip(frame, 0)
    return frame


from .CameraRgbVideoSource import CameraRgbVideoSource
from .MockRgbVideoSource import MockRgbVideoSource
from .IRgbVideoSource import IRgbVideoSource

__all__ = [
    "IRgbVideoSource",
    "CameraRgbVideoSource",
    "MockRgbVideoSource",
]
