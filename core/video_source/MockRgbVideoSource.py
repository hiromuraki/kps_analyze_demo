from .IRgbVideoSource import IRgbVideoSource
from . import _apply_flip
import cv2
import numpy as np


class MockRgbVideoSource(IRgbVideoSource):
    def __init__(self, video_path: str):
        self._video_path = video_path
        self._cap = None
        self.__width = 0
        self.__height = 0
        self.__fps = 0.0
        self.__flip_x = False
        self.__flip_y = False

    @property
    def width(self) -> int:
        return self.__width

    @property
    def height(self) -> int:
        return self.__height

    @property
    def fps(self) -> float:
        return self.__fps

    @property
    def flip_x(self) -> bool:
        return self.__flip_x

    @flip_x.setter
    def flip_x(self, value: bool):
        self.__flip_x = value

    @property
    def flip_y(self) -> bool:
        return self.__flip_y

    @flip_y.setter
    def flip_y(self, value: bool):
        self.__flip_y = value

    def open(self) -> bool:
        import logging
        logger = logging.getLogger("MockRgbVideoSource")
        logger.info(f"Opening video file: {self._video_path}")
        self._cap = cv2.VideoCapture(self._video_path)
        if not self._cap.isOpened():
            logger.error(f"Failed to open video file: {self._video_path}")
            return False
        self.__width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.__height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.__fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self.__fps <= 0:
            self.__fps = 30.0
        # 试读一帧确认解码器正常（生产机可能缺 ffmpeg）
        ret, _ = self._cap.read()
        if not ret:
            logger.error(f"Failed to decode first frame: {self._video_path}")
            self._cap.release()
            self._cap = None
            return False
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        logger.info(f"Opened: {self.__width}x{self.__height}@{self.__fps:.0f}fps")
        return True

    def get_frame(self) -> np.ndarray:
        ret, frame = self._cap.read()  # type: ignore CV 自己的类型标注问题，忽略即可
        if not ret:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # type: ignore CV 自己的类型标注问题，忽略即可
            ret, frame = self._cap.read()  # type: ignore CV 自己的类型标注问题，忽略即可
            if not ret:
                raise RuntimeError("Failed to read frame from video")
        return _apply_flip(frame, self.__flip_x, self.__flip_y)

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> bool:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        return True
