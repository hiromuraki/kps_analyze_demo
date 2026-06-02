from .IRgbVideoSource import IRgbVideoSource
import logging
import platform
import cv2
import numpy as np

logger = logging.getLogger("CameraRgbVideoSource")

_IS_WINDOWS = platform.system() == "Windows"


class CameraRgbVideoSource(IRgbVideoSource):
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480, fps: float = 30.0):
        self.__camera_id = camera_id
        self.__width = width
        self.__height = height
        self.__fps = fps
        self.__cap = None
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
        logger.info(f"Camera {self.__camera_id}: opening...")
        if _IS_WINDOWS:
            self.__cap = cv2.VideoCapture(self.__camera_id, cv2.CAP_DSHOW)
        else:
            self.__cap = cv2.VideoCapture(self.__camera_id)
        if not self.__cap.isOpened():
            logger.error(f"Camera {self.__camera_id}: cv2.VideoCapture failed to open")
            return False

        # 必须在设置分辨率之前先设 FOURCC，Windows DSHOW 才生效；MSMF 后端不支持此设置
        if _IS_WINDOWS:
            self.__cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # type: ignore CV 自己的类型标注问题，忽略即可
        self.__cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.__width)
        self.__cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.__height)
        self.__cap.set(cv2.CAP_PROP_FPS, self.__fps)

        # 回读实际值
        actual_w = int(self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.__cap.get(cv2.CAP_PROP_FPS)
        self.__width = actual_w
        self.__height = actual_h
        self.__fps = actual_fps if actual_fps > 0 else self.__fps
        logger.info(
            f"Camera {self.__camera_id}: {self.__width}x{self.__height}@{self.__fps:.0f}fps "
            f"(backend={'DSHOW' if _IS_WINDOWS else 'default'})"
        )

        # 试读一帧确认能正常工作
        ret, frame = self.__cap.read()
        if not ret:
            logger.error(f"Camera {self.__camera_id}: test read failed")
            self.__cap.release()
            self.__cap = None
            return False
        logger.info(f"Camera {self.__camera_id}: test read ok, shape={frame.shape}")
        return True

    def get_frame(self) -> np.ndarray:
        ret, frame = self.__cap.read()  # type: ignore CV 自己的类型标注问题，忽略即可
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        return CameraRgbVideoSource._apply_flip(frame, self.__flip_x, self.__flip_y)

    def is_open(self) -> bool:
        return self.__cap is not None and self.__cap.isOpened()

    def release(self) -> bool:
        if self.__cap is not None:
            self.__cap.release()
            self.__cap = None
        return True

    @classmethod
    def _apply_flip(cls, frame: np.ndarray, flip_x: bool, flip_y: bool) -> np.ndarray:
        """按需翻转图像。flip_x=水平, flip_y=垂直。"""
        if flip_x and flip_y:
            return cv2.flip(frame, -1)
        if flip_x:
            return cv2.flip(frame, 1)
        if flip_y:
            return cv2.flip(frame, 0)
        return frame
