import numpy as np


class IRgbVideoSource:
    @property
    def width(self) -> int:
        raise NotImplementedError

    @property
    def height(self) -> int:
        raise NotImplementedError

    @property
    def fps(self) -> float:
        raise NotImplementedError

    @property
    def flip_x(self) -> bool:
        return False

    @flip_x.setter
    def flip_x(self, value: bool):
        pass

    @property
    def flip_y(self) -> bool:
        return False

    @flip_y.setter
    def flip_y(self, value: bool):
        pass

    def open(self) -> bool:
        raise NotImplementedError

    def get_frame(self) -> np.ndarray:
        raise NotImplementedError

    def is_open(self) -> bool:
        raise NotImplementedError

    def release(self) -> bool:
        raise NotImplementedError
