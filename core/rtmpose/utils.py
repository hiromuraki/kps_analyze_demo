"""预处理、SimCC 解码等工具函数。"""

import cv2
import numpy as np


def preprocess(
    frame: np.ndarray,
    w: int,
    h: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """BGR uint8 (H,W,3) → 归一化 float32 (1,3,H,W)。"""
    if frame.shape[0] != h or frame.shape[1] != w:
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    blob = frame.astype(np.float32)
    blob = (blob - mean) / std
    blob = np.transpose(blob, (2, 0, 1))  # HWC → CHW
    return blob[np.newaxis, ...]


def decode_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray) -> np.ndarray:
    """
    将 SimCC 输出解码为 (17, 3) [x, y, confidence]。

    Args:
        simcc_x: (17, bins) x 坐标分类 logits。
        simcc_y: (17, bins) y 坐标分类 logits。

    Returns:
        (17, 3) 关键点数组。
    """
    x = _soft_expected(simcc_x)
    y = _soft_expected(simcc_y)
    x_conf = _soft_max(simcc_x)
    y_conf = _soft_max(simcc_y)
    conf = np.minimum(x_conf, y_conf)
    return np.stack([x, y, conf], axis=-1)


def _soft_expected(logits: np.ndarray) -> np.ndarray:
    """softmax 后计算期望值（按概率加权平均 bin 索引）。"""
    probs = _softmax(logits)
    bins = np.arange(logits.shape[-1], dtype=np.float32)
    return (probs * bins).sum(axis=-1)


def _soft_max(logits: np.ndarray) -> np.ndarray:
    """softmax 后取最大概率值作为置信度。"""
    return _softmax(logits).max(axis=-1)


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
