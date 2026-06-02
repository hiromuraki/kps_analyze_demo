"""COCO-17 关键点格式常量和 RTMPose 预处理参数。"""

import numpy as np

#: COCO-17 关键点名称，按索引顺序
COCO_KP_NAMES = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

# RTMPose 默认输入尺寸 (width, height)
DEFAULT_INPUT_SIZE = (256, 192)

# ImageNet 均值 / 标准差（BGR 通道顺序）
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
