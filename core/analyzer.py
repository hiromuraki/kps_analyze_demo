from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from collections.abc import Iterable
from .kp2d_extractor import I2dPoseExtractor
from .kp3d_reconstructor import I3dPoseReconstructor
from .renderer import H36M2dKeypointsRenderer
from .converter import DataConverter
from .pose_judger import judge_pose
from .rep_counter import (
    get_rep_ceiling,
    get_rep_floor,
    get_rep_feature_value,
    get_rep_count_direction,
)
import numpy as np
import logging

logger = logging.getLogger("analyzer")


@dataclass
class AnalysisResult:
    """analyze_frame 的返回结构。"""

    rendered: np.ndarray  # 叠加 2D 骨骼后的 BGR 帧 (H, W, 3)
    kps_3d: np.ndarray  # 3D 骨骼 (17, 3)
    violations: list[str] = field(default_factory=list)  # 违规规则 ID
    rep_counted: bool = False  # 本轮是否完成了一次动作计数


# H36M 关节点名称 → 索引 (0-16)
_H36M_NAME_TO_INDEX: dict[str, int] = {
    "pelvis": 0,
    "right_hip": 1,
    "r_hip": 1,
    "right_knee": 2,
    "r_knee": 2,
    "right_ankle": 3,
    "r_ankle": 3,
    "left_hip": 4,
    "l_hip": 4,
    "left_knee": 5,
    "l_knee": 5,
    "left_ankle": 6,
    "l_ankle": 6,
    "spine": 7,
    "thorax": 8,
    "chest": 8,
    "nose": 9,
    "head": 9,
    "head_top": 10,
    "top": 10,
    "left_shoulder": 11,
    "l_shoulder": 11,
    "left_elbow": 12,
    "l_elbow": 12,
    "left_wrist": 13,
    "l_wrist": 13,
    "right_shoulder": 14,
    "r_shoulder": 14,
    "right_elbow": 15,
    "r_elbow": 15,
    "right_wrist": 16,
    "r_wrist": 16,
}


def _map_kp_names_to_indices(names: Iterable[str]) -> list[int]:
    """将关节点名称转换为 H36M 索引列表 (0-16)，忽略未知名称。"""
    indices = []
    for name in names:
        idx = _H36M_NAME_TO_INDEX.get(name.lower().strip())
        if idx is not None:
            indices.append(idx)
        else:
            logger.warning(f"Unknown keypoint name: '{name}'")
    return indices


class RepPhase(Enum):
    UP = auto()  # 伸展态（站立 / 臂伸直）
    DOWN = auto()  # 收缩态（蹲到底 / 曲臂）


class RepCounter:
    """
    通用动作计数状态机。

    依赖四个外部函数读取规则：
    - get_rep_feature_value(kps_3d, rule) -> float
    - get_rep_ceiling(rule) -> float
    - get_rep_floor(rule) -> float
    - get_rep_count_direction(rule) -> str
    """

    def __init__(self, rule: dict):
        self._rule = rule
        self._ceiling = get_rep_ceiling(rule)
        self._floor = get_rep_floor(rule)
        self._direction = get_rep_count_direction(rule)  # "down_up" | "up_down"
        self._phase = RepPhase.UP
        self._count = 0

    # ------------------------------------------------------------------
    @property
    def count(self) -> int:
        return self._count

    @property
    def phase(self) -> RepPhase:
        return self._phase

    # ------------------------------------------------------------------
    def update(self, kps_3d: np.ndarray) -> bool:
        """
        输入一帧 3D 骨骼，更新状态机。

        Returns:
            True 当本帧完成了一次动作计数时。
        """
        value = get_rep_feature_value(kps_3d, self._rule)
        prev = self._phase

        if value >= self._ceiling:
            self._phase = RepPhase.UP
        elif value <= self._floor:
            self._phase = RepPhase.DOWN

        counted = False
        if self._direction == "down_up":
            if prev == RepPhase.DOWN and self._phase == RepPhase.UP:
                self._count += 1
                counted = True
        else:  # "up_down"
            if prev == RepPhase.UP and self._phase == RepPhase.DOWN:
                self._count += 1
                counted = True

        return counted


class FrameAnalyzer:
    def __init__(
        self,
        kp2d_extractor: I2dPoseExtractor,
        kp3d_reconstructor: I3dPoseReconstructor,
        pose_name: str,
        pose_rule: dict,
    ):
        self._kp2d_extractor = kp2d_extractor
        self._kp3d_reconstructor = kp3d_reconstructor
        self._pose_name = pose_name
        self._rule = pose_rule
        self._frame_buffer = deque[np.ndarray](maxlen=351)
        self._rep_counter: RepCounter | None = None
        if "rep_counting" in pose_rule:
            self._rep_counter = RepCounter(pose_rule)

    @property
    def pose_name(self) -> str:
        return self._pose_name

    @property
    def rep_count(self) -> int:
        """当前动作累计完成次数（无 rep_counting 配置时返回 0）。"""
        return self._rep_counter.count if self._rep_counter is not None else 0

    def analyze_frame(self, frame: np.ndarray) -> AnalysisResult:
        """
        对输入帧进行分析处理。

        Args:
            frame: BGR 图像，shape=(H, W, 3)，dtype=uint8，值域 [0, 255]。
                由 IRgbVideoSource.get_frame() 返回。
            pose_type: 动作类型，指示需要采用什么规则对当前动作进行分析

        Returns:
            0: 进行 2D 骨骼叠加处理后的图像，shape=(H, W, 3)，dtype=uint8。
            1: 3D 骨骼点坐标，shape=(17, 3)，dtype=float32。骨骼点顺序为 H36M 定义的 17 个关键点。
            2: 触发的违反规则的 ID 列表，如 ['R1', 'R2']，用于前端展示告警信息。
        """

        # 分析过程
        # 1. 使用 RTMPose 从帧中提取 2D 骨骼点，输出为 H36M 格式
        # 2. 使用 MHFormer 重建出 3D 骨骼点
        # 3. 根据 3D 骨骼点计算出需要告警的关键点
        # 4. 将 3D 骨骼得出的关键点部位反向映射到 2D 骨骼点上，得到需要告警的 2D 骨骼点索引列表
        # 5. 绘制 2D 骨骼连线到帧上，告警点及其关联骨骼使用高亮颜色，作为返回结果
        kp2d_coco17 = self._kp2d_extractor.extract(frame)
        if self._kp2d_extractor.data_out == "COCO17":
            kp2d_h36m = DataConverter.coco17_to_h36m(kp2d_coco17)
        elif self._kp2d_extractor.data_out == "H36M":
            kp2d_h36m = kp2d_coco17

        self._frame_buffer.append(kp2d_h36m)

        kps_3d = self._kp3d_reconstructor.reconstruct(
            np.stack(list(self._frame_buffer)),
            frame_index=-1,
        )  # out: (17, 3)，取当前帧（-1）的 3D 重建结果

        violations, affected_keypoints = judge_pose(kps_3d, self._rule)

        # 动作计数
        rep_counted = self._rep_counter is not None and self._rep_counter.update(kps_3d)

        # 反向映射：关节点名称 → H36M 索引 (0-16)
        alert_kps_2d = _map_kp_names_to_indices(affected_keypoints)
        logger.debug(f"错误的 3D 关节点：({affected_keypoints}), 对应的 2D 点：{alert_kps_2d}")

        rendered = H36M2dKeypointsRenderer.render_on_frame(frame, kp2d_h36m, alert_kps_2d)

        return AnalysisResult(
            rendered=rendered,
            kps_3d=kps_3d,
            violations=violations,
            rep_counted=rep_counted,
        )
