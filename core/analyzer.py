from __future__ import annotations
from collections import deque
from collections.abc import Iterable
from .kp2d_extractor import I2dPoseExtractor
from .kp3d_reconstructor import I3dPoseReconstructor
from .renderer import H36M2dKeypointsRenderer
from .converter import DataConverter
from .pose_judger import judge_pose
import numpy as np
import logging

logger = logging.getLogger("analyzer")

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
        # 跳帧缓存
        self._do_heavy = True
        self._cached_2d_h36m: np.ndarray = np.zeros((17, 3), dtype=np.float32)
        self._cached_3d: np.ndarray = np.zeros((17, 3), dtype=np.float32)
        self._cached_violations: list[str] = []
        self._cached_alert_indices: list[int] = []

    @property
    def pose_name(self) -> str:
        return self._pose_name

    def analyze_frame(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
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

        self._do_heavy = not self._do_heavy

        if self._do_heavy:
            # 重型帧：2D 提取 + 3D 重建 + 姿态判定 → 全部更新缓存
            kp2d_coco17 = self._kp2d_extractor.extract(frame)
            if self._kp2d_extractor.data_out == "COCO17":
                kp2d_h36m = DataConverter.coco17_to_h36m(kp2d_coco17)
            else:
                kp2d_h36m = kp2d_coco17
            self._frame_buffer.append(kp2d_h36m)

            kps_3d = self._kp3d_reconstructor.reconstruct(
                np.stack(list(self._frame_buffer)), frame_index=-1
            )
            violated_rule_id_set, affected_keypoints = judge_pose(kps_3d, self._rule)
            alert_kps_2d = _map_kp_names_to_indices(affected_keypoints)

            self._cached_2d_h36m = kp2d_h36m
            self._cached_3d = kps_3d
            self._cached_violations = violated_rule_id_set
            self._cached_alert_indices = alert_kps_2d
        else:
            # 轻量帧：全部复用缓存，仅追加一份 2D 缓存到缓冲区（维持 MHFormer 窗口连续性）
            kp2d_h36m = self._cached_2d_h36m
            self._frame_buffer.append(kp2d_h36m)
            kps_3d = self._cached_3d
            violated_rule_id_set = self._cached_violations
            alert_kps_2d = self._cached_alert_indices

        rendered_frame = H36M2dKeypointsRenderer.render_on_frame(frame, kp2d_h36m, alert_kps_2d)
        return rendered_frame, kps_3d, violated_rule_id_set
