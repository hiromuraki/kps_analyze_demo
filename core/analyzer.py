from __future__ import annotations
from collections import deque
from .kp2d_extractor import I2dPoseExtractor
from .kp3d_reconstructor import I3dPoseReconstructor
from .renderer import H36M2dKeypointsRenderer
from .converter import DataConverter
from .pose_judger import judge_pose
import numpy as np
import logging

logger = logging.getLogger("analyzer")


class FrameAnalyzer:
    def __init__(self, kp2d_extractor: I2dPoseExtractor, kp3d_reconstructor: I3dPoseReconstructor):
        self._kp2d_extractor = kp2d_extractor
        self._kp3d_reconstructor = kp3d_reconstructor
        # 用于存储历史帧的骨骼数据，供 MHFormer 重建 3D 骨骼时作为临近帧使用。
        # 实际只需要 176 帧，不过多的一点消耗影响不大，保持数值一致性更容易理解意图
        self._frame_buffer = deque[np.ndarray](maxlen=351)

    def analyze_frame(self, frame: np.ndarray, pose_type: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
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

        violated_rule_id_set, affected_keypoints = judge_pose(
            kps_3d,
            self._get_rule(pose_type),
        )  # TODO: 根据规则判断姿势，得到告警点索引列表

        # TODO: 反向映射 3D 骨骼点索引到 2D 骨骼点索引，得到需要告警的 2D 骨骼点索引列表
        alert_kps_2d = [11, 12, 13]  # TODO: 替换为反向映射的骨骼节点

        # 渲染结果
        rendered_frame = H36M2dKeypointsRenderer.render_on_frame(frame, kp2d_h36m, alert_kps_2d)

        return rendered_frame, kps_3d, violated_rule_id_set

    def _get_rule(self, pose_type: str) -> dict:
        """
        根据 pose_type 获取对应的姿势判定规则。

        Args:
            pose_type: 姿势类型字符串，如 "深蹲" 等。

        Returns:
            对应 pose_type 的规则字典，结构自定。
        """
        return {}
        raise NotImplementedError
