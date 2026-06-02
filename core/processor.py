from __future__ import annotations
import numpy as np
import logging
import cv2

logger = logging.getLogger("core")


class H36MKeypointsRenderer:
    # 默认颜色 —— 中性、低辨识度，不分散注意力
    _KEYPOINT_COLOR = (128, 128, 128)  # 灰色点
    _SKELETON_COLOR = (100, 100, 100)  # 暗灰线
    # 告警颜色 —— 高辨识度，引人关注
    _ALERT_KEYPOINT_COLOR = (0, 255, 0)  # 亮绿点
    _ALERT_SKELETON_COLOR = (0, 0, 255)  # 亮红线

    _CIRCLE_RADIUS = 4
    _ALERT_CIRCLE_RADIUS = 6
    _LINE_THICKNESS = 1
    _ALERT_LINE_THICKNESS = 2

    # Human3.6M 17 点骨骼连接
    #
    #                10 (头顶)
    #                 |
    #                9 (鼻子)
    #                 |
    #                 8 (胸腔/颈底)
    #                /|          \
    #              /  |            \
    #     11 (左肩)  |          14 (右肩)
    #       |        |                |
    #   12 (左肘)    |            15 (右肘)
    #       |        |                |
    #   13 (左腕)    |            16 (右腕)
    #                |
    #              7 (脊椎)
    #                |
    #              0 (骨盆)
    #             /   \
    #           /       \
    #      4 (左髋)   1 (右髋)
    #        |           |
    #      5 (左膝)   2 (右膝)
    #        |           |
    #      6 (左踝)   3 (右踝)
    #
    _H36M_POSE_CONNECTIONS = [
        # 右腿: 0→1→2→3
        (0, 1),
        (1, 2),
        (2, 3),
        # 左腿: 0→4→5→6
        (0, 4),
        (4, 5),
        (5, 6),
        # 脊椎 + 头部: 0→7→8→9→10
        (0, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        # 右臂: 8→14→15→16
        (8, 14),
        (14, 15),
        (15, 16),
        # 左臂: 8→11→12→13
        (8, 11),
        (11, 12),
        (12, 13),
    ]

    @classmethod
    def render_on_frame(cls, frame: np.ndarray, kp2d: np.ndarray, alert_kps: list[int]) -> np.ndarray:
        """
        将 H36M 格式 2D 关键点及骨骼连线渲染到帧上。

        默认使用中性低辨识度颜色，alert_kps 中指定的关键点及其关联的
        骨骼连线使用高辨识度颜色突出显示。

        Args:
            frame: BGR 图像, shape=(H, W, 3), uint8。
            kp2d: 关键点, shape=(17, 2) 或 (17, 3), 每行 [x, y] 或 [x, y, conf]。
            alert_kps: 需要高亮的关键点索引列表，索引范围 [0, 16]。
                None 或空列表表示不高亮任何点。

        Returns:
            渲染后的 BGR 图像 (与输入 frame 同 shape)。
        """
        out = frame.copy()
        alert_set: set[int] = set(alert_kps or ())

        # 构建可见点集合，跳过 (0, 0) 原点
        visible: dict[int, tuple[int, int]] = {}
        for idx, pt in enumerate(kp2d):
            x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
            if x == 0 and y == 0:
                continue
            visible[idx] = (x, y)
            is_alert = idx in alert_set
            radius = cls._ALERT_CIRCLE_RADIUS if is_alert else cls._CIRCLE_RADIUS
            color = cls._ALERT_KEYPOINT_COLOR if is_alert else cls._KEYPOINT_COLOR
            cv2.circle(out, (x, y), radius, color, -1)

        # 骨骼连线：两端都在 alert_kps 中才高亮
        for a, b in cls._H36M_POSE_CONNECTIONS:
            if a not in visible or b not in visible:
                continue
            is_alert = a in alert_set and b in alert_set
            thickness = cls._ALERT_LINE_THICKNESS if is_alert else cls._LINE_THICKNESS
            color = cls._ALERT_SKELETON_COLOR if is_alert else cls._SKELETON_COLOR
            cv2.line(out, visible[a], visible[b], color, thickness)

        return out


class H36MKeypointsExtractor:
    def __init__(self):
        self._kps_npz = np.load("sample_data/example_2d_h36m_kps.npz")
        self._kps_frames: np.ndarray = self._kps_npz[self._kps_npz.files[0]]  # (Frames, 17, 3)
        self._kps_frame_count = self._kps_frames.shape[0]
        self._frame_index = 0
        logger.info(f"Loaded 2D keypoints: {self._kps_frames.shape}")

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """
        从输入帧中提取 2D 关键点。

        Args:
            frame: BGR 图像，shape=(H, W, 3)，dtype=uint8，值域 [0, 255]。

        Returns:
            关键点数组，shape=(17, 3)，每行 [x, y, conf]。
        """

        kps = self._kps_frames[self._frame_index % self._kps_frame_count]
        self._frame_index += 1
        return kps


def judge_pose(kp3d: np.ndarray, rule: dict) -> tuple[tuple[str], tuple[str]]:
    """
    根据规则对 3D 骨骼姿势进行判定，返回触发的规则 ID 和涉及的关节点。

    从 3D 关键点的空间位置出发，检查各关节角度、相对位置等几何关系
    是否在规则定义的正常范围内，超出阈值则触发告警。

    Args:
        kp3d: HMFormer 的 3D 关键点, shape=(17, 3), 每行 [x, y, z]。
        rule: 姿势判定规则字典，结构自定。
              预期包含各关节的角度阈值、相对距离约束等。

    Returns:
        violated_rule_ids: 被违反的规则 ID 元组，如 ('R1', 'R2')，
                           未触发任何规则时为空元组 ()。
        affected_keypoints: 涉及告警的 H36M 关节点名称元组，
                            如 ('left_elbow', 'left_wrist')，
                            会被用于反向映射到 2D 骨骼渲染时高亮。
    """
    raise NotImplementedError


class FrameAnalyzer:
    def __init__(self, pose_name: str = "sample"):
        self._h36mkpe = H36MKeypointsExtractor()
        self._rule: dict = None  # TODO: 加载规则

    def analyze_frame(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[str]]:
        """
        对输入帧进行分析处理。

        Args:
            frame: BGR 图像，shape=(H, W, 3)，dtype=uint8，值域 [0, 255]。
                由 IRgbVideoSource.get_frame() 返回。
            frame_index: 当前帧序号，用于从预加载的骨骼数据中取对应帧。

        Returns:
            处理后的图像，shape=(H, W, 3)，dtype=uint8。
            输出将经 cv2.imencode 压缩为 JPEG 后推送至前端。
        """

        # 分析过程
        # 1. 使用 RTMPose 从帧中提取 2D 骨骼点，输出为 H36M 格式
        # 2. 使用 MHFormer 重建出 3D 骨骼点
        # 3. 根据 3D 骨骼点计算出需要告警的关键点
        # 4. 将 3D 骨骼得出的关键点部位反向映射到 2D 骨骼点上，得到需要告警的 2D 骨骼点索引列表
        # 5. 绘制 2D 骨骼连线到帧上，告警点及其关联骨骼使用高亮颜色，作为返回结果
        kps_h36m = self._h36mkpe.extract(frame)
        kps_3d = np.ndarray((17, 3))  # TODO: MHFormer 重建 3D 骨骼点

        violated_rule_id_set, affected_keypoints = judge_pose(
            kps_3d, self._rule
        )  # TODO: 根据规则判断姿势，得到告警点索引列表

        # TODO: 反向映射 3D 骨骼点索引到 2D 骨骼点索引，得到需要告警的 2D 骨骼点索引列表
        alert_kps_2d = [11, 12, 13]  # TODO: 替换为反向映射的骨骼节点

        # 渲染结果
        rendered_frame = H36MKeypointsRenderer.render_on_frame(frame, kps_h36m, alert_kps_2d)

        return rendered_frame, violated_rule_id_set
