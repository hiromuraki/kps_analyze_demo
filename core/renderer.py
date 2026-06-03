import numpy as np
import cv2


class H36MKeypointsRenderer:
    # 默认颜色 —— 中性、低辨识度
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
