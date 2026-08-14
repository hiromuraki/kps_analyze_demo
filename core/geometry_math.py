"""
core/geometry_math.py — 2D / 3D 几何计算工具

集中管理姿态判定（judge_pose）与动作计数（RepCounter）共用的几何计算，

包含两组类：
- Geometry2DMath : 图像平面（像素坐标，Y 向下）内的计算
- Geometry3DMath : H36M 三维坐标 (x, y, z) 内的计算

统一约定：
- 点 / 向量均为可索引序列（numpy 数组、list、tuple 均可）
- 角度一律返回度数（degree）
- 零向量 / 退化输入返回 0.0，不抛出异常

对应旧函数（嵌入整合时替换关系）：
- pose_judger.calc_three_point_angle(vec_a, vec_b)      →  Geometry*.angle_between_vectors(vec_a, vec_b)
- pose_judger.calculate_angle_between_segments(p1..p4)  →  Geometry*.angle_between_segments(p1..p4)
- rep_counter.calc_2point_distance(p1, p2)              →  Geometry3DMath.distance(p1, p2)
- judge_pose 内联的水平夹角 arctan2 计算               →  Geometry2DMath.horizontal_angle(p1, p2)

已知限制（劣弧角）
--------------------
angle_between_vectors / angle_between_segments 基于 arccos，只能返回 [0°, 180°] 的劣弧角。
由 cos(θ) = cos(360°−θ) 可知 θ 与 360°−θ 在余弦上不可区分：
例如肘关节实际 190°（过伸）会被计算为 170°，产生「170→180→170」的回折现象。

后续若需区分方向（有向角，如过伸检测）：
- 2D：atan2(cross_z, dot) 天然带方向，可直接扩展（无需参考轴）
- 3D：需引入参考轴 axis（旋转平面法向），计算 atan2(dot(cross(v1,v2), axis), dot(v1,v2))
"""

import math


class Geometry2DMath:
    """图像平面（2D 像素坐标）几何计算。

    坐标系：x 向右、y 向下（OpenCV 图像约定）。
    点形如 (x, y) 或 (x, y, z)，计算时只取前两维。
    """

    # ------------------------------------------------------------------
    @staticmethod
    def angle_between_vectors(vec_a, vec_b) -> float:
        """两个平面向量的无向夹角（0~180°）。

        vec_a, vec_b : 平面向量 (x, y)，可为 numpy 数组 / list / tuple。
        返回 : 夹角度数；任一向量近零（<1e-6）时返回 0.0。

        对应旧 calc_three_point_angle(vec_a, vec_b)（共顶点三点场景的向量形式）。
        注意：基于 arccos 的劣弧角，无法区分 θ 与 360°−θ（见模块 docstring）。
        """
        dot = vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1]
        norm_a = math.sqrt(vec_a[0] ** 2 + vec_a[1] ** 2)
        norm_b = math.sqrt(vec_b[0] ** 2 + vec_b[1] ** 2)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        cos_ang = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
        return math.degrees(math.acos(cos_ang))

    @staticmethod
    def three_point_angle(vertex, p_a, p_b) -> float:
        """以 vertex 为顶点的三点内角（0~180°）。

        vertex : 顶点坐标；p_a, p_b : 两个端点坐标。
        即线段 vertex-p_a 与 vertex-p_b 的夹角。
        便捷封装，内部自动构造向量后调用 angle_between_vectors。
        """
        va = (p_a[0] - vertex[0], p_a[1] - vertex[1])
        vb = (p_b[0] - vertex[0], p_b[1] - vertex[1])
        return Geometry2DMath.angle_between_vectors(va, vb)

    # ------------------------------------------------------------------
    @staticmethod
    def angle_between_segments(p1, p2, p3, p4) -> float:
        """两条独立线段 (p1-p2) 与 (p3-p4) 的夹角（0~180°）。

        适用：四点互不相同、无公共顶点的线段夹角规则。
        若 p2 == p3（共顶点）应改用 three_point_angle / angle_between_vectors。
        对应旧 calculate_angle_between_segments(p1..p4)。
        """
        va = (p2[0] - p1[0], p2[1] - p1[1])
        vb = (p4[0] - p3[0], p4[1] - p3[1])
        return Geometry2DMath.angle_between_vectors(va, vb)

    # ------------------------------------------------------------------
    @staticmethod
    def horizontal_angle(p1, p2) -> float:
        """两点连线与水平面（图像 x 轴）的夹角（0~90°）。

        p1, p2 : 两个关节点坐标 (x, y)。
        返回 : 0~90° 度数。

        实现：arctan2(|dy|, |dx|)。atan2 本身带方向，但这里取绝对值，
        只关心倾斜幅度——用于「水平夹角」规则（如左右肩高低差检测）。
        对应 judge_pose 内联实现。
        """
        dy = abs(float(p2[1]) - float(p1[1]))
        dx = abs(float(p2[0]) - float(p1[0]))
        return math.degrees(math.atan2(dy, max(dx, 1e-6)))

    # ------------------------------------------------------------------
    @staticmethod
    def distance(p1, p2) -> float:
        """两点平面欧式距离（忽略 z 分量）。

        p1, p2 : 两个点坐标 (x, y)。
        对应 rep_counter.calc_2point_distance 的 2D 版本（原实现仅 3D）。
        """
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


class Geometry3DMath:
    """三维空间几何计算。

    坐标系：H36M 输出 (x, y, z)，与 2D 一致 y 向下。
    点形如 (x, y, z)。
    """

    # ------------------------------------------------------------------
    @staticmethod
    def angle_between_vectors(vec_a, vec_b) -> float:
        """两个空间向量的无向夹角（0~180°）。

        vec_a, vec_b : 空间向量 (x, y, z)，可为 numpy 数组 / list / tuple。
        返回 : 夹角度数；任一向量近零（<1e-6）时返回 0.0。

        对应旧 calc_three_point_angle(vec_a, vec_b)（及 rep_counter 同名副本）。
        注意：基于 arccos 的劣弧角；三维空间两向量不共面时该夹角为
        「线线角」，本身不含旋转方向信息（见模块 docstring）。
        """
        dot = vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1] + vec_a[2] * vec_b[2]
        norm_a = math.sqrt(vec_a[0] ** 2 + vec_a[1] ** 2 + vec_a[2] ** 2)
        norm_b = math.sqrt(vec_b[0] ** 2 + vec_b[1] ** 2 + vec_b[2] ** 2)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        cos_ang = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
        return math.degrees(math.acos(cos_ang))

    @staticmethod
    def three_point_angle(vertex, p_a, p_b) -> float:
        """以 vertex 为顶点的三点内角（0~180°）。

        vertex : 顶点坐标 (x, y, z)；p_a, p_b : 两个端点坐标。
        即线段 vertex-p_a 与 vertex-p_b 的夹角。
        便捷封装，内部自动构造向量后调用 angle_between_vectors。
        """
        va = (p_a[0] - vertex[0], p_a[1] - vertex[1], p_a[2] - vertex[2])
        vb = (p_b[0] - vertex[0], p_b[1] - vertex[1], p_b[2] - vertex[2])
        return Geometry3DMath.angle_between_vectors(va, vb)

    # ------------------------------------------------------------------
    @staticmethod
    def angle_between_segments(p1, p2, p3, p4) -> float:
        """两条独立线段 (p1-p2) 与 (p3-p4) 的空间夹角（0~180°）。

        适用：四点互不相同、无公共顶点的线段夹角规则。
        若 p2 == p3（共顶点）应改用 three_point_angle / angle_between_vectors。
        对应旧 calculate_angle_between_segments(p1..p4)（及 rep_counter 同名副本）。
        """
        va = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
        vb = (p4[0] - p3[0], p4[1] - p3[1], p4[2] - p3[2])
        return Geometry3DMath.angle_between_vectors(va, vb)

    # ------------------------------------------------------------------
    @staticmethod
    def distance(p1, p2) -> float:
        """两点三维欧式距离。

        对应旧 rep_counter.calc_2point_distance(p1, p2)。
        """
        return math.sqrt(
            (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2
        )
