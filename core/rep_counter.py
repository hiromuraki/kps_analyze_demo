from enum import Enum, auto
import time

import numpy as np


# ===================== 骨骼节点名称=====================
NODE_NAME_TO_INDEX = {
    "骨盆": 0,
    "右髋": 1,
    "右膝": 2,
    "右脚踝": 3,
    "左髋": 4,
    "左膝": 5,
    "左脚踝": 6,
    "脊柱": 7,
    "胸腔": 8,
    "鼻子": 9,
    "头顶": 10,
    "左肩": 11,
    "左肘": 12,
    "左手腕": 13,
    "右肩": 14,
    "右肘": 15,
    "右手腕": 16,
    "地面": -1,
}


# ===================== 三个基础读取函数 =====================
def get_rep_ceiling(rule: dict) -> float:
    """
    返回重复计数的上界阈值（特征值高于此判定为伸展态）。
    rule    : 规则 JSON 的完整 dict。
    return  : 上界阈值。
    """
    rep_cfg = rule["rep_counting"]
    return float(rep_cfg["top_threshold"])


def get_rep_floor(rule: dict) -> float:
    """
    返回重复计数的下界阈值（特征值低于此判定为收缩态）。
    rule    : 规则 JSON 的完整 dict。
    return  : 下界阈值。
    """
    rep_cfg = rule["rep_counting"]
    return float(rep_cfg["bottom_threshold"])


def get_rep_count_direction(rule: dict) -> str:
    """
    返回重复计数的触发方向。
    rule    : 规则 JSON 的完整 dict。
    return  : "down_up"（收缩→伸展 计一次）
              或 "up_down"（伸展→收缩 计一次）。
    """
    rep_cfg = rule["rep_counting"]
    return rep_cfg["count_on"]


# ===================== 核心特征提取函数 =====================
def get_rep_feature_value(kps_3d: np.ndarray, rule: dict) -> float:
    """
    从 3D 骨骼中提取当前帧的重复计数特征值。

    kps_3d  : shape=(17, 3), H36M 格式 xyz 坐标，index 0-16。
    rule    : 规则 JSON 的完整 dict（含 rep_counting 块）。
    return  : 当前帧的特征标量值（角度=度数, 距离=原始单位）。
    """

    def calc_three_point_angle(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        计算同一个顶点出发两个向量的夹角（三点内角，0~180°）
        对应场景：P1-P2(顶点)-P4，P2=P3
        """
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        cos_ang = np.dot(vec_a, vec_b) / (norm_a * norm_b)
        cos_ang = np.clip(cos_ang, -1.0, 1.0)
        return np.degrees(np.arccos(cos_ang))

    def calculate_angle_between_segments(p1, p2, p3, p4) -> float:
        """
        计算两条独立线段 (p1-p2)、(p3-p4) 的异面夹角，返回 0~180°
        适用：四点完全不同、两条无公共顶点线段
        """
        vec1 = p2 - p1
        vec2 = p4 - p3
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def calc_2point_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """计算两点3D欧式距离"""
        return float(np.linalg.norm(p2 - p1))

    # 提取重复计数配置块
    rep_cfg = rule["rep_counting"]
    feat_type = rep_cfg["type"]  # "angle" / "distance"

    # 取出关键点名称，安全读取，distance无p3/p4不会报错
    p1_name = rep_cfg["p1"]
    p2_name = rep_cfg["p2"]
    p3_name = rep_cfg.get("p3", "")
    p4_name = rep_cfg.get("p4", "")

    # 处理地面-1索引，生成地面坐标（X、Z沿用参考点，Y=0）
    def get_keypoint(idx: int, ref_point: np.ndarray) -> np.ndarray:
        if idx == -1:
            return np.array([ref_point[0], 0.0, ref_point[2]])
        return kps_3d[idx]

    # 根据类型分支处理
    if feat_type == "angle":
        # angle类型必须完整四点，读取所有索引与坐标
        idx1 = NODE_NAME_TO_INDEX[p1_name]
        idx2 = NODE_NAME_TO_INDEX[p2_name]
        idx3 = NODE_NAME_TO_INDEX[p3_name]
        idx4 = NODE_NAME_TO_INDEX[p4_name]

        p1 = get_keypoint(idx1, kps_3d[idx2])
        p2 = get_keypoint(idx2, kps_3d[idx1])
        p3 = get_keypoint(idx3, kps_3d[idx2])
        p4 = get_keypoint(idx4, kps_3d[idx3])

        # 区分共顶点 / 异面两线段
        if p2_name == p3_name:
            vec_a = p1 - p2
            vec_b = p4 - p2
            return calc_three_point_angle(vec_a, vec_b)
        else:
            return calculate_angle_between_segments(p1, p2, p3, p4)

    elif feat_type == "distance":
        # distance只用到p1/p2，完全不碰p3/p4
        idx1 = NODE_NAME_TO_INDEX[p1_name]
        idx2 = NODE_NAME_TO_INDEX[p2_name]
        p1 = get_keypoint(idx1, kps_3d[idx2])
        p2 = get_keypoint(idx2, kps_3d[idx1])
        return calc_2point_distance(p1, p2)

    else:
        raise ValueError(f"不支持的rep_counting.type: {feat_type}, 仅支持 angle / distance")


class RepPhase(Enum):
    UP = auto()  # 伸展态（站立 / 臂伸直）
    DOWN = auto()  # 收缩态（蹲到底 / 曲臂）


class RepMotion(Enum):
    DESCENT = auto()     # 下降期（下蹲 / 曲臂）
    ASCENT = auto()      # 上升期（起身 / 伸展）
    STATIC_UP = auto()   # 顶部保持（伸展态静止，如站直锁关节）
    STATIC_DOWN = auto() # 底部保持（收缩态静止，如蹲到底）


class RepCounter:
    """
    通用动作计数状态机。

    依赖四个外部函数读取规则：
    - get_rep_feature_value(kps_3d, rule) -> float
    - get_rep_ceiling(rule) -> float
    - get_rep_floor(rule) -> float
    - get_rep_count_direction(rule) -> str
    """

    def __init__(self, rule: dict, ema_alpha: float = 0.25, motion_debounce: int = 4, delta_threshold: float | None = None):
        self._rule = rule
        self._ceiling = get_rep_ceiling(rule)
        self._floor = get_rep_floor(rule)
        self._direction = get_rep_count_direction(rule)
        self._phase = RepPhase.UP
        self._count = 0
        rep_cfg = rule.get("rep_counting", {})
        self._delta_threshold = delta_threshold if delta_threshold is not None else rep_cfg.get("delta_threshold", 1.0)
        # per-rep ROM 追踪
        self._feature_min: float = float("inf")
        self._feature_max: float = float("-inf")
        self._rom_values: list[float] = []  # 每次完成的 ROM
        self._rep_timestamps: list[float] = []  # 每次完成的时间戳
        self._raw_value: float = 0.0  # 最近一帧原始特征值
        self._smooth_value: float | None = None  # EMA 平滑后的值（首帧直接赋值）
        self._ema_alpha = ema_alpha
        self._motion: RepMotion = RepMotion.STATIC_UP
        # 方向切换去抖
        self._motion_candidate: RepMotion = RepMotion.STATIC_UP
        self._candidate_frames: int = 0
        self._motion_debounce = motion_debounce

    # ------------------------------------------------------------------
    @property
    def count(self) -> int:
        return self._count

    @property
    def phase(self) -> RepPhase:
        return self._phase

    @property
    def motion(self) -> RepMotion:
        """当前动作阶段（下降/上升/静止）。"""
        return self._motion

    @property
    def feature_value(self) -> float:
        """EMA 平滑后的特征值（角度或距离）。"""
        return self._smooth_value if self._smooth_value is not None else self._raw_value

    @property
    def rom_values(self) -> list[float]:
        """每次动作重复的关节活动度（度）。"""
        return list(self._rom_values)

    @property
    def rep_timestamps(self) -> list[float]:
        """每次动作重复完成时的时间戳（秒）。"""
        return list(self._rep_timestamps)

    @property
    def ema_alpha(self) -> float:
        """EMA 平滑系数。"""
        return self._ema_alpha

    @property
    def motion_debounce(self) -> int:
        """方向去抖帧数。"""
        return self._motion_debounce

    @property
    def delta_threshold(self) -> float:
        """方向切换的特征值变化阈值。"""
        return self._delta_threshold

    # ------------------------------------------------------------------
    def update(self, kps_3d: np.ndarray) -> bool:
        """
        输入一帧 3D 骨骼，更新状态机。

        Returns:
            True 当本帧完成了一次动作计数时。
        """
        raw = get_rep_feature_value(kps_3d, self._rule)
        self._raw_value = raw

        # ── EMA 平滑 ──
        if self._smooth_value is None:
            self._smooth_value = raw
        else:
            self._smooth_value = self._ema_alpha * raw + (1 - self._ema_alpha) * self._smooth_value

        # ── 平滑后的一阶导数 → 瞬时方向 ──
        delta = self._smooth_value - (
            self._feature_prev_smooth if hasattr(self, "_feature_prev_smooth") else self._smooth_value
        )
        self._feature_prev_smooth = self._smooth_value
        if delta > self._delta_threshold:
            instant = RepMotion.ASCENT
        elif delta < -self._delta_threshold:
            instant = RepMotion.DESCENT
        else:
            instant = RepMotion.STATIC_UP if self._phase == RepPhase.UP else RepMotion.STATIC_DOWN

        # ── 方向去抖：连续 N 帧同一方向才切换 ──
        if instant == self._motion_candidate:
            self._candidate_frames += 1
        else:
            self._motion_candidate = instant
            self._candidate_frames = 1

        if self._candidate_frames >= self._motion_debounce:
            self._motion = self._motion_candidate

        # ── 用平滑值做 phase 判定（避免尖峰误触发）──
        prev = self._phase
        value = self._smooth_value

        if value >= self._ceiling:
            self._phase = RepPhase.UP
        elif value <= self._floor:
            self._phase = RepPhase.DOWN

        # 追踪本 rep 的特征最值（用平滑值）
        self._feature_min = min(self._feature_min, value)
        self._feature_max = max(self._feature_max, value)

        counted = False
        if self._direction == "down_up":
            if prev == RepPhase.DOWN and self._phase == RepPhase.UP:
                self._count += 1
                counted = True
        else:  # "up_down"
            if prev == RepPhase.UP and self._phase == RepPhase.DOWN:
                self._count += 1
                counted = True

        if counted:
            rom = self._feature_max - self._feature_min
            self._rom_values.append(rom)
            self._rep_timestamps.append(time.monotonic())
            self._feature_min = float("inf")
            self._feature_max = float("-inf")

        return counted

    def reset(self):
        """重置计数器和 ROM 历史。"""
        self._count = 0
        self._phase = RepPhase.UP
        self._feature_min = float("inf")
        self._feature_max = float("-inf")
        self._rom_values.clear()
        self._rep_timestamps.clear()
        self._raw_value = 0.0
        self._smooth_value = None
        self._motion = RepMotion.STATIC_UP
        self._motion_candidate = RepMotion.STATIC_UP
        self._candidate_frames = 0
        if hasattr(self, "_feature_prev_smooth"):
            del self._feature_prev_smooth
