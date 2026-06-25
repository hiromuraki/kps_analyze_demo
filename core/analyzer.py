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
import time
import uuid
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
        self._direction = get_rep_count_direction(rule)
        self._phase = RepPhase.UP
        self._count = 0
        # per-rep ROM 追踪
        self._feature_min: float = float("inf")
        self._feature_max: float = float("-inf")
        self._rom_values: list[float] = []         # 每次完成的 ROM
        self._rep_timestamps: list[float] = []      # 每次完成的时间戳

    # ------------------------------------------------------------------
    @property
    def count(self) -> int:
        return self._count

    @property
    def phase(self) -> RepPhase:
        return self._phase

    @property
    def rom_values(self) -> list[float]:
        """每次动作重复的关节活动度（度）。"""
        return list(self._rom_values)

    @property
    def rep_timestamps(self) -> list[float]:
        """每次动作重复完成时的时间戳（秒）。"""
        return list(self._rep_timestamps)

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

        # 追踪本 rep 的特征最值
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
        self._state: str = "running"  # running | paused | stopped
        self._debounce_frames = 30
        self._frame_n = 0
        self._active_frames = 0      # 仅在 running 态递增，用于 accuracy 计算
        self._v_first_seen: dict[str, int] = {}
        self._v_reported: set[str] = set()
        # 训练统计
        self._started_at: float = time.monotonic()
        self._violated_frame_count = 0
        self._training_id: str = str(uuid.uuid4())[:8]
        self._stats_history: deque[dict] = deque(maxlen=64)
        # 冻结快照（stopped 时留存最后统计，避免属性返回 0）
        self._frozen: dict | None = None

    @property
    def state(self) -> str:
        """当前状态：running / paused / stopped。"""
        return self._state

    @property
    def training_id(self) -> str:
        """当前训练会话 ID（8 位）。"""
        return self._frozen["training_id"] if self._frozen else self._training_id

    @property
    def stats_history(self) -> list[dict]:
        """历史训练统计快照列表。"""
        return list(self._stats_history)

    @property
    def pose_name(self) -> str:
        return self._pose_name

    @property
    def rep_count(self) -> int:
        """当前动作累计完成次数。"""
        if self._frozen:
            return self._frozen.get("total_reps", 0)
        return self._rep_counter.count if self._rep_counter is not None else 0

    @property
    def accuracy(self) -> float:
        if self._frozen:
            return self._frozen.get("accuracy", 0.0)
        if self._active_frames == 0:
            return 1.0
        return 1.0 - (self._violated_frame_count / self._active_frames)

    @property
    def rom(self) -> float:
        if self._frozen:
            return self._frozen.get("rom", 0.0)
        vals = self._rep_counter.rom_values if self._rep_counter else []
        return float(np.mean(vals)) if vals else 0.0

    @property
    def density(self) -> float:
        if self._frozen:
            return self._frozen.get("density", 0.0)
        duration = time.monotonic() - self._started_at
        if duration <= 0:
            return 0.0
        return self.rep_count / (duration / 60.0)

    @property
    def calories(self) -> float:
        if self._frozen:
            return self._frozen.get("calories", 0.0)
        return self.rep_count * self._rule.get("calories_per_rep", 0.0)

    @property
    def balance_score(self) -> float:
        if self._frozen:
            return self._frozen.get("balance_score", 0.0)
        vals = self._rep_counter.rom_values if self._rep_counter else []
        if len(vals) < 2:
            return 0.0
        cv = float(np.std(vals) / max(np.mean(vals), 1.0))
        return max(0.0, 100.0 - cv * 100.0)

    @property
    def duration_seconds(self) -> float:
        if self._frozen:
            return self._frozen.get("duration_seconds", 0.0)
        return time.monotonic() - self._started_at

    @property
    def total_frames(self) -> int:
        if self._frozen:
            return self._frozen.get("total_frames", 0)
        return self._frame_n

    @property
    def violated_frame_count(self) -> int:
        if self._frozen:
            return self._frozen.get("violated_frames", 0)
        return self._violated_frame_count

    @property
    def fatigue_score(self) -> float:
        if self._frozen:
            return self._frozen.get("fatigue_score", 0.0)
        vals = self._rep_counter.rom_values if self._rep_counter else []
        if len(vals) < 4:
            return 0.0
        n = len(vals)
        third = n // 3 or 1
        early_rom = float(np.mean(vals[:third]))
        late_rom = float(np.mean(vals[-third:]))
        rom_decay = max(0.0, (early_rom - late_rom) / max(early_rom, 1.0))

        rc = self._rep_counter
        ts = rc.rep_timestamps if rc else []
        fps = self._frame_n / max(self.duration_seconds, 0.001)
        early_spd = (ts[third] - ts[0]) / third if len(ts) > third else 1.0 / fps
        late_spd = (ts[-1] - ts[-third]) / third if len(ts) > third else 1.0 / fps
        spd_decay = max(0.0, (late_spd - early_spd) / max(early_spd, 0.01))

        mid = self._frame_n // 2
        late_v = sum(1 for _, start in self._v_first_seen.items() if start > mid)
        early_v = len(self._v_first_seen) - late_v
        v_rise = max(0.0, (late_v - early_v) / max(early_v + late_v, 1.0))

        return (rom_decay * 0.4 + spd_decay * 0.3 + v_rise * 0.3) * 100.0

    # ------------------------------------------------------------------
    def pause(self):
        """暂停（停 judge + rep_count，2D/3D 继续）。"""
        self._state = "paused"
        logger.info("Analysis paused")

    def resume(self):
        """开始 / 恢复。"""
        was_stopped = self._state == "stopped"
        self._state = "running"
        self._frozen = None
        if was_stopped:
            self._training_id = str(uuid.uuid4())[:8]
            self._started_at = time.monotonic()
            self._frame_n = 0
            self._active_frames = 0
            self._violated_frame_count = 0
            if self._rep_counter is not None:
                self._rep_counter.reset()
        self._v_first_seen.clear()
        self._v_reported.clear()
        logger.info("Analysis %s (id=%s)", "started" if was_stopped else "resumed", self._training_id)

    def stop(self):
        """停止本次训练：保存统计快照 → 冻结数据 → 进入 stopped 态。"""
        snap = {
            "training_id": self._training_id,
            "pose_name": self._pose_name,
            "total_reps": self.rep_count,
            "total_frames": self._active_frames,
            "violated_frames": self._violated_frame_count,
            "accuracy": self.accuracy,
            "rom": self.rom,
            "density": self.density,
            "duration_seconds": self.duration_seconds,
            "calories": self.calories,
            "balance_score": self.balance_score,
            "fatigue_score": self.fatigue_score,
            "stopped_at": time.monotonic(),
        }
        self._stats_history.append(snap)
        self._frozen = snap
        self._state = "stopped"
        logger.info(f"Training stopped: {snap['training_id']} → history[{len(self._stats_history)-1}]")

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

        self._frame_n += 1

        if self._state != "running":
            violations: list[str] = []
            rep_counted = False
            alert_kps_2d: list[int] = []
        else:
            self._active_frames += 1
            violations_raw, affected_keypoints = judge_pose(kps_3d, self._rule)
            rep_counted = self._rep_counter is not None and self._rep_counter.update(kps_3d)
            alert_kps_2d = _map_kp_names_to_indices(affected_keypoints)

            current = set(violations_raw)
            if current:
                self._violated_frame_count += 1

            # 新出现 → 记录首帧
            for v in current - set(self._v_first_seen):
                self._v_first_seen[v] = self._frame_n

            # 消失 → 抹掉记录
            for v in set(self._v_first_seen) - current:
                del self._v_first_seen[v]
                self._v_reported.discard(v)

            # 持续超过去抖阈值 且 未上报 → 触发
            violations = []
            for v in current:
                if v not in self._v_reported and self._frame_n - self._v_first_seen[v] >= self._debounce_frames:
                    violations.append(v)
                    self._v_reported.add(v)

        rendered = H36M2dKeypointsRenderer.render_on_frame(frame, kp2d_h36m, alert_kps_2d)

        return AnalysisResult(
            rendered=rendered,
            kps_3d=kps_3d,
            violations=violations,
            rep_counted=rep_counted,
        )
