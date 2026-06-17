from __future__ import annotations
from collections import deque
from .kp2d_extractor import I2dPoseExtractor
from .kp3d_reconstructor import I3dPoseReconstructor
from .renderer import H36M2dKeypointsRenderer
from .converter import DataConverter
from .pose_judger import judge_pose
import logging
import queue
import threading
import numpy as np

logger = logging.getLogger("analyzer")


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
        # 2D 骨骼历史缓冲区，供 MHFormer 3D 重建使用
        self._frame_buffer = deque[np.ndarray](maxlen=351)

        # 3D 管线（后台线程）
        self._3d_queue: queue.Queue = queue.Queue(maxsize=4)
        self._result_queue: queue.Queue = queue.Queue(maxsize=32)
        self._lock_3d = threading.Lock()
        self._latest_3d_kps: np.ndarray | None = None
        self._stop_event = threading.Event()
        self._3d_thread: threading.Thread | None = None
        self._judge_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def pose_name(self) -> str:
        return self._pose_name

    @property
    def latest_3d_kps(self) -> np.ndarray | None:
        """返回最新 3D 骨骼的拷贝（线程安全）。"""
        with self._lock_3d:
            return self._latest_3d_kps.copy() if self._latest_3d_kps is not None else None

    def drain_results(self) -> list[list[str]]:
        """清空分析结果队列，返回违规规则 ID 列表的列表。"""
        items: list[list[str]] = []
        while True:
            try:
                items.append(self._result_queue.get_nowait())
            except queue.Empty:
                break
        return items

    def analyze_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        2D 快速路径：提取 + 格式转换 + 骨骼渲染。

        Returns:
            叠加 2D 骨骼后的 BGR 帧，shape=(H, W, 3)。
        """
        kp2d_coco17 = self._kp2d_extractor.extract(frame)
        if self._kp2d_extractor.data_out == "COCO17":
            kp2d_h36m = DataConverter.coco17_to_h36m(kp2d_coco17)
        else:  # "H36M"
            kp2d_h36m = kp2d_coco17

        self._frame_buffer.append(kp2d_h36m)
        # 默认不渲染告警高亮（告警通过消息面板展示）
        return H36M2dKeypointsRenderer.render_on_frame(frame, kp2d_h36m, [])

    def start_3d_pipeline(self):
        """启动 3D 重建和姿态判定后台线程。"""
        self._stop_event.clear()
        self._3d_thread = threading.Thread(target=self._3d_loop, daemon=True, name="mhformer-3d")
        self._judge_thread = threading.Thread(target=self._judge_loop, daemon=True, name="pose-judge")
        self._3d_thread.start()
        self._judge_thread.start()
        logger.info("3D pipeline started (MHFormer + judge)")

    def stop_3d_pipeline(self):
        """停止后台线程。"""
        self._stop_event.set()
        for t in (self._3d_thread, self._judge_thread):
            if t is not None:
                t.join(timeout=3.0)
        logger.info("3D pipeline stopped")

    # ------------------------------------------------------------------
    # Background loops
    # ------------------------------------------------------------------

    def _3d_loop(self):
        """每 0.1s 从 2D 缓冲区取最新帧做 MHFormer 3D 重建。"""
        while not self._stop_event.is_set():
            if len(self._frame_buffer) >= 50:  # 至少积累一些帧再开始
                frames = np.stack(list(self._frame_buffer))
                try:
                    kps_3d = self._kp3d_reconstructor.reconstruct(frames, frame_index=-1)
                    with self._lock_3d:
                        self._latest_3d_kps = kps_3d
                    # 不阻塞放入队列
                    try:
                        self._3d_queue.put_nowait(kps_3d)
                    except queue.Full:
                        # 队列满说明判态来不及消费，丢弃旧帧
                        try:
                            self._3d_queue.get_nowait()
                            self._3d_queue.put_nowait(kps_3d)
                        except queue.Empty:
                            pass
                except Exception:
                    logger.exception("3D reconstruction error")
            self._stop_event.wait(0.1)

    def _judge_loop(self):
        """消费 3D 结果队列，进行姿态判定。"""
        while not self._stop_event.is_set():
            try:
                kps_3d = self._3d_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                violated_rule_ids, _affected = judge_pose(kps_3d, self._rule)
                if violated_rule_ids:
                    try:
                        self._result_queue.put_nowait(violated_rule_ids)
                    except queue.Full:
                        self._result_queue.get_nowait()
                        self._result_queue.put_nowait(violated_rule_ids)
            except Exception:
                logger.exception("Judge error")
