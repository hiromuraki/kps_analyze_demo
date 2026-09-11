from __future__ import annotations
from pathlib import Path
from typing import Literal
from .interface import I2dPoseExtractor
import sys
import numpy as np
import logging

_project_root = Path(__file__).resolve().parent.parent.parent

_RTM_DET_DIR = _project_root / "rtm-det-aidlite"
_DEFAULT_DET_MODEL = str(_RTM_DET_DIR / "qnnout" / "rtmdet_m_raw_qcs8550_fp16.qnn236.ctx.bin.aidem")
_DEFAULT_POSE_MODEL = str(_RTM_DET_DIR / "qnnout" / "rtmpose-l_raw_qcs8550_fp16.qnn236.ctx.bin.aidem")

logger = logging.getLogger("kp2d_extractor")

_rtm_imported = False


def _ensure_rtm_imports() -> None:
    """把 ``rtm-det-aidlite`` 加入 ``sys.path``，使 ``rtm_vision`` 可被导入。

    以模块级标志位保证只执行一次。真正的 ``import`` 推迟到首次 ``extract()``，
    避免无 QNN 硬件、缺 aidlite SDK 的环境在模块导入阶段就失败。
    """
    global _rtm_imported
    if _rtm_imported:
        return
    if str(_RTM_DET_DIR) not in sys.path:
        sys.path.insert(0, str(_RTM_DET_DIR))
    _rtm_imported = True


class RTMPose2dPoseExtractor(I2dPoseExtractor):
    """2D 提取器的 RTMPose 实现：RTMDet 检测 + RTMPose 关键点，QNN DSP 推理。

    支持多目标场景，默认只返回得分最高者的关键点，可用 ``extract()`` 的
    ``return_all`` 取回全部检出结果。

    本实现依赖 QNN 硬件与 aidlite SDK。两个模型都在首次调用 ``extract()`` 时
    才真正加载（惰性），因此构造本类本身不会占用 DSP 资源。

    Args:
        det_model: RTMDet 检测模型的 ``.aidem`` 路径，省略时使用内置默认路径。
        pose_model: RTMPose 关键点模型的 ``.aidem`` 路径，省略时使用内置默认路径。
        det_score_thr: 检测框置信度阈值，低于此值的框直接丢弃。
        person_score_thr: 判定为"人"（label=0）所需的最低得分。
        topk: 最多保留的目标数量，按检测得分从高到低取。

    Note:
        用完请调用 ``close()`` 释放 DSP 资源，或用 ``with`` 语句管理生命周期。
    """

    def __init__(
        self,
        det_model: str | None = None,
        pose_model: str | None = None,
        det_score_thr: float = 0.4,
        person_score_thr: float = 0.3,
        topk: int = 5,
    ) -> None:
        self._det_model = det_model or _DEFAULT_DET_MODEL
        self._pose_model = pose_model or _DEFAULT_POSE_MODEL
        self._det_score_thr = det_score_thr
        self._person_score_thr = person_score_thr
        self._topk = topk

        self._det: RTMDet | None = None
        self._pose: RTMPose | None = None

    @property
    def data_out(self) -> Literal["COCO17", "H36M"]:
        return "COCO17"

    # ------------------------------------------------------------------
    # 惰性构造：把 DSP 加载推迟到首次 extract()
    # ------------------------------------------------------------------

    @property
    def _det_instance(self):
        if self._det is None:
            _ensure_rtm_imports()
            from rtm_vision import RTMDet  # noqa: E402

            self._det = RTMDet(
                self._det_model,
                score_thr=self._det_score_thr,
                max_per_img=50,
            )
        return self._det

    @property
    def _pose_instance(self):
        if self._pose is None:
            _ensure_rtm_imports()
            from rtm_vision import RTMPose  # noqa: E402

            self._pose = RTMPose(self._pose_model)
        return self._pose

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def extract(
        self,
        frame: np.ndarray,
        *,
        return_all: bool = False,
    ) -> np.ndarray:
        """检测人体并估计 COCO-17 2D 关键点。

        Args:
            frame: BGR 图像，``shape=(H, W, 3)``，``dtype=uint8``。
            return_all: 为 ``False``（默认）时只返回得分最高者的 ``(17, 3)``；
                为 ``True`` 时返回全部检出目标的 ``(N, 17, 3)``。
                未检出任何人体时，返回同形状的全零数组。

        Returns:
            ``shape=(17, 3)`` 或 ``(N, 17, 3)``，每行为 ``[x, y, confidence]``
            像素坐标与置信度，关节顺序为 COCO-17。
        """
        det_result = self._det_instance(frame)

        # 筛出置信度达标、且类别为「人」的检测框
        person_mask = (det_result.labels == 0) & (det_result.scores >= self._person_score_thr)
        person_boxes = det_result.bboxes[person_mask]
        person_scores = det_result.scores[person_mask]

        # 按得分降序取前 topk 个
        if self._topk > 0 and person_scores.size > self._topk:
            idx = np.argsort(-person_scores)[: self._topk]
            person_boxes = person_boxes[idx]
            person_scores = person_scores[idx]

        n_persons = len(person_boxes)

        # ---- 未检出人体 ----
        if n_persons == 0:
            return np.zeros((17, 3), dtype=np.float32) if not return_all else np.empty((0, 17, 3), dtype=np.float32)

        # ---- 单人快速路径（默认）----
        if not return_all:
            result = self._pose_instance(frame, person_boxes[:1])[0]
            n = min(17, len(result.keypoints))
            kp = np.zeros((17, 3), dtype=np.float32)
            kp[:n, 0] = result.keypoints[:n, 0]
            kp[:n, 1] = result.keypoints[:n, 1]
            kp[:n, 2] = result.scores[:n]
            return kp

        # ---- 多目标路径 ----
        pose_results = self._pose_instance(frame, person_boxes)
        coco = np.zeros((n_persons, 17, 3), dtype=np.float32)
        for i, result in enumerate(pose_results):
            n = min(17, len(result.keypoints))
            coco[i, :n, 0] = result.keypoints[:n, 0]
            coco[i, :n, 1] = result.keypoints[:n, 1]
            coco[i, :n, 2] = result.scores[:n]
        return coco

    def close(self) -> None:
        """释放两个模型的 QNN 解释器与 DSP 资源。

        可重复调用。释放后再次调用 ``extract()`` 会触发模型重新加载。
        """
        for model in (self._det, self._pose):
            if model is not None:
                model.close()
        self._det = None
        self._pose = None

    # ------------------------------------------------------------------
    # 上下文管理器
    # ------------------------------------------------------------------

    def __enter__(self) -> RTMPose2dPoseExtractor:
        return self

    def __exit__(self, *args: object) -> None:
        """退出 ``with`` 块时释放模型资源。"""
        self.close()
