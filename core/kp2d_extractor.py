from __future__ import annotations
from pathlib import Path
from typing import Literal
import sys
import numpy as np
import logging

_project_root = Path(__file__).resolve().parent.parent

_RTM_DET_DIR = _project_root / "rtm-det-aidlite"
_DEFAULT_DET_MODEL = str(_RTM_DET_DIR / "qnnout" / "rtmdet_m_raw_qcs8550_fp16.qnn236.ctx.bin.aidem")
_DEFAULT_POSE_MODEL = str(_RTM_DET_DIR / "qnnout" / "rtmpose-l_raw_qcs8550_fp16.qnn236.ctx.bin.aidem")

logger = logging.getLogger("kp2d_extractor")

_rtm_imported = False


def _ensure_rtm_imports() -> None:
    """Lazy-import rtm_vision (requires aidlite SDK) on first use."""
    global _rtm_imported
    if _rtm_imported:
        return
    if str(_RTM_DET_DIR) not in sys.path:
        sys.path.insert(0, str(_RTM_DET_DIR))
    _rtm_imported = True


class I2dPoseExtractor:
    @property
    def data_out(self) -> Literal["COCO17", "H36M"]:
        raise NotImplementedError

    def extract(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Mock2dExtractor(I2dPoseExtractor):
    def __init__(self):
        self._kps_npz = np.load("./sample_data/example_2d_h36m_kps.npz")
        self._kps_frames: np.ndarray = self._kps_npz[self._kps_npz.files[0]]  # (Frames, 17, 3)
        self._kps_frame_count = self._kps_frames.shape[0]
        self._frame_index = 0
        logger.info(f"Loaded 2D keypoints: {self._kps_frames.shape}")

    @property
    def data_out(self) -> Literal["COCO17", "H36M"]:
        return "H36M"

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """
        从输入帧中提取 2D 关键点。

        Args:
            frame: BGR 图像，shape=(H, W, 3)，dtype=uint8，值域 [0, 255]。

        Returns:
            关键点数组，shape=(17, 2)，每行 [x, y]。
        """

        kps = self._kps_frames[self._frame_index]
        self._frame_index = (self._frame_index + 1) % self._kps_frame_count
        return kps


class RTMPose2dPoseExtractor(I2dPoseExtractor):
    """Extract COCO-17 2D keypoints from a single RGB image.

    Wraps the RTMDet + RTMPose QNN inference pipeline.  By default returns
    keypoints for the highest-scoring person only; pass ``return_all=True``
    to get all detections.

    Parameters
    ----------
    det_model:
        Path to RTMDet QNN model (``.aidem``).  Auto-detected when omitted.
    pose_model:
        Path to RTMPose QNN model (``.aidem``).  Auto-detected when omitted.
    det_score_thr:
        Bounding-box confidence threshold for the detector.
    person_score_thr:
        Minimum score for a detection to be considered a person (label=0).
    topk:
        Maximum number of persons to consider (highest detection scores first).
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
    # Lazy model instantiation (defers DSP load to first extract())
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
    # Public methods
    # ------------------------------------------------------------------

    def extract(
        self,
        image: np.ndarray,
        *,
        return_all: bool = False,
    ) -> np.ndarray:
        """Detect persons and estimate COCO-17 2D keypoints.

        Parameters
        ----------
        image:
            BGR image array of shape ``(H, W, 3)``, dtype ``uint8``.
        return_all:
            If ``False`` (default), returns ``(17, 3)`` for the top-scoring
            person.  If ``True``, returns ``(N, 17, 3)`` for all detected
            persons.  When no person is found, returns a zero-filled array
            of the same shape (``(17, 3)`` or ``(0, 17, 3)``).

        Returns
        -------
        np.ndarray
            ``(17, 3)`` or ``(N, 17, 3)`` — COCO-17 keypoints in pixel
            coordinates ``(x, y, confidence)``.
        """
        det_result = self._det_instance(image)

        # Keep person detections above threshold
        person_mask = (det_result.labels == 0) & (det_result.scores >= self._person_score_thr)
        person_boxes = det_result.bboxes[person_mask]
        person_scores = det_result.scores[person_mask]

        # Top-k by score
        if self._topk > 0 and person_scores.size > self._topk:
            idx = np.argsort(-person_scores)[: self._topk]
            person_boxes = person_boxes[idx]
            person_scores = person_scores[idx]

        n_persons = len(person_boxes)

        # ---- no-detection path ----
        if n_persons == 0:
            return np.zeros((17, 3), dtype=np.float32) if not return_all else np.empty((0, 17, 3), dtype=np.float32)

        # ---- single-person fast path (default) ----
        if not return_all:
            result = self._pose_instance(image, person_boxes[:1])[0]
            n = min(17, len(result.keypoints))
            kp = np.zeros((17, 3), dtype=np.float32)
            kp[:n, 0] = result.keypoints[:n, 0]
            kp[:n, 1] = result.keypoints[:n, 1]
            kp[:n, 2] = result.scores[:n]
            return kp

        # ---- multi-person path ----
        pose_results = self._pose_instance(image, person_boxes)
        coco = np.zeros((n_persons, 17, 3), dtype=np.float32)
        for i, result in enumerate(pose_results):
            n = min(17, len(result.keypoints))
            coco[i, :n, 0] = result.keypoints[:n, 0]
            coco[i, :n, 1] = result.keypoints[:n, 1]
            coco[i, :n, 2] = result.scores[:n]
        return coco

    def close(self) -> None:
        """Release QNN interpreter / DSP resources."""
        for model in (self._det, self._pose):
            if model is not None:
                model.close()
        self._det = None
        self._pose = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> RTMPose2dPoseExtractor:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
