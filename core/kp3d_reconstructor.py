from __future__ import annotations
import logging
from pathlib import Path
import sys
import numpy as np

_project_root = Path(__file__).resolve().parent.parent

_MHFORMER_DIR = _project_root / "mhformer-aidlite"
_DEFAULT_MODEL_DIR = str(_MHFORMER_DIR / "qnnmodel")

_mhformer_imported = False

logger = logging.getLogger("kp3d_reconstructor")


def _ensure_mhformer_imports() -> None:
    """Lazy-import qnn_reconstruct (requires aidlite SDK) on first use."""
    global _mhformer_imported
    if _mhformer_imported:
        return
    if str(_MHFORMER_DIR) not in sys.path:
        sys.path.insert(0, str(_MHFORMER_DIR))
    _mhformer_imported = True


class I3dPoseReconstructor:
    @property
    def data_out(self) -> str:
        raise NotImplementedError

    def reconstruct(self, kps2d_seq: np.ndarray, frame_index: int) -> np.ndarray:
        raise NotImplementedError


class Mock3dReconstructor(I3dPoseReconstructor):
    def __init__(self):
        self._kps_npz = np.load("./sample_data/example_3d_kps.npz")
        self._kps_frames: np.ndarray = self._kps_npz[self._kps_npz.files[0]]  # (Frames, 17, 3)
        self._kps_frame_count = self._kps_frames.shape[0]
        self._frame_index = 0
        logger.info(f"Loaded 3D keypoints: {self._kps_frames.shape}")

    @property
    def data_out(self) -> str:
        return "h36m_3d"

    def reconstruct(self, kps2d_seq: np.ndarray, frame_index: int) -> np.ndarray:
        """
        从 2D 关键点序列重建 3D 骨骼点。

        Args:
            kps2d_seq: 2D 关键点序列, shape=(T, 17, 2)，每行 [x, y]。
                T 是提供给 MHFormer 的时间维度长度，最大取决于模型需求，不足自动补齐。

        Returns:
            返回第 frame_index 帧的重建结果
        """
        kps = self._kps_frames[self._frame_index]
        self._frame_index = (self._frame_index + 1) % self._kps_frame_count
        return kps


class MHFormer3dPoseReconstructor(I3dPoseReconstructor):
    """Reconstruct 3D keypoints from H36M-format 2D keypoints using MHFormer.

    The underlying model is a temporal Transformer with a receptive field
    of 351 frames (±175).  For sequences shorter than the window, edge
    padding is applied automatically.  Supports both streaming (single
    latest frame via ``frame_index=-1``) and offline (multi-frame via
    ``frame_indices``) modes.

    Parameters
    ----------
    model_dir:
        Directory containing ``qnn_model_info.json`` and the compiled
        QNN model (``.aidem``).  Auto-detected when omitted.
    image_width:
        Image width in pixels for screen→NDC normalisation.  Default 640.
    image_height:
        Image height in pixels for screen→NDC normalisation.  Default 480.
    disable_flip:
        If ``True``, skip test-time horizontal-flip augmentation
        (faster but slightly less accurate).  Recommended for real-time.
    verbose:
        If ``True``, print per-frame inference timing to stdout.
    """

    def __init__(
        self,
        model_dir: str | None = None,
        image_width: float = 640.0,
        image_height: float = 480.0,
        disable_flip: bool = False,
        verbose: bool = False,
    ) -> None:
        self._model_dir = model_dir or _DEFAULT_MODEL_DIR
        self.image_width = image_width
        self.image_height = image_height
        self._disable_flip = disable_flip
        self._verbose = verbose
        self._instance: QNN3DReconstructor | None = None

    # ------------------------------------------------------------------
    # Lazy model instantiation (defers DSP load to first reconstruct())
    # ------------------------------------------------------------------

    @property
    def _reconstructor(self):
        if self._instance is None:
            _ensure_mhformer_imports()
            from lite_demo.qnn_reconstruct import QNN3DReconstructor  # noqa: E402

            self._instance = QNN3DReconstructor(
                model_dir=self._model_dir,
                disable_flip=self._disable_flip,
                verbose=self._verbose,
            )
        return self._instance

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        keypoints_2d: np.ndarray,
        frame_index: int = -1,
        *,
        frame_indices: list[int] | None = None,
        to_world: bool = True,
    ) -> np.ndarray:
        """Lift 2D keypoints to 3D.

        Two calling conventions are supported:

        * **Streaming** (default): ``frame_index`` — reconstruct a single
          frame.  Negative values count from the end (``-1`` = latest).
          Returns ``(17, 3)``.
        * **Offline**: ``frame_indices`` — reconstruct an explicit list of
          frames.  Returns ``(len(frame_indices), 17, 3)``.

        Parameters
        ----------
        keypoints_2d:
            H36M-format 2D keypoints of shape ``(T, 17, 2)`` in pixel
            coordinates, where *T* is the number of frames.
        frame_index:
            Single frame index to reconstruct.  Negative values are
            resolved relative to *T* (e.g. ``-1`` = last frame).
            Ignored when *frame_indices* is given.
        frame_indices:
            Explicit list of frame indices for multi-frame reconstruction.
            When provided, *frame_index* is ignored.
        to_world:
            If ``True`` (default), transform from camera space to world
            space using a fixed rotation and set root-joint *z* = 0.

        Returns
        -------
        np.ndarray
            ``(17, 3)`` when using *frame_index*; ``(N, 17, 3)`` when
            using *frame_indices*.  Coordinates are in world space when
            *to_world* is ``True``.
        """
        _validate_h36m_keypoints(keypoints_2d)
        total_frames = keypoints_2d.shape[0]

        if frame_indices is not None:
            # Multi-frame mode — return (N, 17, 3)
            return self._reconstructor.reconstruct(
                keypoints_2d=keypoints_2d,
                image_width=self.image_width,
                image_height=self.image_height,
                to_world=to_world,
                frame_indices=frame_indices,
            )

        # Single-frame mode — resolve negative index, return (17, 3)
        idx = int(frame_index)
        if idx < 0:
            idx = total_frames + idx

        if idx < 0 or idx >= total_frames:
            raise IndexError(f"frame_index {frame_index} out of range for {total_frames} frames (resolved to {idx})")

        result = self._reconstructor.reconstruct(
            keypoints_2d=keypoints_2d,
            image_width=self.image_width,
            image_height=self.image_height,
            to_world=to_world,
            frame_indices=[idx],
        )
        # result shape (1, 17, 3) → squeeze to (17, 3)
        return result[0]

    def close(self) -> None:
        """Release QNN interpreter / DSP resources."""
        if self._instance is not None:
            self._instance.close()
            self._instance = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> MHFormer3dPoseReconstructor:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _validate_h36m_keypoints(keypoints: np.ndarray) -> None:
    """Raise a clear error if the keypoints don't look like H36M 17-point."""
    if keypoints.ndim != 3:
        raise ValueError(
            f"Expected H36M keypoints with shape (T, 17, 2), got ndim={keypoints.ndim} shape={keypoints.shape}"
        )
    if keypoints.shape[-2:] != (17, 2):
        raise ValueError(f"Expected keypoints ending with (17, 2), got shape {keypoints.shape}")
