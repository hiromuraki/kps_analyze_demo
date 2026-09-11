from __future__ import annotations
from pathlib import Path
from typing import Literal
from .interface import I3dPoseReconstructor
import logging
import sys
import numpy as np

_project_root = Path(__file__).resolve().parent.parent.parent

_MHFORMER_DIR = _project_root / "mhformer-aidlite"
_DEFAULT_MODEL_DIR = str(_MHFORMER_DIR / "qnnmodel")

_mhformer_imported = False

logger = logging.getLogger("kp3d_reconstructor")


def _ensure_mhformer_imports() -> None:
    """把 ``mhformer-aidlite`` 加入 ``sys.path``，使 ``lite_demo.qnn_reconstruct`` 可被导入。

    以模块级标志位保证只执行一次。真正的 ``import`` 推迟到首次 ``reconstruct()``，
    避免无 QNN 硬件、缺 aidlite SDK 的环境在模块导入阶段就失败。
    """
    global _mhformer_imported
    if _mhformer_imported:
        return
    if str(_MHFORMER_DIR) not in sys.path:
        sys.path.insert(0, str(_MHFORMER_DIR))
    _mhformer_imported = True


class MHFormer3dPoseReconstructor(I3dPoseReconstructor):
    """3D 重建器的 MHFormer 实现：时序 Transformer + QNN DSP 推理。

    模型感受野为 351 帧（当前帧前后各 175 帧）。序列短于窗口时自动做边缘
    填充；每帧的预测以该帧为中心的窗口为上下文。

    本实现依赖 QNN 硬件与 aidlite SDK。模型在首次调用 ``reconstruct()`` 时
    才真正加载（惰性），因此构造本类本身不会占用 DSP 资源。

    Args:
        model_dir: 含 ``qnn_model_info.json`` 与编译后 ``.aidem`` 模型的目录。
            省略时使用 ``mhformer-aidlite/qnnmodel``。
        disable_flip: 为 ``True`` 时跳过测试时翻转增强（TTA）。速度更快、
            精度略降，实时场景建议开启。
        verbose: 为 ``True`` 时打印每帧推理耗时。

    Note:
        用完请调用 ``close()`` 释放 DSP 资源，或用 ``with`` 语句管理生命周期。
    """

    def __init__(
        self,
        model_dir: str | None = None,
        disable_flip: bool = False,
        verbose: bool = False,
    ) -> None:
        self._model_dir = model_dir or _DEFAULT_MODEL_DIR
        self._disable_flip = disable_flip
        self._verbose = verbose
        self._instance: QNN3DReconstructor | None = None

    # ------------------------------------------------------------------
    # 惰性构造：把 DSP 加载推迟到首次 reconstruct()
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
    # 公开方法
    # ------------------------------------------------------------------
    @property
    def data_out(self) -> Literal["H36M_3D"]:
        return "H36M_3D"

    def reconstruct(
        self,
        kps2d_seq: np.ndarray,
        frame_indices: list[int],
        frame_size: tuple[int, int],
        *,
        to_world: bool = True,
    ) -> np.ndarray:
        """从 2D 关键点序列重建指定帧的 3D 坐标。

        契约见接口 ``I3dPoseReconstructor.reconstruct``。本实现补充如下：

        - 负索引由**本层**解析。底层 ``QNN3DReconstructor`` 不解析负值，会把它
          当作普通位置参与窗口计算，最终静默落到第 0 帧附近，因此必须在调用
          底层前归一化为绝对索引。
        - 索引越界时抛 ``IndexError``，避免底层在切片为空时报出难以定位的
          numpy 错误。
        - 每帧需一次 DSP 前向推理（开启 TTA 时为两次），开销远高于 mock 实现。

        Args:
            to_world: 为 ``True``（默认）时，把结果从相机坐标系按固定旋转四元数
                转到世界坐标系，并将根关节的 z 归零。注意这是**刚体旋转**——
                它不改变关节夹角，只改变坐标数值与各轴尺度。

        Returns:
            见接口说明：``shape=(N, 17, 3)``，``N == len(frame_indices)``。

        Raises:
            ValueError: ``keypoints_2d`` 形状不是 ``(T, 17, 2)``。
            IndexError: 某个索引（负索引解析后）超出 ``[0, T)``。
        """
        _validate_h36m_keypoints(kps2d_seq)
        total_frame_count = kps2d_seq.shape[0]

        image_width = frame_size[0]
        image_height = frame_size[1]
        
        # 负索引必须在这里解析：底层 QNN3DReconstructor 不解析负值，
        # 它会当成位置参与 max(0, idx - pad)，最终静默钳制到第 0 帧。
        resolved_indices = [i + total_frame_count if i < 0 else i for i in frame_indices]
        for requested, idx in zip(frame_indices, resolved_indices):
            if idx < 0 or idx >= total_frame_count:
                raise IndexError(
                    f"frame index {requested} out of range for {total_frame_count} frames (resolved to {idx})"
                )

        # 接口约定返回 (N, 17, 3)
        return self._reconstructor.reconstruct(
            keypoints_2d=kps2d_seq,
            image_width=image_width,
            image_height=image_height,
            to_world=to_world,
            frame_indices=resolved_indices,
        )

    def close(self) -> None:
        """释放 QNN 解释器与占用的 DSP 资源。

        可重复调用。释放后再次调用 ``reconstruct()`` 会触发模型重新加载。
        """
        if self._instance is not None:
            self._instance.close()
            self._instance = None

    # ------------------------------------------------------------------
    # 上下文管理器
    # ------------------------------------------------------------------

    def __enter__(self) -> MHFormer3dPoseReconstructor:
        return self

    def __exit__(self, *args: object) -> None:
        """退出 ``with`` 块时释放模型资源。"""
        self.close()


def _validate_h36m_keypoints(keypoints: np.ndarray) -> None:
    """校验入参确实是 H36M 17 点的 2D 关键点序列。

    Raises:
        ValueError: 维度不为 3，或末两维不是 ``(17, 2)``。
    """
    if keypoints.ndim != 3:
        raise ValueError(
            f"Expected H36M keypoints with shape (T, 17, 2), got ndim={keypoints.ndim} shape={keypoints.shape}"
        )
    if keypoints.shape[-2:] != (17, 2):
        raise ValueError(f"Expected keypoints ending with (17, 2), got shape {keypoints.shape}")
