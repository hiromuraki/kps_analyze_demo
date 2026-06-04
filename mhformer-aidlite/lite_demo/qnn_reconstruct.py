import json
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

import  aidlite


DEFAULT_ROTATION = np.array(
    [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
    dtype=np.float32,
)
JOINTS_LEFT = [4, 5, 6, 11, 12, 13]
JOINTS_RIGHT = [1, 2, 3, 14, 15, 16]


def normalize_screen_coordinates(points_2d: np.ndarray, width: float, height: float) -> np.ndarray:
    return points_2d / width * 2.0 - np.array([1.0, height / width], dtype=np.float32)


def qrot(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    qvec = quaternion[..., 1:]
    uv = np.cross(qvec, vectors)
    uuv = np.cross(qvec, uv)
    return vectors + 2.0 * (quaternion[..., :1] * uv + uuv)


def camera_to_world(points_3d: np.ndarray, rotation: np.ndarray, translation: float = 0.0) -> np.ndarray:
    tiled_rotation = np.broadcast_to(rotation, points_3d.shape[:-1] + (4,))
    return qrot(tiled_rotation, points_3d) + translation


def resolve_video_size(video_path: str) -> tuple[float, float]:
    capture = cv2.VideoCapture(video_path)
    try:
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Failed to read video size from: {video_path}")
    return float(width), float(height)


def load_keypoints(path: str, person_index: int) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    keypoints = data["reconstruction"] if "reconstruction" in data else data[next(iter(data.files))]
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim == 4:
        keypoints = keypoints[person_index]
    if keypoints.ndim != 3 or keypoints.shape[-2:] != (17, 2):
        raise ValueError(f"Expected [T, 17, 2] or [M, T, 17, 2], got {keypoints.shape}")
    return keypoints


@dataclass
class QNNModelSpec:
    model_path: str
    input_name: str
    input_shape: list[int]
    output_name: str
    output_shape: list[int]
    data_type: aidlite.DataType
    framework_type: aidlite.FrameworkType
    backend_extension_config: str | None = None


def load_model_spec(model_dir: str) -> QNNModelSpec:
    info_path = os.path.join(model_dir, "qnn_model_info.json")
    with open(info_path, "r", encoding="utf-8") as handle:
        info = json.load(handle)

    input_name, input_shape = next(iter(info["inputDimensions"].items()))
    output_name, output_shape = next(iter(info["outputDimensions"].items()))
    data_type = aidlite.DataType.TYPE_FLOAT16 if str(info.get("data_type", "fp16")).lower() == "fp16" else aidlite.DataType.TYPE_FLOAT32
    backend_extension_config = os.path.join(model_dir, "htp_backend_extensions.json")
    if not os.path.exists(backend_extension_config):
        backend_extension_config = None

    return QNNModelSpec(
        model_path=os.path.join(model_dir, info["model_name"]),
        input_name=input_name,
        input_shape=list(map(int, input_shape)),
        output_name=output_name,
        output_shape=list(map(int, output_shape)),
        data_type=data_type,
        framework_type=aidlite.FrameworkType.TYPE_QNN236,
        backend_extension_config=backend_extension_config,
    )


class QNN3DReconstructor:
    def __init__(self, model_dir: str, disable_flip: bool = False, verbose: bool = False):
        self.model_dir = model_dir
        self.spec = load_model_spec(model_dir)
        self.frames = self.spec.input_shape[1]
        self.pad = (self.frames - 1) // 2
        self.disable_flip = disable_flip
        self.verbose = verbose
        env_backend_config = os.environ.get("AIDLITE_BACKEND_EXTENSION_CONFIG")
        self.runtime_backend_config = env_backend_config if env_backend_config and os.path.exists(env_backend_config) else None
        self.interpreter = self._build_interpreter()

    def _build_interpreter(self):
        model = aidlite.Model.create_instance(self.spec.model_path)
        if model is None:
            raise FileNotFoundError(f"Failed to open model: {self.spec.model_path}")
        model.set_model_properties([self.spec.input_shape], self.spec.data_type, [self.spec.output_shape], self.spec.data_type)

        preferred = os.environ.get("AIDLITE_ACCELERATE", "TYPE_DSP").upper()
        if not preferred.startswith("TYPE_"):
            preferred = f"TYPE_{preferred}"
        fallback_order = [preferred, "TYPE_NPU", "TYPE_CPU"]

        errors = []
        for accelerate_name in dict.fromkeys(fallback_order):
            accelerate_type = getattr(aidlite.AccelerateType, accelerate_name, None)
            if accelerate_type is None:
                continue

            config = aidlite.Config.create_instance()
            config.framework_type = self.spec.framework_type
            config.accelerate_type = accelerate_type
            if self.runtime_backend_config:
                config.backend_extension_config = self.runtime_backend_config

            interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(model, config)
            try:
                interpreter.init()
                interpreter.load_model()
                return interpreter
            except Exception as exc:
                errors.append(f"{accelerate_name}: {exc}")

        error_detail = "; ".join(errors) if errors else "no valid AccelerateType found"
        raise RuntimeError(f"Failed to initialize aidlite interpreter ({error_detail})")

    def close(self):
        if self.interpreter is not None:
            self.interpreter.destory()
            self.interpreter = None

    def __del__(self):
        self.close()

    def _infer(self, input_window: np.ndarray) -> np.ndarray:
        self.interpreter.set_input_tensor(self.spec.input_name, np.asarray(input_window, dtype=np.float32))
        t0 = time.time()
        self.interpreter.invoke()
        t1 = time.time()
        self._infer_time = t1 - t0
        if self.verbose:
            print(f"[MHFormer] 推理耗时: {(t1 - t0)*1000:.2f} ms")
        output = self.interpreter.get_output_tensor(self.spec.output_name, aidlite.DataType.TYPE_FLOAT32)
        return np.array(output, dtype=np.float32, copy=True).reshape(self.spec.output_shape)

    def reconstruct(
        self,
        keypoints_2d: np.ndarray,
        image_width: float,
        image_height: float,
        to_world: bool = True,
        frame_indices: list[int] | None = None,
    ) -> np.ndarray:
        total_frames = keypoints_2d.shape[0]
        if frame_indices is None:
            frame_indices = list(range(total_frames))

        outputs = []
        for frame_idx in frame_indices:
            start = max(0, frame_idx - self.pad)
            end = min(frame_idx + self.pad, total_frames - 1)
            window = keypoints_2d[start:end + 1]

            left_pad = max(0, self.pad - frame_idx)
            right_pad = max(0, frame_idx + self.pad - (total_frames - 1))
            if left_pad or right_pad:
                window = np.pad(window, ((left_pad, right_pad), (0, 0), (0, 0)), mode="edge")

            normalized = normalize_screen_coordinates(window, image_width, image_height)
            output = self._infer(normalized[np.newaxis, ...])

            if not self.disable_flip:
                flipped = normalized.copy()
                flipped[:, :, 0] *= -1
                flipped[:, JOINTS_LEFT + JOINTS_RIGHT] = flipped[:, JOINTS_RIGHT + JOINTS_LEFT]
                output_flip = self._infer(flipped[np.newaxis, ...])
                output_flip[:, :, :, 0] *= -1
                output_flip[:, :, JOINTS_LEFT + JOINTS_RIGHT, :] = output_flip[:, :, JOINTS_RIGHT + JOINTS_LEFT, :]
                output = (output + output_flip) / 2.0

            center_frame = output[0, self.pad]
            center_frame[0, :] = 0.0

            if to_world:
                center_frame = camera_to_world(center_frame, DEFAULT_ROTATION)
                center_frame[:, 2] -= np.min(center_frame[:, 2])

            outputs.append(center_frame)
        return np.stack(outputs, axis=0)


def reconstruct_from_files(
    keypoints_path: str,
    video_path: str,
    output_path: str,
    model_dir: str,
    person_index: int = 0,
    frame_indices: list[int] | None = None,
    disable_flip: bool = False,
) -> str:
    keypoints_2d = load_keypoints(keypoints_path, person_index=person_index)
    image_width, image_height = resolve_video_size(video_path)
    reconstructor = QNN3DReconstructor(model_dir=model_dir, disable_flip=disable_flip)
    try:
        output_3d = reconstructor.reconstruct(
            keypoints_2d=keypoints_2d,
            image_width=image_width,
            image_height=image_height,
            to_world=True,
            frame_indices=frame_indices,
        )
    finally:
        reconstructor.close()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, reconstruction=output_3d)
    return output_path
