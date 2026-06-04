from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import time
import numpy as np

from aidlite import (
    AccelerateType,
    Config,
    DataType,
    FrameworkType,
    ImplementType,
    InterpreterBuilder,
    Model,
)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _require_cv2():
    if cv2 is None:
        raise ImportError("opencv-python is required for vision preprocessing")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _to_float32_array(raw_tensor: Any, expected_shape: list[int] | tuple[int, ...]) -> np.ndarray:
    payload = raw_tensor
    if hasattr(payload, "data") and not isinstance(payload, np.ndarray):
        payload = payload.data

    if isinstance(payload, np.ndarray):
        arr = payload.astype(np.float32, copy=True)
    elif isinstance(payload, memoryview):
        arr = np.frombuffer(payload, dtype=np.float32)
    elif isinstance(payload, (bytes, bytearray)):
        arr = np.frombuffer(payload, dtype=np.float32)
    else:
        arr = np.asarray(payload, dtype=np.float32)

    if expected_shape:
        expected_size = int(np.prod(expected_shape))
        if arr.size == expected_size:
            arr = arr.reshape(expected_shape)
        else:
            raise ValueError(
                f"Output tensor has {arr.size} values, expected {expected_size} "
                f"for shape {list(expected_shape)}"
            )
    return np.array(arr, dtype=np.float32, copy=True)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)
        order = order[np.where(iou <= iou_thr)[0] + 1]
    return np.asarray(keep, dtype=np.int64)


def _letterbox(image: np.ndarray, size: tuple[int, int], pad_val: int = 114):
    _require_cv2()
    target_w, target_h = size
    src_h, src_w = image.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    canvas = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(pad_val, pad_val, pad_val),
    )
    return canvas, {"scale": scale, "pad": (left, top), "orig_shape": (src_h, src_w)}


def _build_interpreter(
    model_path: str,
    input_shapes: list[list[int]],
    output_shapes: list[list[int]],
):
    model_name = Path(model_path).name.lower()
    model = Model.create_instance(model_path)
    if model is None:
        raise FileNotFoundError(f"Model not found: {model_path}")
    model.set_model_properties(
        input_shapes,
        DataType.TYPE_FLOAT32,
        output_shapes,
        DataType.TYPE_FLOAT32,
    )

    config = Config.create_instance()
    config.accelerate_type = AccelerateType.TYPE_DSP
    config.implement_type = ImplementType.TYPE_LOCAL
    if "qnn236" in model_name:
        config.framework_type = FrameworkType.TYPE_QNN236
    elif "qnn240" in model_name:
        config.framework_type = FrameworkType.TYPE_QNN240
    elif "qnn231" in model_name:
        config.framework_type = FrameworkType.TYPE_QNN231
    elif "qnn229" in model_name:
        config.framework_type = FrameworkType.TYPE_QNN229
    elif "qnn223" in model_name:
        config.framework_type = FrameworkType.TYPE_QNN223
    elif "qnn216" in model_name:
        config.framework_type = FrameworkType.TYPE_QNN216
    elif ".bin" in model_name:
        config.framework_type = FrameworkType.TYPE_QNN

    interpreter = InterpreterBuilder.build_interpretper_from_model_and_config(
        model, config
    )
    init_result = interpreter.init()
    if init_result is False or (
        isinstance(init_result, int) and not isinstance(init_result, bool) and init_result != 0
    ):
        raise RuntimeError(
            f"aidlite init failed for model {model_path}. Ensure DSP runtime and QNN version are available."
        )
    load_result = interpreter.load_model()
    if load_result is False or (
        isinstance(load_result, int) and not isinstance(load_result, bool) and load_result != 0
    ):
        raise RuntimeError(f"aidlite load_model failed for model {model_path}")
    return interpreter


@dataclass
class DetectionResult:
    bboxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray


class RTMDet:
    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (640, 640),
        score_thr: float = 0.3,
        iou_thr: float = 0.65,
        max_per_img: int = 100,
        nms_pre: int = 1000,
        num_classes: int = 80,
        verbose: bool = False,
    ) -> None:
        self.input_size = input_size
        self.score_thr = score_thr
        self.iou_thr = iou_thr
        self.max_per_img = max_per_img
        self.nms_pre = nms_pre
        self.num_classes = num_classes
        self.verbose = verbose
        self.strides = (8, 16, 32)
        self.output_shapes = [
            [1, 80, 80, num_classes],
            [1, 80, 80, 4],
            [1, 40, 40, num_classes],
            [1, 40, 40, 4],
            [1, 20, 20, num_classes],
            [1, 20, 20, 4],
        ]
        self.interpreter = _build_interpreter(
            model_path,
            input_shapes=[[1, input_size[1], input_size[0], 3]],
            output_shapes=self.output_shapes,
        )
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        interpreter = self.interpreter
        self.interpreter = None
        destroy = getattr(interpreter, "destory", None)
        if destroy is not None:
            destroy()

    def preprocess(self, image_bgr: np.ndarray):
        resized, meta = _letterbox(image_bgr, self.input_size)
        tensor = resized[None].astype(np.float32)
        return tensor, meta

    def infer_raw(self, image_bgr: np.ndarray) -> tuple[list[np.ndarray], dict[str, Any]]:
        tensor, meta = self.preprocess(image_bgr)
        self.interpreter.set_input_tensor(0, tensor)
        t0 = time.time()
        self.interpreter.invoke()
        t1 = time.time()
        self._infer_time = t1 - t0
        if self.verbose:
            print(f"[RTMDet] 推理耗时: {(t1 - t0)*1000:.2f} ms")
        outputs = [
            _to_float32_array(
                self.interpreter.get_output_tensor(i), self.output_shapes[i]
            )
            for i in range(6)
        ]
        return outputs, meta

    def postprocess(self, outputs: list[np.ndarray], meta: dict[str, Any]) -> DetectionResult:
        all_boxes = []
        all_scores = []
        all_labels = []
        input_w, input_h = self.input_size
        for level, stride in enumerate(self.strides):
            cls_score = outputs[level * 2][0]
            bbox_pred = outputs[level * 2 + 1][0]
            feat_h, feat_w, _ = cls_score.shape
            scores = _sigmoid(cls_score).reshape(-1, self.num_classes)
            bbox_pred = bbox_pred.reshape(-1, 4)

            yy, xx = np.meshgrid(
                np.arange(feat_h, dtype=np.float32) * stride,
                np.arange(feat_w, dtype=np.float32) * stride,
                indexing="ij",
            )
            priors = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
            decoded = np.stack(
                [
                    priors[:, 0] - bbox_pred[:, 0],
                    priors[:, 1] - bbox_pred[:, 1],
                    priors[:, 0] + bbox_pred[:, 2],
                    priors[:, 1] + bbox_pred[:, 3],
                ],
                axis=-1,
            )
            decoded[:, 0::2] = np.clip(decoded[:, 0::2], 0, input_w)
            decoded[:, 1::2] = np.clip(decoded[:, 1::2], 0, input_h)

            valid_mask = scores > self.score_thr
            keep_idxs, labels = np.nonzero(valid_mask)
            if keep_idxs.size == 0:
                continue
            picked_scores = scores[keep_idxs, labels]
            if picked_scores.size > self.nms_pre:
                topk = np.argsort(-picked_scores)[: self.nms_pre]
                keep_idxs = keep_idxs[topk]
                labels = labels[topk]
                picked_scores = picked_scores[topk]

            all_boxes.append(decoded[keep_idxs])
            all_scores.append(picked_scores)
            all_labels.append(labels.astype(np.int64))

        if not all_boxes:
            return DetectionResult(
                bboxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                labels=np.empty((0,), dtype=np.int64),
            )

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        max_wh = 4096.0
        nms_boxes = boxes.copy()
        nms_boxes[:, 0::2] += labels[:, None] * max_wh
        keep = _nms(nms_boxes, scores, self.iou_thr)
        keep = keep[: self.max_per_img]

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        pad_x, pad_y = meta["pad"]
        scale = meta["scale"]
        orig_h, orig_w = meta["orig_shape"]
        boxes[:, 0::2] = (boxes[:, 0::2] - pad_x) / max(scale, 1e-6)
        boxes[:, 1::2] = (boxes[:, 1::2] - pad_y) / max(scale, 1e-6)
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, orig_w)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, orig_h)

        return DetectionResult(bboxes=boxes, scores=scores, labels=labels)

    def __call__(self, image_bgr: np.ndarray) -> DetectionResult:
        outputs, meta = self.infer_raw(image_bgr)
        return self.postprocess(outputs, meta)


@dataclass
class PoseResult:
    keypoints: np.ndarray
    scores: np.ndarray
    bbox: np.ndarray


class RTMPose:
    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (192, 256),
        simcc_split_ratio: float = 2.0,
        bbox_padding: float = 1.25,
        verbose: bool = False,
    ) -> None:
        self.input_size = input_size
        self.simcc_split_ratio = simcc_split_ratio
        self.bbox_padding = bbox_padding
        self.verbose = verbose
        self.output_shapes = [[1, 17, 384], [1, 17, 512]]
        self.interpreter = _build_interpreter(
            model_path,
            input_shapes=[[1, input_size[1], input_size[0], 3]],
            output_shapes=self.output_shapes,
        )
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        interpreter = self.interpreter
        self.interpreter = None
        destroy = getattr(interpreter, "destory", None)
        if destroy is not None:
            destroy()

    @staticmethod
    def _fix_aspect_ratio(scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
        w, h = scale
        if w > h * aspect_ratio:
            return np.array([w, w / aspect_ratio], dtype=np.float32)
        return np.array([h * aspect_ratio, h], dtype=np.float32)

    @staticmethod
    def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        direction = a - b
        return b + np.array([-direction[1], direction[0]], dtype=np.float32)

    def _get_warp_matrix(
        self,
        center: np.ndarray,
        scale: np.ndarray,
        output_size: tuple[int, int],
    ) -> np.ndarray:
        _require_cv2()
        src_w = scale[0]
        dst_w, dst_h = output_size
        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center
        src[1, :] = center + np.array([src_w * -0.5, 0.0], dtype=np.float32)
        src[2, :] = self._get_3rd_point(src[0, :], src[1, :])
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = dst[0, :] + np.array([dst_w * -0.5, 0.0], dtype=np.float32)
        dst[2, :] = self._get_3rd_point(dst[0, :], dst[1, :])
        return cv2.getAffineTransform(src, dst)

    def preprocess(self, image_bgr: np.ndarray, bbox_xyxy: np.ndarray):
        _require_cv2()
        bbox = np.asarray(bbox_xyxy, dtype=np.float32)
        center = (bbox[:2] + bbox[2:]) * 0.5
        scale = (bbox[2:] - bbox[:2]) * self.bbox_padding
        scale = self._fix_aspect_ratio(scale, self.input_size[0] / self.input_size[1])
        warp_mat = self._get_warp_matrix(center, scale, self.input_size)
        crop = cv2.warpAffine(
            image_bgr,
            warp_mat,
            self.input_size,
            flags=cv2.INTER_LINEAR,
        )
        tensor = crop[None].astype(np.float32)
        return tensor, {"warp_mat": warp_mat, "bbox": bbox}

    def infer_raw(self, image_bgr: np.ndarray, bbox_xyxy: np.ndarray):
        tensor, meta = self.preprocess(image_bgr, bbox_xyxy)
        self.interpreter.set_input_tensor(0, tensor)
        t0 = time.time()
        self.interpreter.invoke()
        t1 = time.time()
        self._infer_time = t1 - t0
        if self.verbose:
            print(f"[RTMPose] 推理耗时: {(t1 - t0)*1000:.2f} ms")
        simcc_x = _to_float32_array(
            self.interpreter.get_output_tensor(0), self.output_shapes[0]
        )
        simcc_y = _to_float32_array(
            self.interpreter.get_output_tensor(1), self.output_shapes[1]
        )
        return (simcc_x, simcc_y), meta

    def postprocess(
        self,
        outputs: tuple[np.ndarray, np.ndarray],
        meta: dict[str, Any],
    ) -> PoseResult:
        _require_cv2()
        simcc_x, simcc_y = outputs
        simcc_x = simcc_x[0]
        simcc_y = simcc_y[0]
        x_locs = np.argmax(simcc_x, axis=1).astype(np.float32)
        y_locs = np.argmax(simcc_y, axis=1).astype(np.float32)
        max_val_x = np.max(simcc_x, axis=1)
        max_val_y = np.max(simcc_y, axis=1)
        scores = np.minimum(max_val_x, max_val_y).astype(np.float32)

        keypoints = np.stack([x_locs, y_locs], axis=-1) / self.simcc_split_ratio
        inv_warp = cv2.invertAffineTransform(meta["warp_mat"])
        keypoints = cv2.transform(keypoints[None, :, :], inv_warp)[0]
        keypoints[scores <= 0.0] = -1
        return PoseResult(keypoints=keypoints, scores=scores, bbox=meta["bbox"])

    def __call__(self, image_bgr: np.ndarray, bboxes_xyxy: np.ndarray) -> list[PoseResult]:
        results = []
        for bbox in np.asarray(bboxes_xyxy, dtype=np.float32):
            outputs, meta = self.infer_raw(image_bgr, bbox)
            results.append(self.postprocess(outputs, meta))
        return results
