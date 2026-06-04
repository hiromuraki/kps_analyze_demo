"""COCO 17-point ↔ H36M 17-point keypoint format conversion.

Joint orders
------------

COCO 17-point (source: RTMPose)::

    [0]  nose          [6]  right_shoulder  [12] right_hip
    [1]  left_eye      [7]  left_elbow      [13] left_knee
    [2]  right_eye     [8]  right_elbow     [14] right_knee
    [3]  left_ear      [9]  left_wrist      [15] left_ankle
    [4]  right_ear    [10]  right_wrist     [16] right_ankle
    [5]  left_shoulder [11] left_hip

H36M 17-point (target: MHFormer)::

    [0]  pelvis        [6]  left_ankle      [12] left_elbow
    [1]  right_hip     [7]  spine           [13] left_wrist
    [2]  right_knee    [8]  thorax          [14] right_shoulder
    [3]  right_ankle   [9]  neck (nose)     [15] right_elbow
    [4]  left_hip      [10] head_top        [16] right_wrist
    [5]  left_knee     [11] left_shoulder

Usage::

    from core.converter import DataConverter

    h36m = DataConverter.coco17_to_h36m(coco_kp)  # (N,17,3)→(N,17,2) or (17,3)→(17,2)
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Joint-index maps
# ---------------------------------------------------------------------------

_COCO = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

_H36M = {
    "pelvis": 0,
    "r_hip": 1,
    "r_knee": 2,
    "r_ankle": 3,
    "l_hip": 4,
    "l_knee": 5,
    "l_ankle": 6,
    "spine": 7,
    "thorax": 8,
    "neck": 9,
    "head": 10,
    "l_shoulder": 11,
    "l_elbow": 12,
    "l_wrist": 13,
    "r_shoulder": 14,
    "r_elbow": 15,
    "r_wrist": 16,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class DataConverter:
    """Static methods for keypoint format conversion.

    All methods are stateless — no instantiation needed.
    """

    @staticmethod
    def coco17_to_h36m(coco_keypoints: np.ndarray) -> np.ndarray:
        """Convert COCO-17 keypoints to H36M-17 format.

        Directly-mapped joints (11) are copied; interpolated joints (6)
        are computed from their neighbours:

        =============== ==========================================
        H36M joint       Source
        =============== ==========================================
        pelvis           mean of left_hip + right_hip
        thorax           mean of left_shoulder + right_shoulder
        spine            mean of pelvis + thorax
        neck             nose
        head_top         eyes_centre + 1.5×(eyes_centre − nose)
        =============== ==========================================

        Parameters
        ----------
        coco_keypoints:
            ``(17, 3)`` or ``(N, 17, 3)`` — COCO keypoints with
            ``(x, y, confidence)`` per joint.  Only ``x``, ``y``
            are used; confidence is discarded.

        Returns
        -------
        np.ndarray
            ``(17, 2)`` or ``(N, 17, 2)`` — H36M keypoints in
            pixel coordinates, no confidence channel.
        """
        original_ndim = coco_keypoints.ndim
        if original_ndim == 2:
            coco_keypoints = coco_keypoints[np.newaxis, ...]

        if coco_keypoints.ndim != 3 or coco_keypoints.shape[-2:] != (17, 3):
            raise ValueError(f"Expected COCO keypoints of shape (..., 17, 3), got {coco_keypoints.shape}")

        n = coco_keypoints.shape[0]
        h36m = np.zeros((n, 17, 2), dtype=np.float32)

        for i in range(n):
            coco = coco_keypoints[i]

            # ---- direct mappings (11 joints) ----
            h36m[i, _H36M["r_hip"]] = coco[_COCO["right_hip"], :2]
            h36m[i, _H36M["r_knee"]] = coco[_COCO["right_knee"], :2]
            h36m[i, _H36M["r_ankle"]] = coco[_COCO["right_ankle"], :2]
            h36m[i, _H36M["l_hip"]] = coco[_COCO["left_hip"], :2]
            h36m[i, _H36M["l_knee"]] = coco[_COCO["left_knee"], :2]
            h36m[i, _H36M["l_ankle"]] = coco[_COCO["left_ankle"], :2]
            h36m[i, _H36M["l_shoulder"]] = coco[_COCO["left_shoulder"], :2]
            h36m[i, _H36M["l_elbow"]] = coco[_COCO["left_elbow"], :2]
            h36m[i, _H36M["l_wrist"]] = coco[_COCO["left_wrist"], :2]
            h36m[i, _H36M["r_shoulder"]] = coco[_COCO["right_shoulder"], :2]
            h36m[i, _H36M["r_elbow"]] = coco[_COCO["right_elbow"], :2]
            h36m[i, _H36M["r_wrist"]] = coco[_COCO["right_wrist"], :2]

            # ---- interpolated joints (6 joints) ----
            l_hip = coco[_COCO["left_hip"], :2]
            r_hip = coco[_COCO["right_hip"], :2]
            pelvis = (l_hip + r_hip) * 0.5
            h36m[i, _H36M["pelvis"]] = pelvis

            l_shoulder = coco[_COCO["left_shoulder"], :2]
            r_shoulder = coco[_COCO["right_shoulder"], :2]
            thorax = (l_shoulder + r_shoulder) * 0.5
            h36m[i, _H36M["thorax"]] = thorax

            spine = (pelvis + thorax) * 0.5
            h36m[i, _H36M["spine"]] = spine

            nose = coco[_COCO["nose"], :2]
            h36m[i, _H36M["neck"]] = nose

            left_eye = coco[_COCO["left_eye"], :2]
            right_eye = coco[_COCO["right_eye"], :2]
            eyes_centre = (left_eye + right_eye) * 0.5
            head_top = eyes_centre + (eyes_centre - nose) * 1.5
            h36m[i, _H36M["head"]] = head_top

        if original_ndim == 2:
            # Input was (17, 3) → strip the batch dim we added
            return h36m[0]

        return h36m
