import numpy as np


class DataConverter:
    @classmethod
    def coco17_to_h36m_with_confidence(
        cls,
        coco_keypoints: np.ndarray,  # shape: [Person, Frames, 17, 3] (x, y, confidence)
    ) -> np.ndarray:  # shape: [Person, Frames, 17, 3] (x, y, confidence)
        """
        将 COCO (17点) 转换为 H36M (17点) 格式，并保留置信度信息。

        输入：coco_keypoints (np.ndarray): 形状必须严格为 (Frames, 17, 3) 的三维数组，代表 (帧数, 关节数, [x 坐标, y 坐标, confidence])。
        输出：h36m_keypoints (np.ndarray): 形状为 (Frames, 17, 3) 的 H36M 格式数组，包含 (x, y, confidence)。

        注意：由于 COCO 没有骨盆(Pelvis)、脊椎(Spine)和胸骨(Thorax)，该方法通过已知点插值计算得出，并将置信度设置为相关关键点的平均值。
        """

        # 维度验证
        if coco_keypoints.ndim != 3:
            raise ValueError(
                f"输入数据维度错误！预期为 3 维数组 (Frames, 17, 2)，"
                f"但收到了 {coco_keypoints.ndim} 维数组，实际 Shape: {coco_keypoints.shape}"
            )

        if coco_keypoints.shape[1] != 17:
            raise ValueError(
                f"输入关键点数量错误！预期为 17 个关键点，"
                f"但收到了 {coco_keypoints.shape[1]} 个点，实际 Shape: {coco_keypoints.shape}"
            )

        if coco_keypoints.shape[2] < 3:
            raise ValueError(
                f"输入坐标维度错误！预期至少包含 (x, y, confidence) 三个维度，实际 Shape: {coco_keypoints.shape}"
            )

        # 输入 shape 为 (Frames, 17, 3)
        frame_count = coco_keypoints.shape[0]
        h36m_keypoints = np.zeros((frame_count, 17, 3))

        # 提取 COCO 关键点 (参考 COCO 索引)
        l_shoulder, r_shoulder = coco_keypoints[:, 5], coco_keypoints[:, 6]
        l_hip, r_hip = coco_keypoints[:, 11], coco_keypoints[:, 12]

        # 1. 计算 H36M 缺失的核心关节点
        pelvis = (l_hip + r_hip) / 2.0  # 骨盆 = 左右髋关节中点
        thorax = (l_shoulder + r_shoulder) / 2.0  # 胸骨 = 左右肩中点
        spine = (pelvis + thorax) / 2.0  # 脊椎 = 骨盆与胸骨中点

        # 2. 按照 H36M 的 17 点顺序进行映射 (具体索引以 MotionAGFormer 源码定义为准)
        # H36M 常见顺序: 0:Pelvis, 1:R_Hip, 2:R_Knee, 3:R_Foot, 4:L_Hip, 5:L_Knee, 6:L_Foot,
        # 7:Spine, 8:Thorax, 9:Neck/Nose, 10:Head, 11:L_Shoulder, 12:L_Elbow, 13:L_Wrist,
        # 14:R_Shoulder, 15:R_Elbow, 16:R_Wrist

        h36m_keypoints[:, 0] = pelvis
        h36m_keypoints[:, 1] = r_hip
        h36m_keypoints[:, 2] = coco_keypoints[:, 14]  # R_Knee
        h36m_keypoints[:, 3] = coco_keypoints[:, 16]  # R_Ankle (近似为 Foot)
        h36m_keypoints[:, 4] = l_hip
        h36m_keypoints[:, 5] = coco_keypoints[:, 13]  # L_Knee
        h36m_keypoints[:, 6] = coco_keypoints[:, 15]  # L_Ankle
        h36m_keypoints[:, 7] = spine
        h36m_keypoints[:, 8] = thorax
        h36m_keypoints[:, 9] = coco_keypoints[:, 0]  # Nose (作为 Neck 的近似)

        # H36M 的第 10 点 (Head/头顶):
        # 由于COCO里没有头顶，我们用左眼(1)和右眼(2)的中点向上推算
        # 同时为了视觉上更自然，我们将双眼中点稍微向上（Y轴减小）偏移一点点，模拟头顶位置
        l_eye = coco_keypoints[:, 1]
        r_eye = coco_keypoints[:, 2]
        eyes_center = (l_eye + r_eye) / 2.0
        head_top = eyes_center + (eyes_center - coco_keypoints[:, 0]) * 1.5
        h36m_keypoints[:, 10] = head_top

        # 填充手臂
        h36m_keypoints[:, 11] = l_shoulder
        h36m_keypoints[:, 12] = coco_keypoints[:, 7]  # L_Elbow
        h36m_keypoints[:, 13] = coco_keypoints[:, 9]  # L_Wrist
        h36m_keypoints[:, 14] = r_shoulder
        h36m_keypoints[:, 15] = coco_keypoints[:, 8]  # R_Elbow
        h36m_keypoints[:, 16] = coco_keypoints[:, 10]  # R_Wrist

        return h36m_keypoints

    @classmethod
    def coco17_to_h36m(
        cls,
        coco_keypoints: np.ndarray,  # shape: [Person, Frames, 17, 2]
    ) -> np.ndarray:  # shape: [Person, Frames, 17, 2]
        """
        将 COCO (17点) 转换为 H36M (17点) 格式。

        输入：coco_keypoints (np.ndarray): 形状必须严格为 (Frames, 17, 2) 的三维数组，代表 (帧数, 关节数, [x 坐标, y 坐标])。
        输出：h36m_keypoints (np.ndarray): 形状为 (Frames, 17, 2) 的 H36M 格式数组。

        注意：由于 COCO 没有骨盆(Pelvis)、脊椎(Spine)和胸骨(Thorax)，该方法通过已知点插值计算得出。
        """

        # 维度验证
        if coco_keypoints.ndim != 3:
            raise ValueError(
                f"输入数据维度错误！预期为 3 维数组 (Frames, 17, 2)，"
                f"但收到了 {coco_keypoints.ndim} 维数组，实际 Shape: {coco_keypoints.shape}"
            )

        if coco_keypoints.shape[1] != 17:
            raise ValueError(
                f"输入关键点数量错误！预期为 17 个关键点，"
                f"但收到了 {coco_keypoints.shape[1]} 个点，实际 Shape: {coco_keypoints.shape}"
            )

        if coco_keypoints.shape[2] < 2:
            raise ValueError(f"输入坐标维度错误！预期至少包含 (x, y) 两个维度，实际 Shape: {coco_keypoints.shape}")

        # 输入 shape 为 (Frames, 17, 2)
        frames = coco_keypoints.shape[0]
        h36m_keypoints = np.zeros((frames, 17, 2))

        # 提取 COCO 关键点 (参考 COCO 索引)
        l_shoulder, r_shoulder = coco_keypoints[:, 5], coco_keypoints[:, 6]
        l_hip, r_hip = coco_keypoints[:, 11], coco_keypoints[:, 12]

        # 1. 计算 H36M 缺失的核心关节点
        pelvis = (l_hip + r_hip) / 2.0  # 骨盆 = 左右髋关节中点
        thorax = (l_shoulder + r_shoulder) / 2.0  # 胸骨 = 左右肩中点
        spine = (pelvis + thorax) / 2.0  # 脊椎 = 骨盆与胸骨中点

        # 2. 按照 H36M 的 17 点顺序进行映射 (具体索引以 MotionAGFormer 源码定义为准)
        # H36M 常见顺序: 0:Pelvis, 1:R_Hip, 2:R_Knee, 3:R_Foot, 4:L_Hip, 5:L_Knee, 6:L_Foot,
        # 7:Spine, 8:Thorax, 9:Neck/Nose, 10:Head, 11:L_Shoulder, 12:L_Elbow, 13:L_Wrist,
        # 14:R_Shoulder, 15:R_Elbow, 16:R_Wrist

        h36m_keypoints[:, 0] = pelvis
        h36m_keypoints[:, 1] = r_hip
        h36m_keypoints[:, 2] = coco_keypoints[:, 14]  # R_Knee
        h36m_keypoints[:, 3] = coco_keypoints[:, 16]  # R_Ankle (近似为 Foot)
        h36m_keypoints[:, 4] = l_hip
        h36m_keypoints[:, 5] = coco_keypoints[:, 13]  # L_Knee
        h36m_keypoints[:, 6] = coco_keypoints[:, 15]  # L_Ankle
        h36m_keypoints[:, 7] = spine
        h36m_keypoints[:, 8] = thorax
        h36m_keypoints[:, 9] = coco_keypoints[:, 0]  # Nose (作为 Neck 的近似)

        # H36M 的第 10 点 (Head/头顶):
        # 由于COCO里没有头顶，我们用左眼(1)和右眼(2)的中点向上推算
        # 同时为了视觉上更自然，我们将双眼中点稍微向上（Y轴减小）偏移一点点，模拟头顶位置
        l_eye = coco_keypoints[:, 1]
        r_eye = coco_keypoints[:, 2]
        eyes_center = (l_eye + r_eye) / 2.0
        head_top = eyes_center + (eyes_center - coco_keypoints[:, 0]) * 1.5
        h36m_keypoints[:, 10] = head_top

        # 填充手臂
        h36m_keypoints[:, 11] = l_shoulder
        h36m_keypoints[:, 12] = coco_keypoints[:, 7]  # L_Elbow
        h36m_keypoints[:, 13] = coco_keypoints[:, 9]  # L_Wrist
        h36m_keypoints[:, 14] = r_shoulder
        h36m_keypoints[:, 15] = coco_keypoints[:, 8]  # R_Elbow
        h36m_keypoints[:, 16] = coco_keypoints[:, 10]  # R_Wrist

        return h36m_keypoints
