import numpy as np


def judge_pose(kp3d: np.ndarray, rule: dict) -> tuple[list[str], list[str]]:
    """
    根据规则对 3D 骨骼姿势进行判定，返回触发的规则 ID 和涉及的关节点。

    从 3D 关键点的空间位置出发，检查各关节角度、相对位置等几何关系
    是否在规则定义的正常范围内，超出阈值则触发告警。

    Args:
        kp3d: HMFormer 的 3D 关键点, shape=(17, 3), 每行 [x, y, z]。
        rule: 姿势判定规则字典，结构自定。
              预期包含各关节的角度阈值、相对距离约束等。

    Returns:
        violated_rule_ids: 被违反的规则 ID 元组，如 ['R1', 'R2']，
                           未触发任何规则时为空列表 []。
        affected_keypoints: 涉及告警的 H36M 关节点名称元组，
                            如 ['left_elbow', 'left_wrist']，
                            会被用于反向映射到 2D 骨骼渲染时高亮。
    """
    return ([], [])  # 占位返回，实际实现需要根据规则逻辑进行判定
    raise NotImplementedError
