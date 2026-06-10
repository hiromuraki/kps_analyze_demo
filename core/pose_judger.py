import json
import numpy as np

JOINT_INDEX = {
    "骨盆": 0,
    "右髋": 1,
    "右膝": 2,
    "右脚踝": 3,
    "左髋": 4,
    "左膝": 5,
    "左脚踝": 6,
    "脊柱": 7,
    "胸腔": 8,
    "颈": 9,
    "头顶": 10,
    "左肩": 11,
    "左肘": 12,
    "左手腕": 13,
    "右肩": 14,
    "右肘": 15,
    "右手腕": 16,
}

# 关节点名称映射（支持中文到英文）
JOINT_NAME_MAPPING = {
    "左髋": "left_hip",
    "右髋": "right_hip",
    "左膝": "left_knee",
    "右膝": "right_knee",
    "左脚踝": "left_ankle",
    "右脚踝": "right_ankle",
    "左肩": "left_shoulder",
    "右肩": "right_shoulder",
    "左肘": "left_elbow",
    "右肘": "right_elbow",
    "左手腕": "left_wrist",
    "右手腕": "right_wrist",
    "脊柱": "spine",
    "胸腔": "chest",
    "颈": "neck",
    "头顶": "head",
    "骨盆": "pelvis",
}


def judge_pose(kp3d: np.ndarray, rule: dict) -> tuple[list[str], list[str]]:
    """
    根据规则对 3D 骨骼姿势进行判定，返回触发的规则 ID 和涉及的关节点。

    从 3D 关键点的空间位置出发，检查各关节角度、相对位置等几何关系
    是否在规则定义的正常范围内，超出阈值则触发告警。

    Args:
        kp3d: HMFormer 的 3D 关键点，shape=(17, 3)，每行 [x, y, z]。
        rule: 姿势判定规则字典，结构自定。
              预期包含各关节的角度阈值、相对距离约束等。

    Returns:
        violated_rule_ids: 被违反的规则 ID 元组，如 ['01-R1', '01-R2'],
                           未触发任何规则时为空数组 []。
        affected_keypoints: 涉及告警的 H36M 关节点名称元组，
                            如 ['left_elbow', 'left_wrist'],
                            会被用于反向映射到 2D 骨骼渲染时高亮。
    """
    # H36M 关节点索引映射（标准 H36M 17个关键点）
    # 索引顺序: 0:root, 1:rhip, 2:rknee, 3:rfoot, 4:lhip, 5:lknee, 6:lfoot,
    # 7:spine, 8:thorax, 9:neck, 10:head, 11:lshoulder, 12:lelbow, 13:lwrist,
    # 14:rshoulder, 15:relbow, 16:rwrist

    violated_rule_ids = []
    affected_keypoints_set = set()

    # 获取当前动作的规则列表
    action_no = rule.get("action_no", "01")
    rule_list = rule.get("rule_list", [])

    # 如果传入的rule直接包含rule_list，直接使用；否则从文件中加载
    if not rule_list and "action_no" in rule:
        # 从fitness_rules.json加载规则
        try:
            with open("fitness_rules.json", "r", encoding="utf-8") as f:
                rules_data = json.load(f)

            for action in rules_data.get("action_list", []):
                if action.get("action_no") == action_no:
                    rule_list = action.get("rule_list", [])
                    break
        except FileNotFoundError:
            print(f"Warning: fitness_rules.json not found")
            return [], []

    # 辅助函数：计算两个向量之间的夹角（角度制）
    def calculate_angle(point_a, point_b, point_c):
        """计算三点之间的夹角（angle at point_b）"""
        ba = point_a - point_b
        bc = point_c - point_b

        # 防止零向量
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 0.0

        cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        return angle

    # 辅助函数：计算向量与水平面的夹角
    def calculate_angle_to_horizontal(point_a, point_b):
        """计算两点连线与水平面（x-y平面）的夹角"""
        vec = point_b - point_a
        # 计算垂直分量（z轴）与水平面投影的夹角
        horizontal_norm = np.linalg.norm(vec[:2])  # x, y 分量
        vertical_comp = vec[2]  # z 分量

        if horizontal_norm < 1e-6:
            return 90.0 if vertical_comp > 0 else -90.0

        angle = np.degrees(np.arctan2(vertical_comp, horizontal_norm))
        return abs(angle)  # 返回绝对值

    # 辅助函数：计算两个点之间的欧氏距离
    def calculate_distance(point_a, point_b):
        return np.linalg.norm(point_a - point_b)

    # 辅助函数：检查值是否在范围内
    def is_in_range(value, min_val, max_val):
        if min_val is None and max_val is None:
            return True
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True

    # 遍历每条规则进行判断
    for rule_item in rule_list:
        rule_no = rule_item.get("rule_no", "")
        rule_type = rule_item.get("rule_type", "")
        rule_dim = rule_item.get("rule_dim", "")
        node_a = rule_item.get("node_a", "")
        node_b = rule_item.get("node_b", "")
        min_val = rule_item.get("min_value", None)
        max_val = rule_item.get("max_value", None)
        rule_ref = rule_item.get("rule_ref", "")

        # 检查必要节点是否存在
        if node_a not in JOINT_INDEX:
            continue

        idx_a = JOINT_INDEX[node_a]
        point_a = kp3d[idx_a]

        # 根据规则类型进行判断
        is_violated = False
        current_value = None

        if rule_type == "夹角" and rule_dim == "关节间" and node_b:
            # 关节间夹角：node_a 和 node_b 都是关节点，需要计算 node_a 处的角度
            # 实际需要三个点：node_b 为中间点？根据规则语义调整
            if node_b in JOINT_INDEX:
                idx_b = JOINT_INDEX[node_b]
                # 需要找到第三个点来形成夹角
                # 对于肘关节，需要肩-肘-腕；对于膝关节，需要髋-膝-踝
                third_point = None

                # 推断第三个关节点
                if "肘" in node_a and node_a.startswith("左"):
                    third_point = JOINT_INDEX["左手腕"]
                elif "肘" in node_a and node_a.startswith("右"):
                    third_point = JOINT_INDEX["右手腕"]
                elif "膝" in node_a and node_a.startswith("左"):
                    third_point = JOINT_INDEX["左脚踝"]
                elif "膝" in node_a and node_a.startswith("右"):
                    third_point = JOINT_INDEX["右脚踝"]
                elif "肩" in node_a and node_a.startswith("左"):
                    third_point = JOINT_INDEX["左肘"]
                elif "肩" in node_a and node_a.startswith("右"):
                    third_point = JOINT_INDEX["右肘"]
                elif "髋" in node_a and node_a.startswith("左"):
                    third_point = JOINT_INDEX["左膝"]
                elif "髋" in node_a and node_a.startswith("右"):
                    third_point = JOINT_INDEX["右膝"]

                if third_point is not None:
                    # 计算夹角：point_third - point_b - point_a
                    point_b_coord = kp3d[idx_b]
                    point_third = kp3d[third_point]
                    angle = calculate_angle(point_third, point_b_coord, point_a)
                    current_value = angle
                    is_violated = not is_in_range(angle, min_val, max_val)

            elif node_b in JOINT_INDEX:
                # 简单两点连线与水平面的夹角
                point_b_coord = kp3d[JOINT_INDEX[node_b]]
                angle = calculate_angle_to_horizontal(point_a, point_b_coord)
                current_value = angle
                is_violated = not is_in_range(angle, min_val, max_val)

        elif rule_type == "夹角" and rule_dim == "对地角度":
            # 对地角度：计算关节点与地面的夹角
            if node_b == "" or rule_ref == "水平面":
                # 计算关节与水平面的夹角
                if node_a in JOINT_INDEX:
                    # 需要找到相邻关节点来形成线段
                    adjacent_joint = None
                    if "腕" in node_a:
                        if node_a == "左手腕":
                            adjacent_joint = JOINT_INDEX["左肘"]
                        elif node_a == "右手腕":
                            adjacent_joint = JOINT_INDEX["右肘"]
                    elif "踝" in node_a:
                        if node_a == "左脚踝":
                            adjacent_joint = JOINT_INDEX["左膝"]
                        elif node_a == "右脚踝":
                            adjacent_joint = JOINT_INDEX["右膝"]

                    if adjacent_joint is not None:
                        angle = calculate_angle_to_horizontal(kp3d[adjacent_joint], point_a)
                        current_value = angle
                        is_violated = not is_in_range(angle, min_val, max_val)

        elif rule_type == "距离" and rule_dim == "关节间":
            # 距离判断
            if node_b in JOINT_INDEX:
                idx_b = JOINT_INDEX[node_b]
                distance = calculate_distance(point_a, kp3d[idx_b])
                # 对于距离规则，min_val和max_val表示距离范围
                current_value = distance
                is_violated = not is_in_range(distance, min_val, max_val)

        # 如果规则被违反，记录规则ID和涉及的关节点
        if is_violated:
            # 格式化为 "动作编号-规则编号"，如 "01-R1"
            full_rule_id = f"{action_no}-{rule_no}"
            violated_rule_ids.append(full_rule_id)

            # 添加涉及的关节点（用于反向映射到2D骨骼渲染）
            if node_a:
                affected_keypoints_set.add(JOINT_NAME_MAPPING.get(node_a, node_a))
            if node_b and node_b in JOINT_NAME_MAPPING:
                affected_keypoints_set.add(JOINT_NAME_MAPPING.get(node_b, node_b))

    # 转换为列表格式
    affected_keypoints = list(affected_keypoints_set)

    return violated_rule_ids, affected_keypoints
