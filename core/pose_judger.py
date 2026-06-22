import numpy as np
import json


def get_rep_feature_value(kps_3d: np.ndarray, rule: dict) -> float:
    """从 3D 骨骼中提取当前帧的重复计数特征值。\n\n    kps_3d  : shape=(17, 3), H36M 格式 xyz 坐标，index 0-16。\n    rule    : 规则 JSON 的完整 dict（含 rep_counting 块）。\n    return  : 当前帧的特征标量值（角度=度数, 距离=原始单位）。\n"""
    raise NotImplementedError


def get_rep_ceiling(rule: dict) -> float:
    """返回重复计数的上界阈值（特征值高于此判定为伸展态）。\n\n    rule    : 规则 JSON 的完整 dict。\n    return  : 上界阈值。\n"""
    raise NotImplementedError


def get_rep_floor(rule: dict) -> float:
    """返回重复计数的下界阈值（特征值低于此判定为收缩态）。\n\n    rule    : 规则 JSON 的完整 dict。\n    return  : 下界阈值。\n"""
    raise NotImplementedError


def get_rep_count_direction(rule: dict) -> str:
    """返回重复计数的触发方向。\n\n    rule    : 规则 JSON 的完整 dict。\n    return  : \"down_up\"（收缩→伸展 计一次）或 \"up_down\"（伸展→收缩 计一次）。\n"""
    raise NotImplementedError


def judge_pose(kp3d: np.ndarray, rule: dict) -> tuple[list[str], list[str]]:
    """
    根据规则对 3D 骨骼姿势进行判定，返回触发的规则 ID 和涉及的关节点。
    仅实现：两条线段夹角计算（支持共顶点三点关节角、异面独立线段夹角）
    对地角度、距离类规则暂存other_rules，暂不实现判定逻辑

    Args:
        kp3d: HMFormer 的 3D 关键点，shape=(17, 3)，每行 [x, y, z]。
        rule: 姿势判定规则字典，结构自定。
              预期包含各关节的角度阈值、相对距离约束等。

    Returns:
        violated_rule_ids: 被违反的规则 ID 列表，如 ['01-R1', '01-R2'],
                           未触发任何规则时为空数组 []。
        affected_keypoints: 涉及告警的 H36M 关节点名称列表，
                            如 ['left_elbow', 'left_wrist'],
                            会被用于反向映射到 2D 骨骼渲染时高亮。
    """

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
        "鼻子": 9,
        "头顶": 10,
        "左肩": 11,
        "左肘": 12,
        "左手腕": 13,
        "右肩": 14,
        "右肘": 15,
        "右手腕": 16,
    }

    # 关节点名称映射（中文 -> 英文渲染名）
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
        "胸腔": "thorax",
        "头顶": "head_top",
        "骨盆": "pelvis",
        "鼻子": "nose",
    }

    violated_rule_ids = []
    affected_keypoints_set = set()

    # 提取动作编号、规则列表
    action_no = rule.get("action_no", "01")
    rule_list = rule.get("rule_list", [])

    # 入参兼容：如果rule没有直接带rule_list，则从json文件加载对应动作规则
    if not rule_list and "action_no" in rule:
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

    # ====================== 辅助函数 ======================
    def calc_three_point_angle(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        计算同一个顶点出发两个向量的夹角（三点内角，0~180°）
        对应场景：P1-P2(顶点)-P4，P2=P3
        """
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        cos_ang = np.dot(vec_a, vec_b) / (norm_a * norm_b)
        cos_ang = np.clip(cos_ang, -1.0, 1.0)
        return np.degrees(np.arccos(cos_ang))

    def calculate_angle_between_segments(p1, p2, p3, p4) -> float:
        """
        计算两条独立线段 (p1-p2)、(p3-p4) 的异面夹角，返回 0~180°
        适用：四点完全不同、两条无公共顶点线段
        """
        vec1 = p2 - p1
        vec2 = p4 - p3
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def is_in_range(value, min_val, max_val, eps=1e-4) -> bool:
        """
        带浮点容错的区间判断，避免浮点精度临界值误判
        """
        if min_val is None and max_val is None:
            return True
        if min_val is not None and value < (min_val - eps):
            return False
        if max_val is not None and value > (max_val + eps):
            return False
        return True

    # ====================== 规则分组：按两条线段四元组分组 ======================
    segment_groups = {}
    segment_groups["other_rules"] = []

    for rule_item in rule_list:
        rule_type = rule_item.get("rule_type", "")
        rule_dim = rule_item.get("rule_dim", "")

        # 仅处理【线段夹角】类型规则
        if rule_type == "夹角" and rule_dim == "线段夹角":
            p1 = rule_item.get("p1", "")
            p2 = rule_item.get("p2", "")
            p3 = rule_item.get("p3", "")
            p4 = rule_item.get("p4", "")

            # 校验关节名称合法性
            if not all(p in JOINT_INDEX for p in [p1, p2, p3, p4]):
                print(f"[WARN] 规则{rule_item.get('rule_no')}包含无效关节名称，跳过该规则")
                continue

            seg_key = (p1, p2, p3, p4)
            if seg_key not in segment_groups:
                segment_groups[seg_key] = []

            segment_groups[seg_key].append(
                {
                    "rule_no": rule_item.get("rule_no", ""),
                    "min_value": rule_item.get("min_value", None),
                    "max_value": rule_item.get("max_value", None),
                    "p1": p1,
                    "p2": p2,
                    "p3": p3,
                    "p4": p4,
                }
            )
        else:
            # 对地角度、距离、旧版两点规则全部归入other_rules，暂不判定
            segment_groups["other_rules"].append(rule_item)

    # ====================== 逐组判定角度 ======================
    for group_key, rules in segment_groups.items():
        if group_key == "other_rules":
            # 暂不处理非线段夹角规则，直接跳过
            continue

        # 取出四个关节名称
        p1_name, p2_name, p3_name, p4_name = group_key
        # 取出3D坐标
        p1_coord = kp3d[JOINT_INDEX[p1_name]]
        p2_coord = kp3d[JOINT_INDEX[p2_name]]
        p3_coord = kp3d[JOINT_INDEX[p3_name]]
        p4_coord = kp3d[JOINT_INDEX[p4_name]]

        # 分支1：P2与P3是同一个点 → 三点共顶点内角（人体关节标准模式 ∠P1-P2-P4）
        if p2_name == p3_name:
            vec_a = p1_coord - p2_coord
            vec_b = p4_coord - p2_coord
            current_angle = calc_three_point_angle(vec_a, vec_b)
        # 分支2：四个点互不重合 → 两条独立线段异面夹角
        else:
            current_angle = calculate_angle_between_segments(p1_coord, p2_coord, p3_coord, p4_coord)

        # 本组多条区间逻辑：满足任意一个区间 = 姿态正常；全部不满足 = 违规
        is_violate = True
        for r in rules:
            if is_in_range(current_angle, r["min_value"], r["max_value"]):
                is_violate = False
                break

        if is_violate:
            # 本组所有规则全部触发告警
            for r in rules:
                full_rule_id = f"{action_no}-{r['rule_no']}"
                violated_rule_ids.append(full_rule_id)
                # 收集涉及关节，自动去重
                affected_keypoints_set.add(JOINT_NAME_MAPPING.get(p1_name, p1_name))
                affected_keypoints_set.add(JOINT_NAME_MAPPING.get(p2_name, p2_name))
                affected_keypoints_set.add(JOINT_NAME_MAPPING.get(p3_name, p3_name))
                affected_keypoints_set.add(JOINT_NAME_MAPPING.get(p4_name, p4_name))

    affected_keypoints = list(affected_keypoints_set)
    return violated_rule_ids, affected_keypoints
