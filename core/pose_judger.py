import numpy as np
import json


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


def judge_pose(kp2d: np.ndarray, kp3d: np.ndarray, rule: dict, motion: str = "") -> tuple[list[str], list[str]]:
    """
    根据规则对 3D 骨骼姿势进行判定，返回触发的规则 ID 和涉及的关节点。
    仅实现：两条线段夹角计算（支持共顶点三点关节角、异面独立线段夹角）
    对地角度、距离类规则暂存other_rules，暂不实现判定逻辑

    Args:
        kp2d: H36M 格式的 2D 关键点，shape=(17, 2) 或 (17, 3)，每行 [x, y] 或 (x, y, confidence)。
        kp3d: HMFormer 的 3D 关键点，shape=(17, 3)，每行 [x, y, z]。
        rule: 姿势判定规则字典。
        motion: 当前动作阶段，如 "descent"/"ascent"/"static_up"/"static_down"。
                规则可设 "phase" 字段，仅当 motion 匹配时生效；
                无 phase 字段的规则始终生效。

    Returns:
        violated_rule_ids: 被违反的规则 ID 列表。
        affected_keypoints: 涉及告警的 H36M 关节点名称列表。
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

    # 提取动作编号、规则列表（兼容 "rule_set" 和 "rule_list"）
    action_no = rule.get("action_no", "01")
    raw_rule_list = rule.get("rule_set") or rule.get("rule_list", [])

    # 入参兼容：如果rule没有直接带rule_list，则从json文件加载对应动作规则
    if not raw_rule_list and "action_no" in rule:
        try:
            with open("fitness_rules.json", "r", encoding="utf-8") as f:
                rules_data = json.load(f)
            for action in rules_data.get("action_list", []):
                if action.get("action_no") == action_no:
                    raw_rule_list = action.get("rule_list", [])
                    break
        except FileNotFoundError:
            print(f"Warning: fitness_rules.json not found")
            return [], []

    # ── 展平 parts 结构 + 规范化 phase 为列表 ──
    def _normalize_phase(val) -> list[str]:
        """将 phase 字段统一为列表。字符串→单元素列表，空/None→空列表。"""
        if not val:
            return []
        if isinstance(val, list):
            return val
        return [val]

    flat_rules: list[dict] = []
    for item in raw_rule_list:
        phase_list = _normalize_phase(item.get("phase"))
        kp_mode = item.get("keypoints_mode", "3d")
        if "parts" in item:
            for part in item["parts"]:
                part_copy = dict(part)
                part_copy["phase"] = phase_list
                part_copy["rule_no"] = item.get("rule_no", "")
                part_copy["keypoints_mode"] = kp_mode
                flat_rules.append(part_copy)
        else:
            item["phase"] = phase_list
            item["keypoints_mode"] = item.get("keypoints_mode", kp_mode)
            flat_rules.append(item)

    # ====================== 规则分组 ======================
    # 组键前缀: "segment4"=线段夹角(四点), "horizontal"=水平夹角(两点)
    segment_groups = {}
    segment_groups["other_rules"] = []

    for rule_item in flat_rules:
        rule_type = rule_item.get("rule_type", "")

        if rule_type == "夹角":
            p1 = rule_item.get("p1", "")
            p2 = rule_item.get("p2", "")
            p3 = rule_item.get("p3", "")
            p4 = rule_item.get("p4", "")

            if not all(p in JOINT_INDEX for p in [p1, p2]):
                print(f"[WARN] 规则{rule_item.get('rule_no')}包含无效关节名称，跳过该规则")
                continue

            seg_key = ("segment4", p1, p2, p3, p4)
            if seg_key not in segment_groups:
                segment_groups[seg_key] = []

            segment_groups[seg_key].append(
                {
                    "rule_no": rule_item.get("rule_no", ""),
                    "phase": rule_item.get("phase", []),
                    "keypoints_mode": rule_item.get("keypoints_mode", "3d"),
                    "min_value": rule_item.get("min_value", None),
                    "max_value": rule_item.get("max_value", None),
                    "p1": p1,
                    "p2": p2,
                    "p3": p3,
                    "p4": p4,
                }
            )

        elif rule_type == "水平夹角":
            p1 = rule_item.get("p1", "")
            p2 = rule_item.get("p2", "")

            if not all(p in JOINT_INDEX for p in [p1, p2]):
                print(f"[WARN] 规则{rule_item.get('rule_no')}包含无效关节名称，跳过该规则")
                continue

            seg_key = ("horizontal", p1, p2)
            if seg_key not in segment_groups:
                segment_groups[seg_key] = []

            segment_groups[seg_key].append(
                {
                    "rule_no": rule_item.get("rule_no", ""),
                    "phase": rule_item.get("phase", []),
                    "keypoints_mode": rule_item.get("keypoints_mode", "3d"),
                    "min_value": rule_item.get("min_value", None),
                    "max_value": rule_item.get("max_value", None),
                    "p1": p1,
                    "p2": p2,
                }
            )

        else:
            segment_groups["other_rules"].append(rule_item)

    # ====================== 逐组判定 ======================
    for group_key, rules in segment_groups.items():
        if group_key == "other_rules":
            continue

        grp_type = group_key[0]

        # 筛选当前阶段适用的规则（空列表 或 ["all"] = 始终生效）
        active = []
        for r in rules:
            phases = r.get("phase", [])
            if not phases or "all" in phases or motion in phases:
                active.append(r)
        if not active:
            continue

        kp_mode = active[0].get("keypoints_mode", "3d")
        kps = kp2d if kp_mode == "2d" else kp3d

        # ── 水平夹角：两点连线与水平面夹角 ──
        if grp_type == "horizontal":
            _, p1_name, p2_name = group_key
            c1 = kps[JOINT_INDEX[p1_name]]
            c2 = kps[JOINT_INDEX[p2_name]]
            dy = abs(float(c2[1]) - float(c1[1]))
            dx = abs(float(c2[0]) - float(c1[0]))
            current_angle = float(np.degrees(np.arctan2(dy, max(dx, 1e-6))))

            is_violate = True
            for r in active:
                if is_in_range(current_angle, r["min_value"], r["max_value"]):
                    is_violate = False
                    break

            if is_violate:
                for r in active:
                    full_rule_id = f"{action_no}-{r['rule_no']}"
                    violated_rule_ids.append(full_rule_id)
                    affected_keypoints_set.add(JOINT_NAME_MAPPING.get(p1_name, p1_name))
                    affected_keypoints_set.add(JOINT_NAME_MAPPING.get(p2_name, p2_name))
            continue

        # ── 线段夹角：四点两组线段 ──
        _, p1_name, p2_name, p3_name, p4_name = group_key

        p1_coord = kps[JOINT_INDEX[p1_name]]
        p2_coord = kps[JOINT_INDEX[p2_name]]
        p3_coord = kps[JOINT_INDEX[p3_name]]
        p4_coord = kps[JOINT_INDEX[p4_name]]
        if kp_mode == "2d":
            p1_coord = np.append(p1_coord[:2], 0.0)
            p2_coord = np.append(p2_coord[:2], 0.0)
            p3_coord = np.append(p3_coord[:2], 0.0)
            p4_coord = np.append(p4_coord[:2], 0.0)

        if p2_name == p3_name:
            vec_a = p1_coord - p2_coord
            vec_b = p4_coord - p2_coord
            current_angle = calc_three_point_angle(vec_a, vec_b)
        else:
            current_angle = calculate_angle_between_segments(p1_coord, p2_coord, p3_coord, p4_coord)

        is_violate = True
        for r in active:
            if is_in_range(current_angle, r["min_value"], r["max_value"]):
                is_violate = False
                break

        if is_violate:
            for r in active:
                full_rule_id = f"{action_no}-{r['rule_no']}"
                violated_rule_ids.append(full_rule_id)
                affected_keypoints_set.add(JOINT_NAME_MAPPING.get(p1_name, p1_name))
                affected_keypoints_set.add(JOINT_NAME_MAPPING.get(p2_name, p2_name))
                affected_keypoints_set.add(JOINT_NAME_MAPPING.get(p3_name, p3_name))
                affected_keypoints_set.add(JOINT_NAME_MAPPING.get(p4_name, p4_name))

    affected_keypoints = list(affected_keypoints_set)
    return violated_rule_ids, affected_keypoints
