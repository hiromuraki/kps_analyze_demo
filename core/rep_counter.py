import numpy as np


# ===================== 骨骼节点名称=====================
NODE_NAME_TO_INDEX = {
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
    "地面": -1
}

# ===================== 三个基础读取函数 =====================
def get_rep_ceiling(rule: dict) -> float:
    """
    返回重复计数的上界阈值（特征值高于此判定为伸展态）。
    rule    : 规则 JSON 的完整 dict。
    return  : 上界阈值。
    """
    rep_cfg = rule["rep_counting"]
    return float(rep_cfg["top_threshold"])


def get_rep_floor(rule: dict) -> float:
    """
    返回重复计数的下界阈值（特征值低于此判定为收缩态）。
    rule    : 规则 JSON 的完整 dict。
    return  : 下界阈值。
    """
    rep_cfg = rule["rep_counting"]
    return float(rep_cfg["bottom_threshold"])


def get_rep_count_direction(rule: dict) -> str:
    """
    返回重复计数的触发方向。
    rule    : 规则 JSON 的完整 dict。
    return  : "down_up"（收缩→伸展 计一次）
              或 "up_down"（伸展→收缩 计一次）。
    """
    rep_cfg = rule["rep_counting"]
    return rep_cfg["count_on"]


# ===================== 核心特征提取函数 =====================
def get_rep_feature_value(kps_3d: np.ndarray, rule: dict) -> float:
    """
    从 3D 骨骼中提取当前帧的重复计数特征值。

    kps_3d  : shape=(17, 3), H36M 格式 xyz 坐标，index 0-16。
    rule    : 规则 JSON 的完整 dict（含 rep_counting 块）。
    return  : 当前帧的特征标量值（角度=度数, 距离=原始单位）。
    """

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

    def calc_2point_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """计算两点3D欧式距离"""
        return float(np.linalg.norm(p2 - p1))

    # 提取重复计数配置块
    rep_cfg = rule["rep_counting"]
    feat_type = rep_cfg["type"]  # "angle" / "distance"

    # 取出关键点名称，安全读取，distance无p3/p4不会报错
    p1_name = rep_cfg["p1"]
    p2_name = rep_cfg["p2"]
    p3_name = rep_cfg.get("p3", "")
    p4_name = rep_cfg.get("p4", "")

    # 处理地面-1索引，生成地面坐标（X、Z沿用参考点，Y=0）
    def get_keypoint(idx: int, ref_point: np.ndarray) -> np.ndarray:
        if idx == -1:
            return np.array([ref_point[0], 0.0, ref_point[2]])
        return kps_3d[idx]

    # 根据类型分支处理
    if feat_type == "angle":
        # angle类型必须完整四点，读取所有索引与坐标
        idx1 = NODE_NAME_TO_INDEX[p1_name]
        idx2 = NODE_NAME_TO_INDEX[p2_name]
        idx3 = NODE_NAME_TO_INDEX[p3_name]
        idx4 = NODE_NAME_TO_INDEX[p4_name]

        p1 = get_keypoint(idx1, kps_3d[idx2])
        p2 = get_keypoint(idx2, kps_3d[idx1])
        p3 = get_keypoint(idx3, kps_3d[idx2])
        p4 = get_keypoint(idx4, kps_3d[idx3])

        # 区分共顶点 / 异面两线段
        if p2_name == p3_name:
            vec_a = p1 - p2
            vec_b = p4 - p2
            return calc_three_point_angle(vec_a, vec_b)
        else:
            return calculate_angle_between_segments(p1, p2, p3, p4)

    elif feat_type == "distance":
        # distance只用到p1/p2，完全不碰p3/p4
        idx1 = NODE_NAME_TO_INDEX[p1_name]
        idx2 = NODE_NAME_TO_INDEX[p2_name]
        p1 = get_keypoint(idx1, kps_3d[idx2])
        p2 = get_keypoint(idx2, kps_3d[idx1])
        return calc_2point_distance(p1, p2)

    else:
        raise ValueError(f"不支持的rep_counting.type: {feat_type}, 仅支持 angle / distance")