import json
from pathlib import Path

_RULES_DIR = Path("./data/rules")


def get_rule_names() -> list[str]:
    """返回 ``data/rules/`` 目录下所有规则名（即 .json 文件名去扩展名）。"""
    if not _RULES_DIR.is_dir():
        return []
    return sorted(p.stem for p in _RULES_DIR.glob("*.json"))


def load_rule(pose_type: str) -> dict:
    """
    根据动作类型加载对应的规则。

    Args:
        pose_type: 动作类型，对应 ``data/rules/<pose_type>.json`` 文件。

    Returns:
        包含规则信息的字典，供 pose_judger.judge_pose() 使用。
        文件不存在或 JSON 解析失败时返回空字典。

    Example:
        >>> rule = load_rule("深蹲")
        >>> rule["max_knee_angle"]
    """
    path = _RULES_DIR / f"{pose_type}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        import logging

        logging.getLogger("rules_loader").warning(f"Failed to load rule '{pose_type}': {e}")
        return {}
