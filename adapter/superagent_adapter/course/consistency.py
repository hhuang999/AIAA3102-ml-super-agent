"""Compare only explicitly defined Core summary fields to status anchors."""
from __future__ import annotations

import math


def _score(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and 0 <= value <= 1
    except (OverflowError, ValueError):
        return False


def _safe(value):
    """Retain simple finite scalars; describe complex or nonfinite values by type."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    return "<type: " + type(value).__name__ + ">"


def check_summary(status, summary, selected_result=None):
    """Return a JSON-safe comparison; never mutate either input."""
    result = {"status": "not_verifiable", "checked_fields": [], "conflicts": [],
              "note": "缺少可信状态锚点或可核对的已知摘要字段，无法验证。"}
    if not isinstance(status, dict):
        return result
    node = status.get("best_node_id")
    metric = status.get("best_score") if "best_score" in status else status.get("best_metric")
    best_anchored = isinstance(node, str) and bool(node.strip()) and _score(metric)
    if not isinstance(summary, dict):
        return result
    best = summary.get("best")
    if best_anchored and "best" in summary and not isinstance(best, dict):
        result["status"] = "conflict"
        result["checked_fields"].append("/best")
        result["conflicts"].append({"path": "/best", "expected": "object", "actual": "<type: " + type(best).__name__ + ">", "reason": "摘要 best 必须为对象"})
        result["note"] = "已知摘要字段存在冲突；不构成全文事实核验。"
        return result
    for key in ("node_id", "best_node_id", "metric", "validation_score", "best_metric", "score"):
        if not best_anchored or not isinstance(best, dict) or key not in best:
            continue
        path, actual = "/best/" + key, best[key]
        result["checked_fields"].append(path)
        is_node = key in ("node_id", "best_node_id")
        expected = node if is_node else metric
        if is_node:
            valid = isinstance(actual, str) and actual == node
            reason = "节点标识类型错误或与状态不一致"
        else:
            valid = _score(actual) and abs(actual - metric) <= 1e-12
            reason = "分数类型或范围无效，或与状态相差超过容差"
        if not valid:
            result["conflicts"].append({"path": path, "expected": _safe(expected), "actual": _safe(actual), "reason": reason})
    final = summary.get("final")
    final_anchor = any(k in status for k in ("final_available", "final_kind", "final_method", "final_metric", "final_validation_score"))
    if "final" in summary and final_anchor and not isinstance(final, dict) and not (final is None and status.get("final_available") is False):
        path = "/final"
        result["checked_fields"].append(path)
        result["conflicts"].append({"path": path, "expected": "object", "actual": _safe(final), "reason": "摘要 final 必须为对象"})
    if isinstance(final, dict):
        if type(status.get("final_available")) is bool and "available" in final:
            actual, expected = final["available"], status["final_available"]
            path = "/final/available"
            result["checked_fields"].append(path)
            if type(actual) is not bool or actual != expected:
                result["conflicts"].append({"path": path, "expected": expected, "actual": _safe(actual), "reason": "最终方案可用状态不一致"})
        for key in ("kind", "method", "metric"):
            anchor = "final_" + key
            if anchor not in status or key not in final:
                continue
            expected, actual = status[anchor], final[key]
            if expected is not None and (not isinstance(expected, str) or not expected):
                continue
            path = "/final/" + key
            result["checked_fields"].append(path)
            if actual != expected or (expected is not None and not isinstance(actual, str)):
                result["conflicts"].append({"path": path, "expected": _safe(expected), "actual": _safe(actual), "reason": "最终方案字段与状态不一致"})
        validation = final.get("validation")
        if "final_validation_score" in status and isinstance(validation, dict) and "score" in validation:
            expected, actual = status["final_validation_score"], validation["score"]
            if expected is None or _score(expected):
                path = "/final/validation/score"
                result["checked_fields"].append(path)
                valid = actual is None if expected is None else _score(actual) and abs(actual - expected) <= 1e-12
                if not valid:
                    result["conflicts"].append({"path": path, "expected": _safe(expected), "actual": _safe(actual), "reason": "最终方案验证分数与状态不一致"})
        if isinstance(selected_result, dict) and selected_result.get("source") == "final" and "members" in final:
            expected, members = selected_result.get("node_ids"), final["members"]
            if isinstance(expected, list) and all(isinstance(x, str) for x in expected):
                path = "/final/members"
                result["checked_fields"].append(path)
                valid = isinstance(members, list) and all(isinstance(x, dict) and isinstance(x.get("node_id"), str) for x in members)
                if not valid or sorted(x["node_id"] for x in members) != sorted(expected):
                    result["conflicts"].append({"path": path, "expected": "selected_result.node_ids", "actual": _safe(members), "reason": "最终方案成员与已收集结果不一致"})
    if result["conflicts"]:
        result["status"] = "conflict"
        result["note"] = "已知摘要字段存在冲突；不构成全文事实核验。"
    elif result["checked_fields"]:
        result["status"] = "consistent"
        result["note"] = "仅表示已检查字段与状态一致，不是全文事实核验。"
    return result
