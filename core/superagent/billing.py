"""Token settlement on the same card MLEvolve uses.

DeepSeek off-peak (谷时) prices, RMB per 1,000,000 tokens:

- Flash: input 1, cache 0.02, output 4
- Pro (model name contains ``pro``): input 4.5, cache 0.15, output 13.5

Input is prompt tokens that missed the cache. Output includes reasoning.
Cache hits are billed on their own and are not also counted as input.
Amounts that are not a DeepSeek model are left unpriced.

One task writes ``logs/api_usage.json`` and ``api_usage.html``. A 24 hour run
is split into one-hour intervals counted from the first call, and every call
is filed under the agent role that made it so the heaviest part is visible.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Mapping

from .usage_page import render_task_html

log = logging.getLogger("superagent")

_MILLION = Decimal(1_000_000)
_QUANT = Decimal("0.00000001")
_INTERVAL = timedelta(hours=1)
_PRICE_SOURCE = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing"
_OFF_PEAK: dict[str, dict[str, Any]] = {
    "flash": {
        "model": "deepseek-flash",
        "label": "DeepSeek-V4.1-Flash",
        "input": Decimal("1"),
        "cache": Decimal("0.02"),
        "output": Decimal("4"),
    },
    "pro": {
        "model": "deepseek-v4-pro",
        "label": "DeepSeek-V4-Pro",
        "input": Decimal("4.5"),
        "cache": Decimal("0.15"),
        "output": Decimal("13.5"),
    },
}

# Role or stage name -> short label on the usage pages.
PART_LABELS: dict[str, str] = {
    "desc_cleaner": "描述清洗",
    "direction_judge": "指标方向",
    "draft_writer.data_prep": "初稿·数据",
    "draft_writer.model_def": "初稿·模型",
    "draft_writer.train_eval": "初稿·训练",
    "draft_integrator": "初稿整合",
    "planner.free": "规划",
    "planner.structured": "规划（结构化）",
    "patch_coder": "补丁改进",
    "script_coder": "整段重写",
    "repair_coder": "修复",
    "reviewer": "代码评审",
    "result_reader": "结果读取",
    "leak_checker": "泄露检查",
    "format_fixer": "格式修复",
    "draft": "初稿",
    "improve": "改进",
    "debug": "修复",
    "evolution": "分支进化",
    "fusion": "跨分支融合",
    "aggregation": "多分支汇总",
    "code_review": "代码评审",
    "result_parse": "结果解析",
    "data_leakage": "泄露检查",
    "task_desc": "任务描述清洗",
    "generate": "代码生成",
    "query": "评审与其它调用",
}

PROJECT = "SuperAgent"
GENERIC_RUN_DIRS = frozenset({"logs", "superagent_out", "core_out", "outs", "output"})


def task_label(run_dir: str | Path) -> str:
    """A readable task name. A run directory with a generic name is labelled after its task folder."""
    path = Path(run_dir)
    if path.name in GENERIC_RUN_DIRS or path.name.endswith("_out"):
        if path.parent.is_dir() and path.parent.name not in GENERIC_RUN_DIRS:
            return f"{path.parent.name}/{path.name}"
    return path.name


def part_label(name: str) -> str:
    return PART_LABELS.get(name, name or "未标部分")


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        data = dump()
        return data if isinstance(data, dict) else {}
    return {}


def split_usage(usage: Any) -> dict[str, int]:
    """Split one response usage object into input, output, and cache tokens.

    Cache hits are not included in input. Reasoning tokens stay inside output
    and are also reported on their own.
    """
    data = _as_dict(usage)
    prompt = int(data.get("prompt_tokens") or 0)
    output = int(data.get("completion_tokens") or 0)
    details = _as_dict(data.get("prompt_tokens_details"))
    cached = int(details.get("cached_tokens") or 0)
    hit = int(data.get("prompt_cache_hit_tokens") or 0)
    if hit and not cached:
        cached = hit
    miss = data.get("prompt_cache_miss_tokens")
    fresh = int(miss or 0) if miss is not None else max(prompt - cached, 0)
    completion_details = _as_dict(data.get("completion_tokens_details"))
    reasoning = int(completion_details.get("reasoning_tokens") or 0)
    return {
        "input_tokens": fresh,
        "output_tokens": output,
        "cache_tokens": cached,
        "reasoning_tokens": reasoning,
    }


def rates_for(model: str) -> dict[str, Any] | None:
    """Return the off-peak RMB card for a DeepSeek model name."""
    name = (model or "").lower()
    if not name.startswith("deepseek"):
        return None
    if "pro" in name:
        return _OFF_PEAK["pro"]
    return _OFF_PEAK["flash"]


def _q(value: Decimal) -> Decimal:
    return value.quantize(_QUANT, rounding=ROUND_HALF_UP)


def cost_rmb(input_tokens: int, output_tokens: int, cache_tokens: int, model: str) -> dict[str, str] | None:
    """Settle one bucket at the off-peak RMB price. Amounts are decimal strings."""
    rates = rates_for(model)
    if rates is None:
        return None
    return _cost_with_rates(input_tokens, output_tokens, cache_tokens, rates)


def call_cost_float(model: str, input_tokens: int | None, output_tokens: int | None, cache_tokens: int | None) -> float | None:
    """Same settlement as :func:`cost_rmb`, as a float for the run log. None when unpriced or usage is missing."""
    if input_tokens is None and output_tokens is None and cache_tokens is None:
        return None
    settled = cost_rmb(int(input_tokens or 0), int(output_tokens or 0), int(cache_tokens or 0), model)
    if settled is None:
        return None
    return float(settled["total"])


def _cost_with_rates(input_tokens: int, output_tokens: int, cache_tokens: int, rates: dict[str, Any]) -> dict[str, str]:
    input_cost = _q(Decimal(input_tokens) * rates["input"] / _MILLION)
    output_cost = _q(Decimal(output_tokens) * rates["output"] / _MILLION)
    cache_cost = _q(Decimal(cache_tokens) * rates["cache"] / _MILLION)
    total = input_cost + output_cost + cache_cost
    return {"input": f"{input_cost:.8f}", "output": f"{output_cost:.8f}", "cache": f"{cache_cost:.8f}", "total": f"{total:.8f}"}


def _rate_public(rates: dict[str, Any]) -> dict[str, str]:
    return {
        "model": rates["model"],
        "label": rates["label"],
        "input": format(rates["input"], "f"),
        "cache": format(rates["cache"], "f"),
        "output": format(rates["output"], "f"),
        "tier": "off_peak",
        "currency": "CNY",
        "unit": "RMB per 1,000,000 tokens",
        "source": _PRICE_SOURCE,
    }


def _parse_ts(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _zero_cost() -> dict[str, str]:
    return {"input": "0.00000000", "output": "0.00000000", "cache": "0.00000000", "total": "0.00000000"}


def _call_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise one ``llm_calls.jsonl`` row or one history entry."""
    prompt = row.get("prompt_tokens")
    fresh = row.get("input_tokens")
    cache = int(row.get("cache_tokens") or row.get("cached_tokens") or 0)
    if fresh is None and prompt is not None:
        fresh = max(int(prompt) - cache, 0)
    output = row.get("output_tokens")
    if output is None:
        output = row.get("completion_tokens")
    part = str(row.get("part") or row.get("agent") or row.get("kind") or "unknown")
    return {
        "ts": row.get("ts") or "",
        "part": part,
        "kind": str(row.get("kind") or part),
        "model": str(row.get("model") or ""),
        "input_tokens": int(fresh or 0),
        "output_tokens": int(output or 0),
        "cache_tokens": cache,
        "reasoning_tokens": int(row.get("reasoning_tokens") or 0),
        "missing": bool(row.get("missing")) or (prompt is None and fresh is None and output is None and row.get("completion_tokens") is None),
    }


def snapshot_from_calls(calls: list[Mapping[str, Any]], *, task: str, project: str = PROJECT) -> dict[str, Any]:
    """Build the usage document: totals, per-part ranking, and one-hour intervals."""
    history = [_call_from_row(row) for row in calls]
    by_model: dict[str, dict[str, int]] = {}
    parts: dict[str, dict[str, Any]] = {}
    intervals: dict[int, dict[str, Any]] = {}
    total_in = total_out = total_cache = total_reason = 0
    missing = 0
    unpriced = False
    cost_dec = {key: Decimal(0) for key in ("input", "output", "cache", "total")}
    t0: datetime | None = None
    for item in history:
        if item["missing"] and item["input_tokens"] == 0 and item["output_tokens"] == 0 and item["cache_tokens"] == 0:
            missing += 1
        total_in += item["input_tokens"]
        total_out += item["output_tokens"]
        total_cache += item["cache_tokens"]
        total_reason += item["reasoning_tokens"]
        model = item["model"] or "unknown"
        bucket = by_model.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_tokens": 0})
        bucket["calls"] += 1
        bucket["input_tokens"] += item["input_tokens"]
        bucket["output_tokens"] += item["output_tokens"]
        bucket["cache_tokens"] += item["cache_tokens"]
        settled = cost_rmb(item["input_tokens"], item["output_tokens"], item["cache_tokens"], model)
        if settled is None:
            unpriced = True
        else:
            for key in cost_dec:
                cost_dec[key] += Decimal(settled[key])
        _add_part(parts, item, settled)
        ts = _parse_ts(item["ts"])
        if ts is not None and t0 is None:
            t0 = ts
        index = 0
        if ts is not None and t0 is not None:
            index = int((ts - t0).total_seconds() // _INTERVAL.total_seconds())
            if index < 0:
                index = 0
        elif ts is None:
            index = -1
        _add_interval(intervals, index, item, settled, t0)
    total_tokens = total_in + total_out + total_cache
    by_part = _finish_parts(parts, total_tokens)
    by_interval = _finish_intervals(intervals)
    top_part = by_part[0] if by_part else None
    top_interval = max(by_interval, key=lambda row: row["total_tokens"]) if by_interval else None
    priced_models = [name for name in by_model if rates_for(name)]
    card = rates_for(priced_models[0]) if priced_models else _OFF_PEAK["flash"]
    return {
        "project": project,
        "task": task,
        "currency": "CNY",
        "price_tier": "off_peak",
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cache_tokens": total_cache,
        "total_tokens": total_tokens,
        "output_details": {"reasoning_tokens": total_reason},
        "calls": len(history),
        "calls_missing_usage": missing,
        "cost_rmb": {key: f"{cost_dec[key]:.8f}" for key in cost_dec} if history else _zero_cost(),
        "price_rmb_per_million": _rate_public(card),
        "unpriced_model": unpriced,
        "by_model": by_model,
        "by_part": by_part,
        "by_interval": by_interval,
        "top_part": top_part,
        "top_interval": top_interval,
        "history": history,
        "cache_hit_rate": _hit_rate(total_in, total_cache),
        "updated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "note": (
            "input excludes cache hits. output includes reasoning tokens. "
            "Intervals are one hour, counted from the first call of this task. "
            "Amounts are RMB at DeepSeek off-peak prices, the same card MLEvolve uses."
        ),
    }


def _add_part(parts: dict[str, dict[str, Any]], item: Mapping[str, Any], settled: dict[str, str] | None) -> None:
    name = str(item["part"])
    acc = parts.setdefault(name, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_tokens": 0,
                                  "reasoning_tokens": 0, "cost": {key: Decimal(0) for key in ("input", "output", "cache", "total")}})
    acc["calls"] += 1
    for key in ("input_tokens", "output_tokens", "cache_tokens", "reasoning_tokens"):
        acc[key] += int(item[key])
    if settled is not None:
        for key in acc["cost"]:
            acc["cost"][key] += Decimal(settled[key])


def _add_interval(intervals: dict[int, dict[str, Any]], index: int, item: Mapping[str, Any],
                  settled: dict[str, str] | None, t0: datetime | None) -> None:
    acc = intervals.setdefault(index, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_tokens": 0,
                                       "parts": {}, "cost": Decimal(0)})
    acc["calls"] += 1
    tokens = int(item["input_tokens"]) + int(item["output_tokens"]) + int(item["cache_tokens"])
    acc["input_tokens"] += int(item["input_tokens"])
    acc["output_tokens"] += int(item["output_tokens"])
    acc["cache_tokens"] += int(item["cache_tokens"])
    acc["parts"][str(item["part"])] = acc["parts"].get(str(item["part"]), 0) + tokens
    if settled is not None:
        acc["cost"] += Decimal(settled["total"])
    acc["t0"] = t0


def _finish_parts(parts: dict[str, dict[str, Any]], total_tokens: int) -> list[dict[str, Any]]:
    rows = []
    for name, acc in parts.items():
        tokens = acc["input_tokens"] + acc["output_tokens"] + acc["cache_tokens"]
        share = (tokens / total_tokens) if total_tokens else 0.0
        rows.append({
            "name": name,
            "label": part_label(name),
            "calls": acc["calls"],
            "input_tokens": acc["input_tokens"],
            "output_tokens": acc["output_tokens"],
            "cache_tokens": acc["cache_tokens"],
            "reasoning_tokens": acc["reasoning_tokens"],
            "total_tokens": tokens,
            "share": round(share, 6),
            "cost_rmb": {key: f"{acc['cost'][key]:.8f}" for key in ("input", "output", "cache", "total")},
        })
    rows.sort(key=lambda row: (-row["total_tokens"], row["name"]))
    return rows


def _finish_intervals(intervals: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index in sorted(intervals):
        acc = intervals[index]
        tokens = acc["input_tokens"] + acc["output_tokens"] + acc["cache_tokens"]
        top_name, top_tokens = ("", 0)
        if acc["parts"]:
            top_name, top_tokens = max(acc["parts"].items(), key=lambda kv: kv[1])
        t0 = acc.get("t0")
        start = None
        if isinstance(t0, datetime) and index >= 0:
            start = t0 + _INTERVAL * index
        if index < 0:
            label = "未记录时间"
        else:
            clock = start.strftime("%m-%d %H:%M") if start is not None else ""
            label = f"第{index + 1}小时 · {index}–{index + 1}h"
            if clock:
                label = f"{label} · {clock}起"
        rows.append({
            "index": index,
            "label": label,
            "start": start.isoformat(timespec="seconds") if start is not None else "",
            "calls": acc["calls"],
            "input_tokens": acc["input_tokens"],
            "output_tokens": acc["output_tokens"],
            "cache_tokens": acc["cache_tokens"],
            "total_tokens": tokens,
            "cost_rmb": f"{acc['cost']:.8f}",
            "top_part": top_name,
            "top_part_label": part_label(top_name) if top_name else "",
            "top_part_tokens": top_tokens,
        })
    return rows


def _hit_rate(input_tokens: int, cache_tokens: int) -> float:
    denom = input_tokens + cache_tokens
    if denom <= 0:
        return 0.0
    return 100.0 * cache_tokens / denom


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


class UsageLedger:
    """Append-only bill for one task. Reloads an existing file so a restarted run keeps its history."""

    def __init__(self, log_dir: str | Path, task: str | None = None) -> None:
        logs = Path(log_dir)
        self.json_path = logs / "api_usage.json"
        parent = logs.parent if logs.name == "logs" else logs
        self.html_path = parent / "api_usage.html"
        self.task = task if task and task not in GENERIC_RUN_DIRS and not task.endswith("_out") else task_label(parent)
        self._lock = threading.Lock()
        self.calls: list[dict[str, Any]] = []
        if self.json_path.is_file():
            try:
                saved = json.loads(self.json_path.read_text(encoding="utf-8"))
                self.calls = list(saved.get("history") or [])
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("could not reload %s (%s); starting a new bill", self.json_path, exc)
        self.write()

    def record_from_row(self, row: Mapping[str, Any]) -> None:
        with self._lock:
            self.calls.append(_call_from_row(row))
            self.write()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return snapshot_from_calls(self.calls, task=self.task, project=PROJECT)

    def write(self) -> None:
        data = snapshot_from_calls(self.calls, task=self.task, project=PROJECT)
        _atomic_write(self.json_path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        _atomic_write(self.html_path, render_task_html(data))
