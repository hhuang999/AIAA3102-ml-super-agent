"""Recompute the model cost of a Core run from its token logs with a given price table.

The price table is a YAML file:

    currency: CNY
    models:                       # chat models: price per 1M input / output tokens
      deepseek-flash: {input_per_million: 2.0, output_per_million: 8.0}

Output tokens include reasoning tokens. Local embedding records report compute seconds.
Historical hosted embedding records can use an optional embeddings price table.
Missing usage remains unknown; totals below are known subtotals.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import yaml


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_prices(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "models" not in data:
        raise ValueError(f"{path}: missing 'models'")
    return data


def recompute(core_out: Path, prices: Mapping[str, Any]) -> dict[str, Any]:
    """Cost per agent and model at ``prices``, next to the cost the Core recorded."""
    logs = Path(core_out) / "logs"
    models = prices.get("models", {})
    emb_prices = prices.get("embeddings", {})
    by_agent: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0})
    missing: set[str] = set()
    total = recorded = 0.0
    pt_total = ct_total = unknown = 0
    late = {r["attempt_id"]:r for r in _rows(logs / "late_usage.jsonl")}
    for r in _rows(logs / "llm_calls.jsonl"):
        if r.get("attempt_id") in late:
            r = {**r, **late[r["attempt_id"]]}
        pt, ct = int(r.get("prompt_tokens") or 0), int(r.get("completion_tokens") or 0)
        unknown += int(r.get("prompt_tokens") is None or r.get("completion_tokens") is None)
        recorded += float(r.get("cost") or 0.0)
        price = models.get(str(r.get("model")))
        cost = 0.0
        if price is None:
            if pt or ct:
                missing.add(str(r.get("model")))
        else:
            hit = int(r.get("cached_tokens") or 0)
            cost = ((pt-hit)*float(price["input_per_million"]) + hit*float(price.get("cached_input_per_million",price["input_per_million"])) + ct*float(price["output_per_million"]))/1e6
        b = by_agent[str(r.get("agent"))]
        b["calls"] += 1
        b["prompt_tokens"] += pt
        b["completion_tokens"] += ct
        b["cost"] += cost
        total += cost
        pt_total += pt
        ct_total += ct
    emb_cost = emb_recorded = 0.0
    emb_tokens = 0
    local_seconds = 0.0
    for r in _rows(logs / "embedding_calls.jsonl"):
        tokens = int(r.get("tokens") or 0)
        emb_tokens += tokens
        if r.get("backend") == "local":
            local_seconds += float(r.get("seconds") or 0.0)
            continue
        unknown += int(r.get("tokens") is None)
        emb_recorded += float(r.get("cost") or 0.0)
        price = emb_prices.get(str(r.get("model")))
        if price is None:
            if tokens:
                missing.add(str(r.get("model")))
            continue
        emb_cost += tokens * float(price["per_million"]) / 1e6
    return {
        "currency": prices.get("currency", ""),
        "llm": {"prompt_tokens": pt_total, "completion_tokens": ct_total, "cost": round(total, 6),
                "by_agent": {k: {**v, "cost": round(v["cost"], 6)} for k, v in sorted(by_agent.items())}},
        "embedding": {"tokens": emb_tokens, "local_compute_seconds":round(local_seconds,3), "cost": round(emb_cost, 6)},
        "total_cost": round(total + emb_cost, 6),
        "recorded_by_core": {"llm": round(recorded, 6), "embedding": round(emb_recorded, 6),
                             "total": round(recorded + emb_recorded, 6)},
        "unknown_usage_attempts":unknown, "cost_complete": not unknown and not missing,
        "models_without_price": sorted(missing),
    }
