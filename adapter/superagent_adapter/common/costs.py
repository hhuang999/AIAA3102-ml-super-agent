"""Recompute the model cost of a Core run from its token logs with a given price table.

The price table is a YAML file:

    currency: CNY
    aliases:
      deepseek-chat: deepseek-v4.1-flash-1017
    models:                       # chat models: price per 1M input / output tokens
      deepseek-v4.1-flash-1017: {input_per_million: 2.0, output_per_million: 8.0, cached_input_per_million: 0.2}
    embeddings:                   # embedding models: price per 1M tokens
      qwen3-embedding-0.6b: {per_million: 0.5}

Output tokens include reasoning tokens. Non-billing statuses (not_sent, rate_limited, overloaded,
http_rejected) and failed requests without token counts cost nothing.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import yaml

NO_BILLING_STATUSES = frozenset({"not_sent", "rate_limited", "overloaded", "http_rejected"})


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _receipts(db_path: Path) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    uri = f"file:{db_path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.OperationalError:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    rows: list[dict[str, Any]] = []
    try:
        try:
            cur = conn.execute("SELECT role, status, usage FROM attempts ORDER BY started")
        except sqlite3.OperationalError:
            conn.close()
            conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
            cur = conn.execute("SELECT role, status, usage FROM attempts ORDER BY started")
        for role, status, payload in cur.fetchall():
            usage = json.loads(payload) if payload else {}
            rows.append({
                "agent": role,
                "status": status,
                "model": usage.get("model"),
                "returned_model": usage.get("returned_model"),
                "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
                "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
                "cached_tokens": usage.get("cached_tokens", usage.get("cached_input_tokens")),
                "reasoning_tokens": usage.get("reasoning_tokens"),
                "cost": usage.get("cost"),
            })
    finally:
        conn.close()
    return rows


def read_usage(core_out: Path) -> list[dict[str, Any]]:
    logs = Path(core_out) / "logs"
    jsonl = logs / "llm_calls.jsonl"
    if jsonl.exists():
        raw = _rows(jsonl)
    else:
        raw = _receipts(logs / "operations.sqlite")
    out: list[dict[str, Any]] = []
    for r in raw:
        status = r.get("status")
        if status in NO_BILLING_STATUSES:
            continue
        out.append(r)
    return out


def _resolve_price(row: Mapping[str, Any], models: Mapping[str, Any], aliases: Mapping[str, str]) -> tuple[Mapping[str, Any] | None, str]:
    for key in ("returned_model", "model"):
        raw = row.get(key)
        if not raw:
            continue
        s = str(raw)
        canonical = str(aliases.get(s, s))
        if canonical in models:
            return models[canonical], canonical
        if s in models:
            return models[s], s
    fallback = str(row.get("returned_model") or row.get("model") or "")
    return None, str(aliases.get(fallback, fallback))


def load_prices(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "models" not in data:
        raise ValueError(f"{path}: missing 'models'")
    return data


def recompute(core_out: Path, prices: Mapping[str, Any]) -> dict[str, Any]:
    """Cost per agent and model at ``prices``, next to the cost the Core recorded."""
    logs = Path(core_out) / "logs"
    models = prices.get("models", {})
    aliases = prices.get("aliases") or {}
    emb_prices = prices.get("embeddings", {})
    by_agent: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0})
    missing: set[str] = set()
    total = recorded = 0.0
    pt_total = ct_total = 0
    input_complete = True
    for r in read_usage(core_out):
        pt, ct = int(r.get("prompt_tokens") or 0), int(r.get("completion_tokens") or 0)
        raw_cached = r.get("cached_tokens")
        recorded += float(r.get("cost") or 0.0)
        price, resolved_name = _resolve_price(r, models, aliases)
        cost = 0.0
        if price is None:
            if pt or ct:
                missing.add(resolved_name)
        else:
            inp_rate = float(price["input_per_million"])
            out_rate = float(price["output_per_million"])
            cache_rate = float(price.get("cached_input_per_million", inp_rate))
            if raw_cached is not None:
                cached = max(0, min(int(raw_cached), pt))
                uncached = max(0, pt - cached)
                cost = (uncached * inp_rate + cached * cache_rate + ct * out_rate) / 1e6
            elif pt > 0 and cache_rate != inp_rate:
                input_complete = False
                cost = (pt * min(inp_rate, cache_rate) + ct * out_rate) / 1e6
            else:
                cost = (pt * inp_rate + ct * out_rate) / 1e6
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
    for r in _rows(logs / "embedding_calls.jsonl"):
        tokens = int(r.get("tokens") or 0)
        emb_tokens += tokens
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
                "input_complete": input_complete,
                "by_agent": {k: {**v, "cost": round(v["cost"], 6)} for k, v in sorted(by_agent.items())}},
        "embedding": {"tokens": emb_tokens, "cost": round(emb_cost, 6)},
        "total_cost": round(total + emb_cost, 6),
        "recorded_by_core": {"llm": round(recorded, 6), "embedding": round(emb_recorded, 6),
                             "total": round(recorded + emb_recorded, 6)},
        "models_without_price": sorted(missing),
    }
