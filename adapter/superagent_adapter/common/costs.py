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
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
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
    aliases = prices.get("aliases", {})
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
        returned_m = r.get("returned_model")
        req_m = r.get("model")
        price = None
        for candidate in (returned_m, req_m):
            if not candidate:
                continue
            resolved = aliases.get(str(candidate), str(candidate))
            entry = models.get(resolved)
            if isinstance(entry, dict):
                price = entry
                break
            if isinstance(entry, list) and entry:
                price = (_price_rows(prices, str(candidate), r["ts"]) if r.get("ts") else entry[-1]) or entry[-1]
                break
        cost = 0.0
        if price is None:
            if pt or ct:
                missing.add(str(returned_m or req_m))
        else:
            hit = int(r.get("cached_tokens") or 0)
            if hit < 0 or hit > pt:
                hit = 0
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


def _utc(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def _price_rows(prices, model, started_at):
    aliases = prices.get("aliases", {})
    model = aliases.get(model, model)
    source = prices.get("models", {}).get(model)
    if source is None:
        return None
    rows = source if isinstance(source, list) else [source]
    started = _utc(started_at)
    eligible = []
    for row in rows:
        begins = row.get("effective_from", prices.get("effective_from"))
        ends = row.get("effective_until", prices.get("effective_until"))
        if (begins is None or _utc(begins) <= started) and (ends is None or started < _utc(ends)):
            eligible.append((_utc(begins) if begins is not None else datetime.min.replace(tzinfo=timezone.utc), row))
    return max(eligible, key=lambda item: item[0])[1] if eligible else None


def _decimal(row, field):
    value = row.get(field)
    if value is None:
        return None
    try:
        price = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"invalid price {field}") from exc
    if not price.is_finite() or price < 0:
        raise ValueError(f"invalid price {field}")
    return price


def _attempt_cost(prices, model, at, usage):
    price = _price_rows(prices, model, at)
    if price is None:
        return None, Decimal(0), False
    currency = price.get("currency", prices.get("currency"))
    if not currency:
        raise ValueError(f"price for {model} requires a currency")
    total, hit, miss, output = (usage.get(k) for k in (
        "input_tokens", "cached_input_tokens", "uncached_input_tokens", "output_tokens",
    ))
    if any(n is not None and (type(n) is not int or n < 0) for n in (total, hit, miss, output)):
        raise ValueError("invalid token count in usage receipt")
    if total is not None and (hit is not None and hit > total or miss is not None and miss > total
                              or hit is not None and miss is not None and hit + miss != total):
        hit = miss = None
    elif total is not None and (hit is None) != (miss is None):
        hit, miss = (total - miss, miss) if hit is None else (hit, total - hit)
    inp = _decimal(price, "input_per_million")
    cache = _decimal(price, "cached_input_per_million") if "cached_input_per_million" in price else inp
    out = _decimal(price, "output_per_million")
    if total is not None and inp is not None and (inp == cache or total == 0):
        amount, input_complete = total * inp, True
    elif total is not None and hit is None and miss is None and inp is not None and cache is not None:
        # Cache hit/miss breakdown missing while prices differ: record the guaranteed lower bound and keep incomplete.
        amount, input_complete = total * min(inp, cache), False
    else:
        amount = (hit * cache if hit is not None and cache is not None else Decimal(0))
        amount += miss * inp if miss is not None and inp is not None else Decimal(0)
        input_complete = None not in (hit, miss, inp, cache)
    if output is not None and out is not None:
        amount += output * out
    return currency, amount / Decimal(1_000_000), input_complete and output is not None and out is not None


def _receipts(core_output):
    db = Path(core_output) / "logs/operations.sqlite"
    if not db.is_file():
        return None
    uri = db.resolve().as_uri()
    conn = None
    try:
        try:
            conn = sqlite3.connect(uri + "?mode=ro", uri=True)
            conn.execute("BEGIN")
        except sqlite3.OperationalError:
            if conn is not None:
                conn.close()
            conn = sqlite3.connect(uri + "?mode=ro&immutable=1", uri=True)
            conn.execute("BEGIN")
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='view' AND name='api_usage_events_v1'").fetchone() is None:
            return None
        identities = conn.execute("SELECT run_id FROM api_usage_run_v1").fetchall()
        if len(identities) != 1:
            raise ValueError("usage database requires exactly one run ID")
        rows = conn.execute("SELECT run_id, receipt_seq, event_type, event_at, attempt_id, payload_json "
                            "FROM api_usage_events_v1 ORDER BY receipt_seq").fetchall()
        return identities[0][0], rows
    finally:
        if conn is not None:
            conn.close()


def read_usage(core_output, prices, cursor=None, process_exited=False):
    """Read one transaction of charge receipts without importing Core."""
    if not Path(core_output).is_dir():
        raise ValueError(f"Core output directory not found: {core_output}")
    snapshot = _receipts(core_output)
    if snapshot is None:
        if cursor is not None:
            raise ValueError("legacy usage reports have no cursor")
        return {"schema_version": 1, "mode": "legacy", "cursor": None, **recompute(core_output, prices)}
    run_id, rows = snapshot
    after = 0
    if cursor is not None:
        if isinstance(cursor, str):
            run, sep, raw_seq = cursor.rpartition(":")
            if not sep or not raw_seq.isdecimal():
                raise ValueError("expected usage cursor run_id:receipt_seq")
            cursor = {"run_id": run, "receipt_seq": int(raw_seq)}
        after = cursor.get("receipt_seq")
        if cursor.get("run_id") != run_id or type(after) is not int or not 0 <= after <= len(rows):
            raise ValueError("usage cursor refers to a different run or an unknown receipt sequence")
    attempts = {}
    changed = {}
    for expected, (run, seq, kind, at, attempt, payload) in enumerate(rows, 1):
        if run != run_id or seq != expected:
            raise ValueError(f"usage receipt gap at sequence {expected}")
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError(f"invalid usage payload at sequence {seq}")
        if kind == "attempt_started":
            if attempt in attempts:
                raise ValueError(f"duplicate attempt start at sequence {seq}")
            attempts[attempt] = {"started": at, "model": data["model"]}
        elif kind in ("attempt_finished", "usage_received") and attempt in attempts:
            attempts[attempt][kind] = data
        else:
            raise ValueError(f"invalid receipt {kind} at sequence {seq}")
        if seq > after:
            changed[attempt] = None
    status_path = Path(core_output) / "status.json"
    if status_path.is_file():
        state = json.loads(status_path.read_text(encoding="utf8")).get("status")
        process_exited |= state in {"success", "failed", "invalid_input", "no_valid_solution"}
    counts = {"total": len(attempts), "in_flight": 0, "unknown": 0, "settled": 0, "not_sent": 0}
    usage = {"input_tokens_known": 0, "output_tokens_known": 0, "reasoning_tokens_known": 0}
    totals = defaultdict(Decimal)
    models_without_price = set()
    no_billing_statuses = {"rate_limited", "overloaded", "http_rejected"}
    for row in attempts.values():
        finished = row.get("attempt_finished", {})
        receipt = row.get("usage_received", {})
        if finished.get("not_sent"):
            counts["not_sent"] += 1
            continue
        if not receipt and finished.get("status") in no_billing_statuses:
            counts["settled"] += 1
            continue
        returned_model = receipt.get("model")
        model = returned_model or row["model"]
        if returned_model and _price_rows(prices, returned_model, row["started"]) is None and _price_rows(prices, row["model"], row["started"]) is not None:
            model = row["model"]
        for field in usage:
            usage[field] += receipt.get(field.removesuffix("_known")) or 0
        currency, partial, fully_priced = _attempt_cost(prices, model, row["started"], receipt)
        if currency:
            totals[currency] += partial
        else:
            models_without_price.add(model)
        counts["in_flight" if not finished and not process_exited else "settled" if fully_priced else "unknown"] += 1
    complete = not (counts["unknown"] or counts["in_flight"])
    currency = prices.get("currency")
    subtotals = {cur: float(round(value, 10)) for cur, value in sorted(totals.items())}
    single = len(totals) <= 1
    return {
        "schema_version": 1, "mode": "receipts", "cursor": {"run_id": run_id, "receipt_seq": len(rows)},
        "price_table": {"id": prices.get("id") or "legacy-price-table", "currency": next(iter(totals), currency) if single else None},
        "attempts": counts, "usage": usage,
        "cost": {"known_subtotal": next(iter(subtotals.values()), 0.0) if single else None,
                 "by_currency": subtotals, "complete": complete},
        "changed_attempt_ids": list(changed), "models_without_price": sorted(models_without_price),
    }
