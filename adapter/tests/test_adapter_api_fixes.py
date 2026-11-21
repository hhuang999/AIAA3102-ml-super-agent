"""Unit tests for adapter analyze and cost accounting fixes."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from superagent.llm.ledger import Ledger
from superagent_adapter.analysis import analyze
from superagent_adapter.common.costs import read_usage, recompute


def test_analyze_handles_formal_none_llm_cost(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "report").mkdir()
    (tmp_path / "logs" / "search_tree.json").write_text(json.dumps({
        "higher_is_better": True,
        "nodes": [
            {"id": "root", "kind": "root", "parent_id": None, "branch_id": None, "failed": False,
             "score": None, "step": 0, "failure_reasons": [], "error_type": None, "code": ""},
            {"id": "n1", "kind": "draft", "parent_id": "root", "branch_id": 1, "failed": False,
             "score": 0.91, "step": 1, "failure_reasons": [], "error_type": None, "code": "x = 1",
             "method_summary": "baseline"},
        ],
    }), encoding="utf-8")
    (tmp_path / "logs" / "decisions.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "status.json").write_text(json.dumps({
        "status": "success", "best_node_id": "n1", "elapsed_secs": 12.0,
    }), encoding="utf-8")
    (tmp_path / "report" / "run_summary.json").write_text(json.dumps({
        "best": {"path": [{"kind": "draft", "score": 0.91}]},
        "score_timeline": [],
        "key_events": [],
        "resources": {
            "llm": {
                "calls": 2, "failed_calls": 0, "prompt_tokens": 500, "completion_tokens": 100,
                "reasoning_tokens": 20, "cost": None, "currency": "CNY", "accounting": "adapter",
                "by_agent": {"script_coder": {"calls": 2, "cost": None}},
            },
            "embedding": {"calls": 0, "tokens": 0, "cost": 0.0, "currency": "CNY"},
            "exec": {"num_runs": 1, "num_timeouts": 0, "total_script_seconds": 3.0},
        },
    }), encoding="utf-8")

    result = analyze(tmp_path)
    assert result["usage"]["llm_cost"] is None
    assert result["usage"]["cost_per_node"] is None


def test_read_usage_rejections_model_fallback_and_missing_cache_split(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    ledger = Ledger(logs / "operations.sqlite", run_id="run-cost-1")
    ledger.register_operation("op1", "sig", ("script_coder",))
    opts = {
        "model": "deepseek-flash", "thinking": True, "reasoning_effort": "low",
        "max_tokens": 4096, "messages": [{"role": "user", "content": "hi"}],
    }

    # 1. Pre-inference HTTP rejections (429, 503, 400) + pre-send not_sent
    for st in ("rate_limited", "overloaded", "http_rejected", "not_sent"):
        att = ledger.begin_request("op1", "script_coder", "c1", "f1", opts)
        ledger.finish_request(att, st, None, 0.1)

    # 2. Successful call returning a versioned model name not listed in prices.yaml
    att_ok = ledger.begin_request("op1", "script_coder", "c1", "f1", opts)
    ledger.receive_usage(att_ok, {
        "provider_id": "p1",
        "model": "deepseek-v4.1-flash-202611",
        "input_tokens": 2000,
        "cached_input_tokens": 500,
        "uncached_input_tokens": 1500,
        "output_tokens": 500,
        "reasoning_tokens": 100,
    })
    ledger.finish_request(att_ok, "received", "stop", 1.2)

    course_prices = {
        "id": "course-v2",
        "currency": "CNY",
        "models": {"deepseek-flash": {"input_per_million": 2.0, "output_per_million": 8.0}},
    }
    report = read_usage(tmp_path, course_prices, process_exited=True)
    assert report["cost"]["complete"] is True
    assert report["attempts"]["unknown"] == 0
    assert report["attempts"]["not_sent"] == 1
    assert report["attempts"]["settled"] == 4  # 3 clean HTTP rejections + 1 priced call
    assert report["attempts"]["total"] == 5
    assert report["models_without_price"] == []
    assert report["cost"]["known_subtotal"] == pytest.approx((2000 * 2.0 + 500 * 8.0) / 1e6)

    # 3. Missing cache hit/miss fields when cached price differs -> lower bound + complete=False
    att_nocache = ledger.begin_request("op1", "script_coder", "c2", "f2", opts)
    ledger.receive_usage(att_nocache, {
        "provider_id": "p2",
        "model": "deepseek-flash",
        "input_tokens": 1000,
        "output_tokens": 100,
    })
    ledger.finish_request(att_nocache, "received", "stop", 0.5)

    invoice_prices = {
        "id": "invoice-v1",
        "currency": "CNY",
        "models": {"deepseek-flash": {
            "input_per_million": 2.0,
            "cached_input_per_million": 0.04,
            "output_per_million": 8.0,
        }},
    }
    inv = read_usage(tmp_path, invoice_prices, process_exited=True)
    assert inv["cost"]["complete"] is False
    assert inv["attempts"]["unknown"] == 1
    expected_att_ok = (1500 * 2.0 + 500 * 0.04 + 500 * 8.0) / 1e6
    expected_att_nocache_min = (1000 * 0.04 + 100 * 8.0) / 1e6
    assert inv["cost"]["known_subtotal"] == pytest.approx(expected_att_ok + expected_att_nocache_min)


def test_legacy_recompute_guards_cached_tokens_and_uses_aliases(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "llm_calls.jsonl").write_text(
        json.dumps({
            "attempt_id": "a1",
            "agent": "script_coder",
            "model": "deepseek-flash",
            "returned_model": "deepseek-v4.1-flash",
            "prompt_tokens": 100,
            "cached_tokens": 500,  # malformed > prompt_tokens; must not go negative!
            "completion_tokens": 50,
            "cost": 0.0006,
        }) + "\n",
        encoding="utf-8",
    )
    prices = {
        "currency": "CNY",
        "aliases": {"deepseek-v4.1-flash": "deepseek-flash"},
        "models": {"deepseek-flash": {"input_per_million": 2.0, "cached_input_per_million": 0.04, "output_per_million": 8.0}},
    }
    res = recompute(tmp_path, prices)
    assert res["llm"]["cost"] == pytest.approx((100 * 2.0 + 50 * 8.0) / 1e6)
    assert res["cost_complete"] is True
