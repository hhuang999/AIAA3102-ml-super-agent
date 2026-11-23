"""Off-peak billing, per-part ranking, and hourly intervals."""

from datetime import datetime, timedelta, timezone

from superagent.billing import cost_rmb, snapshot_from_calls, split_usage
from superagent.usage_page import render_board_html, render_task_html
from superagent.usage_board import collect_tasks


def test_split_keeps_cache_out_of_input():
    parts = split_usage({
        "prompt_tokens": 1000,
        "completion_tokens": 400,
        "prompt_tokens_details": {"cached_tokens": 250},
        "completion_tokens_details": {"reasoning_tokens": 80},
    })
    assert parts == {"input_tokens": 750, "output_tokens": 400, "cache_tokens": 250, "reasoning_tokens": 80}


def test_flash_and_pro_cards():
    assert cost_rmb(1_000_000, 1_000_000, 1_000_000, "deepseek-v4.1-flash")["total"] == "5.02000000"
    assert cost_rmb(1_000_000, 0, 0, "deepseek-v4-pro")["total"] == "4.50000000"
    assert cost_rmb(10, 10, 10, "gpt-4.1") is None


def test_part_rank_and_hourly_intervals():
    start = datetime(2026, 11, 12, 8, 0, tzinfo=timezone.utc)
    later = start + timedelta(hours=1, minutes=5)
    calls = [
        {"ts": start.isoformat(), "agent": "script_coder", "model": "deepseek-v4.1-flash",
         "input_tokens": 1000, "completion_tokens": 100, "cache_tokens": 0},
        {"ts": start.isoformat(), "agent": "reviewer", "model": "deepseek-v4.1-flash",
         "prompt_tokens": 50, "completion_tokens": 10, "cache_tokens": 0},
        {"ts": later.isoformat(), "part": "draft", "model": "deepseek-v4.1-flash",
         "input_tokens": 2_000_000, "output_tokens": 0, "cache_tokens": 0},
    ]
    data = snapshot_from_calls(calls, task="demo", project="SuperAgent")
    assert data["by_part"][0]["name"] == "draft"
    assert data["by_part"][0]["label"] == "初稿"
    assert data["top_part"]["total_tokens"] == 2_000_000
    assert [row["index"] for row in data["by_interval"]] == [0, 1]
    assert data["top_interval"]["index"] == 1
    assert data["cost_rmb"]["total"] == "2.00149000"
    page = render_board_html([data])
    assert "demo" in page and "初稿" in page and "第2小时" in page


def test_board_reads_api_usage_once(tmp_path):
    run = tmp_path / "logs"
    run.mkdir()
    bill = snapshot_from_calls(
        [{"ts": "2026-11-12T08:00:00+08:00", "part": "improve", "model": "deepseek-v4.1-flash",
          "input_tokens": 100, "output_tokens": 20, "cache_tokens": 5}],
        task="leaf", project="MLEvolve",
    )
    (run / "api_usage.json").write_text(__import__("json").dumps(bill), encoding="utf-8")
    (run / "llm_calls.jsonl").write_text('{"agent":"script_coder","model":"deepseek-v4.1-flash","prompt_tokens":1,"completion_tokens":1}\n', encoding="utf-8")
    tasks = collect_tasks([tmp_path])
    assert len(tasks) == 1
    assert tasks[0]["task"] == "leaf"
    assert tasks[0]["by_part"][0]["label"] == "改进"
