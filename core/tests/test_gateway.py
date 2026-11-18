"""LLM gateway: parsing, validation and cost are tested as plain functions; everything that
needs a reply uses the real endpoint (marked ``llm``)."""

import os

import pytest

from superagent.config import load_defaults
from superagent.errors import LLMUnavailableError, StructuredOutputError
from superagent.llm import Field, abort_decision, call_cost, make_gateway
from superagent.llm.jsonparse import parse_json_object
from superagent.llm.schema import validate
from superagent.utils import load_env_file, read_jsonl

SCHEMA = {"has_error": Field("boolean", "x"), "score": Field("number", "y", nullable=True), "note": Field("string", "z")}
GOAL_SCHEMA = {"goal": Field("enum", "minimize or maximize", values=("minimize", "maximize")), "explanation": Field("string", "why")}


def test_parse_fenced():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_with_surrounding_text():
    assert parse_json_object('Here you go: {"a": {"b": "}"}} hope it helps {"c": 2}') == {"a": {"b": "}"}}


def test_parse_unescaped_newline():
    assert parse_json_object('{"a": "line1\nline2\tx"}') == {"a": "line1\nline2\tx"}


def test_parse_no_object():
    assert parse_json_object("nope") is None


def test_bool_string_coerced():
    obj, errors = validate(parse_json_object('{"has_error": "true", "score": "0.5", "note": "n"}'), SCHEMA)
    assert not errors and obj["has_error"] is True and obj["score"] == 0.5


def test_missing_field_reported():
    _obj, errors = validate({"has_error": False}, SCHEMA)
    assert errors and any("note" in e for e in errors)


def test_bad_enum_reported():
    _obj, errors = validate({"goal": "bigger", "explanation": "x"}, GOAL_SCHEMA)
    assert errors


def test_call_cost():
    group = load_defaults()["llm"]["generation"]
    expected = (1000 * group["input_price_per_million"] + 500 * group["output_price_per_million"]) / 1e6
    assert call_cost(group, 1000, 500) == pytest.approx(expected)
    assert call_cost(group, None, None) is None


def test_abort_decision():
    # Generating roles stop when the finalize reserve begins; assessing roles may use half of it.
    assert abort_decision("script_coder", 100, 300, False, False)
    assert not abort_decision("script_coder", 301, 300, False, False)
    assert not abort_decision("result_reader", 200, 300, False, False)
    assert abort_decision("result_reader", 150, 300, False, False)
    assert abort_decision("script_coder", 1000, 300, False, True)
    assert abort_decision("leak_checker", 1000, 300, True, False)


def test_load_env_file(tmp_path):
    f = tmp_path / ".env"
    f.write_text("# comment\nA_KEY=abc\nexport B_KEY='quoted value'\nEXISTING=new\nbroken line\n")
    env = {"EXISTING": "old"}
    assert load_env_file(f, env) == ["A_KEY", "B_KEY"]
    assert env == {"EXISTING": "old", "A_KEY": "abc", "B_KEY": "quoted value"}
    assert load_env_file(tmp_path / "missing.env", env) == []


def real_gateway(tmp_path, **llm_over):
    cfg = load_defaults()["llm"]
    cfg.update(llm_over)
    return make_gateway(cfg, os.environ[cfg["api_key_env"]], tmp_path / "calls.jsonl")


@pytest.mark.llm
def test_real_llm_text_and_structured(tmp_path):
    g = real_gateway(tmp_path)
    t = g.text("script_coder", "You are terse.", "Say the word hello and nothing else.")
    assert "hello" in t.content.lower()
    obj = g.structured("direction_judge", "You judge metrics.", "The metric is accuracy. Should it be minimized or maximized?", GOAL_SCHEMA)
    assert obj["goal"] == "maximize"
    rows = read_jsonl(tmp_path / "calls.jsonl")
    assert all(r["ok"] for r in rows)
    assert rows[0]["thinking"] is True and rows[1]["thinking"] is False
    assert sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows) > 0
    assert all(r["cost"] and r["cost"] > 0 for r in rows)
    for r in rows:
        for k in ("ts", "call_id", "attempt", "agent", "model", "thinking", "prompt_tokens", "completion_tokens",
                  "reasoning_tokens", "latency_secs", "error"):
            assert k in r


@pytest.mark.llm
def test_real_llm_retry_after_rejected_answer(tmp_path):
    g = real_gateway(tmp_path)
    seen = []

    def reject_first(obj, is_last):
        seen.append(obj)
        return (obj, ["rejected once on purpose: answer again"]) if len(seen) == 1 else (obj, [])

    obj = g.structured("direction_judge", "You judge metrics.", "The metric is log loss. Should it be minimized or maximized?",
                       GOAL_SCHEMA, node_id="n1", extra_check=reject_first)
    assert obj["goal"] == "minimize"
    rows = read_jsonl(tmp_path / "calls.jsonl")
    assert len(rows) == 2 and rows[0]["call_id"] == rows[1]["call_id"]
    assert [r["attempt"] for r in rows] == [1, 2]
    assert rows[0]["ok"] is False and rows[1]["ok"] is True and rows[0]["node_id"] == "n1"


@pytest.mark.llm
def test_real_llm_always_rejected_raises(tmp_path):
    g = real_gateway(tmp_path)
    with pytest.raises(StructuredOutputError):
        g.structured("direction_judge", "You judge metrics.", "The metric is accuracy. Minimize or maximize?", GOAL_SCHEMA,
                     extra_check=lambda obj, is_last: (obj, ["never accepted"]))
    assert len(read_jsonl(tmp_path / "calls.jsonl")) == load_defaults()["llm"]["json_max_attempts"]


def test_unreachable_endpoint_gives_up(tmp_path):
    """No reply is involved: the endpoint address does not accept connections."""
    cfg = load_defaults()["llm"]
    cfg.update(base_url="http://127.0.0.1:9/v1", api_max_retries=2, api_retry_delay_seconds=0.1, request_timeout_seconds=5)
    g = make_gateway(cfg, "not-a-real-key", tmp_path / "calls.jsonl")
    with pytest.raises(LLMUnavailableError):
        g.text("desc_cleaner", "s", "u")
    rows = read_jsonl(tmp_path / "calls.jsonl")
    assert len(rows) == 2 and all(not r["ok"] and r["cost"] is None for r in rows)
    assert "not-a-real-key" not in (tmp_path / "calls.jsonl").read_text()
