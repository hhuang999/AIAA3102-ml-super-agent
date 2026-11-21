"""Unit tests covering API review fixes (P0/P1/P2) across LLM gateway, transport, ledger, prompts and config."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

from superagent.codegen import CodeGenerator
from superagent.config import ConfigError, build_config, load_defaults
from superagent.control.state import build_state
from superagent.errors import LLMUnavailableError
from superagent.llm import Field
from superagent.llm.async_transport import AsyncDeepSeekTransport
from superagent.llm.gateway import (
    LLMGateway,
    OpenAITransport,
    TransientAPIError,
    TransportResponse,
)
from superagent.llm.ledger import CancelledCall, DeferredCall, Ledger, PermanentCall
from superagent.llm.schema import validate
from superagent.operations import Dispatch, Operations
from superagent.control.adaptive import Proposal
from superagent.prompts import blocks, builders
from superagent.report import _llm_usage
from superagent.run import _clean_description, _judge_direction
from superagent.utils import read_jsonl


def _mock_async_client(monkeypatch, handler):
    real_async = openai.AsyncOpenAI
    monkeypatch.setattr(
        openai,
        "AsyncOpenAI",
        lambda **kw: real_async(
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kw
        ),
    )


def _mock_sync_client(monkeypatch, handler):
    real_sync = openai.OpenAI
    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kw: real_sync(
            http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kw
        ),
    )


def _sse_body(frames):
    return "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"


def _make_formal_gateway(tmp_path, handler, monkeypatch, **cfg_overrides):
    _mock_async_client(monkeypatch, handler)
    cfg = load_defaults()["llm"]
    cfg.update(cfg_overrides)
    tr = AsyncDeepSeekTransport(cfg["base_url"], "test-key", cfg["request_limits"], slots=2)
    prompts_dir = tmp_path / "runs"

    def pdir(node_id):
        d = prompts_dir / node_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    gw = LLMGateway(
        cfg,
        tmp_path / "logs" / "llm_calls.jsonl",
        tr,
        prompt_log_dir=pdir,
        sleep=lambda _s: None,
        run_id="run-test-1",
    )
    gw.ledger.register_operation("op1", "sig1", ("draft_writer.data_prep", "reviewer"))
    return gw, tr


# --------------------------------------------------------------------------- P0: 429 / 503 / 524 & receipts


def test_formal_429_backoff_preserves_extra_sends_and_keeps_cost_complete(tmp_path, monkeypatch):
    from superagent_adapter.common.costs import read_usage

    calls = []

    def handler(req):
        calls.append(req)
        if len(calls) <= 3:
            return httpx.Response(429, json={"error": {"message": "rate limit reached"}})
        frames = [
            {"id": "resp-1", "model": "deepseek-flash", "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]},
            {"id": "resp-1", "model": "deepseek-flash", "choices": [], "usage": {
                "prompt_tokens": 1000, "completion_tokens": 200,
                "prompt_cache_hit_tokens": 400, "prompt_cache_miss_tokens": 600,
            }},
        ]
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=_sse_body(frames))

    gw, tr = _make_formal_gateway(tmp_path, handler, monkeypatch, api_retry_delay_seconds=0.05, operation_extra_sends=2)
    try:
        delays = []
        for _ in range(3):
            with pytest.raises(DeferredCall) as exc_info:
                gw.text("draft_writer.data_prep", "sys", "usr", "op1")
            delays.append(exc_info.value.resume_at - time.monotonic())
            gw.cooldown_until = 0.0  # simulate waiting out cooldown
        # 3 rate limits exceeded operation_extra_sends=2, yet 4th call still succeeds!
        res = gw.text("draft_writer.data_prep", "sys", "usr", "op1")
        assert res.content == "ok"
        assert len(calls) == 4
        # Exponential backoff should increase delay across consecutive 429s
        assert delays[2] > delays[0]

        prices = {"id": "course-v2", "currency": "CNY", "models": {"deepseek-flash": {"input_per_million": 2.0, "output_per_million": 8.0}}}
        usage = read_usage(tmp_path, prices, process_exited=True)
        assert usage["cost"]["complete"] is True
        assert usage["attempts"]["unknown"] == 0
        assert usage["attempts"]["total"] == 4
        assert usage["cost"]["known_subtotal"] == pytest.approx((1000 * 2.0 + 200 * 8.0) / 1e6)

        ctx = SimpleNamespace(
            flags={"formal": True},
            gateway=gw,
            paths=SimpleNamespace(llm_calls=tmp_path / "logs" / "llm_calls.jsonl"),
            cfg={"llm": gw.cfg},
        )
        rep = _llm_usage(ctx)
        assert rep["unknown_usage"] == 0
        assert rep["cost"] is None
    finally:
        tr.close()


def test_formal_rate_limit_retries_are_bounded(tmp_path, monkeypatch):
    def handler(_req):
        return httpx.Response(429, json={"error": {"message": "throttled"}})

    gw, tr = _make_formal_gateway(tmp_path, handler, monkeypatch, operation_rate_limit_retries=2, api_retry_delay_seconds=0.01)
    try:
        for _ in range(2):
            with pytest.raises(DeferredCall):
                gw.text("draft_writer.data_prep", "s", "u", "op1")
            gw.cooldown_until = 0.0
        with pytest.raises(PermanentCall, match="rate_limit_exhausted"):
            gw.text("draft_writer.data_prep", "s", "u", "op1")
    finally:
        tr.close()


def test_formal_503_honors_api_max_retries_then_defers_not_permanent(tmp_path, monkeypatch):
    from superagent_adapter.common.costs import read_usage

    calls = []

    def handler(_req):
        calls.append(1)
        return httpx.Response(503, json={"error": {"message": "server overloaded"}})

    gw, tr = _make_formal_gateway(tmp_path, handler, monkeypatch, api_max_retries=2, api_retry_delay_seconds=0.01)
    try:
        with pytest.raises(DeferredCall, match="http_503") as exc_info:
            gw.text("draft_writer.data_prep", "s", "u", "op1")
        assert exc_info.value.resume_at is not None
        # Initial attempt + 2 retries = 3 sends (proves api_max_retries > 1 works)
        assert len(calls) == 3

        prices = {"id": "course-v2", "currency": "CNY", "models": {"deepseek-flash": {"input_per_million": 2.0, "output_per_million": 8.0}}}
        usage = read_usage(tmp_path, prices, process_exited=True)
        assert usage["cost"]["complete"] is True
        assert usage["attempts"]["unknown"] == 0
    finally:
        tr.close()


def test_formal_524_marks_unknown_usage_and_sets_cooldown(tmp_path, monkeypatch):
    from superagent_adapter.common.costs import read_usage

    def handler(_req):
        return httpx.Response(524, text="gateway timeout")

    gw, tr = _make_formal_gateway(tmp_path, handler, monkeypatch)
    try:
        with pytest.raises(DeferredCall, match="http_524") as exc_info:
            gw.text("draft_writer.data_prep", "s", "u", "op1")
        assert exc_info.value.resume_at - time.monotonic() > 100

        prices = {"id": "course-v2", "currency": "CNY", "models": {"deepseek-flash": {"input_per_million": 2.0, "output_per_million": 8.0}}}
        usage = read_usage(tmp_path, prices, process_exited=True)
        assert usage["cost"]["complete"] is False
        assert usage["attempts"]["unknown"] == 1
    finally:
        tr.close()


def test_formal_http_400_preserves_error_message_and_logs_prompt(tmp_path, monkeypatch):
    def handler(_req):
        return httpx.Response(400, json={"error": {"message": "invalid parameter: reasoning_effort"}})

    gw, tr = _make_formal_gateway(tmp_path, handler, monkeypatch)
    try:
        with pytest.raises(PermanentCall, match="invalid parameter: reasoning_effort"):
            gw.text("draft_writer.data_prep", "sys-prompt", "usr-prompt", "op1")
        prompts = read_jsonl(tmp_path / "runs" / "op1" / "prompts.jsonl")
        assert len(prompts) == 1
        assert prompts[0]["messages"][0]["content"] == "sys-prompt"
        calls = read_jsonl(tmp_path / "logs" / "llm_calls.jsonl")
        assert len(calls) == 1
        assert calls[0]["ok"] is False and calls[0]["http_status"] == 400
        assert "invalid parameter: reasoning_effort" in calls[0]["error"]
    finally:
        tr.close()


def test_formal_retry_after_from_body(tmp_path, monkeypatch):
    def handler(_req):
        return httpx.Response(429, json={"error": {"message": "slow down"}, "retry_after": 11})

    gw, tr = _make_formal_gateway(tmp_path, handler, monkeypatch)
    try:
        with pytest.raises(DeferredCall) as exc_info:
            gw.text("draft_writer.data_prep", "s", "u", "op1")
        assert 9.5 < exc_info.value.resume_at - time.monotonic() <= 12.0
    finally:
        tr.close()


def test_legacy_retries_beyond_one_and_releases_rejected_reserve(tmp_path, monkeypatch):
    calls = []

    def handler(_req):
        calls.append(1)
        if len(calls) <= 2:
            return httpx.Response(503, json={"error": {"message": "busy"}})
        return httpx.Response(200, json={
            "id": "r1", "model": "deepseek-flash", "object": "chat.completion", "created": 1,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        })

    _mock_sync_client(monkeypatch, handler)
    cfg = load_defaults()["llm"]
    cfg.update(stream=False, api_max_retries=2, api_retry_delay_seconds=0)
    gw = LLMGateway(cfg, tmp_path / "calls.jsonl", OpenAITransport(cfg["base_url"], "k", 5), sleep=lambda _s: None)
    gw.ledger.limit = 1.0  # tight cap: 503 rejections must NOT permanently lock reserve!
    res = gw.text("script_coder", "s", "u", "op1")
    assert res.content == "done"
    assert len(calls) == 3
    summary = gw.ledger.summary()
    assert summary["unknown_or_inflight"] == 0
    assert summary["risk_reserve"] == 0


# --------------------------------------------------------------------------- P0: Async cancel & non-blocking SQLite


def test_async_transport_sqlite_off_event_loop_and_clean_cancellation(tmp_path):
    entered_stream = threading.Event()
    release_server = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            if body.get("stream"):
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    self.wfile.write(b'data: {"id":"s1","model":"deepseek-flash","choices":[{"delta":{"content":"hi"}}]}\n\n')
                    self.wfile.flush()
                    entered_stream.set()
                    release_server.wait(3)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                payload = json.dumps({
                    "id": "ok1", "model": "deepseek-flash", "object": "chat.completion", "created": 1,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "fast"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                }).encode()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        def log_message(self, *_):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        t.start()
        limits = load_defaults()["llm"]["request_limits"]
        tr = AsyncDeepSeekTransport(f"http://127.0.0.1:{server.server_port}", "k", limits, slots=2)
        ledger = Ledger(tmp_path / "operations.sqlite", run_id="r-cancel")
        ledger.register_operation("op-c", "s", ("draft_writer.data_prep",))
        begin_thread_names = []
        cancel_ev = threading.Event()

        def begin():
            begin_thread_names.append(threading.current_thread().name)
            return ledger.begin_request("op-c", "draft_writer.data_prep", "c1", "f1", {
                "model": "deepseek-flash", "thinking": False, "reasoning_effort": None,
                "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}],
            })

        req = {
            "role": "draft_writer.data_prep", "model": "deepseek-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100, "temperature": 0.0, "thinking": False,
            "reasoning_effort": None, "json_mode": False, "stream": True,
        }

        def trigger_cancel():
            assert entered_stream.wait(3)
            cancel_ev.set()

        ct = threading.Thread(target=trigger_cancel, daemon=True)
        ct.start()
        try:
            with pytest.raises(CancelledCall):
                tr.invoke(req, cancel_ev.is_set, begin, ledger.receive_usage, ledger.finish_request)
            assert begin_thread_names and all("deepseek-ledger" in n for n in begin_thread_names)
            with ledger.db() as db:
                statuses = [r[0] for r in db.execute("SELECT status FROM attempts").fetchall()]
            assert statuses == ["cancelled"]
            # close() must be safe and idempotent
            tr.close()
            tr.close()
        finally:
            release_server.set()
            server.shutdown()
            t.join(timeout=2)


def test_async_cancel_during_slow_begin_still_finishes_attempt(tmp_path):
    begin_entered = threading.Event()
    allow_begin_return = threading.Event()
    ledger = Ledger(tmp_path / "operations.sqlite", run_id="r-begin-cancel")
    ledger.register_operation("op-b", "s", ("desc_cleaner",))
    limits = load_defaults()["llm"]["request_limits"]
    tr = AsyncDeepSeekTransport("http://127.0.0.1:1", "k", limits, slots=1)
    cancel_ev = threading.Event()

    def slow_begin():
        begin_entered.set()
        allow_begin_return.wait(2)
        return ledger.begin_request("op-b", "desc_cleaner", "c1", "f1", {
            "model": "deepseek-flash", "thinking": False, "reasoning_effort": None,
            "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}],
        })

    def cancel_mid_begin():
        assert begin_entered.wait(2)
        cancel_ev.set()
        time.sleep(0.1)
        allow_begin_return.set()

    threading.Thread(target=cancel_mid_begin, daemon=True).start()
    req = {
        "role": "desc_cleaner", "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100, "temperature": 0.0, "thinking": False,
        "reasoning_effort": None, "json_mode": False, "stream": True,
    }
    try:
        with pytest.raises(CancelledCall):
            tr.invoke(req, cancel_ev.is_set, slow_begin, ledger.receive_usage, ledger.finish_request)
        with ledger.db() as db:
            statuses = [r[0] for r in db.execute("SELECT status FROM attempts").fetchall()]
        assert statuses == ["cancelled"]
    finally:
        tr.close()


# --------------------------------------------------------------------------- P1: non-streaming timeout, model_idle, cached_tokens, APIError


def test_async_non_streaming_uses_attempt_cap_not_read_idle(tmp_path):
    class SlowHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            time.sleep(0.35)
            payload = json.dumps({
                "id": "ns1", "model": "deepseek-flash", "object": "chat.completion", "created": 1,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                          "prompt_cache_hit_tokens": None, "prompt_tokens_details": {"cached_tokens": 4}},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), SlowHandler) as server:
        t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        t.start()
        limits = dict(load_defaults()["llm"]["request_limits"], read_idle_seconds=0.15, short_attempt_seconds=5.0)
        tr = AsyncDeepSeekTransport(f"http://127.0.0.1:{server.server_port}", "k", limits, slots=1)
        ledger = Ledger(tmp_path / "operations.sqlite", run_id="r-ns")
        ledger.register_operation("op-ns", "s", ("desc_cleaner",))
        req = {
            "role": "desc_cleaner", "model": "deepseek-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100, "temperature": 0.0, "thinking": False,
            "reasoning_effort": None, "json_mode": False, "stream": False,
        }
        try:
            resp = tr.invoke(
                req,
                lambda: False,
                lambda: ledger.begin_request("op-ns", "desc_cleaner", "c1", "f1", req),
                ledger.receive_usage,
                ledger.finish_request,
            )
            assert resp.content == "ok"
            # Null prompt_cache_hit_tokens falls back to prompt_tokens_details.cached_tokens
            assert resp.cached_tokens == 4
        finally:
            tr.close()
            server.shutdown()
            t.join(timeout=2)


def test_async_model_idle_preserves_partial_metadata_and_retries_empty(tmp_path):
    release = threading.Event()
    hits = []

    class IdleHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            hits.append(body)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                if body["messages"][-1]["content"] == "partial":
                    self.wfile.write(b'data: {"id":"mid-1","model":"deepseek-flash","choices":[{"delta":{"content":"abc"}}]}\n\n')
                    self.wfile.flush()
                    release.wait(2)
                elif len(hits) == 1:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    release.wait(2)
                else:
                    self.wfile.write(_sse_body([
                        {"id": "ok-2", "model": "deepseek-flash", "choices": [{"delta": {"content": "recovered"}, "finish_reason": "stop"}]},
                        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
                    ]).encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), IdleHandler) as server:
        t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        t.start()
        cfg = load_defaults()["llm"]
        cfg["request_limits"] = dict(cfg["request_limits"], model_idle_seconds=0.2, read_idle_seconds=5.0)
        cfg["api_retry_delay_seconds"] = 0
        tr = AsyncDeepSeekTransport(f"http://127.0.0.1:{server.server_port}", "k", cfg["request_limits"], slots=2)
        gw = LLMGateway(cfg, tmp_path / "calls.jsonl", tr, sleep=lambda _s: None, run_id="r-idle")
        gw.ledger.register_operation("op-empty", "s1", ("desc_cleaner",))
        gw.ledger.register_operation("op-part", "s2", ("desc_cleaner",))
        try:
            # Empty idle timeout raises TransientAPIError and retries to success
            res1 = gw.text("desc_cleaner", "s", "empty", "op-empty")
            assert res1.content == "recovered"
            assert len(hits) == 2

            # Partial idle timeout preserves provider_id and model
            res2 = gw.text("desc_cleaner", "s", "partial", "op-part")
            assert res2.content == "abc" and res2.finish_reason == "model_idle_timeout"
            rows = read_jsonl(tmp_path / "calls.jsonl")
            part_row = [r for r in rows if r["operation_id"] == "op-part"][-1]
            assert part_row["provider_id"] == "mid-1"
            assert part_row["returned_model"] == "deepseek-flash"
        finally:
            release.set()
            tr.close()
            server.shutdown()
            t.join(timeout=2)


def test_async_transport_does_not_misclassify_ledger_valueerror_and_handles_base_api_error(tmp_path, monkeypatch):
    def handler(_req):
        return httpx.Response(200, json={
            "id": "r1", "model": "deepseek-flash", "object": "chat.completion", "created": 1,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    _mock_async_client(monkeypatch, handler)
    limits = load_defaults()["llm"]["request_limits"]
    tr = AsyncDeepSeekTransport("https://api.deepseek.com", "k", limits, slots=1)
    req = {
        "role": "desc_cleaner", "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100, "temperature": 0.0, "thinking": False,
        "reasoning_effort": None, "json_mode": False, "stream": False,
    }
    try:
        # ValueError from receive() must NOT be wrapped as invalid_transport_frame
        with pytest.raises(ValueError, match="ledger boom"):
            tr.invoke(
                req,
                lambda: False,
                lambda: "att-1",
                lambda _a, _u: (_ for _ in ()).throw(ValueError("ledger boom")),
                lambda *a, **kw: None,
            )

        # Base openai.APIError becomes TransientAPIError
        async def raise_api_error(**_kw):
            raise openai.APIError("unexpected sdk error", request=httpx.Request("POST", "https://api.deepseek.com"), body=None)

        monkeypatch.setattr(tr.client.chat.completions, "create", raise_api_error)
        with pytest.raises(TransientAPIError, match="APIError"):
            tr.invoke(req, lambda: False, lambda: "att-2", lambda *a: None, lambda *a, **kw: None)
    finally:
        tr.close()


# --------------------------------------------------------------------------- P1: reasoning_effort downgrade & truncation expansion


def test_text_downgrades_max_to_high_to_low_to_off_and_handles_whitespace(tmp_path):
    asks = []
    replies = [
        TransportResponse("  \n", "length", 10, 32768, 32765),   # off-by-3 tokens, still reasoning exhausted!
        TransportResponse("", "length", 10, 32768, None),        # missing reasoning_tokens, still exhausted!
        TransportResponse("", "length", 10, 32768, 32768),       # low -> off!
        TransportResponse("final answer", "stop", 10, 20, 0),
    ]

    def send(req):
        asks.append(req)
        return replies.pop(0)

    cfg = load_defaults()["llm"]
    cfg["operation_extra_sends"] = 4
    cfg["role_policy"]["draft_writer.data_prep"]["reasoning_effort"] = "max"
    gw = LLMGateway(cfg, tmp_path / "calls.jsonl", send)
    res = gw.text("draft_writer.data_prep", "s", "u", "op1")
    assert res.content == "final answer"
    assert [(a["thinking"], a["reasoning_effort"]) for a in asks] == [
        (True, "max"),
        (True, "high"),
        (True, "low"),
        (False, None),
    ]

    # Also verify low -> off works within default operation_extra_sends=2
    asks_low = []
    replies_low = [
        TransportResponse("", "length", 10, 32768, 32765),
        TransportResponse("recovered", "stop", 10, 20, 0),
    ]
    gw_low = LLMGateway(load_defaults()["llm"], tmp_path / "calls_low.jsonl",
                        lambda r: (asks_low.append(r), replies_low.pop(0))[1])
    assert gw_low.text("draft_writer.data_prep", "s", "u", "op2").content == "recovered"
    assert [(a["thinking"], a["reasoning_effort"]) for a in asks_low] == [(True, "low"), (False, None)]


def test_structured_downgrades_low_to_off_and_expands_on_json_truncation(tmp_path):
    # Case 1: reviewer with initial_effort="low" exhausts reasoning -> downgrades to "off"
    asks1 = []
    replies1 = [
        TransportResponse("   ", "length", 10, 16384, 16380),
        TransportResponse('{"n": 7}', "stop", 10, 5, 0),
    ]
    gw1 = LLMGateway(load_defaults()["llm"], tmp_path / "c1.jsonl", lambda r: (asks1.append(r), replies1.pop(0))[1])
    out1 = gw1.structured("reviewer", "s", "u", {"n": Field("number", "n")}, "op1", initial_effort="low", expanded=True)
    assert out1 == {"n": 7}
    assert [(a["thinking"], a["reasoning_effort"]) for a in asks1] == [(True, "low"), (False, None)]
    assert "Prior response (preserve its findings):" not in asks1[1]["messages"][-1]["content"]

    # Case 2: non-sensitive reviewer (thinking=disabled, 8192 -> 16384) gets truncated JSON -> expands!
    asks2 = []
    replies2 = [
        TransportResponse('{"n": ', "length", 10, 8192, 0),
        TransportResponse('{"n": 9}', "stop", 10, 10, 0),
    ]
    gw2 = LLMGateway(load_defaults()["llm"], tmp_path / "c2.jsonl", lambda r: (asks2.append(r), replies2.pop(0))[1])
    out2 = gw2.structured("reviewer", "s", "u", {"n": Field("number", "n")}, "op2")
    assert out2 == {"n": 9}
    assert [a["max_tokens"] for a in asks2] == [8192, 16384]


def test_full_script_expands_on_empty_length_when_recovery_limit_available(tmp_path):
    asks = []
    replies = [
        TransportResponse("", "length", 10, 65536, 0),
        TransportResponse("Plan overview.\n```python\nx = 1\n```", "stop", 10, 50, 0),
    ]
    cfg = load_defaults()["llm"]
    cfg["lower_effort_on_truncation"] = False
    gw = LLMGateway(cfg, tmp_path / "calls.jsonl", lambda r: (asks.append(r), replies.pop(0))[1])
    res = CodeGenerator(gw, 3, 2).full_script("script_coder", "sys", "usr", "op1")
    assert res.extracted is not None and "x = 1" in res.extracted.code
    assert [a["max_tokens"] for a in asks] == [65536, 131072]


# --------------------------------------------------------------------------- P2: Prompt prefix cache, config, schema, legacy OpenAITransport, startup


def test_prompt_prefix_cache_ordering(tmp_path):
    # 1. full_script retry appends note at tail, preserving user prefix
    prompts = []
    replies = [
        TransportResponse("no code here", "stop", 10, 5),
        TransportResponse("Overview.\n```python\nx = 1\n```", "stop", 10, 10),
    ]
    gw = LLMGateway(load_defaults()["llm"], tmp_path / "calls.jsonl", lambda r: (prompts.append(r["messages"][-1]["content"]), replies.pop(0))[1])
    res = CodeGenerator(gw, 2, 2).full_script("script_coder", "sys", "BASE_PROMPT_BODY", "op1")
    assert res.extracted is not None
    assert prompts[1].startswith("BASE_PROMPT_BODY")
    assert "# Note" in prompts[1]

    # 2. patch_prompt & repair_patch_prompt place retry_note at tail
    op = builders.refine_op("task", "preview", "sum", "x = 1\n", "out", "rules", False, None, 0, 0)
    _, p0 = builders.patch_prompt(op, "plan", "preview", "x = 1\n", "out", "")
    _, p1 = builders.patch_prompt(op, "plan", "preview", "x = 1\n", "out", "# Retry note\nfix anchor")
    prefix_before_output = p0.split("# Output format")[0]
    assert p1.startswith(prefix_before_output)

    rop = builders.repair_op("task", "preview", "x = 1\n", "err", "rules", "")
    _, r0 = builders.repair_patch_prompt(rop, "preview", "x = 1\n", "err", "")
    _, r1 = builders.repair_patch_prompt(rop, "preview", "x = 1\n", "err", "# Retry note\nfix anchor")
    assert r1.startswith(r0.split("# Output format")[0])

    # 3. reviewer places static task+rules prefix before dynamic run_limit_secs and code
    _, rev1 = builders.reviewer("SAME_TASK", "code_v1()", "off", 1000)
    _, rev2 = builders.reviewer("SAME_TASK", "code_v2()", "off", 400)
    common_prefix = rev1.split("# Environment facts")[0]
    assert len(common_prefix) > 500
    assert rev2.startswith(common_prefix)

    # 4. environment_block is deterministic (no shuffling)
    import random
    e1 = blocks.environment_block("catalog", random.Random(1))
    e2 = blocks.environment_block("catalog", random.Random(2))
    assert e1 == e2


def test_config_judgement_temp_role_overrides_and_deepseek_v1(tmp_path):
    defaults = load_defaults()
    assert defaults["llm"]["judgement"]["temperature"] == 0.0
    assert defaults["llm"]["max_context_tokens"] == 1_000_000
    assert defaults["llm"]["operation_rate_limit_retries"] == 8

    # Role policy can override temperature, recovery_max_output_tokens, and model
    user_cfg = tmp_path / "u.yaml"
    user_cfg.write_text(
        "llm:\n"
        "  role_policy:\n"
        "    result_reader: {temperature: 0.2, recovery_max_output_tokens: 8192}\n"
        "    planner.structured: {recovery_max_output_tokens: 32768}\n",
        encoding="utf-8",
    )
    cfg = build_config(user_cfg, environ={})
    assert cfg["llm"]["role_policy"]["result_reader"]["temperature"] == 0.2
    assert cfg["llm"]["role_policy"]["planner.structured"]["recovery_max_output_tokens"] == 32768

    # single_model / bubblewrap accepts https://api.deepseek.com/v1
    bw_cfg = tmp_path / "bw.yaml"
    bw_cfg.write_text(
        "verification:\n  runner: bubblewrap\nllm:\n  base_url: 'https://api.deepseek.com/v1'\n",
        encoding="utf-8",
    )
    built = build_config(bw_cfg, environ={})
    assert built["llm"]["base_url"] == "https://api.deepseek.com/v1"

    # Rejects non-deepseek-flash model override inside role_policy under bubblewrap
    bad_bw = tmp_path / "bad_bw.yaml"
    bad_bw.write_text(
        "verification:\n  runner: bubblewrap\nllm:\n  role_policy:\n    reviewer: {model: 'gpt-4o'}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="deepseek-flash"):
        build_config(bad_bw, environ={})


def test_schema_rejects_nonfinite_non_nullable_number():
    schema = {
        "req_num": Field("number", "required finite"),
        "opt_num": Field("number", "optional", nullable=True),
    }
    _, errors = validate({"req_num": "NaN", "opt_num": "Infinity"}, schema)
    assert any("req_num" in e and "finite" in e for e in errors)
    obj, errors2 = validate({"req_num": "1.5", "opt_num": "Infinity"}, schema)
    assert not errors2 and obj == {"req_num": 1.5, "opt_num": None}


def test_openai_transport_provider_gating_and_sse_error_frame(tmp_path, monkeypatch):
    seen_bodies = []

    def handler(req):
        body = json.loads(req.content)
        seen_bodies.append(body)
        if body["model"] == "gpt-4o":
            return httpx.Response(200, json={
                "id": "g1", "model": "gpt-4o", "object": "chat.completion", "created": 1,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            })
        # DeepSeek mid-stream error frame AFTER partial text must raise TransientAPIError
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=(
                'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
                'data: {"error":{"message":"upstream crash"}}\n\n'
            ),
        )

    _mock_sync_client(monkeypatch, handler)
    tr_openai = OpenAITransport("https://api.openai.com/v1", "k", 5)
    try:
        resp = tr_openai({
            "model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32, "temperature": 0.0, "thinking": False, "reasoning_effort": None,
            "json_mode": False, "stream": False,
        })
        assert resp.content == "hi"
        assert "thinking" not in seen_bodies[0]
    finally:
        tr_openai.close()

    tr_ds = OpenAITransport("https://api.deepseek.com", "k", 5)
    try:
        with pytest.raises(TransientAPIError, match="upstream crash"):
            tr_ds({
                "model": "deepseek-flash", "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 32, "temperature": 0.0, "thinking": False, "reasoning_effort": None,
                "json_mode": False, "stream": True,
            })
        assert seen_bodies[1]["thinking"] == {"type": "disabled"}
    finally:
        tr_ds.close()


def test_startup_helpers_fail_fast_on_auth_error_and_avoid_retry_amplification(tmp_path):
    calls = []

    def bad_auth(_req):
        calls.append(1)
        err = TransientAPIError("unused")
        raise err

    # 401 FatalAPIError -> PermanentCall sets endpoint_failure -> must propagate!
    from superagent.llm.gateway import FatalAPIError

    def auth_401(_req):
        calls.append(1)
        err = FatalAPIError("HTTP 401: invalid api key")
        err.status = 401
        raise err

    cfg = load_defaults()["llm"]
    gw = LLMGateway(cfg, tmp_path / "c1.jsonl", auth_401)
    with pytest.raises(PermanentCall, match="401"):
        _clean_description(gw, "raw description")
    assert len(calls) == 1

    calls.clear()
    gw2 = LLMGateway(cfg, tmp_path / "c2.jsonl", auth_401)
    with pytest.raises(PermanentCall, match="401"):
        _judge_direction(gw2, "accuracy metric")
    assert len(calls) == 1

    # Invalid JSON in _judge_direction should only make 3 total attempts (not 3 * json_max_attempts)
    bad_json_calls = []
    gw3 = LLMGateway(cfg, tmp_path / "c3.jsonl", lambda _r: (bad_json_calls.append(1), TransportResponse("not json", "stop", 5, 5))[1])
    assert _judge_direction(gw3, "accuracy metric") is True
    assert len(bad_json_calls) == 3


def test_context_length_guard_and_thread_safe_state_totals(tmp_path):
    cfg = load_defaults()["llm"]
    cfg["max_context_tokens"] = 2048
    gw = LLMGateway(cfg, tmp_path / "calls.jsonl", lambda _r: TransportResponse("ok", "stop", 10, 5))
    with pytest.raises(PermanentCall, match="required_context_too_large"):
        gw.text("desc_cleaner", "s", "x" * 3000)

    # build_state reads gateway totals under lock cleanly
    from superagent.tree import Direction, SearchTree
    ctx = SimpleNamespace(
        tree=SearchTree(Direction(True)),
        state_lock=threading.RLock(),
        step_timings=[],
        gateway=gw,
        experience=None,
        elapsed=lambda: 1.0,
        time_limit=100.0,
        time_left=lambda: 99.0,
        reserve=10.0,
        progress=lambda: 0.01,
        cfg=load_defaults(),
        active_tasks=0,
        direction=Direction(True),
        gpu_info=None,
    )
    st = build_state(ctx)
    assert st.cost == 0.0


def test_operations_staggers_deferred_resumes(tmp_path):
    import random
    from concurrent.futures import Future

    ledger = Ledger(tmp_path / "operations.sqlite", run_id="r-ops")
    ledger.register_operation("op-a", "sa", ("planner.structured",))
    ledger.register_operation("op-b", "sb", ("planner.structured",))
    ctx = SimpleNamespace(
        gateway=SimpleNamespace(ledger=ledger, endpoint_failure=None, cooldown_until=0.0),
        flags={},
        tree=SimpleNamespace(search_start=0.0),
        stop_event=threading.Event(),
        rng=random.Random(42),
    )
    ops = Operations(SimpleNamespace(ctx=ctx))
    p1 = Proposal("p1", "root", "draft", "h1", "generated", False, 0)
    p2 = Proposal("p2", "root", "draft", "h2", "generated", False, 0)
    d1 = Dispatch("op-a", p1, "sa", "dec1")
    d2 = Dispatch("op-b", p2, "sb", "dec2")
    base = time.monotonic() + 5.0
    for d in (d1, d2):
        fut = Future()
        fut.set_exception(DeferredCall("http_429", base))
        ops._complete(d, fut)
    assert d2.resume_at > d1.resume_at


def test_legacy_ledger_enables_wal_and_stalled_stream_cancels_promptly(tmp_path, monkeypatch):
    import superagent.llm.gateway as gw_mod

    # 1. Legacy ledger enables WAL mode
    leg = Ledger(tmp_path / "legacy.sqlite")
    with leg.db() as db:
        mode = db.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"

    # 2. Stalled legacy stream unblocks promptly when should_abort fires
    stream_started = threading.Event()
    release_server = threading.Event()

    class StallHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')
                self.wfile.flush()
                stream_started.set()
                release_server.wait(5)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_):
            pass

    monkeypatch.setattr(gw_mod, "ABORT_POLL_SECONDS", 0.02)
    with ThreadingHTTPServer(("127.0.0.1", 0), StallHandler) as server:
        t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        t.start()
        cfg = load_defaults()["llm"]
        cfg["api_max_retries"] = 0
        tr = OpenAITransport(f"http://127.0.0.1:{server.server_port}", "k", 30.0)
        gw = LLMGateway(cfg, tmp_path / "calls.jsonl", tr, should_abort=lambda _: stream_started.is_set())
        try:
            t0 = time.monotonic()
            with pytest.raises(LLMUnavailableError, match="deadline reached"):
                gw.text("patch_coder", "s", "u", "op1")
            assert time.monotonic() - t0 < 1.5
        finally:
            release_server.set()
            tr.close()
            server.shutdown()
            t.join(timeout=2)


def test_operations_complete_contains_runtime_and_sqlite_errors(tmp_path):
    from concurrent.futures import Future
    from superagent.tree import Direction, SearchTree

    ledger = Ledger(tmp_path / "operations.sqlite", run_id="r-err")
    ledger.register_operation("op-r", "sr", ("planner.structured",))
    tree = SearchTree(Direction(True))
    ctx = SimpleNamespace(
        gateway=SimpleNamespace(ledger=ledger, endpoint_failure=None, cooldown_until=0.0),
        flags={"active_parent_leases": set(), "artifact_leases": set()},
        tree=tree,
        task_spec=SimpleNamespace(signature="sp1", metric="accuracy"),
        cfg=load_defaults(),
        direction=Direction(True),
        executor=SimpleNamespace(run_dir=lambda _nid: tmp_path / "runs" / _nid),
        experience=None,
        paths=SimpleNamespace(decisions=tmp_path / "decisions.jsonl"),
        stop_event=threading.Event(),
    )
    ops = Operations(SimpleNamespace(ctx=ctx))
    p = Proposal("p1", tree.root.id, "draft", "h1", "generated", False, 0)
    d = Dispatch("op-r", p, "sr", "dec1", started=time.monotonic())
    fut = Future()
    fut.set_exception(sqlite3.OperationalError("database is locked"))
    ops._complete(d, fut)
    assert len(ops.facts) == 1
    assert ops.facts[0]["status"] == "failed"
    assert "database is locked" in ops.facts[0]["reason"]

