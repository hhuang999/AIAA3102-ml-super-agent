"""Regressions at the model-response and generated-code boundaries."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import openai
import pytest

from helpers import make_ctx, mk
from superagent.codegen import CodeGenerator, extract_code
from superagent.config import load_defaults
from superagent.llm import Field
from superagent.errors import LLMUnavailableError
from superagent.llm.gateway import LLMGateway, OpenAITransport, TransportResponse, TransientAPIError
from superagent.llm.ledger import DeferredCall
from superagent.patching import apply_patch
from superagent.planner import structured_plan
from superagent.review import review
from superagent.utils import read_jsonl


def block(old, new):
    return f"<<<<<<< SEARCH\n{old}\n=======\n{new}\n>>>>>>> REPLACE\n"


@pytest.mark.parametrize("stream", [False, True])
def test_invalid_usage_detail_keeps_body_and_known_tokens(tmp_path, monkeypatch, stream):
    usage = {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14,
             "completion_tokens_details": "invalid"}
    if stream:
        frames = [{"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]},
                  {"choices": [], "usage": usage}]
        wire = httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            text="".join(f"data: {json.dumps(frame)}\n\n" for frame in frames) + "data: [DONE]\n\n",
        )
    else:
        wire = httpx.Response(200, json={
            "id": "fixture", "model": "deepseek-flash", "object": "chat.completion", "created": 1,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": usage,
        })
    client = openai.OpenAI
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: client(
        http_client=httpx.Client(transport=httpx.MockTransport(lambda req: wire)), **kw,
    ))
    cfg = load_defaults()["llm"]
    cfg["stream"] = stream
    g = LLMGateway(cfg, tmp_path / "calls.jsonl", OpenAITransport("https://test.invalid", "fixture", 2))
    assert g.text("patch_coder", "system", "user", "operation").content == "ok"
    row, = read_jsonl(g.calls_log_path)
    assert (row["prompt_tokens"], row["completion_tokens"], row["reasoning_tokens"]) == (12, 2, None)
    assert g.ledger.summary()["unknown_or_inflight"] == 0


def test_invalid_plan_values_are_rejected_before_patch(tmp_path):
    requests = []
    replies = iter([None, "Use a reproducible split."])

    def send(req):
        requests.append(req)
        body = {"components": ["data_prep"], "changes": {"data_prep": next(replies)}}
        return TransportResponse(json.dumps(body), "stop", 10, 11)

    g = LLMGateway(load_defaults()["llm"], tmp_path / "calls.jsonl", send)
    assert structured_plan(g, "system", "user", "op")["changes"] == {"data_prep": "Use a reproducible split."}
    assert len(requests) == 2
    assert "nonempty" in requests[1]["messages"][-1]["content"]


def test_code_extraction_rejects_script_with_a_dropped_python_block():
    reply = "A new script.\n```python\na = 1\n```\n```python\nif:\n```"
    assert extract_code(reply) is None


@pytest.mark.parametrize(("code", "search"), [
    ("max_epochs = 10\n", "epochs = 1"),
    ("x = 10\nx = 1\n", "x = 1\nmissing = 0"),
    ("def first():\n    return 1\ndef second():\n    return 1\n", "    return 1"),
])
def test_substring_and_ambiguous_anchors_do_not_apply(code, search):
    assert apply_patch(code, block(search, "return 2")) == (code, 0)


def test_wrong_substring_does_not_take_priority_over_exact_line():
    original = "max_epochs = 10\nepochs = 1\n"
    assert apply_patch(original, block("epochs = 1", "epochs = 2")) == (
        "max_epochs = 10\nepochs = 2\n", 1,
    )


def test_complete_but_invalid_python_patch_can_be_corrected(tmp_path):
    replies = iter([block("x = 1", "if:"), block("if:", "x = 2")])
    requests = []

    def send(req):
        requests.append(req)
        return TransportResponse(next(replies), "stop", 20, 12)

    g = LLMGateway(load_defaults()["llm"], tmp_path / "calls.jsonl", send)
    result = CodeGenerator(g, 2, 3).patch("patch_coder", "x = 1\n", lambda c, n: ("patch", c+n), "op")
    assert result.code == "x = 2\n" and not result.partial
    assert "SyntaxError" in requests[1]["messages"][-1]["content"]


def test_review_requests_applicable_python_revision_before_runner(tmp_path):
    ctx = make_ctx(tmp_path, sample=False)
    node = mk(ctx.tree, "draft", ctx.tree.root, failed=None, code="target = 1\n")
    replies = iter([block("target = 1", "if:"), block("target = 1", "target = 2")])
    requests = []

    def send(req):
        requests.append(req)
        body = {"verdict": "revise", "findings": "Fix x.", "patch": next(replies)}
        return TransportResponse(json.dumps(body), "stop", 30, 10)

    ctx.gateway = LLMGateway(load_defaults()["llm"], tmp_path / "calls.jsonl", send)
    outcome = review(ctx, node)
    assert outcome.changed and node.code == "target = 2\n"
    assert "SyntaxError" in requests[1]["messages"][-1]["content"]
    assert [req["max_tokens"] for req in requests] == [16384, 16384]
    assert all(req["thinking"] and req["reasoning_effort"] == "low" for req in requests)


def test_sensitive_effort_override_can_recover_at_same_output_limit(tmp_path):
    requests = []
    replies = iter([TransportResponse("", "length", 10, 16384, 16384),
                    TransportResponse('{"n": 2}', "stop", 10, 3)])

    def send(req):
        requests.append(req)
        return next(replies)

    g = LLMGateway(load_defaults()["llm"], tmp_path / "calls.jsonl", send)
    assert g.structured("reviewer", "s", "u", {"n": Field("number", "count")}, "op",
                        initial_effort="high", expanded=True) == {"n": 2}
    assert [(r["reasoning_effort"], r["max_tokens"]) for r in requests] == [("high", 16384), ("low", 16384)]


def test_review_resume_preserves_feedback_when_run_limit_changes(tmp_path):
    ctx = make_ctx(tmp_path, sample=False)
    node = mk(ctx.tree, "draft", ctx.tree.root, failed=None, code="target = 1\n")
    requests = []

    def send(req):
        requests.append(req)
        if len(requests) == 1:
            return TransportResponse(" ", "stop", 12, 3)
        if len(requests) == 2:
            exc = TransientAPIError("fixture timeout")
            exc.status = 524
            raise exc
        return TransportResponse('{"verdict":"pass","findings":"No issue."}', "stop", 12, 10)

    log = tmp_path / "calls.jsonl"
    ctx.current_run_limit = lambda: 30
    ctx.gateway = LLMGateway(ctx.cfg["llm"], log, send)
    with pytest.raises(DeferredCall):
        review(ctx, node)

    ctx.current_run_limit = lambda: 20
    ctx.gateway = LLMGateway(ctx.cfg["llm"], log, send)
    assert review(ctx, node).verdict == "pass"
    prompt = requests[2]["messages"][-1]["content"]
    assert "20 seconds" in prompt and "blank_stop" in prompt
    assert len({row["call_id"] for row in read_jsonl(log)}) == 1


@pytest.mark.parametrize("structured", [False, True])
def test_524_during_content_recovery_keeps_changed_request(tmp_path, structured):
    requests = []

    def send(req):
        requests.append(req)
        if len(requests) == 1:
            return TransportResponse(" " if structured else "", "stop" if structured else "length",
                                     10, 32768, 32768)
        if len(requests) == 2:
            exc = TransientAPIError("fixture timeout")
            exc.status = 524
            raise exc
        return TransportResponse('{"n": 2}' if structured else "ok", "stop", 10, 3)

    def invoke(g):
        if structured:
            return g.structured("reviewer", "system", "user", {"n": Field("number", "count")}, "op")
        return g.text("patch_coder", "system", "user", "op").content

    cfg = load_defaults()["llm"]
    log = tmp_path / "calls.jsonl"
    with pytest.raises(DeferredCall):
        invoke(LLMGateway(cfg, log, send))
    # Fresh gateway on the same operation after cooldown.
    assert invoke(LLMGateway(cfg, log, send)) == ({"n": 2} if structured else "ok")
    assert len(requests) == 3
    keys = ("messages", "thinking", "reasoning_effort", "max_tokens")
    assert [requests[2][k] for k in keys] == [requests[1][k] for k in keys]
    assert [requests[2][k] for k in keys] != [requests[0][k] for k in keys]
    rows = read_jsonl(log)
    assert len({row["call_id"] for row in rows}) == 1
    assert [row["attempt"] for row in rows] == [1, 2, 3]


@pytest.mark.parametrize("case", ["quiet", "keepalive", "cancel", "broken_tail"])
def test_live_socket_stream_boundaries(tmp_path, monkeypatch, case):
    import superagent.llm.gateway as gateway

    prefix_sent = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    connections = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            connections.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                if case == "keepalive":
                    for _ in range(13):
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        time.sleep(0.04)
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n')
                self.wfile.flush()
                prefix_sent.set()
                if case == "cancel":
                    release.wait(3)
                elif case == "quiet":
                    time.sleep(0.65)
                if case != "broken_tail":
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                finished.set()

        def log_message(self, *_):
            pass

    monkeypatch.setattr(gateway, "ABORT_POLL_SECONDS", 0.02)
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        t.start()
        try:
            cfg = load_defaults()["llm"]
            cfg["api_max_retries"] = 0
            g = LLMGateway(cfg, tmp_path / "calls.jsonl",
                           OpenAITransport(f"http://127.0.0.1:{server.server_port}", "test",
                                           2 if case == "cancel" else 0.4),
                           should_abort=lambda _: case == "cancel" and prefix_sent.is_set())
            if case == "cancel":
                with pytest.raises(LLMUnavailableError, match="deadline reached"):
                    g.text("patch_coder", "s", "u", "op")
                assert not finished.is_set()
                assert g.ledger.summary()["unknown_or_inflight"] == 1
            else:
                result = g.text("patch_coder", "s", "u", "op")
                assert (result.content, result.truncated) == ("ok", case != "keepalive")
            assert len(connections) == 1
        finally:
            release.set()
            server.shutdown()
            t.join(timeout=2)
