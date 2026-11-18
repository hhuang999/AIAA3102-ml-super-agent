import json
import httpx
import openai
import pytest

from superagent.codegen import CodeGenerator
from superagent.config import load_defaults
from superagent.llm.gateway import (
    LLMGateway,
    OpenAITransport,
    TransportResponse,
    TransientAPIError,
)
from superagent.llm.ledger import DeferredCall, Ledger


def test_524_then_resume_only_unapplied_blocks(tmp_path):
    replies = iter(
        [
            TransportResponse(
                "<<<<<<< SEARCH\nx=1\n=======\nx=2\n>>>>>>> REPLACE", "length", 10, 12
            ),
            None,
            TransportResponse(
                "<<<<<<< SEARCH\ny=1\n=======\ny=2\n>>>>>>> REPLACE", "stop", 12, 10
            ),
        ]
    )
    requests = []

    def send(req):
        requests.append(req)
        x = next(replies)
        if x is None:
            err = TransientAPIError("524")
            err.status = 524
            err.retry_after = 120
            raise err
        return x

    cfg = load_defaults()["llm"]

    def gateway():
        return LLMGateway(cfg, tmp_path / "calls.jsonl", send)

    g = gateway()
    op = g.ledger.claim("parent-signature")
    g.ledger.put(op, "validated_plan", "parent", {"change": "x and y"})

    def run(g):
        return CodeGenerator(g, 3, 3).patch(
            "patch_coder", "x=1\ny=1", lambda c, n: ("patch", c + "\n" + n), op
        )

    with pytest.raises(DeferredCall):
        run(g)
    g.ledger.state(op, "suspended")
    # Re-open after route recovery, using a fresh caller and same parent signature.
    g = gateway()
    assert g.ledger.claim("parent-signature") == op
    assert g.ledger.get(op, "validated_plan", "parent") == {"change": "x and y"}
    result = run(g)
    assert result.code == "x=2\ny=2" and not result.partial
    assert len(requests) == 3 and requests[-1]["max_tokens"] == requests[1]["max_tokens"]
    assert requests[-1]["messages"] == requests[1]["messages"]
    assert g.ledger.summary()["unknown_or_inflight"] == 1
    assert g.ledger.summary()["risk_reserve"] > 0


def test_stream_usage_only_tail_and_interrupted_json(monkeypatch, tmp_path):
    frames = [
        {"choices": [{"delta": {"content": "yes"}, "finish_reason": "stop"}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 3,
                "prompt_cache_hit_tokens": 40,
            },
        },
    ]

    def send(req):
        assert json.loads(req.content)["stream_options"]["include_usage"]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join("data: " + json.dumps(x) + "\n\n" for x in frames)
            + "data: [DONE]\n\n",
        )

    real = openai.OpenAI
    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kw: real(
            http_client=httpx.Client(transport=httpx.MockTransport(send)), **kw
        ),
    )
    g = LLMGateway(
        load_defaults()["llm"],
        tmp_path / "calls.jsonl",
        OpenAITransport("https://test.invalid", "test", 3),
    )
    assert g.text("patch_coder", "s", "u", "op").content == "yes"
    row = json.loads((tmp_path / "calls.jsonl").read_text())
    assert row["cost"] == pytest.approx((10 * 2 + 40 * 0.04 + 3 * 8) / 1e6)
    assert row["cached_tokens"] == 40 and row["finish_reason"] == "stop"


def test_unknown_reserve_prevents_overspend(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite")
    ledger.limit = 1
    item = ledger.begin("o", "plan", "a", "s", 0.7)
    ledger.finish(item, None, None, "abandoned")
    with pytest.raises(DeferredCall, match="cost limit"):
        ledger.begin("p", "plan", "b", "s", 0.4)
    ledger.finish(item, 0.1, {}, "reconciled")
    assert ledger.begin("p", "plan", "b", "s", 0.4)
