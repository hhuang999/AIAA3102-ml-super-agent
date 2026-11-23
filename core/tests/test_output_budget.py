from superagent.codegen import CodeGenerator, PatchState, patch_step
from superagent.config import load_defaults
from superagent.llm.gateway import LLMGateway, TransportResponse
from superagent.llm import Field
from superagent.planner import normalize_and_check


def block(old, new):
    return f"<<<<<<< SEARCH\n{old}\n=======\n{new}\n>>>>>>> REPLACE"


def gateway(tmp_path, replies):
    asks = []

    def send(req):
        asks.append(req)
        return replies.pop(0)

    return LLMGateway(load_defaults()["llm"], tmp_path / "calls.jsonl", send), asks


def test_visible_cut_without_complete_block_buys_headroom(tmp_path):
    g, asks = gateway(
        tmp_path,
        [
            TransportResponse("<<<<<<< SEARCH", "length", 20, 32768),
            TransportResponse(block("x=0", "x=1"), "stop", 30, 22),
        ],
    )
    # This case exercises visible-output recovery with thinking explicitly off.
    g.cfg["role_policy"]["patch_coder"]["thinking"] = "disabled"
    result = CodeGenerator(g, 2, 3).patch(
        "patch_coder", "x=0", lambda c, n: ("s", c + n), "op"
    )
    assert result.code == "x=1" and not result.partial
    assert [(a["thinking"], a["max_tokens"]) for a in asks] == [
        (False, 32768),
        (False, 65536),
    ]


def test_patch_default_retains_thinking_during_policy_evaluation(tmp_path):
    g, asks = gateway(tmp_path, [TransportResponse("complete", "stop", 10, 2)])
    assert g.text("patch_coder", "s", "u").content == "complete"
    assert asks[0]["thinking"] is True
    assert asks[0]["reasoning_effort"] == "low"


def test_visible_fragment_can_recover_even_with_mostly_reasoning(tmp_path):
    g, _ = gateway(tmp_path, [TransportResponse("x", "length", 10, 32768, 32765)])
    assert g.text("repair_coder", "s", "u").body_limited


def test_business_retry_does_not_repeat_empty_exhaustion(tmp_path):
    g, asks = gateway(
        tmp_path,
        [
            TransportResponse("", "length", 10, 32768, 32768),
            TransportResponse("", "length", 10, 32768, 0),
        ],
    )
    gen = CodeGenerator(g, 2, 3)
    for _ in range(2):
        assert not gen.patch(
            "repair_coder", "x=1", lambda c, n: ("s", c + n), "op"
        ).code
    assert [(a["thinking"], a["max_tokens"]) for a in asks] == [
        (True, 32768),
        (False, 32768),
    ]


def test_unrelated_block_does_not_complete_known_missing_edit():
    st = PatchState("x=0\ny=0\nz=0", "x=0\ny=0\nz=0")
    assert not patch_step(st, block("x=0", "x=1") + "\n" + block("y=00", "y=1"))
    assert not patch_step(st, block("z=0", "z=1"))
    assert patch_step(st, block("y=0", "y=1"))


def test_protocol_failure_does_not_shrink_structured_budget(tmp_path):
    g, asks = gateway(
        tmp_path,
        [
            TransportResponse('{"n":1}', "interrupted", 10, 10),
            TransportResponse('{"n":2}', "stop", 10, 10),
        ],
    )
    assert g.structured(
        "reviewer", "s", "u", {"n": Field("number", "number")}, initial_effort="low"
    ) == {"n": 2}
    assert [r["max_tokens"] for r in asks] == [8192, 8192]
    assert all(r["thinking"] and r["reasoning_effort"] == "low" for r in asks)
    assert "interrupted" in asks[1]["messages"][-1]["content"]


def test_invalid_plan_cannot_drive_patch_even_on_last_attempt():
    for changes in ({"model_def": "a"}, {"alien": "b"}):
        assert normalize_and_check(
            {"components": ["train_eval"], "changes": changes}, True
        )[1]


def test_interrupted_named_block_is_still_due_after_unrelated_patch():
    st = PatchState("x=0\ny=0\nz=0", "x=0\ny=0\nz=0")
    assert not patch_step(
        st, block("x=0", "x=1") + "\n<<<<<<< SEARCH\ny=0\n=======\ny="
    )
    assert not patch_step(st, block("z=0", "z=1"))
    assert patch_step(st, block("y=0", "y=1"))


def test_retried_block_can_expand_context_and_rephrase_replacement():
    original = "def train():\n    x = 1\n    return x\n\ndef infer():\n    x = 1\n    return x\n"
    st = PatchState(original, original)
    ambiguous = "    x = 1\n    return x"
    assert not patch_step(st, block(ambiguous, "    return 2"))
    expanded = "def infer():\n" + ambiguous
    assert not patch_step(st, block(expanded, "def infer(mode=None):\n" + ambiguous))
    updated = "def infer(mode=None):\n" + ambiguous
    assert patch_step(st, block(updated, "def infer(mode=None):\n    result = 2\n    return result"))
    assert st.current == "def train():\n    x = 1\n    return x\n\ndef infer(mode=None):\n    result = 2\n    return result\n"


def test_null_stream_frame_closes_attempt(tmp_path, monkeypatch):
    import httpx
    import openai
    import pytest
    from superagent.errors import LLMUnavailableError
    from superagent.llm.gateway import OpenAITransport

    client = openai.OpenAI

    def send(req):
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text="data: null\n\n"
        )

    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kw: client(
            http_client=httpx.Client(transport=httpx.MockTransport(send)), **kw
        ),
    )
    cfg = load_defaults()["llm"]
    cfg["api_max_retries"] = 0
    g = LLMGateway(
        cfg,
        tmp_path / "calls.jsonl",
        OpenAITransport("https://test.invalid", "fixture", 1),
    )
    with pytest.raises(LLMUnavailableError):
        g.text("patch_coder", "s", "u", "op")
    with g.ledger.db() as db:
        assert db.execute("SELECT status,cost FROM attempts").fetchall() == [
            ("transport_error", None)
        ]


def test_role_override_can_disable_repair_without_disabling_planning(tmp_path):
    from superagent.config import build_config

    user = tmp_path / "roles.yaml"
    user.write_text("llm:\n  role_policy:\n    repair_coder: {thinking: disabled}\n")
    cfg = build_config(user, environ={})["llm"]
    g = LLMGateway(cfg, None, lambda req: None)
    assert g._group("repair_coder")["thinking"] == "disabled"
    assert g._group("planner.structured")["thinking"] == "enabled"


def test_expanded_window_survives_partial_progress(tmp_path):
    g, asks = gateway(
        tmp_path,
        [
            TransportResponse("<<<<<<< SEARCH", "length", 90, 32768),
            TransportResponse(block("x=0", "x=1"), "length", 110, 65536),
            TransportResponse(block("y=0", "y=1"), "stop", 115, 40),
        ],
    )
    result = CodeGenerator(g, 2, 3).patch(
        "patch_coder", "x=0\ny=0", lambda c, n: ("patch", c + n), "op"
    )
    assert result.code == "x=1\ny=1" and not result.partial
    assert [r["max_tokens"] for r in asks] == [32768, 65536, 65536]
    assert all(r["reasoning_effort"] == "low" for r in asks)


def test_tight_fee_cap_refuses_long_patch_but_allows_short_judgement(tmp_path):
    import pytest
    from superagent.llm.ledger import DeferredCall

    g, asks = gateway(tmp_path, [TransportResponse("success", "stop", 10, 4)])
    g.ledger.limit = 0.05
    with pytest.raises(DeferredCall, match="cost limit"):
        g.text("patch_coder", "s", "u", "op")
    assert not asks
    assert g.text("direction_judge", "s", "u", "other").content == "success"
    assert asks[0]["max_tokens"] == 512


def test_sdk_retry_keeps_wire_limits_and_task(monkeypatch, tmp_path):
    import json
    import httpx
    import openai
    from superagent.llm.gateway import OpenAITransport

    requests = []

    def handle(req):
        requests.append(json.loads(req.content))
        if len(requests) == 1:
            return httpx.Response(503, json={"error": {"message": "unavailable"}})
        return httpx.Response(
            200,
            json={
                "id": "test",
                "model": "deepseek-flash",
                "object": "chat.completion",
                "created": 1,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "total_tokens": 11,
                },
            },
        )

    real = openai.OpenAI
    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kw: real(
            http_client=httpx.Client(transport=httpx.MockTransport(handle)), **kw
        ),
    )
    cfg = load_defaults()["llm"]
    cfg.update(stream=False, api_retry_delay_seconds=0)
    g = LLMGateway(
        cfg,
        tmp_path / "calls.jsonl",
        OpenAITransport("https://fixture.invalid", "fixture", 2),
    )
    assert g.text("patch_coder", "s", "u", "op").content == "ok"
    assert len(requests) == 2 and requests[0] == requests[1]
    assert (
        requests[1]["max_tokens"] == 32768 and requests[1]["reasoning_effort"] == "low"
    )
