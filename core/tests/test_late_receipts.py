import json
import threading
import time
from types import SimpleNamespace

import httpx
import openai
import pytest

from superagent.config import load_defaults
from superagent.errors import LLMUnavailableError
from superagent.llm.gateway import LLMGateway, OpenAITransport, TransportResponse
from superagent.llm.ledger import DeferredCall, PermanentCall
from superagent.report import _llm_usage


def test_local_deadline_reconciles_late_usage_without_second_send(
    tmp_path, monkeypatch
):
    import superagent.llm.gateway as gw

    monkeypatch.setattr(gw, "ABORT_POLL_SECONDS", 0.01)
    started, release = threading.Event(), threading.Event()

    def send(_):
        started.set()
        assert release.wait(3)
        return TransportResponse("late", "stop", 100, 50, cached_tokens=25)

    llm_cfg = load_defaults()["llm"]
    for group in ("generation", "judgement"):
        llm_cfg[group] = {**llm_cfg[group], "input_price_per_million": 2.0, "cached_input_price_per_million": 0.04, "output_price_per_million": 8.0}
    g = LLMGateway(
        llm_cfg,
        tmp_path / "calls.jsonl",
        send,
        should_abort=lambda _: started.is_set(),
    )
    with pytest.raises(LLMUnavailableError, match="deadline"):
        g.text("patch_coder", "s", "u", "op")
    assert g.ledger.summary()["unknown_or_inflight"] == 1
    ctx = SimpleNamespace(paths=SimpleNamespace(llm_calls=tmp_path / "calls.jsonl"),
                          cfg={"llm": load_defaults()["llm"]})
    assert _llm_usage(ctx)["unknown_usage"] == 1
    release.set()
    until = time.monotonic() + 2
    file = tmp_path / "late_usage.jsonl"
    while not file.exists() and time.monotonic() < until:
        time.sleep(0.01)
    row = json.loads(file.read_text())
    orig = json.loads((tmp_path / "calls.jsonl").read_text())
    assert orig["attempt_id"] == row["attempt_id"] and row["cost"] == pytest.approx(
        0.000551
    )
    assert g.ledger.summary()["unknown_or_inflight"] == 0
    usage = _llm_usage(ctx)
    assert usage["calls"] == usage["failed_calls"] == 1
    assert usage["unknown_usage"] == 0
    assert usage["completion_tokens"] == 50
    assert usage["cost"] == pytest.approx(0.000551)
    assert g.totals["unknown_cost_attempts"] == 0
    assert g.totals["cost"] == pytest.approx(usage["cost"])


@pytest.mark.parametrize(
    "status,error", [(401, PermanentCall), (400, PermanentCall), (402, PermanentCall),
                     (429, DeferredCall), (524, DeferredCall)]
)
def test_real_sdk_failure_classification(tmp_path, monkeypatch, status, error):
    calls = []

    def send(req):
        calls.append(req.url.path)
        return httpx.Response(
            status,
            headers={"retry-after": "7"},
            json={"error": {"message": "synthetic"}},
        )

    client = openai.OpenAI
    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kw: client(
            http_client=httpx.Client(transport=httpx.MockTransport(send)), **kw
        ),
    )
    g = LLMGateway(
        load_defaults()["llm"],
        tmp_path / "calls.jsonl",
        OpenAITransport("https://route.invalid", "key", 3),
    )
    with pytest.raises(error):
        g.text("patch_coder", "s", "u", "op")
    if status in (401, 402, 429, 524):
        with pytest.raises(LLMUnavailableError):
            g.text("patch_coder", "s", "u", "another")
    assert len(calls) == 1 and g.ledger.summary()["unknown_or_inflight"] == (1 if status == 524 else 0)
    if status in (429, 524):
        assert g.cooldown_until - time.monotonic() > 6
