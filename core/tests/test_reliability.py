import os

import pytest

from helpers import make_ctx, mk
from superagent.assessment import ReaderResult, apply_assessment, jump_triggered
from superagent.catalog import render_entry
from superagent.config import load_defaults
from superagent.executor import ExecResult, Executor
from superagent.llm.gateway import LLMGateway, TransportResponse
from superagent.utils import read_jsonl


def _exec(tmp_path):
    prep = tmp_path / "prepared"
    prep.mkdir(exist_ok=True)
    return Executor(tmp_path / "runs", prep, 1, len(os.sched_getaffinity(0)), 0, {})


SMOKE_FAILS = '''import os
if os.environ.get("SUPERAGENT_SMOKE") == "1":
    raise ValueError("broken in smoke mode")
open("./out/full.txt", "w").write("full")
'''

SMOKE_PASSES = '''import os
name = "smoke.txt" if os.environ.get("SUPERAGENT_SMOKE") == "1" else "full.txt"
open("./out/" + name, "w").write("x")
print("VALIDATION_SCORE: 0.5")
'''

SMOKE_SLOW = '''import os, time
if os.environ.get("SUPERAGENT_SMOKE") == "1":
    time.sleep(30)
open("./out/full.txt", "w").write("full")
'''


def test_smoke_failure_skips_the_full_run(tmp_path):
    ex = _exec(tmp_path)
    res = ex.run("n1", SMOKE_FAILS, timeout=30, smoke_timeout=20)
    assert res.smoke == "failed" and res.error_type == "ValueError"
    assert "SMOKE RUN FAILED" in res.output
    assert not (ex.run_dir("n1") / "out" / "full.txt").exists()
    assert (ex.run_dir("n1") / "smoke_stdout.log").exists()


def test_smoke_pass_then_full_run_with_clean_outputs(tmp_path):
    ex = _exec(tmp_path)
    res = ex.run("n2", SMOKE_PASSES, timeout=30, smoke_timeout=20)
    out = ex.run_dir("n2") / "out"
    assert res.smoke == "passed" and res.error_type is None
    assert res.output.startswith("[superagent] Smoke run (SUPERAGENT_SMOKE=1) passed")
    assert (out / "full.txt").exists() and not (out / "smoke.txt").exists()


def test_smoke_timeout_does_not_stop_the_full_run(tmp_path):
    ex = _exec(tmp_path)
    res = ex.run("n3", SMOKE_SLOW, timeout=30, smoke_timeout=1)
    assert res.smoke == "timed_out" and res.error_type is None
    assert (ex.run_dir("n3") / "out" / "full.txt").exists()


def test_no_smoke_run_by_default(tmp_path):
    ex = _exec(tmp_path)
    res = ex.run("n4", SMOKE_FAILS, timeout=30)
    assert res.smoke is None and res.error_type is None


class _Transport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, request):
        self.requests.append(dict(request))
        return self.replies.pop(0)


def test_empty_truncated_reply_is_retried_with_lower_effort(tmp_path):
    cfg = load_defaults()["llm"]
    tr = _Transport([TransportResponse("", "length", 10, 32768, 32768), TransportResponse("code", "stop", 10, 100, 50)])
    gw = LLMGateway(cfg, tmp_path / "calls.jsonl", tr)
    res = gw.text("patch_coder", "s", "u")
    assert res.content == "code"
    assert [r["reasoning_effort"] for r in tr.requests] == ["high", "low"]
    rows = read_jsonl(tmp_path / "calls.jsonl")
    assert len(rows) == 2 and rows[0]["ok"] is False and "retrying with reasoning effort low" in rows[0]["error"]
    assert rows[1]["reasoning_effort"] == "low" and rows[1]["ok"]


def test_no_retry_for_judgement_roles_or_when_switched_off(tmp_path):
    cfg = load_defaults()["llm"]
    tr = _Transport([TransportResponse("", "length", 10, 8192, 0)])
    assert LLMGateway(cfg, None, tr).text("desc_cleaner", "s", "u").content == ""     # thinking disabled
    cfg = {**cfg, "lower_effort_on_truncation": False}
    tr = _Transport([TransportResponse("", "length", 10, 32768, 32768)])
    assert LLMGateway(cfg, None, tr).text("patch_coder", "s", "u").content == ""
    assert len(tr.requests) == 1


def test_jump_trigger():
    assert jump_triggered(0.65, 0.5, True, 0.2)
    assert not jump_triggered(0.55, 0.5, True, 0.2)
    assert jump_triggered(0.2, 0.28, False, 0.2)
    assert not jump_triggered(0.25, 0.28, False, 0.2)
    assert not jump_triggered(0.9, None, True, 0.2)
    assert not jump_triggered(0.9, 0.1, True, 0.0)


def test_large_jump_asks_for_a_leak_check(tmp_path):
    ctx = make_ctx(tmp_path)
    ctx.tree.best = mk(ctx.tree, "draft", ctx.tree.root, score=0.5)
    n = mk(ctx.tree, "draft", ctx.tree.root, failed=None)
    n.code = "print(1)"
    d = ctx.executor.prepare_run_dir(n.id, n.code)
    import pandas as pd

    pd.DataFrame({"id": [f"r{i}" for i in range(100)], "label": [i % 2 for i in range(100)]}).to_csv(d / "out" / "predictions.csv", index=False)
    res = ExecResult("VALIDATION_SCORE: 0.7\n", 1.0, None, None, False, 0, 100)
    asked = []
    info = apply_assessment(ctx, n, res, ReaderResult(False, "ok", 0.7, "maximize", "m", False),
                            leak_answer=lambda: asked.append(1) or {"leak_found": False, "evidence": "ok", "certainty": "low"})
    assert asked == [1] and info["leak_check_reason"] == "score_jump" and n.ok


def test_catalog_renders_pitfalls():
    text = render_entry({"title": "M", "usage": "finetune", "when_to_use": "w", "pitfalls": "use bf16"})
    assert "- Known pitfalls: use bf16" in text


@pytest.mark.parametrize("gpus, expect", [(1, True), (0, False)])
def test_gpu_habits_only_with_a_gpu(gpus, expect):
    from superagent.prompts.blocks import Resources, implementation_rules

    text = implementation_rules(Resources(100, 1, 100, 1, gpus, "catalog"), 42, 0.2)
    assert ("bf16 autocast" in text) is expect
    assert "Smoke mode" not in text
    assert "Smoke mode" in implementation_rules(Resources(100, 1, 100, 1, gpus, "catalog", smoke_seconds=180), 42, 0.2)


def test_effective_branches():
    from superagent.report import effective_branches

    assert effective_branches([1, 1, 1]) == [1.0, 1.0, 1.0]
    assert effective_branches([1, 2])[-1] == pytest.approx(2.0)
    assert effective_branches([1, 2, 3, 4], window=2)[-1] == pytest.approx(2.0)


def test_summary_has_diagnostics_and_markdown(tmp_path):
    import json

    from superagent.control import RuleController, StepTiming, build_state
    from superagent.report import build_summary, render_markdown

    ctx = make_ctx(tmp_path)
    ctx.control = RuleController(ctx)
    ctx.control.start(build_state(ctx))
    t = ctx.tree
    d = mk(t, "draft", t.root, score=0.5)
    mk(t, "refine", d, score=0.4)
    mk(t, "refine", d, failed=True)
    t.best = d
    ctx.record_step(StepTiming(1, d.id, "draft", True, 300.0, 200.0))
    ctx.paths.decisions.write_text("\n".join(json.dumps(r) for r in [
        {"selection_mode": "initial", "reason": "root_has_capacity", "progress": 0.01, "details": {}},
        {"selection_mode": "chain", "reason": "branch_progressing", "progress": 0.6, "details": {}},
        {"selection_mode": "elite_pick", "reason": "branch_progressing", "progress": 0.8, "details": {"left_chain_at": "x"}},
        {"selection_mode": "trigger", "reason": "synthesis_trigger", "progress": 0.6, "details": {"left_chain_at": "y"}},
    ]) + "\n")
    ctx.flags["synthesis_blockers"] = {"no_global_plateau": 3}
    s = build_summary(ctx)
    dg = s["diagnostics"]
    assert dg["selection_modes"] == {"initial": 1, "chain": 1, "elite_pick": 1, "trigger": 1}
    assert dg["chains_left_for_elite"] == 1 and dg["chains_left_for_synthesis"] == 1 and dg["operators"]["refine"] == 2
    assert "no_global_plateau" in dg["not_triggered"]["synthesis"]
    assert "retrospect_borrow" in dg["not_triggered"] and "elite_pick" not in dg["not_triggered"]
    assert s["control"]["controller"] == "rules"
    md = render_markdown(s)
    assert "## Diagnostics" in md and "## Control" in md
