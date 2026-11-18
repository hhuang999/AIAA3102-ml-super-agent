import json
from dataclasses import replace

import yaml
import pytest
from helpers import make_ctx, mk
from superagent.control.adaptive import AdaptiveController
from superagent.control.state import StepTiming, build_state
from superagent.llm.gateway import TransportResponse
from superagent.run import run_task
from test_contract_pipeline import create


def _slow_final(*args, **kwargs):
    import time

    time.sleep(60)


def test_deadline_kills_slow_ensemble_and_preserves_incumbent(tmp_path, monkeypatch):
    from time import monotonic
    from superagent.run import _write_final
    from superagent.publication import version, publish

    ctx = make_ctx(tmp_path)
    ctx.control = AdaptiveController(ctx)
    ctx.cfg["run"]["time_limit_seconds"] = 2
    n = mk(ctx.tree, "draft", ctx.tree.root, score=0.9)
    ctx.tree.top_list = [n]
    dest = version(ctx.paths.root)
    (dest / "source.json").write_text('{"available":true,"kind":"single"}')
    (dest / "predictions.csv").write_text("id,label\\nx,1")
    publish(ctx.paths.root, dest)
    monkeypatch.setattr("superagent.final.build_final", _slow_final)
    ctx.process_start = monotonic()
    assert _write_final(ctx)["kind"] == "single"
    assert 0.9 < ctx.elapsed() < 3.0
    assert (ctx.paths.root / "final/predictions.csv").read_bytes() == (
        dest / "predictions.csv"
    ).read_bytes()


def test_short_deadline_scales_reserve_without_confusing_method_and_code(tmp_path):
    from superagent.control.adaptive import method_shape, code_signature

    ctx = make_ctx(tmp_path)
    ctx.cfg["run"]["time_limit_seconds"] = 120
    ctrl = AdaptiveController(ctx)
    assert ctrl.reserve() == 12
    ctrl.publish_seconds.append(8)
    assert ctrl.reserve() == 16
    assert method_shape("learning_rate = .1") == method_shape("learning_rate=.2")
    assert code_signature("learning_rate = .1") != code_signature("learning_rate=.2")


def test_tail_rejects_edit_but_allows_verified_rung(tmp_path):
    ctx = make_ctx(tmp_path)
    ctrl = AdaptiveController(ctx)
    n = mk(ctx.tree, "draft", ctx.tree.root, score=0.9)
    ctx.tree.best = n
    n.checkpoint = {
        "next_seconds": 12,
        "delivery_seconds": 8,
        "can_continue": True,
        "progress": {
            "optimizer_steps": 1,
            "model_sha256": "a" * 64,
            "optimizer_sha256": "b" * 64,
        },
    }
    st = replace(
        build_state(ctx),
        time_left_secs=360,
        reserve_secs=300,
        step_timings=tuple(
            StepTiming(i, str(i), "refine", True, 800, 600) for i in range(6)
        ),
    )
    choice, log = ctrl.choose(st)
    assert choice.action == "promote"
    assert any(x["action"] == "refine" for x in log["rejected"])
    choice2, _ = ctrl.choose(st)
    assert choice2 is None  # another worker cannot claim the same promotion


def test_observe_does_not_claim_and_budget_preserves_free_continuation(tmp_path):
    from superagent.llm.ledger import Ledger

    ctx = make_ctx(tmp_path)
    from types import SimpleNamespace

    ctx.gateway = SimpleNamespace(ledger=Ledger(tmp_path / "limits.db"))
    ctx.gateway.ledger.limit = 0.01
    ctrl = AdaptiveController(ctx)
    n = mk(ctx.tree, "draft", ctx.tree.root, score=0.5)
    n.checkpoint = {
        "next_seconds": 2,
        "delivery_seconds": 2,
        "can_continue": True,
        "progress": {
            "optimizer_steps": 1,
            "model_sha256": "a" * 64,
            "optimizer_sha256": "b" * 64,
        },
    }
    offer, evidence = ctrl.choose(build_state(ctx), reserve=False)
    assert offer.action == "promote" and offer.fee == 0 and not ctrl.leases
    assert n.pending_children == 0 and any(
        i["estimated_fee"] > 0.01 for i in evidence["rejected"]
    )


def test_legacy_checkpoint_without_progress_does_not_offer_promotion(tmp_path):
    ctx = make_ctx(tmp_path)
    ctrl = AdaptiveController(ctx)
    node = mk(ctx.tree, "draft", ctx.tree.root, score=0.5)
    node.checkpoint = {"can_continue": True, "next_seconds": 1, "delivery_seconds": 1}
    choice, _ = ctrl.choose(build_state(ctx), reserve=False)
    assert choice is None or choice.action != "promote"


@pytest.mark.parametrize("advance", [True, False])
def test_actual_search_promotes_baseline_without_new_llm_generation(
    tmp_path, monkeypatch, advance
):
    task = tmp_path / "flower"
    create(task)
    if not advance:
        script = task / "baseline.py"
        script.write_text(
            script.read_text().replace(
                'for _, x, y in examples("train"):',
                'for _, x, y in ([] if resumed else examples("train")):',
            )
        )
    config = tmp_path / "cfg.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "experience": {"enabled": False},
                "search": {"initial_drafts": 1},
                "run": {"finalize_reserve_seconds": 5},
                "execution": {"smoke_run": False},
                "resources": {"num_cpus": 1, "parallel": 1, "num_gpus": 0},
            }
        )
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    calls = []

    def transport(self, req):
        calls.append(req["role"])
        assert req["role"] == "reviewer"
        return TransportResponse(
            '{"verdict":"pass","findings":"checked","patch":null}', "stop", 12, 8
        )

    monkeypatch.setattr("superagent.run.OpenAITransport.__call__", transport)
    out = tmp_path / "out"
    result = run_task(task, out, config, time_limit_secs=180, max_steps=2)
    assert result["status"] == "success" and result["num_ok_nodes"] == (
        2 if advance else 1
    )
    assert calls == ["reviewer"]
    decisions = [
        json.loads(l) for l in (out / "logs/decisions.jsonl").read_text().splitlines()
    ]
    assert decisions[-1]["action"] == "promote"
    nodes = json.loads((out / "logs/search_tree.json").read_text())["nodes"]
    promoted = nodes[-1]
    assert promoted["artifact_verification"]["passed"]
    assert promoted["continuation_verification"]["passed"] is advance
    if advance:
        assert (
            promoted["continuation_verification"]["to_steps"]
            > promoted["continuation_verification"]["from_steps"]
        )
    else:
        assert "continuation_no_progress" in promoted["failure_reasons"]
        assert result["best_node_id"] != promoted["id"]
    assert (out / "final/predictions.csv").is_file()
    account = json.loads((out / "logs/accounting.json").read_text())
    assert account["unfinished_nodes"] == [] and account["unassigned_gpu_seconds"] == 0


def test_bad_optional_checkpoint_does_not_discard_verified_single(tmp_path, monkeypatch):
    task = tmp_path / "flower"
    create(task)
    script = task / "baseline.py"
    script.write_text(script.read_text() + """
    if not args.predict_only:
        import os
        (Path(os.environ["SUPERAGENT_CHECKPOINT_DIR"]) / "checkpoint.json").write_text("{bad json")
""")
    config = tmp_path / "cfg.yaml"
    config.write_text(yaml.safe_dump({
        "experience": {"enabled": False}, "search": {"initial_drafts": 1},
        "execution": {"smoke_run": False}, "run": {"finalize_reserve_seconds": 4},
        "review": {"enabled": False}, "final": {"allow_ensemble": False},
    }))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture")
    out = tmp_path / "out"
    status = run_task(task, out, config, time_limit_secs=90, max_steps=1, num_cpus=1, num_gpus=0, parallel=1)
    assert status["status"] == "success"
    node = json.loads((out / "logs/search_tree.json").read_text())["nodes"][-1]
    assert node["artifact_verification"]["passed"] and node["checkpoint"] is None
    assert (out / "final/predictions.csv").read_bytes() == (out / "best/predictions.csv").read_bytes()
