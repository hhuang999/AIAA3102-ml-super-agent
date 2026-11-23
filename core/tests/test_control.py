import json
import time

import pytest

from helpers import make_ctx, mk
from superagent.control import (
    Controller,
    CostCapRule,
    GpuInfo,
    ResourceRule,
    RuleController,
    StepTiming,
    TimeAllocationRule,
    build_state,
    load_controller_class,
    parse_nvidia_smi,
)
from superagent.errors import ConfigError
from superagent.prompts.blocks import implementation_rules


def _ctx(tmp_path, limit=28800.0, elapsed=0.0, **over):
    ctx = make_ctx(tmp_path, cfg_over={"run__time_limit_seconds": limit, **over})
    ctx.process_start = time.monotonic() - elapsed
    ctx.tree.search_start = ctx.process_start
    return ctx


def _timing(i, step, exec_s, ok=True):
    return StepTiming(step=i, node_id=f"n{i}", kind="refine", ok=ok, step_seconds=step, exec_seconds=exec_s)


def test_time_allocation_before_any_step(tmp_path):
    ctx = _ctx(tmp_path, resources__parallel=2, resources__num_cpus=2)
    rule = TimeAllocationRule(ctx.cfg["control"]["time_allocation"], 7200)
    p = rule.plan(build_state(ctx))
    usable = 28800 - 300
    assert p["slot_budget_secs"] == pytest.approx(usable * 2 / 60, abs=1)
    assert p["script_secs"] == pytest.approx(usable * 2 / 60 - 180, abs=1)
    assert not p["step_estimate_measured"]


def test_time_allocation_uses_measured_generation_time(tmp_path):
    ctx = _ctx(tmp_path, resources__parallel=2, resources__num_cpus=2)
    for i in range(4):
        ctx.record_step(_timing(i, 700, 400))       # 300 s outside the script
    rule = TimeAllocationRule(ctx.cfg["control"]["time_allocation"], 7200)
    p = rule.plan(build_state(ctx))
    assert p["other_secs"] == 300
    assert p["step_estimate_measured"]
    assert p["steps_left"] == int((28800 - 300) * 2 // 700)


def test_time_allocation_floor_ceiling_and_time_left(tmp_path):
    ctx = _ctx(tmp_path, control__time_allocation__target_nodes=2000)
    rule = TimeAllocationRule(ctx.cfg["control"]["time_allocation"], 7200)
    assert rule.plan(build_state(ctx))["script_secs"] == 300            # floor
    ctx = _ctx(tmp_path / "b", control__time_allocation__target_nodes=1)
    rule = TimeAllocationRule(ctx.cfg["control"]["time_allocation"], 7200)
    assert rule.plan(build_state(ctx))["script_secs"] == 7200           # script timeout
    ctx = _ctx(tmp_path / "c", limit=1000.0, elapsed=400.0)               # 300 s usable
    rule = TimeAllocationRule(ctx.cfg["control"]["time_allocation"], 7200)
    p = rule.plan(build_state(ctx))
    assert p["script_secs"] == pytest.approx(300 - 180, abs=2)           # never beyond the time left


def test_time_allocation_start_rule_uses_short_tail_steps(tmp_path):
    ctx = _ctx(tmp_path, limit=1000.0, elapsed=300.0)                    # 400 s usable, 180 s generation
    rule = TimeAllocationRule(ctx.cfg["control"]["time_allocation"], 7200)
    assert rule.start(build_state(ctx)).allow                           # 220 s of script time >= 120 s
    ctx = _ctx(tmp_path / "b", limit=1000.0, elapsed=450.0)               # 250 s usable
    assert not rule.start(build_state(ctx)).allow                       # 70 s of script time < 120 s


def test_static_controller_matches_fixed_rules(tmp_path):
    ctx = _ctx(tmp_path, limit=10000.0, elapsed=1000.0)
    c = Controller(ctx)
    st = build_state(ctx)
    lim = c.step_limits(st)
    assert lim.run_limit_secs == pytest.approx(min(7200, 10000 - 1000 - 300), abs=2)
    assert lim.steps_left == 500 and not lim.budget_text
    assert c.may_start_step(st).allow
    ctx.record_step(_timing(1, 9000, 6000))                               # 1.5 x 6000 s does not fit
    assert not c.may_start_step(build_state(ctx)).allow
    assert c.search_config(st) is ctx.cfg
    assert not c.should_finish(st).allow


def test_rule_controller_with_rules_off_behaves_statically(tmp_path):
    ctx = _ctx(tmp_path, control__time_allocation__mode="off", control__cost_cap__mode="off", control__resources__mode="off")
    rc, sc = RuleController(ctx), Controller(ctx)
    st = build_state(ctx)
    assert rc.step_limits(st).run_limit_secs == pytest.approx(sc.step_limits(st).run_limit_secs, abs=1)
    assert rc.may_start_step(st).allow == sc.may_start_step(st).allow


def test_observe_mode_logs_but_does_not_apply(tmp_path):
    ctx = _ctx(tmp_path, control__time_allocation__mode="observe")
    rc = RuleController(ctx)
    lim = rc.step_limits(build_state(ctx))
    assert lim.run_limit_secs == 7200 and not lim.budget_text          # static value applied
    rows = [json.loads(x) for x in ctx.paths.control_log.read_text().splitlines()]
    proposed = rows[-1]["rules"]["time_allocation"]
    assert proposed["mode"] == "observe" and proposed["proposed"]["run_limit_secs"] < 7200


def test_enforce_mode_applies_the_budget(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.control = RuleController(ctx)
    lim = ctx.begin_step()
    assert lim.budget_text and lim.steps_left_estimated
    assert ctx.current_run_limit() == pytest.approx(lim.run_limit_secs, abs=0.5)
    text = implementation_rules(ctx.resources(), 42, 0.2)
    assert "Time budget for this script" in text and "aim for the best result" not in text


def test_static_prompt_text_is_unchanged(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.control = Controller(ctx)
    ctx.begin_step()
    text = implementation_rules(ctx.resources(), 42, 0.2)
    assert "Steps left: 500." in text and "aim for the best result" in text


def test_cost_cap_rule():
    rule = CostCapRule({"mode": "enforce", "limit": 10.0, "currency": "CNY", "rate_to_cap_currency": 2.0, "warn_fraction": 0.8})

    class S:
        def __init__(self, cost, nodes, active):
            self.cost, self.nodes_total, self.active_tasks = cost, nodes, active

    assert rule.evaluate(S(1.0, 10, 1)).reason == "within_limit"     # 2.0 CNY
    assert rule.evaluate(S(4.1, 10, 1)).reason == "cost_warning"     # 8.2 CNY
    d = rule.evaluate(S(4.6, 10, 1))                                  # 9.2 + 0.92 projected for the running step
    assert d.allow and d.reason == "cost_limit"


def test_cost_cap_observe_and_enforce(tmp_path):
    ctx = _ctx(tmp_path, control__cost_cap__limit=0.001, control__cost_cap__mode="observe")
    ctx.gateway = type("G", (), {"totals": {"cost": 0.01, "prompt_tokens": 1, "completion_tokens": 1}})()
    rc = RuleController(ctx)
    d = rc.should_finish(build_state(ctx))
    assert not d.allow and d.reason == "cost_limit"                  # observe: never finishes
    assert rc.summary()["cost_cap"]["events"]["cost_limit"]["would_finish"] is True
    ctx2 = _ctx(tmp_path / "b", control__cost_cap__limit=0.001, control__cost_cap__mode="enforce")
    ctx2.gateway = ctx.gateway
    assert RuleController(ctx2).should_finish(build_state(ctx2)).allow


def test_resource_rule_gpu_share(tmp_path):
    ctx = _ctx(tmp_path, resources__parallel=2, resources__num_cpus=2, resources__num_gpus=1)
    ctx.gpu_info = GpuInfo(count=1, memory_mb=(23034.0,), names=("NVIDIA L4",))
    rule = ResourceRule(ctx.cfg["control"]["resources"])
    mem, sharing = rule.gpu_share(build_state(ctx))
    assert sharing == 2 and mem == pytest.approx(23034 / 1024 / 2, abs=0.1)
    ctx.cfg["resources"]["num_gpus"] = 0
    assert rule.gpu_share(build_state(ctx)) == (None, 1)


def test_parse_nvidia_smi():
    info = parse_nvidia_smi("NVIDIA L4, 23034\nNVIDIA A100-SXM4-80GB, 81920\nbroken line\n")
    assert info.count == 2 and info.memory_mb == (23034.0, 81920.0) and info.names[0] == "NVIDIA L4"


class CustomController(Controller):
    name = "custom"


def test_load_controller_class():
    assert load_controller_class("static") is Controller
    assert load_controller_class("rules") is RuleController
    assert load_controller_class("test_control:CustomController") is CustomController
    with pytest.raises(ConfigError):
        load_controller_class("nope")
    with pytest.raises(ConfigError):
        load_controller_class("json:JSONDecoder")


def test_state_counts_nodes_and_timings(tmp_path):
    ctx = _ctx(tmp_path)
    t = ctx.tree
    a = mk(t, "draft", t.root, score=0.5)
    mk(t, "refine", a, failed=True)
    ctx.record_step(_timing(1, 100, 60))
    st = build_state(ctx)
    assert st.nodes_total == 2 and st.nodes_ok == 1 and st.nodes_failed == 1
    assert st.nodes_by_kind == {"draft": 1, "refine": 1}
    assert st.median_other_seconds(8) == 40 and st.median_ok_script_seconds() == 60


def test_plan_written_at_start(tmp_path):
    ctx = _ctx(tmp_path)
    rc = RuleController(ctx)
    rc.start(build_state(ctx))
    row = json.loads(ctx.paths.control_log.read_text().splitlines()[0])
    assert row["event"] == "plan" and row["rules"]["time_allocation"] == "enforce"
    assert row["time_allocation"]["script_secs"] > 0


def test_static_preset_is_valid_and_switches_off_new_mechanisms():
    from pathlib import Path

    from superagent.config import build_config

    c = build_config(Path(__file__).resolve().parents[1] / "configs" / "static.yaml")
    assert c["control"]["controller"] == "static" and c["search"]["reward_mode"] == "chain_end"
    assert not c["search"]["chain_soft_switch"] and not c["execution"]["smoke_run"] and not c["final"]["allow_ensemble"]
    assert build_config(Path(__file__).resolve().parents[1] / "configs" / "cost_cap_enforced.yaml")["control"]["cost_cap"]["mode"] == "enforce"


def test_example_controller(tmp_path, monkeypatch):
    import sys
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples" / "controllers"))
    cls = load_controller_class("budget_controller:BudgetController")
    ctx = _ctx(tmp_path, control__cost_cap__limit=1.0, control__cost_cap__mode="observe")
    ctx.gateway = type("G", (), {"totals": {"cost": 0.95, "prompt_tokens": 1, "completion_tokens": 1}})()
    c = cls(ctx)
    st = build_state(ctx)
    assert c.should_finish(st).reason == "cost_share_reached"
    assert c.step_limits(st).run_limit_secs <= 0.25 * st.usable_secs + 0.1
    sys.modules.pop("budget_controller", None)


class BrokenController(Controller):
    name = "broken"

    def step_limits(self, state):
        raise RuntimeError("boom")

    def may_start_step(self, state):
        raise RuntimeError("boom")

    def search_config(self, state=None):
        raise RuntimeError("boom")


def test_broken_controller_falls_back_to_fixed_rules(tmp_path):
    from superagent.search import SearchController

    ctx = _ctx(tmp_path)
    ctx.control = BrokenController(ctx)
    assert ctx.begin_step() is None
    assert ctx.current_run_limit() == pytest.approx(min(7200, 28800 - 300), abs=2)
    assert ctx.search_cfg() is ctx.cfg
    assert SearchController(ctx)._may_start() == ctx.can_start_task()


def test_default_resources_and_cost_warning_config():
    from pathlib import Path
    from superagent.config import build_config, load_defaults

    d = load_defaults()
    assert d["resources"]["num_gpus"] == 4
    assert d["resources"]["parallel"] == 4
    assert d["resources"]["num_cpus"] == 16
    cap = d["control"]["cost_cap"]
    assert cap["mode"] == "enforce"
    assert cap["limit"] == 20.0
    assert cap["warn_fraction"] == 0.9
    assert cap["stop_on_warning"] is True
    for p in sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml")):
        build_config(p, environ={})


def test_cost_cap_stop_on_warning_at_18():
    rule = CostCapRule({"mode": "enforce", "limit": 20.0, "currency": "CNY", "rate_to_cap_currency": 1.0,
                        "warn_fraction": 0.9, "stop_on_warning": True})

    class S:
        def __init__(self, cost, nodes, active):
            self.cost, self.nodes_total, self.active_tasks = cost, nodes, active

    d_under = rule.evaluate(S(17.5, 10, 1))
    assert not d_under.allow and d_under.reason == "within_limit"
    d_warn = rule.evaluate(S(18.0, 20, 1))
    assert d_warn.allow and d_warn.reason == "cost_warning"
    assert d_warn.details["warn_limit"] == 18.0
    d_warn2 = rule.evaluate(S(19.5, 10, 1))
    assert d_warn2.allow and d_warn2.reason == "cost_warning"
    d_proj = rule.evaluate(S(17.5, 4, 1))
    assert d_proj.allow and d_proj.reason == "cost_limit"
    d_hard = rule.evaluate(S(20.0, 10, 1))
    assert d_hard.allow and d_hard.reason == "cost_limit"


def test_adaptive_controller_stops_new_branches_at_18(tmp_path):
    from superagent.control.adaptive import AdaptiveController

    ctx = _ctx(tmp_path)
    ctx.gateway = type("G", (), {"totals": {"cost": 18.1, "prompt_tokens": 100, "completion_tokens": 100}})()
    ctrl = AdaptiveController(ctx)
    st = build_state(ctx)
    d = ctrl.should_finish(st)
    assert d.allow and d.reason == "cost_warning"

    choice, info = ctrl.choose(st, reserve=False)
    assert choice is None and info["reason"] == "cost_warning"
    rows = [json.loads(line) for line in ctx.paths.control_log.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["cost_warning"]
    assert rows[0]["finished"] is True and rows[0]["warn_limit"] == 18.0
    assert "cost_warning" in ctrl.summary()["cost_cap"]["events"]

    ctrl.leases.add(("root", "draft"))
    choice2, info2 = ctrl.choose(st, reserve=False)
    assert choice2 is None and info2["reason"] == "cost_warning"


def test_search_controller_winds_down_on_cost_warning(tmp_path, monkeypatch):
    from superagent.control.adaptive import AdaptiveController
    from superagent.executor import ExecResult
    from superagent.operators import GenOutcome
    from superagent.search import SearchController
    from superagent.tree import Node

    ctx = _ctx(
        tmp_path,
        resources__parallel=2,
        resources__num_cpus=2,
        search__initial_drafts=1,
        search__concurrent_initial_drafts=False,
        execution__draft_launch_stagger_seconds=0,
        run__max_steps=10,
    )
    gw = type("G", (), {"totals": {"cost": 5.0, "prompt_tokens": 10, "completion_tokens": 10}})()
    ctx.gateway = gw
    ctx.control = AdaptiveController(ctx)
    sc = SearchController(ctx)

    generated = []

    def fake_generate(action, parent, **_kw):
        node = Node(id=f"n{len(generated) + 1}", kind=action, code="print('VALIDATION_SCORE: 0.8')", plan="p")
        generated.append(node.id)
        gw.totals["cost"] = 18.2
        return GenOutcome(node=node, kind=action, details={})

    def fake_exec(node_id, code, limit, **_kw):
        return ExecResult("VALIDATION_SCORE: 0.8\n", 0.05, None, None, False, 0, limit)

    def fake_assess(_ctx, node, _res):
        node.failed = False
        node.score = 0.8
        return {}

    monkeypatch.setattr(sc.ops, "generate", fake_generate)
    monkeypatch.setattr(sc, "_prerun", lambda node, rec: None)
    monkeypatch.setattr(ctx.executor, "run", fake_exec)
    monkeypatch.setattr("superagent.search.assess", fake_assess)

    sc.run()
    assert len(generated) == 1
    assert len(ctx.tree.ok_nodes()) == 1
    assert sc.stop_reason() == "cost_warning"


def test_adaptive_script_timeout_ignores_fast_crashes_and_honors_min_script_floor(tmp_path):
    from superagent.control.adaptive import AdaptiveController

    ctx = _ctx(tmp_path, run__time_limit_seconds=21600)
    ctrl = AdaptiveController(ctx)
    st = build_state(ctx)
    st = type(st)(
        **{
            **st.__dict__,
            "step_timings": (
                StepTiming(1, "c1", "draft", False, 10.0, 4.2),
                StepTiming(2, "ok1", "draft", True, 30.0, 18.0),
                StepTiming(3, "c2", "refine", False, 12.0, 5.0),
                StepTiming(4, "ok2", "refine", True, 35.0, 20.0),
            ),
        }
    )
    _, draft_run = ctrl.estimate("draft", st)
    assert draft_run == 18.0  # 4.2s crash is excluded from ok_script estimate
    # Even when expected execution is 20s, hard kill timeout keeps min_script_seconds (300s) headroom
    assert ctrl.script_timeout(20.0, st) >= 300.0


def test_adaptive_script_timeout_scales_dynamically_with_run_budget(tmp_path):
    from superagent.control.adaptive import AdaptiveController

    # Short 100s run -> bounded safely below usable_secs
    ctx_short = _ctx(tmp_path / "s", run__time_limit_seconds=100, run__finalize_reserve_seconds=20)
    ctrl_short = AdaptiveController(ctx_short)
    t_short = ctrl_short.script_timeout(15.0, build_state(ctx_short))
    assert 35.0 <= t_short <= 80.0

    # Long 6-hour (21600s) run with 4 parallel slots -> generous slot-budget floor (>800s) for heavy multi-fold models
    ctx_long = _ctx(tmp_path / "l", run__time_limit_seconds=21600, resources__parallel=4)
    ctrl_long = AdaptiveController(ctx_long)
    t_long = ctrl_long.script_timeout(30.0, build_state(ctx_long))
    assert t_long >= 900.0
