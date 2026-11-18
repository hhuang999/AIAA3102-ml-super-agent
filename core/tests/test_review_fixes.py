"""Regression tests for bookkeeping edge cases and the late-start guard."""

import json

from helpers import make_ctx, mk
from superagent.bookkeeping import after_assessment_files, update_rankings
from superagent.context import MIN_SCRIPT_SECONDS, required_start_seconds


def _ok_node(ctx, score):
    n = mk(ctx.tree, "draft", ctx.tree.root, score)
    d = ctx.executor.prepare_run_dir(n.id, f"# {score}\n")
    n.code = f"# {score}\n"
    (d / "out" / "predictions.csv").write_text("id,label\nr0,1\n")
    return n


def test_best_never_overwritten_by_older_state(tmp_path):
    ctx = make_ctx(tmp_path)
    a = _ok_node(ctx, 0.9)
    b = _ok_node(ctx, 0.8)
    update_rankings(ctx, a)
    update_rankings(ctx, b)
    assert json.loads((ctx.paths.best / "score.json").read_text())["node_id"] == a.id
    c = _ok_node(ctx, 0.95)
    ctx.tree.best = c  # tree moved on without best/ being rewritten yet
    update_rankings(ctx, None)
    assert json.loads((ctx.paths.best / "score.json").read_text())["node_id"] == c.id
    assert (ctx.paths.top / "rank_01" / "solution.py").read_text() == "# 0.95\n"


def test_sorted_ok_skips_scoreless(tmp_path):
    ctx = make_ctx(tmp_path)
    a = _ok_node(ctx, 0.9)
    a.failed, a.score = True, None
    assert ctx.tree.sorted_ok([a]) == []


def test_required_start_seconds():
    assert required_start_seconds(1.5, None) == MIN_SCRIPT_SECONDS          # no successful script yet
    assert required_start_seconds(1.5, 10) == MIN_SCRIPT_SECONDS            # never below the minimum
    assert required_start_seconds(1.5, 400) == 600
    assert required_start_seconds(0, 400) == MIN_SCRIPT_SECONDS             # guard switched off


def test_can_start_task_uses_typical_run_time(tmp_path):
    ctx = make_ctx(tmp_path, cfg_over={"run__time_limit_seconds": 1000, "run__finalize_reserve_seconds": 100})
    assert ctx.can_start_task()                     # 900 s usable, nothing has run yet
    mk(ctx.tree, "draft", ctx.tree.root, 0.9, exec_seconds=700.0)
    assert not ctx.can_start_task()                 # 1.5 x 700 s would not fit into 900 s
    ctx.cfg["run"]["late_start_guard"] = 0.0
    assert ctx.can_start_task()


def test_model_files_removed_for_nodes_outside_the_kept_ranks(tmp_path):
    ctx = make_ctx(tmp_path, cfg_over={"execution__keep_artifacts_top_k": 1, "search__top_per_branch": 1})
    first = _ok_node(ctx, 0.9)
    weaker = mk(ctx.tree, "refine", first, 0.8)     # same branch: never enters the top list
    for n in (first, weaker):
        d = ctx.executor.prepare_run_dir(n.id, "# x\n")
        (d / "out" / "predictions.csv").write_text("id,label\nr0,1\n")
        (d / "out" / "model").mkdir()
        (d / "out" / "model" / "m.bin").write_text("w")
        after_assessment_files(ctx, n)
    update_rankings(ctx, first)
    update_rankings(ctx, weaker)
    assert (ctx.executor.run_dir(first.id) / "out" / "model" / "m.bin").exists()
    assert not (ctx.executor.run_dir(weaker.id) / "out" / "model").exists() and not weaker.has_artifacts
