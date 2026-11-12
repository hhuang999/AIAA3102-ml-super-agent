import time

import pytest

from helpers import FixedRng, cfg, make_ctx, mk, tree
from superagent.chain import add_reward, chain_control, step_reward
from superagent.scheduling import (
    attempts_since_branch_best,
    branch_plateau,
    dispatch,
    synthesis_blocker,
    synthesis_pending,
)
from superagent.search import SearchController, StepRecord, Task

# (kind, parent step or None for the root, failed, score) of the first 17 nodes of a recorded 8-hour run
# on a lower-is-better task; steps are numbered from 1 in this order.
RECORDED = [
    ("draft", None, True, None), ("draft", None, False, 0.344217), ("draft", None, True, None),
    ("repair", 1, True, None), ("refine", 2, False, 0.329035), ("repair", 4, True, None),
    ("repair", 3, False, 0.272718), ("refine", 5, True, None), ("repair", 6, False, 0.318056),
    ("repair", 8, False, 0.326103), ("refine", 7, False, 0.266477), ("refine", 9, False, 0.277273),
    ("refine", 11, False, 0.28295), ("refine", 10, True, None), ("refine", 12, False, 0.284467),
    ("refine", 13, False, 0.28315), ("repair", 14, False, 0.274104),
]


def recorded_tree():
    t = tree(higher=False)
    by_step = {}
    for i, (kind, parent, failed, score) in enumerate(RECORDED, 1):
        p = t.root if parent is None else by_step[parent]
        n = mk(t, kind, p, score=score, failed=failed)
        by_step[i] = n
    best = t.sorted_ok()[0]
    t.best = best
    return t, by_step


def test_recorded_run_would_trigger_synthesis():
    t, _ = recorded_tree()
    assert t.best.score == 0.266477 and t.best.step == 11
    new, old = cfg(), cfg(operators__synthesis__trigger_on_stagnation=False)
    assert synthesis_blocker(t, new, 0.58) is None
    assert dispatch(t, t.root, old, 0.58, FixedRng()).action == "draft"       # fixed rule: a fifth draft
    assert dispatch(t, t.root, new, 0.58, FixedRng()).action == "synthesis"   # new rule: synthesis first
    assert synthesis_blocker(t, new, 0.4) == "outside_time_window"


def test_step_reward_levels():
    t = tree()
    d = mk(t, "draft", t.root, score=0.5)
    assert step_reward(d, t, 1e-4) == 0.5                 # first success of the branch
    better = mk(t, "refine", d, score=0.6)
    assert step_reward(better, t, 1e-4) == 1.0            # new branch best
    worse = mk(t, "refine", better, score=0.55)
    assert step_reward(worse, t, 1e-4) == 0.0             # success without progress
    bad = mk(t, "refine", worse, failed=True)
    assert step_reward(bad, t, 1e-4) == -1.0


def test_add_reward_updates_the_path():
    t = tree()
    d = mk(t, "draft", t.root, score=0.5)
    r = mk(t, "refine", d, score=0.6)
    add_reward(r, 1.0)
    add_reward(d, 0.5)
    assert (r.visits, r.reward_sum) == (1, 1.0)
    assert (d.visits, d.reward_sum) == (2, 1.5)
    assert (t.root.visits, t.root.reward_sum) == (2, 1.5)


def test_per_step_mode_does_not_count_a_second_visit_at_chain_end():
    t = tree()
    c = cfg(search__max_no_gain=0)
    d = mk(t, "draft", t.root, score=0.5, append=False)
    d.chain_best = d
    t.append(d)
    r = mk(t, "refine", d, score=0.4, append=False)        # no gain: with max_no_gain 0 the chain ends
    out = chain_control(t, r, c, 0.1, FixedRng(0.99))
    assert out.returned_to_root and out.reward is None
    assert r.visits == 0 and t.root.visits == 0


def test_branch_plateau_counts_failed_attempts():
    t = tree()
    d = mk(t, "draft", t.root, score=0.5)
    best = mk(t, "refine", d, score=0.7)
    mk(t, "refine", best, failed=True)
    w = mk(t, "refine", best, score=0.6)
    b = d.branch_id
    assert attempts_since_branch_best(t, b) == 2
    assert not branch_plateau(t, b, 3, True, count_failed=True)
    assert not branch_plateau(t, b, 3, True, count_failed=False)  # the fixed rule never sees the failed attempt
    mk(t, "refine", w, score=0.65)
    assert branch_plateau(t, b, 3, True, count_failed=True)
    mk(t, "refine", w, score=0.8)                                  # a new branch best resets the count
    assert not branch_plateau(t, b, 3, True, count_failed=True)


def test_dispatch_uses_the_new_plateau_rule():
    t = tree()
    d = mk(t, "draft", t.root, score=0.5)
    best = mk(t, "refine", d, score=0.7)
    a = mk(t, "refine", best, failed=True)
    a2 = mk(t, "repair", a, failed=True)
    last = mk(t, "repair", a2, score=0.6)
    assert dispatch(t, last, cfg(), 0.2, FixedRng(0.9)).action == "retrospect"
    assert dispatch(t, last, cfg(plateau__count_failed_attempts=False), 0.2, FixedRng(0.9)).action == "refine"


def test_synthesis_blockers():
    t = tree()
    c = cfg()
    assert synthesis_blocker(t, c, None) == "search_not_started"
    assert synthesis_blocker(t, c, 0.6) == "too_few_branches_with_successes"
    assert synthesis_blocker(t, cfg(operators__synthesis__enabled=False), 0.6) == "disabled"
    t.synthesis_count = 2
    assert synthesis_blocker(t, c, 0.6) == "max_count_reached"
    assert synthesis_pending(t) == 2


# --------------------------------------------------------------------------- search controller

class _Captured(Exception):
    pass


def _controller(tmp_path, progress, **over):
    ctx = make_ctx(tmp_path, higher=False, cfg_over=over)
    ctx.tree = recorded_tree()[0]
    ctx.direction = ctx.tree.direction
    ctx.tree.search_start = time.monotonic() - progress * ctx.time_limit
    sc = SearchController(ctx)
    calls = []

    def fake_step(rec, parent, pre_generated=None, dispatched=None):
        calls.append((rec, parent, dispatched))
        return None

    sc._step = fake_step  # type: ignore[assignment]
    return ctx, sc, calls


def test_synthesis_trigger_in_the_search_loop(tmp_path):
    ctx, sc, calls = _controller(tmp_path, 0.58)
    follow = ctx.tree.nodes[-1]
    sc._execute(Task("search", follow, "chain"))
    rec, parent, d = calls[-1]
    assert rec.mode == "trigger" and parent is ctx.tree.root and d.action == "synthesis"
    assert ctx.tree.synthesis_count == 1 and rec.details["left_chain_at"] == follow.id
    sc._execute(Task("search", follow, "chain"))                  # one synthesis at a time
    assert calls[-1][0].mode != "trigger"
    assert ctx.flags["synthesis_blockers"]["synthesis_running"] == 1


def test_leaving_a_chain_for_an_elite_node(tmp_path):
    ctx, sc, calls = _controller(tmp_path, 0.9, operators__synthesis__enabled=False)
    ctx.rng = FixedRng(0.95)                                      # above the explore weight 0.2: exploit
    follow = ctx.tree.nodes[-1]
    sc._execute(Task("search", follow, "chain"))
    rec, parent, d = calls[-1]
    assert rec.mode == "elite_pick" and rec.details["left_chain_at"] == follow.id and d is not None
    ctx.rng = FixedRng(0.05)                                      # below the explore weight: stay on the chain
    sc._execute(Task("search", follow, "chain"))
    assert calls[-1][0].mode == "chain" and calls[-1][1] is follow


@pytest.mark.parametrize("progress, over", [(0.3, {}), (0.9, {"search__chain_soft_switch": False})])
def test_chain_is_kept(tmp_path, progress, over):
    ctx, sc, calls = _controller(tmp_path, progress, operators__synthesis__enabled=False, **over)
    ctx.rng = FixedRng(0.99)
    follow = ctx.tree.nodes[-1]
    sc._execute(Task("search", follow, "chain"))
    assert calls[-1][0].mode == "chain"


def test_step_record_has_a_start_time():
    rec = StepRecord()
    assert rec.started <= time.monotonic() and rec.limits is None
