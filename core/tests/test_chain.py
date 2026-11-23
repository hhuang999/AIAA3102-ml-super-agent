import pytest

from helpers import FixedRng, cfg, mk, tree
from superagent.chain import backpropagate, chain_control

C = cfg(search__reward_mode="chain_end")  # rewards at the end of a chain (the per-step mode is tested in test_search_mechanisms)


def started(t):
    t.search_start = 0.0
    return t


def fresh(t, kind, parent, score=None, failed=False):
    """A node that is evaluated but not yet appended to the node list."""
    return mk(t, kind, parent, score, failed=failed, append=False)


def test_draft_success_starts_chain():
    t = started(tree())
    d = fresh(t, "draft", t.root, 0.80)
    out = chain_control(t, d, C, 0.1, FixedRng())
    assert d.chain_best is d and not out.returned_to_root and out.reward is None


def setup_d(no_gain=0):
    t = started(tree())
    d = mk(t, "draft", t.root, 0.80)
    d.chain_best = d
    d.no_gain_count = no_gain
    return t, d


def test_no_gain_increments():
    t, d = setup_d()
    r = fresh(t, "refine", d, 0.80005)
    out = chain_control(t, r, C, 0.1, FixedRng())
    assert r.no_gain_count == 1 and d.no_gain_count == 0 and r.chain_best is d and not out.returned_to_root


def test_no_gain_limit_ends_chain():
    t, d = setup_d(no_gain=3)
    r = fresh(t, "refine", d, 0.80005)
    out = chain_control(t, r, C, 0.1, FixedRng())
    assert r.dead_end and out.returned_to_root and out.reward == 1.0


def test_gain_moves_chain_best():
    t, d = setup_d()
    r = fresh(t, "refine", d, 0.81)
    out = chain_control(t, r, C, 0.1, FixedRng())
    assert r.chain_best is r and not out.returned_to_root


def failed_chain(depth):
    t = started(tree())
    p = mk(t, "draft", t.root, failed=True)
    for _ in range(depth - 1):
        p = mk(t, "repair", p, failed=True)
    n = fresh(t, "repair", p, failed=True) if depth > 0 else p
    return t, n


def test_failed_depth_two_continues():
    t, n = failed_chain(2)
    assert n.repair_depth == 2
    out = chain_control(t, n, C, 0.1, FixedRng())
    assert not out.returned_to_root and out.reward is None


def test_failed_depth_three_returns():
    t, n = failed_chain(3)
    out = chain_control(t, n, C, 0.1, FixedRng())
    assert out.returned_to_root and out.reward == -1.0 and not n.dead_end


def test_failed_depth_twenty_dead_end():
    t, n = failed_chain(20)
    out = chain_control(t, n, C, 0.1, FixedRng())
    assert out.returned_to_root and out.reward == -1.0 and n.dead_end


def test_repair_success_below_chain_best():
    t, d = setup_d()
    p = mk(t, "refine", d, failed=True)
    n = fresh(t, "repair", p, 0.78)
    out = chain_control(t, n, C, 0.1, FixedRng())
    assert n.keep_going and n.no_gain_count == 1 and d.no_gain_count == 0 and not out.returned_to_root


def test_forced_draft_reward_zero():
    t = started(tree())
    d = fresh(t, "draft", t.root, 0.8)
    t.branch_finished_count[d.branch_id] = 2
    out = chain_control(t, d, C, 0.5, FixedRng())
    assert out.forced_return and out.reward == 0.0


def test_repair_success_reward():
    t = started(tree())
    p = mk(t, "draft", t.root, failed=True)
    n = fresh(t, "repair", p, 0.7)
    n.chain_best = t.root
    t.branch_finished_count[n.branch_id] = 2
    out = chain_control(t, n, C, 0.5, FixedRng())
    assert out.returned_to_root and out.reward == 1.5


def forced_setup(count_before, recent_scores, n_score):
    t = started(tree())
    d = mk(t, "draft", t.root, 0.5)
    d.chain_best = d
    for s in recent_scores:
        if s is None:
            mk(t, "refine", d, failed=True)
        else:
            mk(t, "refine", d, s)
    t.branch_finished_count[d.branch_id] = count_before
    n = fresh(t, "refine", d, n_score)
    return t, n


def test_forced_return_rules():
    t, n = forced_setup(2, [None, 0.70, None, 0.75], 0.72)
    assert chain_control(t, n, C, 0.5, FixedRng()).forced_return
    t, n = forced_setup(2, [None, 0.70, None, 0.75], 0.75)
    assert not chain_control(t, n, C, 0.5, FixedRng()).forced_return
    t, n = forced_setup(3, [None, 0.70, None, 0.75], 0.72)
    assert not chain_control(t, n, C, 0.5, FixedRng()).forced_return
    t, n = forced_setup(0, [None, 0.70, None, 0.75], 0.72)
    assert chain_control(t, n, C, 0.85, FixedRng(0.3)).forced_return
    t, n = forced_setup(0, [None, 0.70, None, 0.75], 0.72)
    assert not chain_control(t, n, C, 0.85, FixedRng(0.7)).forced_return
    t, n = forced_setup(2, [None, None, None, None], 0.72)
    assert chain_control(t, n, C, 0.5, FixedRng()).forced_return


def test_backpropagate():
    t = tree()
    d = mk(t, "draft", t.root, 0.5, in_progress=True, keep_going=True)
    r1 = mk(t, "refine", d, 0.6, keep_going=True, no_gain_count=2)
    r2 = mk(t, "refine", r1, 0.6, keep_going=False)
    t.root.keep_going = True
    backpropagate(r2, 1.0, True)
    for x in (r2, r1, d, t.root):
        assert x.visits == 1 and x.reward_sum == 1.0
    assert not r1.keep_going and not d.keep_going and t.root.keep_going
    assert not d.in_progress and r1.no_gain_count == 2


def test_backpropagate_no_count():
    t = tree()
    d = mk(t, "draft", t.root, 0.5, in_progress=True, keep_going=True)
    r1 = mk(t, "refine", d, 0.6, keep_going=True, no_gain_count=2)
    r2 = mk(t, "refine", r1, 0.6, keep_going=False)
    backpropagate(r2, 1.0, False)
    assert all(x.visits == 0 and x.reward_sum == 0 for x in (r2, r1, d, t.root))
    assert not r1.keep_going and not d.in_progress and r1.no_gain_count == 2
