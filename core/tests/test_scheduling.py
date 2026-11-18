import pytest

from helpers import FixedRng, cfg, mk, tree
from superagent.scheduling import branch_plateau, dispatch, global_plateau, synthesis_conditions

C = cfg()


def branch_with(scores):
    t = tree()
    d = mk(t, "draft", t.root, scores[0])
    prev = d
    for s in scores[1:]:
        prev = mk(t, "refine", prev, s)
    return t, d.branch_id, prev


@pytest.mark.parametrize("scores,window,expected", [
    ([0.80, 0.85, 0.84, 0.83, 0.82], 3, True),
    ([0.80, 0.85, 0.84, 0.83], 3, False),
    ([0.80, 0.85, 0.84, 0.83, 0.82], 6, False),
    ([0.80, 0.70], 3, False),
    ([0.80], 3, False),
    ([0.85, 0.84, 0.85], 2, False),
])
def test_branch_plateau(scores, window, expected):
    t, b, _ = branch_with(scores)
    assert branch_plateau(t, b, window) is expected


def test_global_plateau():
    t = tree()
    nodes = [mk(t, "draft", t.root, 0.5) for _ in range(5)]
    t.best = nodes[4]  # step 5
    for _ in range(3):
        mk(t, "draft", t.root, 0.1)
    assert t.nodes[-1].step == 8
    assert not global_plateau(t, True, 4)
    mk(t, "draft", t.root, 0.1)
    assert global_plateau(t, True, 4)


def syn_tree():
    t = tree()
    a = mk(t, "draft", t.root, 0.9)
    mk(t, "refine", a, 0.8)
    b = mk(t, "draft", t.root, 0.7)
    mk(t, "refine", b, 0.6)
    t.best = a
    for _ in range(4):
        mk(t, "refine", b, 0.5)
    return t, a, b


def test_synthesis_conditions():
    t, a, b = syn_tree()
    assert synthesis_conditions(t, C, 0.6)
    assert not synthesis_conditions(t, C, 0.45)
    assert not synthesis_conditions(t, C, 0.9)
    t.synthesis_count = 2
    assert not synthesis_conditions(t, C, 0.6)
    t.synthesis_count = 0
    t.best = t.nodes[-1]
    assert not synthesis_conditions(t, C, 0.6)


def test_synthesis_needs_two_rich_branches():
    t = tree()
    a = mk(t, "draft", t.root, 0.9)
    mk(t, "refine", a, 0.8)
    mk(t, "draft", t.root, 0.7)
    t.best = a
    for _ in range(4):
        mk(t, "refine", a, 0.5)
    assert not synthesis_conditions(t, C, 0.6)


def test_dispatch_root():
    t = tree()
    assert dispatch(t, t.root, C, 0.1, FixedRng()).action == "draft"
    assert t.root.pending_children == 1
    t.root.pending_children = 0
    for _ in range(5):
        mk(t, "draft", t.root, 0.5)
    d = dispatch(t, t.root, C, 0.1, FixedRng())
    assert (d.action, d.reason) == ("none", "root_full_no_synthesis")


def test_dispatch_root_synthesis():
    t, a, b = syn_tree()
    for _ in range(3):
        mk(t, "draft", t.root, 0.1)
    t.best = a
    for _ in range(4):
        mk(t, "refine", b, 0.4)
    d = dispatch(t, t.root, C, 0.6, FixedRng())
    assert d.action == "synthesis" and t.synthesis_count == 1


def test_dispatch_failed_and_bad_predictions():
    t = tree()
    f = mk(t, "draft", t.root, failed=True)
    assert dispatch(t, f, C, 0.1, FixedRng()).action == "repair"
    g = mk(t, "draft", t.root, failed=True, predictions_ok=False)
    assert dispatch(t, g, C, 0.1, FixedRng()).reason == "node_failed"


def test_dispatch_plateau_variants():
    t, b, last = branch_with([0.80, 0.85, 0.84, 0.83, 0.82])
    assert dispatch(t, last, C, 0.3, FixedRng()).action == "retrospect"
    assert dispatch(t, last, C, 0.6, FixedRng(0.2)).action == "borrow"
    assert dispatch(t, last, C, 0.6, FixedRng(0.5)).action == "retrospect"
    c2 = cfg(operators__retrospect__enabled=False, operators__borrow__enabled=False)
    d = dispatch(t, last, c2, 0.6, FixedRng())
    assert (d.action, d.reason) == ("refine", "plateau_no_alternative")


def test_dispatch_progressing_and_elite_window():
    t, b, last = branch_with([0.80, 0.85])
    assert dispatch(t, last, C, 0.3, FixedRng()).action == "refine"
    t2, b2, last2 = branch_with([0.80, 0.85, 0.84, 0.83, 0.82])
    last2.picked_by_elite = True
    assert dispatch(t2, last2, C, 0.3, FixedRng()).action == "refine"


def test_dispatch_dead_end():
    t = tree()
    d = mk(t, "draft", t.root, 0.5, dead_end=True)
    assert dispatch(t, d, C, 0.3, FixedRng()).action == "none"
