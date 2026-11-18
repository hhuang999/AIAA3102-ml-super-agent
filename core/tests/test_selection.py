import collections

import pytest

from helpers import FixedRng, cfg, mk, seeded, tree
from superagent.selection import elite_candidates, elite_pick, explore_weight, is_full, tree_walk, uct_c, uct_value
from superagent.tree import Node

C = cfg()


@pytest.mark.parametrize("p,w", [(0.30, 1.0), (0.50, 1.0), (0.60, 0.6), (0.65, 0.4), (0.70, 0.2), (0.90, 0.2)])
def test_explore_weight(p, w):
    assert explore_weight(p, C) == pytest.approx(w, abs=1e-4)


@pytest.mark.parametrize("t,c", [(10, 1.414), (44, 1.414), (45, 1.414), (100, 0.864), (136, 0.504), (137, 0.5), (350, 0.5), (400, 0.5)])
def test_uct_c(t, c):
    assert uct_c(t, C) == pytest.approx(c, abs=1e-4)


def test_uct_value():
    assert uct_value(Node(id="a", kind="draft"), 10, 1.414) == float("inf")
    n = Node(id="b", kind="draft", visits=2, reward_sum=1.0)
    assert uct_value(n, 10, 1.414) == pytest.approx(2.0172, abs=1e-4)


def test_capacity():
    t = tree()
    for _ in range(4):
        mk(t, "draft", t.root, 0.5)
    t.root.pending_children = 1
    assert is_full(t.root, C)
    t.root.pending_children = 0
    mk(t, "synthesis", t.root, 0.5)
    assert not is_full(t.root, C)
    f = mk(t, "draft", t.root, failed=True)
    t.root.children.remove(f)  # keep root counts simple
    assert not is_full(f, C)
    mk(t, "repair", f, failed=True)
    assert is_full(f, C)
    ok = t.root.children[0]
    for _ in range(3):
        mk(t, "refine", ok, 0.5)
    assert is_full(ok, C)
    ok2 = t.root.children[1]
    mk(t, "refine", ok2, 0.5)
    mk(t, "refine", ok2, 0.5)
    mk(t, "refine", ok2, 0.5, via_elite=True)
    mk(t, "refine", ok2, 0.5, via_elite=True)
    assert not is_full(ok2, C)
    from superagent.selection import has_elite_capacity

    assert has_elite_capacity(ok2, C)


def no_syn():
    return False


def test_walk_only_root():
    t = tree()
    assert tree_walk(t, C, no_syn, seeded()) is t.root


def test_walk_unvisited_draft():
    t = tree()
    ds = [mk(t, "draft", t.root, 0.5 + i / 100) for i in range(5)]
    ds[0].visits = 1
    got = tree_walk(t, C, no_syn, seeded())
    assert got in ds[1:] and got.in_progress


def test_walk_all_in_progress_returns_root():
    t = tree()
    for _ in range(5):
        mk(t, "draft", t.root, 0.5, in_progress=True)
    assert tree_walk(t, C, no_syn, seeded()) is t.root


def test_walk_follows_chain():
    t = tree()
    ds = [mk(t, "draft", t.root, 0.5, in_progress=True) for _ in range(4)]
    d = mk(t, "draft", t.root, 0.5, keep_going=True)
    r1 = mk(t, "refine", d, 0.6, keep_going=True)
    r2 = mk(t, "refine", r1, 0.7, keep_going=True)
    assert tree_walk(t, C, no_syn, seeded()) is r2
    assert d.in_progress
    assert all(x.in_progress for x in ds)


def test_walk_dead_end():
    t = tree()
    d = mk(t, "draft", t.root, 0.5, dead_end=True)
    for _ in range(4):
        mk(t, "draft", t.root, 0.5, in_progress=True)
    assert tree_walk(t, C, no_syn, seeded()) is d


def elite_tree():
    t = tree()
    a = mk(t, "draft", t.root, 0.95)
    b = mk(t, "draft", t.root, 0.91)
    c = mk(t, "draft", t.root, 0.90)
    nodes = {"A95": a, "B91": b, "C90": c}
    for s in (0.94, 0.93, 0.92):
        nodes[f"A{int(s * 100)}"] = mk(t, "refine", a, s)
    return t, nodes


def test_elite_candidates():
    t, n = elite_tree()
    early = elite_candidates(t, C, 0.5)
    assert early == [n["A95"], n["A94"], n["A93"], n["B91"], n["C90"]]
    late = elite_candidates(t, C, 0.8)
    assert late == [n["A95"], n["A94"], n["B91"]]


def _freq(t, p, n=10000):
    rng = seeded(1)
    cnt = collections.Counter()
    for _ in range(n):
        node, mode = elite_pick(t, C, p, no_syn, rng)
        node.picked_by_elite = False
        cnt[node.id] += 1
    return {k: v / n for k, v in cnt.items()}


def test_elite_probabilities_three():
    t, n = elite_tree()
    f = _freq(t, 0.8)
    for key, prob in (("A95", 0.5455), ("A94", 0.2727), ("B91", 0.1818)):
        assert abs(f[n[key].id] - prob) < 0.02


def test_elite_probabilities_five():
    t, n = elite_tree()
    f = _freq(t, 0.5)
    for key, prob in (("A95", 0.4380), ("A94", 0.2190), ("A93", 0.1460), ("B91", 0.1095), ("C90", 0.0876)):
        assert abs(f[n[key].id] - prob) < 0.02


def test_elite_full_candidate_excluded():
    t, n = elite_tree()
    n["A95"].pending_children = 10
    f = _freq(t, 0.8)
    assert n["A95"].id not in f
    assert abs(f[n["A94"].id] - 0.6) < 0.02
    assert abs(f[n["B91"].id] - 0.4) < 0.02


def test_elite_without_success_is_tree_walk():
    t = tree()
    node, mode = elite_pick(t, C, 0.8, no_syn, seeded())
    assert node is t.root and mode == "tree_walk"
