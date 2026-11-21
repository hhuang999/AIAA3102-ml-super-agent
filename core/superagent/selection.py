"""Node selection: tree_walk (upper confidence bound) and elite_pick."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .tree import Node, SearchTree

INF = float("inf")


@dataclass
class Selection:
    node: Node
    mode: str                 # tree_walk | elite_pick
    explore_weight: float | None


def explore_weight(p: float, cfg: Mapping[str, Any]) -> float:
    ex = cfg["search"]["explore"]
    start, end, wmin = float(ex["switch_start"]), float(ex["switch_end"]), float(ex["min_weight"])
    if p < start:
        return 1.0
    if p < end:
        return 1.0 - (1.0 - wmin) * (p - start) / (end - start)
    return wmin


def uct_c(t: int, cfg: Mapping[str, Any]) -> float:
    s = cfg["search"]
    u = s["uct"]
    r1, r2 = u["phase_ratios"]
    max_steps = int(cfg["run"]["max_steps"])
    t1 = min(int(s["max_drafts"]) * int(s["max_refinements_per_node"]) ** 2, round(max_steps * float(r1)))
    t2 = round(max_steps * float(r2))
    c_init, c_min, decay = float(u["c_init"]), float(u["c_min"]), float(u["decay_per_step"])
    if t < t1:
        return c_init
    if t <= t2:
        return max(c_init - decay * (t - t1), c_min)
    return c_min


def uct_value(child: Node, parent_visits: int, c: float) -> float:
    if child.visits == 0:
        return INF
    explore = math.sqrt(math.log(parent_visits) / child.visits) if parent_visits > 0 else 0.0
    return child.reward_sum / child.visits + c * explore


def is_full(n: Node, cfg: Mapping[str, Any]) -> bool:
    s = cfg["search"]
    if n.is_root:
        drafts = sum(1 for c in n.children if c.kind == "draft")
        return drafts + n.pending_children >= int(s["max_drafts"])
    if n.is_failed:
        if any(c.ok for c in n.children):
            return True
        return len(n.children) + n.pending_children >= int(s["max_repairs_per_node"])
    normal = sum(1 for c in n.children if not c.via_elite)
    return normal + n.pending_children >= int(s["max_refinements_per_node"])


def has_elite_capacity(n: Node, cfg: Mapping[str, Any]) -> bool:
    return len(n.children) + n.pending_children < int(cfg["search"]["elite_max_children"])


def best_child(tree: SearchTree, n: Node, c: float, rng: random.Random) -> Node:
    """Child with the highest UCB value; returns ``n`` itself when there is none to pick."""
    cands = [ch for ch in n.children if ch.evaluated]
    if n.is_root:
        cands = [ch for ch in cands if not ch.in_progress]
    if not cands:
        return n
    values = [uct_value(ch, n.visits, c) for ch in cands]
    top = max(values)
    best = [ch for ch, v in zip(cands, values) if v == top]
    chosen = best[0] if len(best) == 1 else rng.choice(best)
    if n.is_root:
        chosen.in_progress = True
    return chosen


def walk_from(
    tree: SearchTree,
    start: Node,
    cfg: Mapping[str, Any],
    synthesis_ok: Callable[[], bool],
    rng: random.Random,
) -> Node:
    """The tree_walk descent rules, starting at ``start``. Caller holds the tree lock."""
    c = uct_c(len(tree.nodes), cfg)
    n = start
    synthesis_enabled = bool(cfg["operators"]["synthesis"]["enabled"])
    for _ in range(100_000):
        if n.dead_end:
            return n
        if not is_full(n, cfg):
            if n.is_failed and n.repair_succeeded:
                nxt = best_child(tree, n, c, rng)
            elif n.keep_going and n.children:
                nxt = best_child(tree, n, c, rng)
            else:
                return n
            if nxt is n:
                return n
            n = nxt
            continue
        if n.is_root and synthesis_enabled and synthesis_ok():
            return n
        nxt = best_child(tree, n, c, rng)
        if nxt is n:
            return n
        n = nxt
    return n  # pragma: no cover - safety net


def tree_walk(tree: SearchTree, cfg: Mapping[str, Any], synthesis_ok: Callable[[], bool], rng: random.Random) -> Node:
    with tree.lock:
        return walk_from(tree, tree.root, cfg, synthesis_ok, rng)


def elite_candidates(tree: SearchTree, cfg: Mapping[str, Any], p: float) -> list[Node]:
    e = cfg["search"]["elite"]
    late = p >= float(cfg["search"]["explore"]["switch_end"])
    k = int(e["late_k"] if late else e["early_k"])
    per_branch = int(e["late_per_branch"] if late else e["early_per_branch"])
    out: list[Node] = []
    used: dict[int, int] = {}
    for n in tree.sorted_ok():
        b = n.branch_id or 0
        if used.get(b, 0) >= per_branch:
            continue
        used[b] = used.get(b, 0) + 1
        out.append(n)
        if len(out) >= k:
            break
    return out


def weighted_rank_choice(cands: list[tuple[Node, int]], rng: random.Random) -> Node:
    weights = [1.0 / rank for _n, rank in cands]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for (node, _rank), w in zip(cands, weights):
        acc += w
        if r < acc:
            return node
    return cands[-1][0]


def elite_pick(
    tree: SearchTree, cfg: Mapping[str, Any], p: float, synthesis_ok: Callable[[], bool], rng: random.Random
) -> tuple[Node, str]:
    """Returns (node, mode). Falls back to tree_walk when there are no candidates."""
    with tree.lock:
        cands = elite_candidates(tree, cfg, p)
        if not cands:
            return walk_from(tree, tree.root, cfg, synthesis_ok, rng), "tree_walk"
        ranked = [(n, i + 1) for i, n in enumerate(cands)]
        expandable = [(n, r) for n, r in ranked if has_elite_capacity(n, cfg)]
        if expandable:
            chosen = weighted_rank_choice(expandable, rng)
        else:
            start = weighted_rank_choice(ranked, rng)
            chosen = walk_from(tree, start, cfg, synthesis_ok, rng)
        return chosen, "elite_pick"


def select(
    tree: SearchTree, cfg: Mapping[str, Any], p: float | None, synthesis_ok: Callable[[], bool], rng: random.Random
) -> Selection:
    if p is None:
        return Selection(tree_walk(tree, cfg, synthesis_ok, rng), "tree_walk", None)
    w = explore_weight(p, cfg)
    if rng.random() < w:
        return Selection(tree_walk(tree, cfg, synthesis_ok, rng), "tree_walk", w)
    node, mode = elite_pick(tree, cfg, p, synthesis_ok, rng)
    return Selection(node, mode, w)
