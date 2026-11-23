"""Chain control, rewards and backpropagation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping

from .tree import Node, SearchTree


@dataclass
class ChainOutcome:
    forced_return: bool
    returned_to_root: bool
    reward: float | None
    chain_ended_no_gain: bool = False


def reward_for(n: Node) -> float:
    if not n.ok or n.score is None:
        return -1.0
    parent = n.parent
    if parent is None or parent.is_root:
        return 0.0
    if parent.is_failed:
        return 1.5
    return 1.0


def step_reward(n: Node, tree: SearchTree, min_gain: float) -> float:
    """Reward of one evaluated node: -1 failed, 0.5 first success, smooth [0.75, 1.5] new branch best, 0 otherwise."""
    if not n.ok or n.score is None:
        return -1.0
    with tree.lock:
        earlier = [x for x in tree.branch_ok.get(n.branch_id or 0, []) if x is not n and x.score is not None]
        if not earlier:
            return 0.5
        prev = tree.sorted_ok(earlier)[0]
        g = tree.direction.gain(n.score, prev.score)  # type: ignore[arg-type]
        if g < min_gain:
            return 0.0
        rel = g / max(abs(prev.score), 0.05)  # type: ignore[arg-type]
        return min(1.5, 0.75 + max(0.0, rel))


def add_reward(n: Node, value: float) -> None:
    """Count one visit with ``value`` on ``n`` and every ancestor. Caller holds the tree lock."""
    x: Node | None = n
    while x is not None:
        x.visits += 1
        x.reward_sum += value
        x = x.parent


def backpropagate(n: Node, value: float, count_visit: bool) -> None:
    """Walk from ``n`` up to the root (inclusive). Caller holds the tree lock."""
    x: Node | None = n
    while x is not None:
        parent = x.parent
        if parent is not None:
            if x.ok and parent.is_failed:
                parent.repair_succeeded = True
            elif x.is_failed and x.repair_succeeded and parent.is_failed:
                parent.repair_succeeded = True
            if not parent.is_root:
                parent.keep_going = x.keep_going
        if x.kind in ("draft", "synthesis") and x.in_progress:
            x.in_progress = False
        if count_visit:
            x.visits += 1
            x.reward_sum += value
        x = parent


def _finish(n: Node, per_step: bool = False) -> ChainOutcome:
    """End the chain at ``n``. With per-step rewards the visit was already counted."""
    if per_step:
        backpropagate(n, 0.0, False)
        return ChainOutcome(False, True, None)
    r = reward_for(n)
    backpropagate(n, r, True)
    return ChainOutcome(False, True, r)


def chain_control(
    tree: SearchTree, n: Node, cfg: Mapping[str, Any], p: float | None, rng: random.Random
) -> ChainOutcome:
    """Decide whether the chain continues from ``n``. ``n`` is not yet in the node list."""
    s = cfg["search"]
    d = tree.direction
    per_step = str(s["reward_mode"]) == "per_step"
    with tree.lock:
        # Part one: forced return to the root.
        if p is not None and tree.search_start is not None and n.branch_id is not None:
            c = tree.branch_finished_count.get(n.branch_id, 0) + 1
            tree.branch_finished_count[n.branch_id] = c
            fr = s["forced_return"]
            force = False
            if p >= float(fr["late_start"]):
                force = rng.random() < float(fr["late_prob"])
            elif p >= float(fr["mid_start"]) and c % int(fr["mid_every"]) == 0:
                force = True
            if force:
                recent = [x for x in tree.nodes[-int(fr["recent_window"]):] if x.ok and x.score is not None]
                exempt = False
                if recent and n.ok and n.score is not None:
                    best_recent = max(recent, key=lambda x: -d.sort_key(x.score))  # type: ignore[arg-type]
                    exempt = not d.better(best_recent.score, n.score)
                if not exempt:
                    if n.ok and n.score is not None:
                        cb = n.chain_best
                        if cb is None or cb.score is None or d.better(n.score, cb.score):
                            n.chain_best = n
                    out = _finish(n, per_step)
                    out.forced_return = True
                    return out

        # Part two: the normal decision.
        cb = n.chain_best
        parent = n.parent
        prev_no_gain = parent.no_gain_count if parent is not None else (cb.no_gain_count if cb is not None else 0)
        s_score = n.score
        must_return = False
        ended_no_gain = False
        if n.ok:
            if parent is not None and parent.is_failed:
                if s_score is not None:
                    if cb is not None and cb.score is not None and d.better(s_score, cb.score):
                        n.chain_best = n
                    n.keep_going = True
                else:
                    must_return = True
            cb_score = cb.score if cb is not None else None
            if s_score is not None and cb_score is not None:
                g = d.gain(s_score, cb_score)
                if g < float(s["min_gain"]):
                    if prev_no_gain < int(s["max_no_gain"]):
                        n.no_gain_count = prev_no_gain + 1
                        n.keep_going = True
                    else:
                        n.no_gain_count = prev_no_gain + 1
                        n.keep_going = False
                        n.dead_end = True
                        must_return = True
                        ended_no_gain = True
                else:
                    n.no_gain_count = 0
                    n.chain_best = n
                    n.keep_going = True
            elif s_score is not None:
                n.no_gain_count = 0
                n.chain_best = n
                n.keep_going = True
            else:
                must_return = True
        elif n.failed is None:
            must_return = True
        else:
            n.no_gain_count = prev_no_gain
            depth = n.repair_depth
            if depth >= int(s["repair_return_depth"]):
                must_return = True
            if depth >= int(s["max_repair_depth"]):
                n.dead_end = True
        if must_return:
            out = _finish(n, per_step)
            out.chain_ended_no_gain = ended_no_gain
            return out
        return ChainOutcome(False, False, None)
