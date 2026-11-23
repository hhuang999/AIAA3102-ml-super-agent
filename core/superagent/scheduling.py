"""Operator scheduling, plateau detection and the synthesis condition."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Mapping

from .selection import is_full
from .tree import Node, SearchTree


@dataclass
class Dispatch:
    action: str                        # draft | refine | repair | retrospect | borrow | synthesis | none
    reason: str
    details: dict[str, Any] = field(default_factory=dict)
    via_elite: bool = False


def _after_best_finished(n: Node, best: Node) -> bool:
    if n.created_at and best.finished_at:
        return n.created_at >= best.finished_at
    return True


def attempts_since_branch_best(tree: SearchTree, branch: int) -> int | None:
    """Evaluated nodes (successful or failed) of ``branch`` after its best node; None without a success."""
    with tree.lock:
        best = tree.branch_best(branch)
        if best is None or best.step is None:
            return None
        return sum(
            1 for n in tree.nodes
            if n.branch_id == branch and n.step is not None and n.step > best.step and _after_best_finished(n, best)
        )


def branch_plateau(tree: SearchTree, branch: int, window: int, enabled: bool = True, count_failed: bool = False) -> bool:
    if not enabled:
        return False
    if count_failed:
        since = attempts_since_branch_best(tree, branch)
        return since is not None and window > 0 and since >= window
    with tree.lock:
        ok = tree.branch_ok.get(branch, [])
        if not ok:
            return False
        recent = ok[-window:] if window > 0 else []
        if len(recent) < 2:
            return False
        best = tree.branch_best(branch)
        assert best is not None
        return not any(n.score == best.score for n in recent)


def global_plateau(tree: SearchTree, enabled: bool, window: int) -> bool:
    if not enabled:
        return False
    with tree.lock:
        if tree.best is None or tree.best.step is None:
            return False
        latest = tree.nodes[-1].step or 0
        return latest - tree.best.step >= window


def synthesis_representatives(tree: SearchTree) -> list[Node]:
    with tree.lock:
        reps = [b for b in (tree.branch_best(br) for br in tree.branches()) if b is not None]
        return tree.sorted_ok(reps)


def synthesis_pending(tree: SearchTree) -> int:
    """Synthesis steps reserved or running but not yet evaluated."""
    with tree.lock:
        done = sum(1 for n in tree.nodes if n.kind == "synthesis")
        return max(0, tree.synthesis_count - done)


def synthesis_blocker(tree: SearchTree, cfg: Mapping[str, Any], p: float | None) -> str | None:
    """The first unmet synthesis condition, or None when synthesis may run."""
    syn = cfg["operators"]["synthesis"]
    if not syn["enabled"]:
        return "disabled"
    if p is None:
        return "search_not_started"
    with tree.lock:
        if tree.synthesis_count >= int(syn["max_count"]):
            return "max_count_reached"
        lo, hi = syn["window"]
        if not (float(lo) <= p <= float(hi)):
            return "outside_time_window"
        if not tree.synthesis_time_ok:
            return "not_enough_time"
        rich = [b for b in tree.branches() if len(tree.branch_ok.get(b, [])) >= int(syn["min_ok_per_branch"])]
        if len(rich) < int(syn["min_branches"]):
            return "too_few_branches_with_successes"
        pl = cfg["plateau"]
        if not global_plateau(tree, bool(pl["enabled"]), int(pl["global_window"])):
            return "no_global_plateau"
        if len(synthesis_representatives(tree)) < 2:
            return "too_few_representatives"
    return None


def synthesis_conditions(tree: SearchTree, cfg: Mapping[str, Any], p: float | None) -> bool:
    syn = cfg["operators"]["synthesis"]
    if not syn["enabled"] or p is None:
        return False
    with tree.lock:
        if tree.synthesis_count >= int(syn["max_count"]):
            return False
        lo, hi = syn["window"]
        if not (float(lo) <= p <= float(hi)):
            return False
        if not tree.synthesis_time_ok:
            return False
        rich = [b for b in tree.branches() if len(tree.branch_ok.get(b, [])) >= int(syn["min_ok_per_branch"])]
        if len(rich) < int(syn["min_branches"]):
            return False
        pl = cfg["plateau"]
        if not global_plateau(tree, bool(pl["enabled"]), int(pl["global_window"])):
            return False
        return len(synthesis_representatives(tree)) >= 2


def dispatch(
    tree: SearchTree,
    node: Node,
    cfg: Mapping[str, Any],
    p: float | None,
    rng: random.Random,
    via_elite: bool | None = None,
) -> Dispatch:
    """Choose the operator for ``node``. Reserves capacity under the tree lock:
    a creating action increments ``node.pending_children`` (or ``synthesis_count``)."""
    elite = bool(node.picked_by_elite if via_elite is None else via_elite)
    with tree.lock:
        if node.dead_end:
            return Dispatch("none", "dead_end", via_elite=elite)
        if node.is_root:
            if bool(cfg["operators"]["synthesis"]["trigger_on_stagnation"]) and synthesis_conditions(tree, cfg, p):
                tree.synthesis_count += 1
                return Dispatch("synthesis", "synthesis_conditions_met")
            if not is_full(node, cfg):
                node.pending_children += 1
                return Dispatch("draft", "root_has_capacity")
            if synthesis_conditions(tree, cfg, p):
                tree.synthesis_count += 1
                return Dispatch("synthesis", "synthesis_conditions_met")
            return Dispatch("none", "root_full_no_synthesis")
        if node.is_failed or node.predictions_ok is False:
            node.pending_children += 1
            return Dispatch("repair", "node_failed", via_elite=elite)
        pl = cfg["plateau"]
        window = int(pl["elite_branch_window"] if elite else pl["branch_window"])
        details: dict[str, Any] = {"plateau_window": window}
        if branch_plateau(tree, node.branch_id or 0, window, bool(pl["enabled"]), bool(pl["count_failed_attempts"])):
            ops = cfg["operators"]
            borrow_ok = bool(ops["borrow"]["enabled"]) and p is not None and p >= float(ops["borrow"]["start_fraction"])
            retro_ok = bool(ops["retrospect"]["enabled"])
            node.pending_children += 1
            if borrow_ok and retro_ok:
                action = "borrow" if rng.random() < float(ops["borrow"]["probability"]) else "retrospect"
                return Dispatch(action, "branch_plateau", details, via_elite=elite)
            if borrow_ok:
                return Dispatch("borrow", "branch_plateau", details, via_elite=elite)
            if retro_ok:
                return Dispatch("retrospect", "branch_plateau", details, via_elite=elite)
            return Dispatch("refine", "plateau_no_alternative", details, via_elite=elite)
        node.pending_children += 1
        return Dispatch("refine", "branch_progressing", details, via_elite=elite)
