"""Search tree data structures and score comparison helpers."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .utils import new_id, now_iso

NODE_KINDS = ("root", "draft", "refine", "repair", "retrospect", "borrow", "synthesis","promote")


@dataclass(eq=False)
class Node:
    id: str
    kind: str
    parent: "Node | None" = None
    children: list["Node"] = field(default_factory=list)
    step: int | None = None
    branch_id: int | None = None
    plan: str = ""
    structured_plan: dict[str, Any] | None = None
    code: str = ""
    output: str = ""
    exec_seconds: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    summary: str = ""
    method_summary: str = ""
    score: float | None = None
    score_source: str | None = None
    failed: bool | None = None
    predictions_ok: bool | None = None
    failure_reasons: list[str] = field(default_factory=list)
    visits: int = 0
    reward_sum: float = 0.0
    dead_end: bool = False
    chain_best: "Node | None" = None
    no_gain_count: int = 0
    keep_going: bool = False
    repair_succeeded: bool = False
    in_progress: bool = False
    pending_children: int = 0
    via_elite: bool = False
    picked_by_elite: bool = False
    has_artifacts: bool = False
    val_files_ok: bool | None = None
    parent_gain: float | None = None
    global_gain: float | None = None
    complementarity: float | None = None
    trial_id: str | None = None
    rung: int = 0
    checkpoint: dict | None = None
    artifact_verification: dict | None = None
    continuation_verification: dict | None = None
    smoke: str | None = None
    created_at: str = field(default_factory=now_iso)
    finished_at: str | None = None
    created_mono: float = field(default_factory=time.monotonic)
    in_tree: bool = False

    @property
    def is_root(self) -> bool:
        return self.kind == "root"

    @property
    def ok(self) -> bool:
        """A successful node: evaluated and not failed (it always has a score)."""
        return self.failed is False

    @property
    def is_failed(self) -> bool:
        return self.failed is True

    @property
    def evaluated(self) -> bool:
        return self.failed is not None

    @property
    def repair_depth(self) -> int:
        depth = 0
        p = self.parent
        while p is not None and p.is_failed:
            depth += 1
            p = p.parent
        return depth

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Node({self.id[:6]} {self.kind} b{self.branch_id} score={self.score} failed={self.failed})"


class Direction:
    """Score comparison that respects the metric direction."""

    def __init__(self, higher_is_better: bool) -> None:
        self.higher_is_better = bool(higher_is_better)

    def better(self, a: float | None, b: float | None) -> bool:
        if a is None:
            return False
        if b is None:
            return True
        return a > b if self.higher_is_better else a < b

    def gain(self, new: float, old: float) -> float:
        return new - old if self.higher_is_better else old - new

    def sort_key(self, score: float) -> float:
        return -score if self.higher_is_better else score


class SearchTree:
    """All nodes plus indexes. Every access from worker threads must hold ``lock``."""

    def __init__(self, direction: Direction) -> None:
        self.lock = threading.RLock()
        self.direction = direction
        self.root = Node(id=new_id(), kind="root", step=0, in_tree=True)
        self.nodes: list[Node] = [self.root]
        self.by_id: dict[str, Node] = {self.root.id: self.root}
        self.branch_all: dict[int, list[Node]] = {}
        self.branch_ok: dict[int, list[Node]] = {}
        self.branch_finished_count: dict[int, int] = {}
        self.best: Node | None = None
        self.top_list: list[Node] = []
        self.synthesis_count = 0
        # Set by the search before each step: False when the usable time left is too short for a synthesis
        # step (it writes a full script, then runs it).
        self.synthesis_time_ok = True
        self.search_start: float | None = None
        self._next_branch = 1
        self.score_timeline: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ creation
    def new_branch_id(self) -> int:
        with self.lock:
            b = self._next_branch
            self._next_branch += 1
            return b

    def attach(self, node: Node, parent: Node) -> None:
        """Hang a freshly generated node under ``parent`` and register it in branch_all."""
        with self.lock:
            node.parent = parent
            parent.children.append(node)
            self.by_id[node.id] = node
            assert node.branch_id is not None
            self.branch_all.setdefault(node.branch_id, []).append(node)

    def append(self, node: Node) -> None:
        """Add an evaluated node to the ordered node list (gives it its step number)."""
        with self.lock:
            node.step = len(self.nodes)
            node.in_tree = True
            self.nodes.append(node)

    def discard(self, node: Node) -> None:
        """Remove a created node that will never be added to the tree."""
        with self.lock:
            if node.parent is not None and node in node.parent.children:
                node.parent.children.remove(node)
            if node.branch_id is not None and node in self.branch_all.get(node.branch_id, []):
                self.branch_all[node.branch_id].remove(node)
            if node in self.branch_ok.get(node.branch_id or -1, []):
                self.branch_ok[node.branch_id].remove(node)  # type: ignore[index]
            self.by_id.pop(node.id, None)

    def register_ok(self, node: Node) -> None:
        with self.lock:
            assert node.branch_id is not None
            self.branch_ok.setdefault(node.branch_id, []).append(node)

    # ------------------------------------------------------------------ queries
    def solution_nodes(self) -> list[Node]:
        with self.lock:
            return [n for n in self.nodes if not n.is_root]

    def ok_nodes(self) -> list[Node]:
        with self.lock:
            return [n for n in self.nodes if n.ok and n.score is not None]

    def branches(self) -> list[int]:
        with self.lock:
            return sorted(self.branch_all)

    def branch_best(self, b: int) -> Node | None:
        with self.lock:
            best: Node | None = None
            for n in self.branch_ok.get(b, []):
                if best is None or self.direction.better(n.score, best.score):
                    best = n
            return best

    def sorted_ok(self, nodes: list[Node] | None = None) -> list[Node]:
        """Successful nodes ordered best first; ties keep the earlier node first."""
        with self.lock:
            pool = [n for n in (nodes if nodes is not None else self.ok_nodes()) if n.ok and n.score is not None]
            return sorted(pool, key=lambda n: (self.direction.sort_key(n.score), n.step if n.step is not None else 10**9))  # type: ignore[arg-type]
