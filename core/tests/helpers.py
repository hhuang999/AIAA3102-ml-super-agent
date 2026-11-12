"""Helpers to build search trees by hand in tests."""

from __future__ import annotations

import copy
import random

from superagent.config import load_defaults
from superagent.tree import Direction, Node, SearchTree
from superagent.utils import new_id

DEFAULTS = load_defaults()


def cfg(**over):
    c = copy.deepcopy(DEFAULTS)
    for path, value in over.items():
        cur = c
        keys = path.split("__")
        for k in keys[:-1]:
            cur = cur[k]
        cur[keys[-1]] = value
    return c


def tree(higher=True) -> SearchTree:
    return SearchTree(Direction(higher))


def mk(t: SearchTree, kind: str, parent: Node, score=None, failed=False, append=True, **kw) -> Node:
    n = Node(id=new_id(), kind=kind)
    if parent.is_root:
        n.branch_id = t.new_branch_id()
        n.chain_best = t.root
    else:
        n.branch_id = parent.branch_id
        n.chain_best = parent.chain_best
    t.attach(n, parent)
    n.failed = failed
    n.score = None if failed else score
    if failed is False:
        t.register_ok(n)
    for k, v in kw.items():
        setattr(n, k, v)
    if append and failed is not None:
        t.append(n)
    return n


class FixedRng:
    """Returns the given values from random(), cycling; choice() picks the first element."""

    def __init__(self, *values: float) -> None:
        self.values = list(values) or [0.0]
        self.i = 0

    def random(self) -> float:
        v = self.values[self.i % len(self.values)]
        self.i += 1
        return v

    def choice(self, seq):
        return seq[0]

    def shuffle(self, seq):
        pass


def seeded(seed=0):
    return random.Random(seed)


def make_ctx(tmp_path, higher=True, cfg_over=None, sample=True):
    """A RunContext without an LLM gateway, for testing code that makes no LLM call."""
    import time
    from pathlib import Path

    import pandas as pd

    from superagent.context import LockedRandom, OutputPaths, RunContext
    from superagent.executor import Executor

    c = cfg(**(cfg_over or {}))
    paths = OutputPaths(Path(tmp_path) / "out")
    paths.make_dirs()
    if sample:
        pd.DataFrame({"id": [f"r{i}" for i in range(100)], "label": [0] * 100}).to_csv(paths.prepared / "sample_predictions.csv", index=False)
    ex = Executor(paths.runs, paths.prepared, 1, 1, 0, {})
    t = SearchTree(Direction(higher))
    return RunContext(cfg=c, paths=paths, task_dir=Path(tmp_path), direction=t.direction, direction_source="cli", tree=t,
                      gateway=None, executor=ex, experience=None, task_desc="Classify.", data_preview="preview",
                      catalog_text="", modalities=["tabular"], sample_path=(paths.prepared / "sample_predictions.csv") if sample else None,
                      rng=LockedRandom(0), process_start=time.monotonic(), started_at="now")
