"""Retrospect, borrow and synthesis with the real LLM on a hand-built tree (slow, small cost)."""

import ast
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [pytest.mark.llm, pytest.mark.slow]

TEMPLATE = '''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.{module} import {cls}
import os, pickle
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
features = [c for c in train.columns if c not in ("id", "species")]
X_tr, X_va, y_tr, y_va = train_test_split(train[features], train["species"], test_size=0.2, random_state=42, stratify=train["species"])
model = {cls}({args}).fit(X_tr, y_tr)
print("split: stratified 80/20, seed 42; metric: accuracy")
score = accuracy_score(y_va, model.predict(X_va))
os.makedirs("./out/model", exist_ok=True)
pickle.dump(model, open("./out/model/model.pkl", "wb"))
pd.DataFrame({{"id": test["id"], "species": model.predict(test[features])}}).to_csv("./out/predictions.csv", index=False)
print(f"VALIDATION_SCORE: {{score:.6f}}")
'''


@pytest.fixture(scope="module")
def ctx_and_tree(tmp_path_factory):
    from helpers import cfg as mkcfg
    from superagent.context import LockedRandom, OutputPaths, RunContext
    from superagent.executor import Executor
    from superagent.llm.gateway import LLMGateway, OpenAITransport
    from superagent.profiler import profile
    from superagent.task_io import prepare_data
    from superagent.tree import Direction, SearchTree

    base = tmp_path_factory.mktemp("ops")
    subprocess.run([sys.executable, str(ROOT / "examples" / "make_example_tasks.py"), "--out", str(base), "--tasks", "tabular_small"],
                   check=True, capture_output=True)
    task = base / "tabular_small" / "task"
    c = mkcfg(experience__enabled=False, review__enabled=False)
    paths = OutputPaths(base / "out")
    paths.make_dirs()
    prepare_data(task / "data", paths.prepared)
    gw = LLMGateway(c["llm"], paths.llm_calls, OpenAITransport(c["llm"]["base_url"], os.environ["DEEPSEEK_API_KEY"], 600))
    t = SearchTree(Direction(True))
    ctx = RunContext(cfg=c, paths=paths, task_dir=task, direction=t.direction, direction_source="cli", tree=t, gateway=gw,
                     executor=Executor(paths.runs, paths.prepared, 1, 2, 0, {}), experience=None,
                     task_desc=(task / "description.md").read_text(), data_preview=profile(paths.prepared, c["profiler"]).preview,
                     catalog_text="", modalities=["tabular"], sample_path=paths.prepared / "sample_predictions.csv",
                     rng=LockedRandom(0), process_start=time.monotonic(), started_at="now")
    t.search_start = time.monotonic()
    from superagent.search import SearchController, StepRecord
    from superagent.tree import Node

    sc = SearchController(ctx)

    def add(kind, parent, module, cls, args, plan):
        n = Node(id=f"{kind}{len(t.by_id):04d}x", kind=kind, plan=plan, code=TEMPLATE.format(module=module, cls=cls, args=args))
        if parent.is_root:
            n.branch_id = t.new_branch_id()
            n.chain_best = t.root
        else:
            n.branch_id, n.chain_best = parent.branch_id, parent.chain_best
        t.attach(n, parent)
        sc._step(StepRecord(selected=parent, mode="initial", action=kind), parent, pre_generated=n)
        assert n.ok, n.summary
        return n

    d1 = add("draft", t.root, "linear_model", "LogisticRegression", "max_iter=1000", "Logistic regression on the raw measurements.")
    r1 = add("refine", d1, "linear_model", "LogisticRegression", "max_iter=1000, C=0.1", "Stronger L2 regularisation.")
    r2 = add("refine", r1, "linear_model", "LogisticRegression", "max_iter=1000, C=10.0", "Weaker regularisation.")
    d2 = add("draft", t.root, "ensemble", "RandomForestClassifier", "n_estimators=200, random_state=0", "Random forest on the raw measurements.")
    return ctx, sc, {"d1": d1, "r1": r1, "r2": r2, "d2": d2}


def check(node, kind):
    assert node is not None and node.kind == kind
    assert node.code.strip(), node.summary
    ast.parse(node.code)
    assert node.plan.strip()


def test_retrospect(ctx_and_tree):
    ctx, sc, n = ctx_and_tree
    out = sc.ops.retrospect(n["r2"])
    check(out.node, "retrospect")
    assert out.details["trajectory_steps"] == 3


def test_borrow(ctx_and_tree):
    ctx, sc, n = ctx_and_tree
    out = sc.ops.borrow(n["r2"])
    check(out.node, "borrow")
    assert out.details["reference_ids"] == [n["d2"].id]


def test_synthesis_runs(ctx_and_tree):
    ctx, sc, n = ctx_and_tree
    ctx.tree.synthesis_count = 1
    out = sc.ops.synthesis(ctx.tree.root)
    check(out.node, "synthesis")
    res = ctx.executor.run(out.node.id, out.node.code, 600)
    assert res.error_type is None, res.output[-2000:]
    assert "VALIDATION_SCORE:" in res.output
