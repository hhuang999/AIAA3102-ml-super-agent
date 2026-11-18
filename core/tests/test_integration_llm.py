"""End-to-end tests with the real LLM endpoint (slow, cost a little; need DEEPSEEK_API_KEY)."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [pytest.mark.llm, pytest.mark.slow]

FAST_CFG = """run:
  finalize_reserve_seconds: 20
search:
  initial_drafts: 2
experience:
  vector:
    enabled: false
"""


@pytest.fixture(scope="module")
def small_task(tmp_path_factory):
    out = tmp_path_factory.mktemp("tasks")
    subprocess.run([sys.executable, str(ROOT / "examples" / "make_example_tasks.py"), "--out", str(out), "--tasks", "tabular_small"],
                   check=True, capture_output=True)
    return out / "tabular_small" / "task"


def run_cli(task, out, cfg_path, *extra, timeout=900):
    cmd = [sys.executable, "-m", "superagent.run", "--task-dir", str(task), "--output-dir", str(out), "--config", str(cfg_path),
           "--num-gpus", "0", "--num-cpus", str(max(2, len(os.sched_getaffinity(0)))), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def write_cfg(tmp_path, text=FAST_CFG):
    p = tmp_path / "cfg.yaml"
    p.write_text(text)
    return p


def test_full_run(small_task, tmp_path):
    out = tmp_path / "out"
    r = run_cli(small_task, out, write_cfg(tmp_path), "--max-steps", "6", "--parallel", "2", "--time-limit-secs", "600")
    assert r.returncode == 0, r.stderr[-3000:]
    st = json.loads((out / "status.json").read_text())
    assert st["status"] == "success"
    for f in ("best/solution.py", "best/predictions.csv", "best/score.json"):
        assert (out / f).exists()
    pred = pd.read_csv(out / "best" / "predictions.csv")
    sample = pd.read_csv(small_task / "data" / "sample_predictions.csv")
    assert list(pred.columns) == list(sample.columns) and len(pred) == len(sample)
    tree = json.loads((out / "logs" / "search_tree.json").read_text())
    assert len(tree["nodes"]) >= 3
    for n in tree["nodes"][1:]:
        assert (out / "runs" / n["id"]).is_dir()
    dec_fields = {"seq", "ts", "progress", "selected_node_id", "selection_mode", "explore_weight", "action", "reason", "details", "new_node_id", "outcome"}
    for line in (out / "logs" / "decisions.jsonl").read_text().splitlines():
        assert dec_fields <= set(json.loads(line))
    call_fields = {"ts", "call_id", "attempt", "agent", "node_id", "model", "thinking", "prompt_tokens", "completion_tokens",
                   "reasoning_tokens", "latency_secs", "ok", "error"}
    rows = [json.loads(line) for line in (out / "logs" / "llm_calls.jsonl").read_text().splitlines()]
    assert all(call_fields <= set(r) for r in rows)
    assert sum((r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0) for r in rows) > 0
    summary = json.loads((out / "report" / "run_summary.json").read_text())
    for k in ("interface_version", "task", "run", "best", "score_timeline", "branches", "actions", "key_events",
              "top_solutions", "ensemble", "failures", "resources"):
        assert k in summary
    assert summary["score_timeline"]
    key = os.environ["DEEPSEEK_API_KEY"]
    for p in out.rglob("*"):
        if p.is_file() and p.suffix in (".json", ".jsonl", ".log", ".yaml", ".md"):
            assert key not in p.read_text(errors="ignore")


def test_time_limit(small_task, tmp_path):
    out = tmp_path / "out"
    cfg = write_cfg(tmp_path, FAST_CFG + "codegen:\n  staged_drafting: false\n")
    t0 = time.monotonic()
    r = run_cli(small_task, out, cfg, "--time-limit-secs", "150", timeout=400)
    elapsed = time.monotonic() - t0
    assert elapsed < 150 + 15, elapsed
    st = json.loads((out / "status.json").read_text())
    assert st["stop_reason"] == "time_limit"
    assert r.returncode in (0, 2)
    assert (out / "report" / "run_summary.json").exists() and (out / "logs" / "search_tree.json").exists()


def test_interrupt(small_task, tmp_path):
    out = tmp_path / "out"
    cfg = write_cfg(tmp_path, FAST_CFG + "codegen:\n  staged_drafting: false\n")
    cmd = [sys.executable, "-m", "superagent.run", "--task-dir", str(small_task), "--output-dir", str(out), "--config", str(cfg),
           "--num-gpus", "0", "--num-cpus", "2", "--time-limit-secs", "1200"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    best_before = None
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        time.sleep(3)
        try:
            st = json.loads((out / "status.json").read_text())
        except (OSError, ValueError):
            continue
        if st.get("best_node_id"):
            best_before = st["best_node_id"]
            break
    assert best_before, "no solution found before the interrupt"
    proc.send_signal(signal.SIGINT)
    proc.wait(timeout=120)
    st = json.loads((out / "status.json").read_text())
    assert st["stop_reason"] == "interrupted"
    assert json.loads((out / "best" / "score.json").read_text())["node_id"] == st["best_node_id"]
    assert proc.returncode in (0, 2)


def test_repair_with_real_llm(tmp_path, small_task):
    """A broken draft is run, fails, and the real LLM repairs it through the normal search step."""
    from helpers import cfg as mkcfg
    from superagent.context import LockedRandom, OutputPaths, RunContext
    from superagent.executor import Executor
    from superagent.llm.gateway import LLMGateway, OpenAITransport
    from superagent.profiler import profile
    from superagent.search import SearchController, StepRecord
    from superagent.task_io import prepare_data
    from superagent.tree import Direction, Node, SearchTree

    c = mkcfg(experience__enabled=False)
    paths = OutputPaths(tmp_path / "out")
    paths.make_dirs()
    prepare_data(small_task / "data", paths.prepared)
    gw = LLMGateway(c["llm"], paths.llm_calls, OpenAITransport(c["llm"]["base_url"], os.environ["DEEPSEEK_API_KEY"], 600))
    ex = Executor(paths.runs, paths.prepared, 1, 2, 0, {})
    t = SearchTree(Direction(True))
    ctx = RunContext(cfg=c, paths=paths, task_dir=small_task, direction=t.direction, direction_source="cli", tree=t, gateway=gw,
                     executor=ex, experience=None, task_desc=(small_task / "description.md").read_text(),
                     data_preview=profile(paths.prepared, c["profiler"]).preview, catalog_text="", modalities=["tabular"],
                     sample_path=paths.prepared / "sample_predictions.csv", rng=LockedRandom(0), process_start=time.monotonic(),
                     started_at="now")
    t.search_start = time.monotonic()
    broken = '''import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
X = train.drop(columns=["id", "species"])
y = train["species"]
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
print("split: stratified 80/20, seed 42; metric: accuracy")
raise ValueError("bad")
'''
    d = Node(id="brokendraft0001", kind="draft", plan="Logistic regression on the raw measurements.", code=broken)
    d.branch_id = t.new_branch_id()
    d.chain_best = t.root
    t.attach(d, t.root)
    sc = SearchController(ctx)
    sc._step(StepRecord(selected=t.root, mode="initial", action="draft", reason="root_has_capacity"), t.root, pre_generated=d)
    assert d.is_failed and d.error_type == "ValueError"
    for _ in range(3):
        fail = [n for n in t.nodes if n.is_failed and not n.repair_succeeded and not n.children]
        if not fail:
            break
        sc._step(StepRecord(selected=fail[-1], mode="chain"), fail[-1])
    repairs = [n for n in t.nodes if n.kind == "repair"]
    assert repairs and any(n.ok for n in repairs)
    assert d.repair_succeeded
