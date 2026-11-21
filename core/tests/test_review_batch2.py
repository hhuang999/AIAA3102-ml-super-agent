"""Unit tests for Batch 2 code review fixes (A-P2, scoped B-P1 & B-P2)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from superagent.bookkeeping import save_experience
from superagent.catalog import CatalogState, MethodMatch, match_methods
from superagent.chain import step_reward
from superagent.config import load_defaults
from superagent.context import LockedRandom, OutputPaths, RunContext
from superagent.control.adaptive import Proposal, Snapshot, rank_proposals
from superagent.ensemble import _align
from superagent.executor import _kill_group
from superagent.experience import ExperienceStore, Record, decision_context, effect_label, tokenize
from superagent.final import _spec_encoding, build_final, stratified_halves
from superagent.helper_api import reference as helper_reference
from superagent.llm.ledger import CancelledCall, Ledger
from superagent.operations import Dispatch, Operations
from superagent.operators import Operators
from superagent.planner import plan_experiment
from superagent.profiler import TREE_LINE_COUNT_MAX_BYTES, build_tree, detect_modalities
from superagent.report import OPERATORS, build_summary
from superagent.review import review
from superagent.scheduling import attempts_since_branch_best
from superagent.tree import Direction, Node, SearchTree


def test_profiler_skips_large_line_counts_and_survives_corrupt_tfrec(tmp_path, monkeypatch):
    big = tmp_path / "huge.csv"
    big.write_text("a,b\n1,2\n", encoding="utf-8")
    small = tmp_path / "small.csv"
    small.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    bad_tfr = tmp_path / "broken.tfrec"
    bad_tfr.write_bytes(b"not-a-valid-tfrecord")

    orig_stat = Path.stat

    class FakeStat:
        def __init__(self, real, fake_size):
            self._r = real
            self.st_size = fake_size
        def __getattr__(self, name):
            return getattr(self._r, name)

    def fake_stat(self, *a, **kw):
        st = orig_stat(self, *a, **kw)
        if self == big:
            return FakeStat(st, TREE_LINE_COUNT_MAX_BYTES + 1024)
        return st

    monkeypatch.setattr(Path, "stat", fake_stat)
    counted = []
    import superagent.profiler as prof_mod
    real_count = prof_mod._count_lines

    def spy_count(p):
        counted.append(p.name)
        return real_count(p)

    monkeypatch.setattr(prof_mod, "_count_lines", spy_count)
    tree_text, _ = build_tree(tmp_path)
    assert "small.csv (3 lines)" in tree_text
    assert "huge.csv (50.0 MB)" in tree_text
    assert "huge.csv" not in counted

    # Corrupt .tfrec must not crash detect_modalities
    mods = detect_modalities(tmp_path, load_defaults()["profiler"])
    assert isinstance(mods, list)


def test_ensemble_align_rejects_stripped_duplicate_ids():
    f1 = pd.DataFrame({"id": ["1", "2"], "p": ["0.9", "0.1"]})
    # Stripped ids in f2 are both "1" (duplicate after strip!), so ID alignment must fall back cleanly to 2 rows
    f2 = pd.DataFrame({"id": ["1", " 1 "], "p": ["0.8", "0.2"]})
    aligned = _align([f1, f2], "id")
    assert len(aligned[0]) == 2
    assert len(aligned[1]) == 2


def test_report_includes_promote_and_operations_records_step_timing(tmp_path):
    assert "promote" in OPERATORS
    cfg = load_defaults()
    ledger = Ledger(tmp_path / "operations.sqlite", run_id="run-timing")
    ledger.register_operation("op-p", "sp", ())
    paths = OutputPaths(tmp_path / "out")
    paths.make_dirs()
    ctx = RunContext(
        cfg=cfg, paths=paths, task_dir=tmp_path, direction=Direction(True), direction_source="cli",
        tree=SearchTree(Direction(True)), gateway=SimpleNamespace(ledger=ledger, endpoint_failure=None, cooldown_until=0.0),
        executor=SimpleNamespace(run_dir=lambda nid: tmp_path / "runs" / nid),
        experience=None, task_desc="t", data_preview="p", catalog_text="c", modalities=[],
        sample_path=None, rng=LockedRandom(1), process_start=time.monotonic() - 2.0, started_at="2026-01-01",
    )
    ctx.task_spec = SimpleNamespace(signature="sig1", metric="accuracy")
    ctx.flags = {"formal": True, "active_parent_leases": set(), "artifact_leases": set()}
    ops = Operations(SimpleNamespace(ctx=ctx))

    n1 = Node(id="n1", kind="promote", parent=ctx.tree.root, branch_id=1, score=0.88, failed=False, exec_seconds=1.5)
    prop = Proposal("pp", ctx.tree.root.id, "promote", "promote test", "generated", True, 0)
    disp = Dispatch("op-p", prop, "sp", "dp", node=n1, started=time.monotonic() - 2.5)
    fut = Future()
    fut.set_result((n1, {"score_verified": True, "deliverable": True}))

    import superagent.single_model as sm
    orig_finish = sm.finish
    try:
        # Run _complete end-to-end with minimal finish hook
        sm.finish = lambda _ctx, nd, _det, rec: (ctx.tree.append(nd), rec())
        ops._complete(disp, fut)
    finally:
        sm.finish = orig_finish

    assert len(ctx.step_timings) == 1
    assert ctx.step_timings[0].kind == "promote"
    assert ctx.step_timings[0].ok is True
    assert ctx.step_timings[0].exec_seconds == 1.5
    summary = build_summary(ctx)
    assert summary["actions"]["promote"]["count"] == 1
    assert summary["diagnostics"]["throughput"]["mean_step_seconds"] is not None


def test_repairable_id_retained_on_cancelled_repair_and_cleared_on_completion(tmp_path):
    cfg = load_defaults()
    cfg["search"]["max_repairs_per_node"] = 1
    ledger = Ledger(tmp_path / "operations.sqlite", run_id="run-rep")
    ledger.register_operation("op-c", "sc", ("repair_coder",))
    ledger.register_operation("op-ok", "sok", ("repair_coder",))
    paths = OutputPaths(tmp_path / "out")
    paths.make_dirs()
    ctx = RunContext(
        cfg=cfg, paths=paths, task_dir=tmp_path, direction=Direction(True), direction_source="cli",
        tree=SearchTree(Direction(True)), gateway=SimpleNamespace(ledger=ledger, endpoint_failure=None, cooldown_until=0.0),
        executor=SimpleNamespace(run_dir=lambda nid: tmp_path / "runs" / nid),
        experience=None, task_desc="t", data_preview="p", catalog_text="c", modalities=[],
        sample_path=None, rng=LockedRandom(1), process_start=time.monotonic(), started_at="2026-01-01",
    )
    ctx.task_spec = SimpleNamespace(signature="sig1", metric="accuracy")
    ctx.flags = {"formal": True, "repairable_id": "bad_parent", "active_parent_leases": set(), "artifact_leases": {"bad_parent"}}
    bad_parent = Node(id="bad_parent", kind="draft", parent=ctx.tree.root, branch_id=1, score=0.9, failed=True)
    ctx.tree.by_id["bad_parent"] = bad_parent
    ops = Operations(SimpleNamespace(ctx=ctx))

    # 1. Cancelled repair MUST keep repairable_id intact
    p1 = Proposal("pr1", "bad_parent", "repair", "fix", "generated", False, 0, target="predict_only")
    d1 = Dispatch("op-c", p1, "sc", "dc", started=time.monotonic())
    f1 = Future()
    f1.set_exception(CancelledCall("cancelled"))
    ops._complete(d1, f1)
    assert ctx.flags.get("repairable_id") == "bad_parent"

    # 2. Completed repair MUST clear repairable_id
    fixed = Node(id="fixed_node", kind="repair", parent=bad_parent, branch_id=1, score=0.9, failed=False)
    p2 = Proposal("pr2", "bad_parent", "repair", "fix", "generated", False, 1, target="predict_only")
    d2 = Dispatch("op-ok", p2, "sok", "dok", node=fixed, started=time.monotonic())
    f2 = Future()
    f2.set_result((fixed, {"score_verified": True, "deliverable": True}))
    import superagent.single_model as sm
    orig_finish = sm.finish
    try:
        sm.finish = lambda _ctx, nd, _det, rec: (ctx.tree.append(nd), rec())
        ops._complete(d2, f2)
    finally:
        sm.finish = orig_finish
    assert "repairable_id" not in ctx.flags


def test_executor_kill_group_fallback_without_killpg(monkeypatch):
    calls = []

    class FakeProc:
        def terminate(self):
            calls.append("terminate")
        def kill(self):
            calls.append("kill")

    monkeypatch.delattr(os, "killpg", raising=False)
    _kill_group(12345, signal.SIGINT, FakeProc())
    _kill_group(12345, signal.SIGTERM, FakeProc())
    assert calls == ["terminate", "kill"]


def test_experience_tokenizer_effect_labels_and_richer_decision_context(tmp_path):
    # 1. Tokenizer handles punctuation, snake_case, CamelCase and CJK
    toks = tokenize("lightgbm, learning_rate=1e-3, StandardScaler 中文分类")
    for expected in ("lightgbm", "learning", "rate", "standardscaler", "standard", "scaler", "中", "文", "分", "类"):
        assert expected in toks

    # 2. effect_label grades drafts vs reference_score and failed repairs as -1
    better = lambda a, b: a > b
    assert effect_label("draft", 0.70, None, better, reference_score=0.80) == -1
    assert effect_label("draft", 0.85, None, better, reference_score=0.80) == 1
    assert effect_label("repair", None, None, better) == -1
    assert effect_label("repair", 0.80, None, better) == 1

    # 3. save_experience grades weak drafts negatively once >=2 prior ok nodes exist
    cfg = load_defaults()
    cfg["experience"]["vector"]["enabled"] = False
    store = ExperienceStore(cfg["experience"], tmp_path / "rec.json", embedder=None, background=False)
    tree = SearchTree(Direction(True))
    n1 = Node(id="n1", kind="draft", parent=tree.root, branch_id=1, plan="baseline A", score=0.80, failed=False, step=1)
    n2 = Node(id="n2", kind="draft", parent=tree.root, branch_id=2, plan="baseline B", score=0.84, failed=False, step=2)
    n3 = Node(id="n3", kind="draft", parent=tree.root, branch_id=3, plan="weak draft C", score=0.55, failed=False, step=3)
    tree.nodes.extend([n1, n2, n3])
    ctx = SimpleNamespace(cfg=cfg, tree=tree, direction=Direction(True), experience=store)
    save_experience(ctx, n3)
    assert store.records[-1].effect == -1

    # 4. decision_context retrieves up to 2 positives + 2 negatives including cross-rung same-trial & failed child
    parent = {
        "seq": 1, "operation_id": "op-p", "node_id": "p", "parent_id": "root",
        "action": "draft", "hypothesis": "ResNet baseline", "method_summary": "resnet18",
        "target": None, "trial_id": "t1", "rung": 0, "split_id": "s", "metric": "accuracy",
        "failed_stage": None, "status": "completed", "score": 0.80, "gain": None, "branch_id": 1,
    }
    facts = [
        parent,
        {
            "seq": 2, "operation_id": "op-g1", "node_id": "g1", "parent_id": "p",
            "action": "refine", "hypothesis": "cosine lr", "method_summary": "cosine schedule",
            "target": "train_eval", "trial_id": "t2", "rung": 0, "split_id": "s", "metric": "accuracy",
            "failed_stage": None, "status": "completed", "score": 0.84, "gain": 0.04, "branch_id": 1,
        },
        {
            "seq": 3, "operation_id": "op-g2", "node_id": "g2", "parent_id": "p",
            "action": "promote", "hypothesis": "promoted rung 1", "method_summary": "resnet18 epoch 4",
            "target": None, "trial_id": "t1", "rung": 1, "split_id": "s", "metric": "accuracy",
            "failed_stage": None, "status": "completed", "score": 0.86, "gain": 0.06, "branch_id": 1,
        },
        {
            "seq": 4, "operation_id": "op-b1", "node_id": "b1", "parent_id": "p",
            "action": "refine", "hypothesis": "bad aug", "method_summary": "cutmix crash",
            "target": "data_prep", "trial_id": "t3", "rung": 0, "split_id": "s", "metric": "accuracy",
            "failed_stage": "executed", "reason": "ValueError shape", "status": "failed", "score": None, "gain": None, "branch_id": 1,
        },
        {
            "seq": 5, "operation_id": "op-b2", "node_id": "b2", "parent_id": "other",
            "action": "refine", "hypothesis": "over-regularized", "method_summary": "dropout 0.9",
            "target": "model_def", "trial_id": "t4", "rung": 0, "split_id": "s", "metric": "accuracy",
            "failed_stage": None, "status": "completed", "score": 0.78, "gain": -0.02, "branch_id": 1,
        },
    ]
    prop = Proposal("pr", "p", "refine", "improve resnet training", "generated", False, 5)
    ev = decision_context(facts, prop, store=None)
    ev_ids = {r["operation_id"] for r in ev}
    assert {"op-p", "op-g1", "op-g2", "op-b1", "op-b2"} <= ev_ids


def test_step_reward_concurrent_plateau_and_promote_probe_tier():
    # 1. step_reward distinguishes small gain from large gain smoothly in [0.75, 1.5]
    tree = SearchTree(Direction(True))
    b = Node(id="b", kind="draft", parent=tree.root, branch_id=1, score=0.80, failed=False, step=1,
             created_at="2026-11-21T17:00:00+08:00", finished_at="2026-11-21T17:05:00+08:00")
    tree.nodes.append(b)
    tree.branch_ok[1] = [b]

    small = Node(id="s", kind="refine", parent=b, branch_id=1, score=0.8005, failed=False, step=2)
    large = Node(id="l", kind="refine", parent=b, branch_id=1, score=0.86, failed=False, step=3)
    assert 0.75 < step_reward(small, tree, 0.0001) < step_reward(large, tree, 0.0001) == 1.5

    # 2. attempts_since_branch_best ignores concurrent sibling started before best finished
    inflight_sib = Node(id="sib", kind="refine", parent=b, branch_id=1, score=0.79, failed=False, step=2,
                        created_at="2026-11-21T17:01:00+08:00", finished_at="2026-11-21T17:06:00+08:00")
    later_child = Node(id="lat", kind="refine", parent=b, branch_id=1, score=0.795, failed=False, step=3,
                       created_at="2026-11-21T17:06:30+08:00", finished_at="2026-11-21T17:07:00+08:00")
    tree.nodes.extend([inflight_sib, later_child])
    assert attempts_since_branch_best(tree, 1) == 1

    # 3. promote probe uses tier=1
    cfg = load_defaults()
    facts = ({
        "seq": 1, "operation_id": "op1", "node_id": "n1", "parent_id": "root",
        "action": "draft", "method": "tab_mlp", "source": "generated",
        "hypothesis": "mlp", "target": None, "branch_id": 1, "seed": 42,
        "split_id": "s", "metric": "accuracy", "status": "completed", "failed_stage": None,
        "reason": None, "seconds": 10.0, "score": 0.80, "score_verified": True,
        "deliverable": True, "trial_id": "n1", "rung": 0,
        "checkpoint": {
            "signature": "sig1", "sha256": "a" * 64, "epoch": 2, "next_seconds": 8.0,
            "delivery_seconds": 2.0, "can_continue": True, "rung": 0,
            "progress": {"optimizer_steps": 50, "model_sha256": "b" * 64, "optimizer_sha256": "c" * 64},
        },
        "trial_signature": "sig1", "source_sha256": "src1", "gain": None,
    },)
    snap = Snapshot(1, facts, frozenset(), None, 0.80, True, 1, frozenset({"n1"}))
    _props, ranked, _ = rank_proposals(snap, (), cfg, True)
    prom = [r for r in ranked if r["action"] == "promote"]
    assert len(prom) == 1 and prom[0]["tier"] == 1


def test_review_narrow_sensitive_trigger_and_gpu_catalog_ordering(tmp_path):
    cfg = load_defaults()
    cfg["review"]["mode"] = "always"
    cfg["review"]["sensitive_effort"] = "high"

    seen_efforts = []

    class FakeGW:
        ledger = None
        def structured(self, _role, _sys, _usr, _schema, **kw):
            seen_efforts.append(kw.get("initial_effort"))
            return {"verdict": "pass", "findings": "ok", "patch": None}

    parent_code = (
        "from sklearn.model_selection import StratifiedKFold\n"
        "def read_split():\n    return [1, 2]\n"
        "def build_model():\n    return dict(lr=0.01)\n"
        "def predict_only():\n    pass\n"
    )
    parent = Node(id="p", kind="draft", code=parent_code)
    ctx = SimpleNamespace(
        cfg=cfg, gateway=FakeGW(), task_desc="t", mode="off",
        task_spec=object(), current_run_limit=lambda: 100.0, flags={"formal": True},
    )

    # 1. Pure hyperparameter change inside build_model does NOT trigger sensitive_effort
    n_hparam = Node(id="c1", kind="refine", parent=parent,
                    code=parent_code.replace("lr=0.01", "lr=0.005"))
    review(ctx, n_hparam)
    assert seen_efforts[-1] is None

    # 2. Editing predict_only or train_test_split DOES trigger sensitive_effort
    n_split = Node(id="c2", kind="refine", parent=parent,
                   code=parent_code.replace("pass", "return 1"))
    review(ctx, n_split)
    assert seen_efforts[-1] == "high"

    # 3. Hardware-aware catalog ordering puts tab_gbdt ahead of tab_forest & scratch NNs on GPU
    cat = CatalogState(entries=[], text="", missing=[])
    facts = {
        "metric": "roc_auc",
        "capabilities": {
            "modalities": ["tabular"],
            "dependencies": {"scikit-learn": "1.5", "lightgbm": "4.0", "torch": "2.4"},
            "num_gpus": 1,
        },
        "input_manifest": {"splits": {
            "train": {"format": "csv", "file_features": [], "features": ["x1", "x2"]},
            "validation": {"format": "csv", "file_features": [], "features": ["x1", "x2"]},
            "test": {"format": "csv", "file_features": [], "features": ["x1", "x2"]},
        }},
    }
    matches = [m.method_id for m in match_methods(facts, cat) if m.eligible]
    assert "tab_gbdt" in matches and "tab_forest" in matches
    assert matches.index("tab_gbdt") < matches.index("tab_forest")


def test_final_spec_encoding_and_stratified_halves(tmp_path, monkeypatch):
    sp_bin = SimpleNamespace(probability=True, columns=["id", "target"], metric="log_loss")
    sp_multi = SimpleNamespace(probability=True, columns=["id", "c0", "c1"], metric="log_loss")
    sp_lbl = SimpleNamespace(probability=False, columns=["id", "target"], metric="accuracy")
    assert _spec_encoding(sp_bin) == "binary_prob"
    assert _spec_encoding(sp_multi) == "multi_prob"
    assert _spec_encoding(sp_lbl) == "label"

    if os.name == "nt":
        import shutil
        import superagent.publication as pub_mod

        def copy_publish(root_p, directory):
            fin = Path(root_p) / "final"
            if fin.exists():
                shutil.rmtree(fin)
            shutil.copytree(directory, fin)

        monkeypatch.setattr(pub_mod, "publish", copy_publish)

    # Unknown custom metric in spec must not raise StopIteration
    cands = []
    for nid, score in (("n1", 0.9), ("n2", 0.85)):
        d = tmp_path / nid / "out"
        d.mkdir(parents=True)
        pd.DataFrame({"id": [str(i) for i in range(60)], "target": ["a"] * 60}).to_csv(d / "predictions.csv", index=False)
        pd.DataFrame({"id": [str(i) for i in range(60)], "target": ["a"] * 60}).to_csv(d / "val_predictions.csv", index=False)
        pd.DataFrame({"id": [str(i) for i in range(60)], "target": ["a"] * 60}).to_csv(d / "val_labels.csv", index=False)
        cands.append(SimpleNamespace(rank=len(cands) + 1, node_id=nid, score=score, out_dir=d, code="x=1"))
    sp_unknown = SimpleNamespace(
        metric="custom_unknown_metric", classes=["a", "b"], columns=["id", "target"],
        truth=pd.Series(["a"] * 60, index=[str(i) for i in range(60)]),
        test_ids=[str(i) for i in range(60)], probability=False, predictions=lambda *a, **k: None,
    )
    info = build_final(tmp_path / "out", cands, True, load_defaults()["final"], spec=sp_unknown)
    assert info["kind"] == "single"

    # Stratified halves preserve rare-class distribution symmetrically
    rows = [f"r{i}" for i in range(20)]
    truth = pd.DataFrame({"target": ["major"] * 16 + ["rare"] * 4}, index=rows)
    h0, h1 = stratified_halves(rows, truth)
    assert len(h0) == 10 and len(h1) == 10
    assert (truth.loc[h0, "target"] == "rare").sum() == 2
    assert (truth.loc[h1, "target"] == "rare").sum() == 2


def test_prompt_deduplication_and_cpu_tabular_helper_api(tmp_path):
    cfg = load_defaults()
    seen_users = []

    class FakeGW:
        def __init__(self, ldg):
            self.ledger = ldg
        def structured(self, _role, _sys, user, _schema, **_kw):
            seen_users.append(user)
            return {
                "schema_version": 1, "action": "refine", "evidence_ids": ["op1"],
                "hypothesis": "tune depth",
                "changes": [{"id": "c1", "component": "model_def", "target": "fit", "edit": "depth=8", "check": "ok"}],
                "expected_observation": "better", "preserve": ["split"],
            }

    ledger = Ledger(tmp_path / "ops.sqlite", run_id="r-prompt")
    ledger.register_operation("op1", "s1", ("planner.structured",))
    ctx = SimpleNamespace(
        gateway=FakeGW(ledger),
        cfg=cfg,
        task_desc="Task desc with embedded input_manifest",
        data_preview="tabular cols",
        catalog_text="tab_gbdt",
        tree=SearchTree(Direction(True)),
        flags={"task_facts": {
            "metric": "accuracy", "maximize": True,
            "input_manifest": {"SHOULD_NOT_REPEAT_IN_TASK_SUMMARY": True},
            "capabilities": {"modalities": ["tabular"], "num_gpus": 0},
            "methods": [{"method_id": "tab_gbdt", "eligible": True}],
        }},
    )
    parent = Node(id="n1", kind="draft", code="def fit():\n    pass\n", output="ok")
    prop = Proposal("p1", "n1", "refine", "tune", "generated", False, 1)
    plan_experiment(ctx, Dispatch("op1", prop, "s1", "d1"), parent, [{"operation_id": "op1"}])
    assert "SHOULD_NOT_REPEAT_IN_TASK_SUMMARY" not in seen_users[0]
    assert '"eligible_methods": ["tab_gbdt"]' in seen_users[0]

    # CPU tabular rules omit save_torch/restore_torch while keeping save_checkpoint & inference contract
    ref_no_torch = helper_reference(include_torch=False)
    assert "save_torch" not in ref_no_torch and "restore_torch" not in ref_no_torch
    assert "save_checkpoint" in ref_no_torch
    assert "--predict-only" in ref_no_torch

    cpu_ctx = SimpleNamespace(
        cfg=cfg,
        gateway=None,
        executor=SimpleNamespace(num_gpus=0, cores_per_slot=4),
        modalities=["tabular"],
        task_spec=None,
        resources=lambda: SimpleNamespace(
            time_left_secs=600, steps_left=5, run_limit_secs=120, cpus_per_run=4, num_gpus=0,
            mode="off", steps_left_estimated=False, budget_text=False, gpu_memory_gb=None,
            gpu_sharing=1, smoke_seconds=0.0, external_stop=False,
        ),
    )
    rules_cpu = Operators(cpu_ctx)._rules()
    assert "save_torch" not in rules_cpu

    gpu_ctx = SimpleNamespace(
        cfg=cfg,
        gateway=None,
        executor=SimpleNamespace(num_gpus=1, cores_per_slot=4),
        modalities=["image"],
        task_spec=None,
        resources=cpu_ctx.resources,
    )
    rules_gpu = Operators(gpu_ctx)._rules()
    assert "save_torch" in rules_gpu and "restore_torch" in rules_gpu
