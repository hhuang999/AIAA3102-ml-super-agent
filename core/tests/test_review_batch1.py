"""Unit tests for Batch 1 code review fixes (A-P0, A-P1, B-P0)."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from superagent.artifacts import PROB_ATOL, PROB_RTOL, same_predictions
from superagent.catalog import MethodMatch, stage_method_assets
from superagent.chain import backpropagate, chain_control
from superagent.checks import check_content
from superagent.codegen import apply_planned_reply, plan_coverage
from superagent.config import build_config, load_defaults
from superagent.control.adaptive import Proposal, Snapshot, rank_proposals
from superagent.control.base import ControlLog
from superagent.ensemble import weights as ensemble_weights
from superagent.experience import ExperienceStore, Record, decision_context
from superagent.helper_api import reference as helper_reference
from superagent.isolation import Bubblewrap
from superagent.llm.ledger import Ledger
from superagent.operations import Dispatch, Operations
from superagent.operators import Operators
from superagent.planner import _smart_source_excerpt, plan_experiment, validate_experiment_plan
from superagent.prompts import builders
from superagent.publication import publish_if_better
from superagent.scheduling import dispatch as legacy_dispatch
from superagent.selection import elite_pick
from superagent.training import restore_torch, save_torch
from superagent.tree import Direction, Node, SearchTree


def test_save_torch_uses_checkpoint_dir_instead_of_cwd(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    monkeypatch.setenv("SUPERAGENT_CHECKPOINT_DIR", str(ckpt_dir))
    monkeypatch.setenv("SUPERAGENT_TRIAL_SIGNATURE", "sig-1")
    monkeypatch.setenv("SUPERAGENT_RUNG", "0")
    monkeypatch.delenv("SUPERAGENT_RESUME_DIR", raising=False)
    monkeypatch.delenv("SUPERAGENT_SMOKE", raising=False)

    ro_cwd = tmp_path / "ro_job"
    ro_cwd.mkdir()
    monkeypatch.chdir(ro_cwd)

    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    restore_torch(model, opt)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    opt.step()

    seen_dirs = []
    real_tmpdir = __import__("tempfile").TemporaryDirectory

    def spy_tmpdir(*args, **kwargs):
        seen_dirs.append(kwargs.get("dir"))
        assert kwargs.get("dir") != "."
        return real_tmpdir(*args, **kwargs)

    monkeypatch.setattr("superagent.training.tempfile.TemporaryDirectory", spy_tmpdir)
    save_torch(model, opt, epoch=1, next_seconds=5.0, delivery_seconds=2.0)

    assert seen_dirs == [str(ckpt_dir)]
    assert (ckpt_dir / "checkpoint.json").is_file()


def test_restore_torch_handles_cuda_device_count_mismatch(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    monkeypatch.setenv("SUPERAGENT_CHECKPOINT_DIR", str(ckpt_dir))
    monkeypatch.setenv("SUPERAGENT_TRIAL_SIGNATURE", "sig-cuda")
    monkeypatch.setenv("SUPERAGENT_RUNG", "0")

    payload = ckpt_dir / "dummy.pt"
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": 1,
            "rng": torch.get_rng_state(),
            "cuda": [torch.ByteTensor([1, 2]), torch.ByteTensor([3, 4])],
            "python": random.getstate(),
            "numpy": ("MT19937", torch.tensor(np.random.get_state()[1].astype("int64")), 0, 0, 0.0),
            "scheduler": None,
            "scaler": None,
            "generator": None,
            "sampler": None,
            "optimizer_steps": 5,
        },
        payload,
    )
    from superagent.training import digest

    sha = digest(payload)
    shutil_target = ckpt_dir / sha
    payload.rename(shutil_target)
    (ckpt_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "signature": "sig-cuda",
                "sha256": sha,
                "epoch": 1,
                "next_seconds": 5.0,
                "delivery_seconds": 1.0,
                "can_continue": True,
                "rung": 0,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("SUPERAGENT_RESUME_DIR", str(ckpt_dir))
    # Simulate CUDA being reported available with only 1 GPU while checkpoint recorded 2
    called = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda _s: (_ for _ in ()).throw(AssertionError("should not call set_rng_state_all")))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda st: called.append(st))

    epoch, _ = restore_torch(model, opt)
    assert epoch == 1
    assert len(called) == 1


def test_bubblewrap_gpu_binds_and_env_preservation(tmp_path, monkeypatch):
    import superagent.isolation as iso

    monkeypatch.setattr(iso.sys, "platform", "linux")
    monkeypatch.setattr(iso.shutil, "which", lambda _name: "/usr/bin/bwrap")
    monkeypatch.setattr(iso.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, "", ""))

    existing_devs = {"/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia0", "/usr/local/nvidia", "/usr/local/cuda"}
    orig_exists = Path.exists
    orig_is_symlink = Path.is_symlink

    def fake_exists(self):
        s = self.as_posix()
        if s in existing_devs or s in ("/usr", "/bin", "/lib", "/lib64", "/sbin"):
            return True
        if s.endswith("/out/ready"):
            return True
        return orig_exists(self)

    def fake_is_symlink(self):
        if self.as_posix() in ("/usr/local/nvidia", "/usr/local/cuda", "/usr", "/bin", "/lib", "/lib64", "/sbin"):
            return False
        return orig_is_symlink(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)

    train, inf, forb = tmp_path / "tr", tmp_path / "inf", tmp_path / "forb"
    for p in (train, inf, forb):
        p.mkdir()

    # 1. CPU-only keeps original isolation without nvidia mounts or CUDA_VISIBLE_DEVICES
    bw_cpu = Bubblewrap(train, inf, forb, num_gpus=0)
    assert "/usr/local/nvidia" not in bw_cpu.runtime
    cmd_cpu, env_cpu = bw_cpu.command(
        tmp_path, ["solution.py"],
        {"CUDA_VISIBLE_DEVICES": "0", "TORCH_NUM_THREADS": "4", "VECLIB_MAXIMUM_THREADS": "4"},
        inference=False,
    )
    assert "CUDA_VISIBLE_DEVICES" not in env_cpu
    assert env_cpu["TORCH_NUM_THREADS"] == "4"
    assert env_cpu["VECLIB_MAXIMUM_THREADS"] == "4"
    assert "--dev-bind" not in cmd_cpu

    # 2. GPU mode binds control + allocated GPU nodes and forwards CUDA_VISIBLE_DEVICES
    bw_gpu = Bubblewrap(train, inf, forb, num_gpus=2)
    assert "/usr/local/nvidia" in bw_gpu.runtime
    assert "/usr/local/cuda" in bw_gpu.runtime
    cmd_gpu, env_gpu = bw_gpu.command(
        tmp_path, ["solution.py"],
        {"CUDA_VISIBLE_DEVICES": "0", "TORCH_NUM_THREADS": "2"},
        inference=False,
    )
    assert env_gpu["CUDA_VISIBLE_DEVICES"] == "0"
    assert env_gpu["TORCH_NUM_THREADS"] == "2"
    dev_binds = [cmd_gpu[i + 1] for i, tok in enumerate(cmd_gpu) if tok == "--dev-bind"]
    assert "/dev/nvidiactl" in dev_binds
    assert "/dev/nvidia-uvm" in dev_binds
    assert "/dev/nvidia0" in dev_binds
    assert "/dev/nvidia1" not in dev_binds


def test_rank_proposals_allows_multi_refine_failed_draft_repair_and_planner_chosen_target():
    cfg = load_defaults()
    cfg["search"]["max_refinements_per_node"] = 3
    cfg["search"]["max_repairs_per_node"] = 1

    # Node n1 (deliverable) already had 1 failed refine (c1) and 1 cancelled refine (c2):
    # should still propose a new refine (slot=2, target=None) up to max_refinements_per_node=3.
    # Node n2 (generated failed draft) exists alongside incumbent n1 -> should propose low-priority repair (tier=2).
    facts = (
        {
            "seq": 1, "operation_id": "op1", "node_id": "n1", "parent_id": "root",
            "action": "draft", "method": "tab_linear", "source": "generated",
            "hypothesis": "linear baseline", "target": None, "branch_id": 1, "seed": 42,
            "split_id": "s1", "metric": "accuracy", "status": "completed", "failed_stage": None,
            "reason": None, "seconds": 10.0, "score": 0.80, "score_verified": True,
            "deliverable": True, "trial_id": "n1", "rung": 0, "checkpoint": None,
            "trial_signature": None, "source_sha256": "abc", "gain": None,
        },
        {
            "seq": 2, "operation_id": "op2", "node_id": "c1", "parent_id": "n1",
            "action": "refine", "method": "generated", "source": "generated",
            "hypothesis": "try depth", "target": "model_def", "branch_id": 1, "seed": 42,
            "split_id": "s1", "metric": "accuracy", "status": "failed", "failed_stage": "executed",
            "reason": "oops", "seconds": 5.0, "score": None, "score_verified": False,
            "deliverable": False, "trial_id": None, "rung": 0, "checkpoint": None,
            "trial_signature": None, "source_sha256": "def", "gain": None,
        },
        {
            "seq": 3, "operation_id": "op3", "node_id": "c2", "parent_id": "n1",
            "action": "refine", "method": "generated", "source": "generated",
            "hypothesis": "cancelled refine", "target": None, "branch_id": 1, "seed": 42,
            "split_id": "s1", "metric": "accuracy", "status": "cancelled", "failed_stage": "generated",
            "reason": "cancelled", "seconds": 1.0, "score": None, "score_verified": False,
            "deliverable": False, "trial_id": None, "rung": 0, "checkpoint": None,
            "trial_signature": None, "source_sha256": None, "gain": None,
        },
        {
            "seq": 4, "operation_id": "op4", "node_id": "n2", "parent_id": "root",
            "action": "draft", "method": "tab_gbdt", "source": "generated",
            "hypothesis": "lgbm draft", "target": None, "branch_id": 2, "seed": 42,
            "split_id": "s1", "metric": "accuracy", "status": "failed", "failed_stage": "executed",
            "reason": "KeyError", "seconds": 4.0, "score": None, "score_verified": False,
            "deliverable": False, "trial_id": "n2", "rung": 0, "checkpoint": None,
            "trial_signature": None, "source_sha256": "ghi", "gain": None,
        },
    )
    snap = Snapshot(
        as_of_seq=4, facts=facts, tried=frozenset(),
        repairable=None, incumbent=0.80, api_available=True, dispatched=4, resumable=frozenset(),
    )
    props, ranked, _rej = rank_proposals(snap, (), cfg, maximize=True)
    n1_refines = [p for p in props if p.parent_id == "n1" and p.action == "refine"]
    assert len(n1_refines) == 1
    assert n1_refines[0].target is None

    # Adding another proposal for n1 later with updated slot must produce a distinct proposal_id
    facts_2 = facts + (
        {
            "seq": 5, "operation_id": "op5", "node_id": "c3", "parent_id": "n1",
            "action": "refine", "method": "generated", "source": "generated",
            "hypothesis": "2nd refine", "target": "data_prep", "branch_id": 1, "seed": 42,
            "split_id": "s1", "metric": "accuracy", "status": "completed", "failed_stage": None,
            "reason": None, "seconds": 6.0, "score": 0.82, "score_verified": True,
            "deliverable": True, "trial_id": "c3", "rung": 0, "checkpoint": None,
            "trial_signature": None, "source_sha256": "jkl", "gain": 0.02,
        },
    )
    snap2 = Snapshot(
        as_of_seq=5, facts=facts_2, tried=frozenset({n1_refines[0].proposal_id}),
        repairable=None, incumbent=0.82, api_available=True, dispatched=5, resumable=frozenset(),
    )
    props2, _, _ = rank_proposals(snap2, (), cfg, maximize=True)
    n1_refines2 = [p for p in props2 if p.parent_id == "n1" and p.action == "refine"]
    assert len(n1_refines2) == 1
    assert n1_refines2[0].proposal_id != n1_refines[0].proposal_id

    # Failed generated draft n2 got a tier=2 repair proposal even though incumbent exists
    n2_repairs = [p for p in props if p.parent_id == "n2" and p.action == "repair"]
    assert len(n2_repairs) == 1
    assert any(r["proposal_id"] == n2_repairs[0].proposal_id and r["tier"] == 2 for r in ranked)


def test_formal_record_decision_context_and_planner_propagate_real_hypothesis_and_donor(tmp_path):
    cfg = load_defaults()
    cfg["experience"]["vector"]["enabled"] = False
    store = ExperienceStore(cfg["experience"], tmp_path / "records.json", embedder=None, background=False)
    ledger = Ledger(tmp_path / "operations.sqlite", run_id="run-ev")
    tree = SearchTree(Direction(True))

    parent = Node(id="n1", kind="draft", parent=tree.root, branch_id=1, plan="TFIDF + LogisticRegression C=1.0",
                  code="def fit():\n    pass\n", output="Accuracy: 0.78", score=0.78, failed=False,
                  method_summary="TFIDF word ngrams + LogReg", in_tree=True)
    donor = Node(id="n2", kind="draft", parent=tree.root, branch_id=2, plan="Char 3-5 TFIDF + LinearSVC",
                 code="def build_char_feats():\n    return 42\n", output="Accuracy: 0.85", score=0.85, failed=False,
                 method_summary="Char ngrams + calibrated SVC", in_tree=True)
    for nd in (parent, donor):
        tree.by_id[nd.id] = nd
        ledger.register_operation(f"op-{nd.id}", f"sig-{nd.id}", ("script_coder",))

    ctx = SimpleNamespace(
        cfg=cfg,
        tree=tree,
        direction=Direction(True),
        task_spec=SimpleNamespace(signature="sig-task", metric="accuracy"),
        executor=SimpleNamespace(run_dir=lambda nid: tmp_path / "runs" / nid),
        experience=store,
        gateway=SimpleNamespace(ledger=ledger, endpoint_failure=None, cooldown_until=0.0),
        flags={"task_facts": {"metric": "accuracy"}, "formal": True},
        task_desc="Classify texts.",
        data_preview="train.csv: 1000 rows, text column",
        catalog_text="### Qwen3-Embedding",
    )
    ops = Operations(SimpleNamespace(ctx=ctx))
    p1 = Proposal("p1", tree.root.id, "draft", "distinct_root_hypothesis", "generated", False, 0)
    ops._record(Dispatch("op-n1", p1, "sig-n1", "d1", node=parent, started=time.monotonic()), parent,
                {"score_verified": True, "deliverable": True}, "completed")
    p2 = Proposal("p2", tree.root.id, "draft", "distinct_root_hypothesis", "generated", False, 1)
    ops._record(Dispatch("op-n2", p2, "sig-n2", "d2", node=donor, started=time.monotonic()), donor,
                {"score_verified": True, "deliverable": True}, "completed")

    assert ops.facts[0]["hypothesis"] == "TFIDF + LogisticRegression C=1.0"
    assert ops.facts[0]["dispatch_reason"] == "distinct_root_hypothesis"
    assert store.records[0].plan == "TFIDF + LogisticRegression C=1.0"

    borrow_prop = Proposal("pb", "n1", "borrow", "adapt a useful method observed on a different branch",
                           "generated", False, 2, evidence_ids=("op-n2",))
    ev = decision_context(ops.facts, borrow_prop, store)
    assert any(r["operation_id"] == "op-n2" and r["method_summary"] == "Char ngrams + calibrated SVC" for r in ev)

    captured_prompts = []
    class FakeGW:
        def __init__(self, ldg):
            self.ledger = ldg
        def structured(self, _role, system, user, _schema, **_kw):
            captured_prompts.append((system, user))
            return {
                "schema_version": 1, "action": "borrow", "evidence_ids": ["op-n1", "op-n2"],
                "hypothesis": "Add char n-grams from n2",
                "changes": [{"id": "c1", "component": "data_prep", "target": "fit", "edit": "add char vectorizer", "check": "score rises"}],
                "expected_observation": "val improves", "preserve": ["split"],
            }

    ctx.gateway = FakeGW(ledger)
    ledger.register_operation("op-b", "sig-b", ("planner.structured",))
    d_borrow = Dispatch("op-b", borrow_prop, "sig-b", "db")
    plan_experiment(ctx, d_borrow, parent, ev)
    assert len(captured_prompts) == 1
    user_prompt = captured_prompts[0][1]
    assert "train.csv: 1000 rows" in user_prompt
    assert "Qwen3-Embedding" in user_prompt
    assert "Accuracy: 0.78" in user_prompt
    assert "build_char_feats" in user_prompt


def test_check_content_catches_binary_empty_or_placeholder_but_allows_imbalanced_binary(tmp_path):
    sample = tmp_path / "sample.csv"
    sample.write_text("id,target\n" + "\n".join(f"{i},0" for i in range(100)), encoding="utf-8")

    # 1. Valid 98% / 2% imbalanced binary predictions MUST PASS
    imbalanced = tmp_path / "imbalanced.csv"
    imbalanced.write_text(
        "id,target\n" + "\n".join(f"{i},{1 if i < 2 else 0}" for i in range(100)),
        encoding="utf-8",
    )
    assert check_content(imbalanced, sample, 0.95).ok is True

    # 2. Mostly empty binary column (96 empty + 4 "1") MUST FAIL
    mostly_empty = tmp_path / "mostly_empty.csv"
    mostly_empty.write_text(
        "id,target\n" + "\n".join(f"{i},{1 if i < 4 else ''}" for i in range(100)),
        encoding="utf-8",
    )
    assert check_content(mostly_empty, sample, 0.95).ok is False

    # 3. Placeholder mask binary sequences ("0 0 0" vs "1 1 1") MUST FAIL
    masks = tmp_path / "masks.csv"
    masks.write_text(
        "id,target\n" + "\n".join(f"{i},{'1 1 1' if i == 0 else '0 0 0'}" for i in range(100)),
        encoding="utf-8",
    )
    assert check_content(masks, sample, 0.95).ok is False


def test_ensemble_weights_handles_negative_and_zero_scores():
    # Positive higher-is-better
    w_pos = ensemble_weights([0.9, 0.8, 0.7], True)
    assert pytest.approx(sum(w_pos)) == 1.0
    assert w_pos[0] > w_pos[1] > w_pos[2] > 0

    # Negative higher-is-better (e.g., negative log loss or R2)
    w_neg_hib = ensemble_weights([-0.1, -0.5, -1.2], True)
    assert pytest.approx(sum(w_neg_hib)) == 1.0
    assert all(w > 0 for w in w_neg_hib)
    assert w_neg_hib[0] > w_neg_hib[1] > w_neg_hib[2]

    # Equal non-positive scores
    w_zero = ensemble_weights([0.0, 0.0], False)
    assert pytest.approx(sum(w_zero)) == 1.0
    assert all(w > 0 for w in w_zero)


def test_elite_pick_and_chain_control_concurrency_isolation():
    cfg = load_defaults()
    tree = SearchTree(Direction(True))
    d1 = Node(id="d1", kind="draft", parent=tree.root, branch_id=1, score=0.80, failed=False, step=1, in_tree=True)
    d1.chain_best = d1
    tree.root.children.append(d1)
    tree.by_id[d1.id] = d1
    tree.nodes.append(d1)
    tree.branch_all[1] = [d1]
    tree.register_ok(d1)

    # 1. elite_pick must not clobber shared picked_by_elite, and dispatch carries via_elite cleanly
    chosen, mode = elite_pick(tree, cfg, 0.8, lambda: False, random.Random(1))
    assert mode == "elite_pick"
    assert chosen is d1
    assert chosen.picked_by_elite is False
    disp = legacy_dispatch(tree, chosen, cfg, 0.8, random.Random(1), via_elite=(mode == "elite_pick"))
    assert disp.via_elite is True
    assert disp.details["plateau_window"] == cfg["plateau"]["elite_branch_window"]

    # 2. chain_control isolates no_gain_count per sibling chain and backpropagate does not reset ancestor count
    c_a = Node(id="ca", kind="refine", parent=d1, branch_id=1, chain_best=d1, score=0.80, failed=False)
    out_a = chain_control(tree, c_a, cfg, 0.1, random.Random(1))
    assert c_a.no_gain_count == 1
    assert d1.no_gain_count == 0

    c_b = Node(id="cb", kind="refine", parent=c_a, branch_id=1, chain_best=d1, score=0.80, failed=False)
    out_b = chain_control(tree, c_b, cfg, 0.1, random.Random(1))
    assert c_b.no_gain_count == 2

    # Ending an unrelated sibling c_sib must not wipe c_a.no_gain_count
    c_sib = Node(id="cs", kind="refine", parent=c_a, branch_id=1, chain_best=d1, score=0.80, failed=True)
    backpropagate(c_sib, 0.0, False)
    assert c_a.no_gain_count == 1


def test_control_log_concurrent_writes(tmp_path):
    ctx = SimpleNamespace(elapsed=lambda: 1.0, paths=SimpleNamespace(control_log=tmp_path / "control.jsonl"))
    clog = ControlLog(ctx)
    errors = []

    def worker(i):
        try:
            for k in range(30):
                clog.write("tick", {"v": k % 3}, key=f"k{i % 2}")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(worker, range(6)))
    assert not errors


def test_stage_method_assets_dereferences_symlinks_with_hardlinks(tmp_path, monkeypatch):
    local_dir = tmp_path / "cache"
    blobs = local_dir / "hub" / "models--demo" / "blobs"
    snap = local_dir / "hub" / "models--demo" / "snapshots" / "rev1"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    blob = blobs / "abc123"
    blob.write_bytes(b"weight-bytes")
    link = snap / "model.safetensors"
    try:
        os.symlink(os.path.relpath(blob, snap), link)
    except OSError:
        # Unprivileged Windows blocks symlinks; create stub and simulate symlink resolution to blob
        link.write_bytes(b"weight-bytes")
        orig_resolve = Path.resolve

        def fake_resolve(self, *a, **kw):
            if orig_resolve(self, *a, **kw) == orig_resolve(link, *a, **kw):
                return orig_resolve(blob, *a, **kw)
            return orig_resolve(self, *a, **kw)

        monkeypatch.setattr(Path, "resolve", fake_resolve)

    from superagent.training import digest
    sha = digest(blob)

    match = MethodMatch(
        method_id="text_qwen3_embedding_features",
        version="1",
        modality="text",
        hypothesis="embed",
        source="catalog",
        eligible=True,
        reason="eligible",
        expected_seconds=60.0,
        recipe_id=None,
        entry={
            "id": "text_qwen3_embedding_features",
            "pretrained": True,
            "snapshot": snap.as_posix(),
            "weight_hashes": {"model.safetensors": sha},
            "example": f"SentenceTransformer('{snap.as_posix()}')",
        },
    )
    mounts = stage_method_assets([match], tmp_path / "staged", local_dir)
    staged_file = tmp_path / "staged" / "text_qwen3_embedding_features" / "model.safetensors"
    assert staged_file.is_file()
    assert not staged_file.is_symlink()
    assert os.stat(staged_file).st_ino == os.stat(blob).st_ino
    assert mounts["/method_assets/text_qwen3_embedding_features"] == tmp_path / "staged" / "text_qwen3_embedding_features"
    assert "/method_assets/text_qwen3_embedding_features" in match.entry["example"]


def test_artifacts_tolerance_helper_api_and_publication_final_swap(tmp_path, monkeypatch):
    assert PROB_ATOL == 1e-5 and PROB_RTOL == 1e-4
    orig = tmp_path / "orig"
    rep = tmp_path / "rep"
    orig.mkdir(); rep.mkdir()
    for fname, d1, d2 in (
        ("predictions.csv", "id,a,b\n1,0.700000,0.300000\n", "id,a,b\n1,0.700005,0.299995\n"),
        ("val_predictions.csv", "id,a,b\n1,0.800000,0.200000\n", "id,a,b\n1,0.799996,0.200004\n"),
    ):
        (orig / fname).write_text(d1, encoding="utf-8")
        (rep / fname).write_text(d2, encoding="utf-8")
    assert same_predictions(orig, rep, probability=True) is True

    (rep / "predictions.csv").write_text("id,a,b\n1,0.75,0.25\n", encoding="utf-8")
    assert same_predictions(orig, rep, probability=True) is False

    ref = helper_reference()
    assert "/method_assets is NOT mounted" in ref

    # Verify stop_event firing AFTER copy/hash check still completes atomic publish
    import shutil
    import superagent.publication as pub_mod

    if os.name == "nt":
        def copy_publish(root_p, directory):
            fin = Path(root_p) / "final"
            if fin.exists():
                shutil.rmtree(fin)
            shutil.copytree(directory, fin)

        monkeypatch.setattr(pub_mod, "publish", copy_publish)

    root = tmp_path / "pub_root"
    run_d = root / "runs" / "n1"
    (run_d / "out" / "model").mkdir(parents=True)
    (run_d / "solution.py").write_text("print('ok')\n", encoding="utf-8")
    (run_d / "out" / "predictions.csv").write_text("id,y\n1,a\n", encoding="utf-8")
    (run_d / "out" / "val_predictions.csv").write_text("id,y\n1,a\n", encoding="utf-8")
    (run_d / "out" / "model" / "inference.json").write_text(json.dumps({"version": 1, "weight_files": ["w.bin"]}), encoding="utf-8")
    (run_d / "out" / "model" / "w.bin").write_bytes(b"123")

    from superagent.artifacts import package_files
    from superagent.training import digest

    inv, _ = package_files(run_d / "out" / "model")
    receipt = {
        "passed": True,
        "code_sha256": digest(run_d / "solution.py"),
        "package_sha256": inv,
        "predictions_sha256": {
            "predictions.csv": digest(run_d / "out" / "predictions.csv"),
            "val_predictions.csv": digest(run_d / "out" / "val_predictions.csv"),
        },
    }
    node = Node(id="n1", kind="draft", score=0.9, failed=False, artifact_verification=receipt)
    stop_ev = threading.Event()

    class FlipLock:
        def __init__(self):
            self._lk = threading.Lock()
        def __enter__(self):
            stop_ev.set()  # signal shutdown right as atomic lock is entered
            return self._lk.__enter__()
        def __exit__(self, *a):
            return self._lk.__exit__(*a)

    ctx = SimpleNamespace(
        paths=SimpleNamespace(root=root),
        task_spec=SimpleNamespace(metric="accuracy", signature="sp", truth=[0, 1]),
        direction=Direction(True),
        executor=SimpleNamespace(run_dir=lambda _id: run_d),
        flags={
            "publish_lock": FlipLock(),
            "input_manifest": SimpleNamespace(payload=lambda inference=True: {}),
            "runtime_environment": {},
        },
        stop_event=stop_ev,
    )
    (root / "logs").mkdir(parents=True, exist_ok=True)
    info = publish_if_better(ctx, node)
    assert info is not None
    assert (root / "final" / "predictions.csv").is_file()

    # Verify stop_event already set BEFORE copy returns None cleanly
    node2 = Node(id="n1", kind="draft", score=0.95, failed=False, artifact_verification=receipt)
    assert publish_if_better(ctx, node2) is None


def test_cold_start_delivery_seconds_is_bounded_for_long_script_timeout(tmp_path):
    from superagent.search import SearchController
    from superagent.context import LockedRandom, OutputPaths, RunContext

    cfg = load_defaults()
    cfg["execution"]["script_timeout_seconds"] = 7200
    captured_envs = []

    class FakeExec:
        cores_per_slot = 2
        num_gpus = 0
        def run_dir(self, nid):
            d = tmp_path / "runs" / nid
            d.mkdir(parents=True, exist_ok=True)
            return d
        def run(self, node_id, code, timeout, **kw):
            captured_envs.append(kw["trial_env"])
            from superagent.executor import ExecResult
            return ExecResult("ok", 1.0, "RuntimeError", "stop", False, 1, timeout)

    paths = OutputPaths(tmp_path / "out")
    paths.make_dirs()
    ctx = RunContext(
        cfg=cfg, paths=paths, task_dir=tmp_path, direction=Direction(True), direction_source="cli",
        tree=SearchTree(Direction(True)), gateway=SimpleNamespace(ledger=None), executor=FakeExec(),
        experience=None, task_desc="t", data_preview="p", catalog_text="c", modalities=[],
        sample_path=None, rng=LockedRandom(1), process_start=time.monotonic(), started_at="2026-01-01",
    )
    sc = SearchController(ctx)
    node = Node(id="n-deliv", kind="draft", code="print(1)\n")
    sc._run(node)
    assert len(captured_envs) == 1
    assert float(captured_envs[0]["SUPERAGENT_DELIVERY_SECONDS"]) == pytest.approx(180.0)


def test_single_model_config_and_staged_builders_formal_mode():
    cfg = build_config(Path(__file__).resolve().parents[1] / "configs" / "single_model.yaml")
    assert cfg["codegen"]["staged_drafting"] is False

    _, u_stage = builders.draft_stage("t", "d", "s", "r", "train_eval", [], "off", declared_split=True)
    assert "score marker" not in u_stage
    assert "--predict-only" in u_stage

    _, u_int = builders.draft_integrator("t", "d", "s", "r", [("data_prep", "p", "x = 1")], declared_split=True)
    assert "score marker" not in u_int
    assert "--predict-only" in u_int


def test_planned_patch_allows_top_level_imports_and_smart_excerpt_preserves_entry():
    src = (
        "import numpy as np\n"
        "\n"
        "def fit_model(x):\n"
        "    return x + 1\n"
        "\n"
        "def helper_untouched(z):\n"
        "    return z * 2\n"
    )
    plan = {
        "schema_version": 1,
        "action": "refine",
        "evidence_ids": ["op1"],
        "hypothesis": "Use pandas in fit_model",
        "changes": [{"id": "c1", "component": "model_def", "target": "fit_model", "edit": "add pandas", "check": "runs"}],
        "expected_observation": "ok",
        "preserve": ["helper_untouched"],
    }
    assert validate_experiment_plan(plan, "refine", {"op1"}, src) == []

    # 1. Valid reply adding a top-level import AND modifying fit_model must SUCCEED
    good_reply = (
        "CHANGE c1\n"
        "<<<<<<< SEARCH\n"
        "import numpy as np\n"
        "=======\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        ">>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\n"
        "def fit_model(x):\n"
        "    return x + 1\n"
        "=======\n"
        "def fit_model(x):\n"
        "    return float(pd.Series(x).sum()) + 1\n"
        ">>>>>>> REPLACE\n"
    )
    updated, done, notes = apply_planned_reply(src, plan, good_reply)
    assert done == ["c1"], notes
    assert "import pandas as pd" in updated
    assert plan_coverage(plan, src, updated) == []

    # 2. Reply sneaking an edit into helper_untouched MUST BE REJECTED
    bad_reply = (
        "CHANGE c1\n"
        "<<<<<<< SEARCH\n"
        "def fit_model(x):\n"
        "    return x + 1\n"
        "=======\n"
        "def fit_model(x):\n"
        "    return x + 2\n"
        ">>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\n"
        "def helper_untouched(z):\n"
        "    return z * 2\n"
        "=======\n"
        "def helper_untouched(z):\n"
        "    return z * 9\n"
        ">>>>>>> REPLACE\n"
    )
    _, done_bad, _ = apply_planned_reply(src, plan, bad_reply)
    assert done_bad == []

    # 3. _smart_source_excerpt keeps predict_only and __main__ even when > 24000 chars
    huge_filler = "\n".join(f"VAR_{i} = {i}  # " + "x" * 60 for i in range(450))
    big_code = (
        "import os\nimport pandas as pd\n\n"
        f"def big_helper():\n    pass\n\n{huge_filler}\n\n"
        "def predict_only(m, d, o):\n    return 'kept'\n\n"
        "if __name__ == '__main__':\n    print('entrypoint')\n"
    )
    assert len(big_code) > 24000
    excerpt = _smart_source_excerpt(big_code, 24000)
    assert len(excerpt) <= 24000
    assert "import pandas as pd" in excerpt
    assert "def predict_only(m, d, o):" in excerpt
    assert "if __name__ == '__main__':" in excerpt
