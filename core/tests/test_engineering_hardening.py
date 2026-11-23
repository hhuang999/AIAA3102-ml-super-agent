"""Hardening tests covering merged 1121 engineering improvements and preserved v1+v2 invariants."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd
import pytest

from helpers import cfg, make_ctx, mk, tree
from superagent.checks import check_content
from superagent.codegen import apply_planned_reply, plan_coverage
from superagent.control.adaptive import Proposal, Snapshot, rank_proposals
from superagent.ensemble import combine, detect_format, weights
from superagent.experience import decision_context, effect_label, tokenize
from superagent.final import _spec_encoding, stratified_halves
from superagent.helper_api import reference
from superagent.isolation import Bubblewrap
from superagent.llm.gateway import LOWER_EFFORT, TransportResponse, _is_reasoning_exhausted
from superagent.llm.ledger import Ledger, NON_QUOTA_STATUSES, NO_BILLING_STATUSES
from superagent.llm.schema import Field, validate
from superagent.planner import _smart_source_excerpt, validate_experiment_plan
from superagent.prompts import blocks
from superagent.review import _changed_symbol_names, review
from superagent.scheduling import attempts_since_branch_best
from superagent.tree import Node


def test_ledger_separates_non_quota_and_no_billing(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite", extras=1, rate_limit_retries=3)
    ledger.limit = 10.0
    ledger.register_operation("op1", "sig1", ("planner.structured",))
    # Multiple rate_limited / overloaded / not_sent retries must NOT exhaust operation_extra_sends
    for st in ("not_sent", "rate_limited", "overloaded"):
        att = ledger.begin("op1", "planner.structured", "c1", "s1", reserve=1.0)
        ledger.finish(att, 99.0, {"model": "m"}, st)
    assert ledger.sends("c1") == 0
    # Regular calls still succeed up to 1 normal + 1 extra
    a1 = ledger.begin("op1", "planner.structured", "c1", "s1", reserve=1.0)
    ledger.finish(a1, 0.25, {"model": "m"}, "received")
    a2 = ledger.begin("op1", "planner.structured", "c1", "s1", reserve=1.0)
    ledger.finish(a2, 0.25, {"model": "m"}, "received")
    assert ledger.sends("c1") == 2
    with ledger.db() as db:
        costs = [r[0] for r in db.execute("SELECT cost FROM attempts WHERE status IN ('not_sent','rate_limited','overloaded')")]
    assert costs == [0.0, 0.0, 0.0]
    assert set(NON_QUOTA_STATUSES) <= set(NO_BILLING_STATUSES)


def test_schema_rejects_nan_and_inf():
    schema = {"score": Field("number", "score", required=True, nullable=False)}
    for bad in (float("nan"), float("inf"), float("-inf"), "nan", "inf"):
        _, errs = validate({"score": bad}, schema)
        assert errs, f"expected {bad!r} to be rejected"
    obj, errs = validate({"score": 0.75}, schema)
    assert not errs and obj["score"] == 0.75


def test_gateway_reasoning_effort_stepdown_and_exhaustion():
    assert LOWER_EFFORT == {"max": "high", "high": "low", "low": "off"}
    r1 = TransportResponse(content="   ", finish_reason="length", prompt_tokens=10, completion_tokens=50, reasoning_tokens=50)
    assert _is_reasoning_exhausted(r1, thinking=True)
    assert not _is_reasoning_exhausted(r1, thinking=False)
    r2 = TransportResponse(content="", finish_reason="length", prompt_tokens=10, completion_tokens=50, reasoning_tokens=0)
    assert not _is_reasoning_exhausted(r2, thinking=True)
    r3 = TransportResponse(content="", finish_reason="length")
    assert _is_reasoning_exhausted(r3, thinking=True)


def test_bubblewrap_gpu_passthrough_args(tmp_path, monkeypatch):
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("#!/bin/sh\n")
    bwrap.chmod(0o755)
    monkeypatch.setattr("superagent.isolation.shutil.which", lambda name: str(bwrap) if name == "bwrap" else None)
    monkeypatch.setattr("superagent.isolation.sys.platform", "linux")
    monkeypatch.setattr("superagent.isolation.Bubblewrap._preflight", lambda self, forbidden: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    tr, inf, forb = tmp_path / "tr", tmp_path / "inf", tmp_path / "forb"
    job = tmp_path / "job"
    for d in (tr, inf, forb, job, job / "out", job / "work"):
        d.mkdir(parents=True, exist_ok=True)
    b0 = Bubblewrap(tr, inf, forb, num_gpus=0)
    b1 = Bubblewrap(tr, inf, forb, num_gpus=1)
    _, env0 = b0.command(job, ["s.py"], {"CUDA_VISIBLE_DEVICES": "0"}, inference=True)
    _, env1 = b1.command(job, ["s.py"], {"CUDA_VISIBLE_DEVICES": "0"}, inference=True)
    assert "CUDA_VISIBLE_DEVICES" not in env0
    assert env1.get("CUDA_VISIBLE_DEVICES") == "0"


def test_rank_proposals_multi_slots_target_none_and_draft_repair():
    c = cfg()
    c["search"]["max_refinements_per_node"] = 3
    c["search"]["max_repairs_per_node"] = 2
    base_row = {
        "operation_id": "op_d1",
        "node_id": "n_d1",
        "parent_id": "root",
        "action": "draft",
        "source": "generated",
        "method": "gbdt",
        "deliverable": True,
        "score_verified": True,
        "score": 0.85,
        "gain": None,
        "seed": 42,
        "split_id": "s1",
        "metric": "accuracy",
        "source_sha256": "abc",
        "trial_id": "t1",
        "rung": 0,
        "trial_signature": None,
        "checkpoint": None,
        "seconds": 10.0,
        "status": "completed",
        "branch_id": 1,
    }
    child_ok = {
        **base_row,
        "operation_id": "op_r1",
        "node_id": "n_r1",
        "parent_id": "n_d1",
        "action": "refine",
        "score": 0.86,
        "gain": 0.01,
        "status": "completed",
    }
    failed_draft = {
        **base_row,
        "operation_id": "op_d2",
        "node_id": "n_d2",
        "action": "draft",
        "deliverable": False,
        "score_verified": False,
        "score": None,
        "status": "failed",
        "branch_id": 2,
    }
    snap = Snapshot(
        as_of_seq=3,
        facts=(base_row, child_ok, failed_draft),
        tried=frozenset(),
        repairable=None,
        incumbent=0.86,
        api_available=True,
        dispatched=3,
        resumable=frozenset(),
    )
    props, _, _ = rank_proposals(snap, (), c, True)
    d1_refines = [p for p in props if p.parent_id == "n_d1" and p.action == "refine"]
    assert d1_refines and d1_refines[0].target is None
    d2_repairs = [p for p in props if p.parent_id == "n_d2" and p.action == "repair"]
    assert len(d2_repairs) == 1


def test_attempts_since_branch_best_ignores_earlier_created_sibling():
    t = tree()
    d = mk(t, "draft", t.root, score=0.5, created_at="2026-11-23T12:00:00", finished_at="2026-11-23T12:00:05")
    early_sibling = mk(t, "refine", d, score=0.4, append=False, created_at="2026-11-23T12:00:06", finished_at="2026-11-23T12:00:20")
    best = mk(t, "refine", d, score=0.8, created_at="2026-11-23T12:00:05", finished_at="2026-11-23T12:00:10")
    # Simulate early_sibling finishing AFTER best was appended
    t.append(early_sibling)
    late_child = mk(t, "refine", best, score=0.75, created_at="2026-11-23T12:00:11", finished_at="2026-11-23T12:00:15")
    assert attempts_since_branch_best(t, d.branch_id) == 1


def test_ast_patch_allows_top_imports_and_imports_target():
    src = "import os\n\ndef feat(x):\n    return x + 1\n\ndef keep(x):\n    return x * 2\n"
    plan = {
        "schema_version": 1,
        "action": "refine",
        "evidence_ids": ["op1"],
        "hypothesis": "Use math.sqrt and numpy import",
        "changes": [
            {"id": "c1", "component": "data_prep", "target": "feat", "edit": "add math.sqrt", "check": "ok"},
        ],
        "expected_observation": "better",
        "preserve": ["keep"],
    }
    assert not validate_experiment_plan(plan, "refine", {"op1"}, src)
    reply = (
        "CHANGE c1\n"
        "<<<<<<< SEARCH\nimport os\n=======\nimport os\nimport math\n>>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\ndef feat(x):\n    return x + 1\n=======\ndef feat(x):\n    return math.sqrt(x + 1)\n>>>>>>> REPLACE\n"
    )
    updated, done, notes = apply_planned_reply(src, plan, reply)
    assert done == ["c1"] and not notes
    assert "import math" in updated and not plan_coverage(plan, src, updated)

    # Disallow editing an unrelated symbol
    bad_reply = (
        "CHANGE c1\n"
        "<<<<<<< SEARCH\ndef keep(x):\n    return x * 2\n=======\ndef keep(x):\n    return x * 3\n>>>>>>> REPLACE\n"
    )
    _, done_bad, _ = apply_planned_reply(src, plan, bad_reply)
    assert done_bad == []

    # __imports__ target works too
    imp_plan = {
        "schema_version": 1,
        "action": "refine",
        "evidence_ids": ["op1"],
        "hypothesis": "import json",
        "changes": [{"id": "i1", "component": "data_prep", "target": "__imports__", "edit": "import json", "check": "ok"}],
        "expected_observation": "ok",
        "preserve": ["keep"],
    }
    assert not validate_experiment_plan(imp_plan, "refine", {"op1"}, src)
    imp_reply = "CHANGE i1\n<<<<<<< SEARCH\nimport os\n=======\nimport os\nimport json\n>>>>>>> REPLACE\n"
    upd_imp, done_imp, _ = apply_planned_reply(src, imp_plan, imp_reply)
    assert done_imp == ["i1"] and not plan_coverage(imp_plan, src, upd_imp)


def test_smart_source_excerpt_preserves_tail_entrypoints():
    filler = "\n".join(f"x_{i} = {i}  # " + ("a" * 80) for i in range(400))
    code = f"import numpy as np\n\ndef fit():\n    return 1\n\n{filler}\n\ndef predict_only():\n    return 42\n"
    assert len(code) > 24000
    excerpt = _smart_source_excerpt(code, 24000)
    assert len(excerpt) <= 24000
    assert "def predict_only():" in excerpt and "import numpy as np" in excerpt


def test_review_risk_mode_preserves_threshold_blend_stack_and_ignores_comments(tmp_path):
    ctx = make_ctx(tmp_path, cfg_over={"review__mode": "risk", "review__enabled": True})
    parent = Node(id="p1", kind="draft", code="def f():\n    lr = 0.05\n    return lr\n")
    # Pure comment mentioning valid/label should fast-pass under risk mode
    benign = Node(id="n1", kind="refine", parent=parent, code="def f():\n    # label valid split comment\n    lr = 0.01\n    return lr\n")
    out_ok = review(ctx, benign)
    assert out_ok.verdict == "pass" and not out_ok.changed
    # Threshold/blend/stack snooping risk must trigger LLM review (and fail when no gateway exists -> None verdict)
    risky = Node(id="n2", kind="refine", parent=parent, code="def f():\n    best_threshold = 0.42\n    return best_threshold\n")
    with pytest.raises(AttributeError):
        review(ctx, risky)  # ctx.gateway is None only reached when high_risk == True
    assert "f" in _changed_symbol_names(parent.code, benign.code)


def test_prompts_deterministic_and_include_torch_gating():
    assert blocks.environment_block("off") == blocks.environment_block("off")
    ref_no_torch = reference(include_torch=False)
    assert "save_torch" not in ref_no_torch and "/method_assets" in ref_no_torch
    ref_torch = reference(include_torch=True)
    assert "save_torch" in ref_torch and "/method_assets" in ref_torch


def test_experience_tokenize_effect_label_and_context():
    toks = tokenize("StratifiedKFold tf-idf, ConvNeXt-Tiny 中文测试")
    assert "stratifiedkfold" in toks and "stratified" in toks and "kfold" in toks
    assert "tf" in toks and "idf" in toks
    assert "中" in toks and "文" in toks and "测" in toks and "试" in toks
    hi = lambda a, b: a > b
    assert effect_label("repair", None, None, hi, ok=False) == -1
    assert effect_label("draft", 0.70, None, hi, reference_scores=[0.6, 0.8, 0.9]) == -1
    assert effect_label("draft", 0.85, None, hi, reference_scores=[0.6, 0.8, 0.9]) == 1

    facts = [
        {"seq": 1, "operation_id": "op0", "node_id": "n0", "parent_id": "root", "action": "draft",
         "method": "m", "hypothesis": "base", "method_summary": "lgbm", "changes_summary": "init",
         "split_id": "s", "metric": "acc", "rung": 0, "trial_id": "t1", "deliverable": True, "status": "completed", "gain": None, "reason": "ok", "failed_stage": None},
        {"seq": 2, "operation_id": "op1", "node_id": "n1", "parent_id": "n0", "action": "promote",
         "method": "m", "hypothesis": "rung1", "method_summary": "lgbm r1", "changes_summary": "more trees",
         "split_id": "s", "metric": "acc", "rung": 1, "trial_id": "t1", "deliverable": True, "status": "completed", "gain": 0.05, "reason": "ok", "failed_stage": None},
        {"seq": 3, "operation_id": "op2", "node_id": "n2", "parent_id": "n0", "action": "refine",
         "method": "m", "hypothesis": "bad idea", "method_summary": "bad", "changes_summary": "drop col",
         "split_id": "s", "metric": "acc", "rung": 0, "trial_id": "t2", "deliverable": False, "status": "failed", "gain": -0.03, "reason": "OOM", "failed_stage": "executed"},
    ]
    prop = Proposal("p", "n0", "refine", "try depth", "generated", False, 3, target=None)
    ctx_rows = decision_context(facts, prop)
    ids = {r["operation_id"] for r in ctx_rows}
    assert {"op0", "op1", "op2"} <= ids
    assert any(r.get("changes_summary") == "more trees" for r in ctx_rows)


def test_checks_content_rejects_half_empty_and_masks_but_allows_imbalanced_binary(tmp_path):
    s = tmp_path / "s.csv"
    pd.DataFrame({"id": [str(i) for i in range(10)], "target": ["0"] * 10}).to_csv(s, index=False)
    half_empty = tmp_path / "half_empty.csv"
    pd.DataFrame({"id": [str(i) for i in range(10)], "target": ["1"] * 5 + [""] * 5}).to_csv(half_empty, index=False)
    assert not check_content(half_empty, s, 0.2).ok

    masks = tmp_path / "masks.csv"
    pd.DataFrame({"id": [str(i) for i in range(10)], "target": ["0 0 0"] * 6 + ["1 1 1"] * 4}).to_csv(masks, index=False)
    assert not check_content(masks, s, 0.99).ok

    imbalanced = tmp_path / "imbal.csv"
    pd.DataFrame({"id": [str(i) for i in range(50)], "target": ["0"] * 49 + ["1"]}).to_csv(imbalanced, index=False)
    assert check_content(imbalanced, None, 0.9).ok


def test_ensemble_negative_scores_and_final_binary_prob_and_stratified_halves():
    w = weights([-0.2, -0.5, -0.9], higher_is_better=True)
    assert len(w) == 3 and all(x > 0 for x in w) and math.isclose(sum(w), 1.0, rel_tol=1e-9)
    assert w[0] > w[1] > w[2]

    a = pd.DataFrame({"id": ["x", "y"], "p": ["0.2", "0.8"]})
    b = pd.DataFrame({"id": [" y ", " x "], "p": ["0.6", "0.4"]})
    comb = combine([a, b], [0.5, 0.5], detect_format(a))
    assert list(comb["id"]) == ["x", "y"] and [float(v) for v in comb["p"]] == pytest.approx([0.3, 0.7])

    spec_bin = SimpleNamespace(probability=True, columns=["id", "target"])
    spec_multi = SimpleNamespace(probability=True, columns=["id", "a", "b"])
    assert _spec_encoding(spec_bin) == "binary_prob"
    assert _spec_encoding(spec_multi) == "multi_prob"

    rows = [f"id{i}" for i in range(20)]
    y = ["maj"] * 16 + ["min"] * 4
    truth = pd.DataFrame({"label": y}, index=rows)
    h0, h1 = stratified_halves(rows, truth)
    assert len(h0) == 10 and len(h1) == 10
    assert sum(1 for r in h0 if truth.loc[r, "label"] == "min") == 2
    assert sum(1 for r in h1 if truth.loc[r, "label"] == "min") == 2
