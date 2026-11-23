"""Hardening tests for adapter cost accounting, batch isolation, submission alignment and grader parsing."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from superagent_adapter.analysis import analyze
from superagent_adapter.batch import _archive_incomplete, _status
from superagent_adapter.common.costs import _receipts, recompute
from superagent_adapter.common.submission import make_submission
from superagent_adapter.common.task_dir import build_task_dir, check_isolation
from superagent_adapter.mlebench.grader import grade_all, parse_report


def test_analysis_handles_none_cost(tmp_path):
    o = tmp_path / "out"
    (o / "logs").mkdir(parents=True)
    (o / "report").mkdir(parents=True)
    (o / "logs" / "search_tree.json").write_text(json.dumps({
        "higher_is_better": True,
        "nodes": [{"id": "root", "kind": "root", "parent_id": None, "failed": None, "score": None, "failure_reasons": [],
                   "error_type": None, "error_message": None, "code": "", "branch_id": None, "step": 0, "method_summary": ""}],
    }))
    (o / "logs" / "decisions.jsonl").write_text("")
    (o / "status.json").write_text(json.dumps({"best_node_id": None, "best_score": None}))
    (o / "report" / "run_summary.json").write_text(json.dumps({
        "score_timeline": [], "best": {"path": []}, "key_events": [],
        "resources": {
            "llm": {"calls": 1, "failed_calls": 0, "prompt_tokens": 10, "completion_tokens": 5, "reasoning_tokens": 0,
                    "cost": None, "currency": "CNY", "by_agent": {"script_coder": {"calls": 1, "cost": None}}},
            "embedding": {"cost": None},
            "exec": {},
        },
    }))
    res = analyze(o)
    assert res["usage"]["cost_per_node"] is None


def test_costs_aliases_cache_lower_bound_non_billing_and_sqlite_immutable(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    # 1. SQLite receipts fallback & immutable=1
    db_path = logs / "operations.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE attempts (id TEXT PRIMARY KEY, op TEXT, role TEXT, call_id TEXT, sig TEXT, status TEXT, reserve REAL, cost REAL, usage TEXT, started REAL, finished REAL)")
    conn.execute(
        "INSERT INTO attempts VALUES ('a1','op1','script_coder','c1','s1','received',0.1,0.0,?,1.0,2.0)",
        (json.dumps({"model": "req-m", "returned_model": "ret-alias", "input_tokens": 1_000_000, "output_tokens": 500_000, "cached_tokens": 2_000_000}),),
    )
    conn.execute(
        "INSERT INTO attempts VALUES ('a2','op1','reviewer','c2','s2','rate_limited',0.1,99.0,?,1.1,2.1)",
        (json.dumps({"model": "req-m", "input_tokens": 1000, "output_tokens": 1000}),),
    )
    conn.execute(
        "INSERT INTO attempts VALUES ('a3','op1','reviewer','c3','s3','received',0.1,0.0,?,1.2,2.2)",
        (json.dumps({"model": "req-m", "input_tokens": 1_000_000, "output_tokens": 0}),),
    )
    conn.commit()
    conn.close()
    assert len(_receipts(db_path)) == 3
    prices = {
        "currency": "CNY",
        "aliases": {"ret-alias": "canon-model", "req-m": "canon-model"},
        "models": {"canon-model": {"input_per_million": 2.0, "cached_input_per_million": 0.2, "output_per_million": 8.0}},
    }
    res = recompute(tmp_path, prices)
    # rate_limited excluded; a1 clamps cached_tokens to 1_000_000 -> 0.2 + 4.0 = 4.2; a3 lacks cache split -> lower bound 1M*0.2 = 0.2 & input_complete=False
    assert res["llm"]["input_complete"] is False
    assert res["llm"]["cost"] == pytest.approx(4.4)
    assert res["models_without_price"] == []


def test_batch_corrupt_status_and_archive_incomplete(tmp_path):
    dirty = tmp_path / "run1"
    dirty.mkdir()
    (dirty / "status.json").write_text("{corrupt")
    assert _status(dirty) is None
    archived = _archive_incomplete(dirty)
    assert archived is not None and archived.exists() and not dirty.exists()
    assert ".incomplete_" in archived.name


def test_task_dir_isolation_keeps_na_ids(tmp_path):
    pub, priv = tmp_path / "pub", tmp_path / "priv"
    pub.mkdir(); priv.mkdir()
    (pub / "description.md").write_text("# Task")
    pd.DataFrame({"id": ["1", "2"], "label": ["a", "b"]}).to_csv(pub / "train.csv", index=False)
    pd.DataFrame({"id": ["NA", "3"], "x": [1, 2]}).to_csv(pub / "test.csv", index=False)
    pd.DataFrame({"id": ["NA", "3"], "label": ["a", "a"]}).to_csv(pub / "sample_submission.csv", index=False)
    pd.DataFrame({"id": ["NA", "3"], "target": ["a", "b"]}).to_csv(priv / "test.csv", index=False)
    t = build_task_dir(pub, tmp_path / "task")
    assert check_isolation(t, priv).ok
    # Leak table with "NA" ID and answer column must be caught even with keep_default_na=False
    pd.DataFrame({"id": ["NA", "3"], "target": ["a", "b"]}).to_csv(t / "data" / "leak.csv", index=False)
    assert not check_isolation(t, priv).ok


def test_make_submission_strips_ids_and_rejects_empty_cells(tmp_path):
    sample = tmp_path / "sample.csv"
    pred = tmp_path / "pred.csv"
    pd.DataFrame({"id": ["b", "a"], "label": ["0", "0"]}).to_csv(sample, index=False)
    pd.DataFrame({"id": [" a ", " b "], "label": ["cat", "dog"]}).to_csv(pred, index=False)
    out = make_submission(pred, sample, tmp_path / "out.csv")
    df = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert list(df["id"]) == ["b", "a"] and list(df["label"]) == ["dog", "cat"]

    bad_pred = tmp_path / "bad_pred.csv"
    pd.DataFrame({"id": ["a", "b"], "label": ["cat", ""]}).to_csv(bad_pred, index=False)
    with pytest.raises(ValueError, match="empty prediction"):
        make_submission(bad_pred, sample, tmp_path / "bad_out.csv")


def test_grader_raw_decode_and_missing_primary(tmp_path):
    log = 'Competition report: [prefix] {"score": 0.85, "gold_medal": false} trailing {braces} here\n'
    rep = parse_report(log)
    assert rep == {"score": 0.85, "gold_medal": False}

    core_out = tmp_path / "core_out"
    (core_out / "best").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="primary submission"):
        grade_all(core_out, tmp_path / "sample.csv", "mlebench", "comp", tmp_path / "data", tmp_path / "res")
