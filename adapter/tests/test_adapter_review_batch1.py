"""Unit tests for Batch 1 adapter fixes (A-P1-6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from superagent_adapter.batch import run_one
from superagent_adapter.common.submission import make_submission
from superagent_adapter.mlebench.grader import grade_all, parse_report


def test_batch_archives_incomplete_nonempty_output_dir(tmp_path, monkeypatch):
    task_dir = tmp_path / "task"
    (task_dir / "data").mkdir(parents=True)
    (task_dir / "description.md").write_text("task desc", encoding="utf-8")

    out_dir = tmp_path / "runs" / "r1"
    out_dir.mkdir(parents=True)
    (out_dir / "status.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    (out_dir / "stale.txt").write_text("leftover", encoding="utf-8")

    def fake_run_core(cmd, _cwd, _log):
        target_out = Path(cmd[cmd.index("--output-dir") + 1])
        assert not target_out.exists()
        return 0, {"status": "success", "final_kind": "single", "final_validation_score": 0.9}

    monkeypatch.setattr("superagent_adapter.batch.run_core", fake_run_core)
    plan = {"core_python": "python"}
    run = {"name": "r1", "task_dir": "task", "output_dir": "runs/r1"}
    row = run_one(plan, run, tmp_path)
    assert row["status"] == "success"
    archived = list((tmp_path / "runs").glob("r1.incomplete_*"))
    assert len(archived) == 1
    assert (archived[0] / "stale.txt").is_file()


def test_parse_report_ignores_trailing_braces_and_grade_all_checks_primary(tmp_path):
    log_text = (
        "[2026-11-21] Competition report:\n"
        '{\n  "competition_id": "demo",\n  "score": 0.875,\n  "gold_medal": false\n}\n'
        "[warning] trailing debug object {'foo': 1} at shutdown\n"
    )
    rep = parse_report(log_text)
    assert rep is not None
    assert rep["score"] == 0.875

    # Formal output with task_facts.json but missing final/predictions.csv raises clear RuntimeError
    core_out = tmp_path / "out"
    (core_out / "logs").mkdir(parents=True)
    (core_out / "logs" / "task_facts.json").write_text("{}", encoding="utf-8")
    sample = tmp_path / "sample.csv"
    sample.write_text("id,label\n1,a\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="primary prediction 'final' is missing"):
        grade_all(core_out, sample, "mlebench", "demo", tmp_path, tmp_path / "grades")


def test_make_submission_strips_ids_and_rejects_empty_predictions(tmp_path):
    sample = tmp_path / "sample.csv"
    sample.write_text("id,label\n001,a\n002,a\n", encoding="utf-8")

    pred_ok = tmp_path / "pred_ok.csv"
    pred_ok.write_text("id,label\n 002 ,cat\n001,dog\n", encoding="utf-8")
    out_csv = tmp_path / "sub.csv"
    make_submission(pred_ok, sample, out_csv)
    lines = out_csv.read_text(encoding="utf-8").splitlines()
    assert lines == ["id,label", "001,dog", "002,cat"]

    pred_empty = tmp_path / "pred_empty.csv"
    pred_empty.write_text("id,label\n001,dog\n002,   \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty prediction cells"):
        make_submission(pred_empty, sample, tmp_path / "bad.csv")


def test_colab_setup_runs_apt_get_update():
    setup_sh = Path(__file__).resolve().parents[1] / "launchers" / "colab" / "remote" / "setup.sh"
    text = setup_sh.read_text(encoding="utf-8")
    assert "apt-get update" in text and "bubblewrap" in text
