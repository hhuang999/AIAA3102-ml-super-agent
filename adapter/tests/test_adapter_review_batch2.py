"""Unit tests for Batch 2 adapter fixes (A-P2-4)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from superagent_adapter.common.task_dir import check_isolation


def test_check_isolation_preserves_na_and_uses_header_first_scan(tmp_path, monkeypatch):
    public_task = tmp_path / "task"
    (public_task / "data").mkdir(parents=True)
    private_dir = tmp_path / "private"
    private_dir.mkdir()

    (public_task / "description.md").write_text("desc", encoding="utf-8")
    # Clean train.csv with "NA" string ID and NO target column -> must skip reading body and pass
    (public_task / "data" / "train.csv").write_text("id,feature\nNA,1.0\n002,2.0\n", encoding="utf-8")
    (public_task / "data" / "test.csv").write_text("id,feature\nNA,3.0\n", encoding="utf-8")
    (private_dir / "answers.csv").write_text("id,secret_label\nNA,pos\n", encoding="utf-8")

    read_calls = []
    real_read_csv = pd.read_csv

    def spy_read_csv(path, *args, **kwargs):
        read_calls.append((Path(path).name, kwargs.get("nrows"), kwargs.get("usecols"), kwargs.get("keep_default_na")))
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy_read_csv)
    rep = check_isolation(public_task, private_dir)
    assert rep.ok is True
    # Clean task CSVs (train.csv) only had nrows=0 header inspect + keep_default_na=False!
    train_reads = [c for c in read_calls if c[0] == "train.csv"]
    assert train_reads == [("train.csv", 0, None, False)]

    # If secret_label + matching "NA" ID leaks into leaked.csv, keep_default_na=False catches it
    (public_task / "data" / "leaked.csv").write_text("id,secret_label\nNA,pos\n", encoding="utf-8")
    rep_bad = check_isolation(public_task, private_dir)
    assert rep_bad.ok is False
    assert any("contains the answer columns" in p or "same name" in p for p in rep_bad.problems)
