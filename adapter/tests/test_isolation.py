from pathlib import Path

import pandas as pd
import pytest

from superagent_adapter.common.task_dir import build_task_dir, check_isolation


@pytest.fixture
def split(tmp_path):
    pub, priv = tmp_path / "public", tmp_path / "private"
    pub.mkdir(); priv.mkdir()
    (pub / "description.md").write_text("# Task\nPredict the author.")
    pd.DataFrame({"id": ["a", "b"], "text": ["x y", "z w"], "author": ["A", "B"]}).to_csv(pub / "train.csv", index=False)
    pd.DataFrame({"id": ["c", "d"], "text": ["q", "r"]}).to_csv(pub / "test.csv", index=False)
    pd.DataFrame({"id": ["c", "d"], "A": [0.5, 0.5], "B": [0.5, 0.5]}).to_csv(pub / "sample_submission.csv", index=False)
    pd.DataFrame({"id": ["c", "d"], "A": [1, 0], "B": [0, 1]}).to_csv(priv / "test.csv", index=False)
    return pub, priv


def test_clean_task_dir_passes(split, tmp_path):
    pub, priv = split
    t = build_task_dir(pub, tmp_path / "task")
    rep = check_isolation(t, priv)
    assert rep.ok, rep.problems
    assert (t / "description.md").exists() and (t / "data" / "sample_predictions.csv").exists()
    assert not (t / "data" / "sample_submission.csv").exists()


def test_private_answers_copied_are_detected(split, tmp_path):
    pub, priv = split
    t = build_task_dir(pub, tmp_path / "task")
    (t / "data" / "answers.csv").write_bytes((priv / "test.csv").read_bytes())
    rep = check_isolation(t, priv)
    assert not rep.ok and any("same content" in p for p in rep.problems)


def test_label_columns_in_test_detected(split, tmp_path):
    pub, priv = split
    t = build_task_dir(pub, tmp_path / "task")
    pd.DataFrame({"id": ["c", "d"], "text": ["q", "r"], "A": [1, 0]}).to_csv(t / "data" / "test.csv", index=False)
    rep = check_isolation(t, priv)
    assert not rep.ok and any("answer columns" in p for p in rep.problems)


def test_answers_with_other_name_detected(split, tmp_path):
    pub, priv = split
    t = build_task_dir(pub, tmp_path / "task")
    pd.DataFrame({"id": ["c", "d"], "A": [1, 0], "B": [0, 1], "extra": [1, 2]}).to_csv(t / "data" / "x.csv", index=False)
    assert not check_isolation(t, priv).ok


def test_real_prepared_split_if_present():
    """Set MLEBENCH_PREPARED_DIR to <data dir>/<competition>/prepared to check a real split."""
    import os

    root = Path(os.environ.get("MLEBENCH_PREPARED_DIR", ""))
    if not os.environ.get("MLEBENCH_PREPARED_DIR") or not root.exists():
        pytest.skip("MLEBENCH_PREPARED_DIR is not set")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        t = build_task_dir(root / "public", Path(d) / "task")
        rep = check_isolation(t, root / "private")
        assert rep.ok, rep.problems
