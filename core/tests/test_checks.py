from pathlib import Path

import pandas as pd
import pytest

from superagent.checks import check_content, check_format, extract_marker_score, reconcile_score


def test_marker_last_wins():
    out = "VALIDATION_SCORE: 0.8\nfoo\n  VALIDATION_SCORE: 0.9123  \n"
    d = reconcile_score(out, 0.9123, 1e-6)
    assert (d.score, d.source, d.mismatch) == (0.9123, "marker", False)
    d = reconcile_score(out, 0.91, 1e-6)
    assert (d.score, d.source, d.mismatch) == (0.9123, "marker", True)


def test_marker_nan_uses_llm():
    d = reconcile_score("VALIDATION_SCORE: nan\n", 0.88, 1e-6)
    assert (d.score, d.source) == (0.88, "llm")


def test_no_marker_no_llm():
    assert reconcile_score("nothing", None, 1e-6).score is None


def test_prefixed_marker_not_counted():
    assert extract_marker_score("Final VALIDATION_SCORE: 0.9") == (False, None)


@pytest.fixture
def sample(tmp_path):
    p = tmp_path / "sample.csv"
    pd.DataFrame({"id": ["a", "b", "c"], "label": ["x", "y", "z"]}).to_csv(p, index=False)
    return p


def write(tmp_path, df):
    p = tmp_path / "pred.csv"
    df.to_csv(p, index=False)
    return p


def test_format_ok_any_row_order(tmp_path, sample):
    assert check_format(write(tmp_path, pd.DataFrame({"id": ["c", "a", "b"], "label": [1, 2, 3]})), sample).ok


def test_format_wrong_name(tmp_path, sample):
    r = check_format(write(tmp_path, pd.DataFrame({"id": ["a", "b", "c"], "prediction": [1, 2, 3]})), sample)
    assert r.rule == 1 and r.fixable_columns


def test_format_reorder(tmp_path, sample):
    p = write(tmp_path, pd.DataFrame({"label": [1, 2, 3], "id": ["a", "b", "c"]}))
    r = check_format(p, sample)
    assert r.ok and r.reordered
    assert list(pd.read_csv(p).columns) == ["id", "label"]


def test_format_rows_ids_empty(tmp_path, sample):
    assert check_format(write(tmp_path, pd.DataFrame({"id": ["a", "b"], "label": [1, 2]})), sample).rule == 2
    assert check_format(write(tmp_path, pd.DataFrame({"id": ["a", "b", "b"], "label": [1, 2, 3]})), sample).rule == 3
    assert check_format(write(tmp_path, pd.DataFrame({"id": ["a", "b", "d"], "label": [1, 2, 3]})), sample).rule == 3
    assert check_format(write(tmp_path, pd.DataFrame({"id": ["a", "b", "c"], "label": ["1", "", "3"]})), sample).rule == 4


@pytest.mark.parametrize("values,ok", [
    (["1"] * 100, False),
    (["cat"] * 96 + ["dog"] * 2 + ["bird"] * 2, False),
    (["0"] * 99 + ["1"], True),
    (["a"] * 60 + ["b"] * 30 + ["c"] * 10, True),
    ([""] * 100, False),
    (["0 0 0 0"] * 60 + [f"{i} 1" for i in range(40)], False),
])
def test_content(tmp_path, values, ok):
    p = write(tmp_path, pd.DataFrame({"id": [str(i) for i in range(100)], "label": values}))
    s = tmp_path / "s.csv"
    pd.DataFrame({"id": [str(i) for i in range(100)], "label": ["x"] * 100}).to_csv(s, index=False)
    assert check_content(p, s, 0.95).ok is ok
