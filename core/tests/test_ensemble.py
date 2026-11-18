import subprocess
import sys

import pandas as pd
import pytest

from superagent.ensemble import combine, detect_format, member_sizes, run_ensemble, weights


def test_member_sizes():
    assert member_sizes(3) == [2, 3]
    assert member_sizes(5) == [2, 3, 4]
    assert member_sizes(6) == [2, 3, 4, 6]
    assert member_sizes(1) == []


def test_weights():
    assert weights([0.90, 0.88, 0.86], True) == pytest.approx([0.5075, 0.3308, 0.1617], abs=1e-4)
    assert weights([0.20, 0.25], False) == pytest.approx([0.7143, 0.2857], abs=1e-4)


def df(vals, cols=("label",)):
    data = {"id": [f"i{k}" for k in range(len(vals))]}
    for j, c in enumerate(cols):
        data[c] = [v[j] if isinstance(v, tuple) else v for v in vals]
    return pd.DataFrame(data).astype(str)


def test_strategies():
    assert detect_format(df(["0", "1", "2"]))["strategy"] == "vote"
    assert detect_format(df(["0.1", "0.9"]))["strategy"] == "average"
    assert detect_format(df(["cat", "dog"]))["strategy"] == "string_vote"
    f = detect_format(df([("0.3", "0.7"), ("0.5", "0.5")], cols=("a", "b")))
    assert f["strategy"] == "average" and f["normalize"]


def test_vote_and_average():
    w = weights([0.90, 0.88, 0.86], True)
    frames = [df(["A"]), df(["B"]), df(["B"])]
    assert combine(frames, w, detect_format(frames[0]))["label"].iloc[0] == "A"
    frames = [df(["0.9"]), df(["0.6"]), df(["0.3"])]
    assert float(combine(frames, w, detect_format(frames[0]))["label"].iloc[0]) == pytest.approx(0.7038, abs=1e-4)


def test_alignment():
    a = pd.DataFrame({"id": ["x", "y", "z"], "label": ["0.1", "0.2", "0.3"]})
    b = pd.DataFrame({"id": ["z", "x", "y"], "label": ["0.3", "0.1", "0.2"]})
    out = combine([a, b], [0.5, 0.5], detect_format(a))
    assert list(out["id"]) == ["x", "y", "z"]
    assert [float(v) for v in out["label"]] == pytest.approx([0.1, 0.2, 0.3])


def test_reproducible_script(tmp_path):
    top = []
    for r, s in enumerate([0.9, 0.85, 0.8], 1):
        d = tmp_path / "top" / f"rank_{r:02d}"
        d.mkdir(parents=True)
        pd.DataFrame({"id": ["a", "b", "c", "d"], "p": [0.1 * r, 0.2, 0.3, 0.4 * r]}).to_csv(d / "predictions.csv", index=False)
        top.append({"rank": r, "node_id": f"n{r}", "score": s})
    block = run_ensemble(tmp_path, top, True, {"max_candidates": 6, "primary_size": 3})
    assert block["available"]
    before = (tmp_path / "ensemble" / "predictions.csv").read_text()
    subprocess.run([sys.executable, str(tmp_path / "ensemble" / "ensemble.py")], check=True)
    assert (tmp_path / "ensemble" / "predictions.csv").read_text() == before
