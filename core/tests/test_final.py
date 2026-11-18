import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from helpers import make_ctx, mk
from superagent.assessment import _check_val_files
from superagent.final import Candidate, build_final, half_of
from superagent.metrics import detect_encoding, identify_metric, load_val_files

CLASSES = ["A", "B", "C"]


def _probs(rng, y, noise):
    logits = rng.normal(0, 1, (len(y), 3))
    logits[np.arange(len(y)), y] += 2.0
    logits += rng.normal(0, noise, logits.shape)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def _member(tmp_path, name, ids, y, p, test_p, cv=False):
    out = tmp_path / name / "out"
    out.mkdir(parents=True)
    cols = ["id"] + CLASSES
    onehot = np.eye(3)[y]
    pd.DataFrame(np.c_[ids, p], columns=cols).to_csv(out / "val_predictions.csv", index=False)
    pd.DataFrame(np.c_[ids, onehot.astype(int)], columns=cols).to_csv(out / "val_labels.csv", index=False)
    tid = [f"t{i}" for i in range(len(test_p))]
    pd.DataFrame(np.c_[tid, test_p], columns=cols).to_csv(out / "predictions.csv", index=False)
    (out / "model").mkdir()
    (out / "model" / "w.bin").write_text("x")
    return out


def _multi_members(tmp_path, n=4, rows=600, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"v{i}" for i in range(rows)]
    y = rng.integers(0, 3, rows)
    cands = []
    for k in range(n):
        p = _probs(rng, y, 1.0 + 0.1 * k)
        test_p = _probs(rng, rng.integers(0, 3, 50), 1.0)
        out = _member(tmp_path, f"m{k}", ids, y, p, test_p)
        cands.append((out, log_loss(y, p, labels=[0, 1, 2])))
    cands.sort(key=lambda t: t[1])
    return [Candidate(rank=i, node_id=f"node{i}", score=s, out_dir=o, code="print(1)") for i, (o, s) in enumerate(cands, 1)]


CFG = {"enabled": True, "allow_ensemble": True, "min_validation_rows": 50, "member_overlap": 0.9, "min_gain": 0.0001,
       "greedy_rounds": 20}


def test_identify_log_loss_and_build_ensemble(tmp_path):
    cands = _multi_members(tmp_path)
    info = build_final(tmp_path / "run", cands, False, CFG)
    assert info["metric"] == "log_loss" and info["encoding"] == "multi_prob"
    assert info["kind"] == "ensemble" and len(info["members"]) >= 2
    v = info["validation"]
    assert v["score"] < v["single_full_score"]
    assert len(v["leave_one_out_gain"]) == len(info["members"])
    assert all(g is not None for g in v["leave_one_out_gain"].values())
    final = tmp_path / "run" / "final"
    pred = pd.read_csv(final / "predictions.csv")
    assert list(pred.columns) == ["id"] + CLASSES and len(pred) == 50
    assert np.allclose(pred[CLASSES].sum(axis=1), 1.0)
    member_dirs = sorted(p.name for p in (final / "members").iterdir())
    assert len(member_dirs) == len(info["members"])
    assert (final / "members" / member_dirs[0] / "artifacts").is_symlink()


def test_combine_script_reproduces_final_predictions(tmp_path):
    cands = _multi_members(tmp_path)
    build_final(tmp_path / "run", cands, False, CFG)
    final = tmp_path / "run" / "final"
    before = pd.read_csv(final / "predictions.csv")
    (final / "predictions.csv").unlink()
    subprocess.run([sys.executable, str(final / "combine.py")], check=True)
    after = pd.read_csv(final / "predictions.csv")
    pd.testing.assert_frame_equal(before, after)


def test_identical_members_keep_the_single_best(tmp_path):
    cands = _multi_members(tmp_path, n=1)
    out0 = cands[0].out_dir
    copy = tmp_path / "copy" / "out"
    copy.mkdir(parents=True)
    for f in out0.iterdir():
        if f.is_file():
            (copy / f.name).write_bytes(f.read_bytes())
    cands.append(Candidate(rank=2, node_id="node2", score=cands[0].score, out_dir=copy))
    cands.append(Candidate(rank=3, node_id="node3", score=cands[0].score, out_dir=copy))
    info = build_final(tmp_path / "run", cands, False, CFG)
    assert info["kind"] == "single"
    assert "both validation halves" in info["reason"]


def test_ensembles_disabled(tmp_path):
    cands = _multi_members(tmp_path)
    info = build_final(tmp_path / "run", cands, False, {**CFG, "allow_ensemble": False})
    assert info["kind"] == "single" and info["members"][0]["node_id"] == cands[0].node_id
    best = pd.read_csv(cands[0].out_dir / "predictions.csv")
    pd.testing.assert_frame_equal(best, pd.read_csv(tmp_path / "run" / "final" / "predictions.csv"))


def test_missing_validation_files_keep_the_single_best(tmp_path):
    cands = _multi_members(tmp_path)
    for c in cands:
        (c.out_dir / "val_labels.csv").unlink()
    info = build_final(tmp_path / "run", cands, False, CFG)
    assert info["kind"] == "single" and info["reason"].startswith("metric not identified")


def test_identify_accuracy_for_label_predictions(tmp_path):
    rng = np.random.default_rng(1)
    ids = [f"r{i}" for i in range(400)]
    y = rng.choice(["cat", "dog", "fox"], 400)
    members = []
    for k in range(3):
        pred = np.where(rng.random(400) < 0.7 + 0.05 * k, y, rng.choice(["cat", "dog", "fox"], 400))
        out = tmp_path / f"l{k}"
        out.mkdir()
        pd.DataFrame({"id": ids, "label": pred}).to_csv(out / "val_predictions.csv", index=False)
        pd.DataFrame({"id": ids, "label": y}).to_csv(out / "val_labels.csv", index=False)
        members.append((load_val_files(out, ["id", "label"]), accuracy_score(y, pred)))
    ident = identify_metric(members, True)
    assert ident.encoding == "label" and ident.metric.name in ("accuracy",)


def test_identify_auc_for_binary_probabilities(tmp_path):
    rng = np.random.default_rng(2)
    ids = list(range(500))
    y = rng.integers(0, 2, 500)
    members = []
    for k in range(3):
        s = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
        out = tmp_path / f"b{k}"
        out.mkdir()
        pd.DataFrame({"id": ids, "target": s}).to_csv(out / "val_predictions.csv", index=False)
        pd.DataFrame({"id": ids, "target": y}).to_csv(out / "val_labels.csv", index=False)
        members.append((load_val_files(out, ["id", "target"]), roc_auc_score(y, s)))
    ident = identify_metric(members, True)
    assert ident.encoding == "binary_prob" and ident.metric.name == "roc_auc"


def test_no_metric_when_scores_do_not_match(tmp_path):
    cands = _multi_members(tmp_path, n=3)
    vals = [(load_val_files(c.out_dir, ["id"] + CLASSES), 0.123) for c in cands]
    ident = identify_metric(vals, False)
    assert ident.metric is None and "reproduces" in ident.reason


@pytest.mark.parametrize("change, message", [
    (lambda p, t: (p.rename(columns={"A": "a"}), t), "columns must be"),
    (lambda p, t: (pd.concat([p, p.iloc[:1]]), t), "duplicate ids"),
    (lambda p, t: (p, t.iloc[1:]), "same ids"),
])
def test_val_file_checks(tmp_path, change, message):
    ids = ["x", "y", "z"]
    p = pd.DataFrame({"id": ids, "A": [0.2, 0.3, 0.5], "B": [0.8, 0.7, 0.5]})
    t = pd.DataFrame({"id": ids, "A": [0, 1, 0], "B": [1, 0, 1]})
    p, t = change(p, t)
    p.to_csv(tmp_path / "val_predictions.csv", index=False)
    t.to_csv(tmp_path / "val_labels.csv", index=False)
    res = load_val_files(tmp_path, ["id", "A", "B"])
    assert not res.ok and message in res.message


def test_detect_encoding():
    def df(**cols):
        return pd.DataFrame(cols).astype(str)

    assert detect_encoding(df(y=["a", "b"]), df(y=["a", "a"])) == "label"
    assert detect_encoding(df(y=[0, 1]), df(y=[0.2, 0.9])) == "binary_prob"
    assert detect_encoding(df(a=[1, 0], b=[0, 1]), df(a=[0.6, 0.1], b=[0.4, 0.9])) == "multi_prob"
    assert detect_encoding(df(a=[1, 1], b=[0, 1]), df(a=[0.6, 0.7], b=[0.4, 0.9])) == "multilabel_prob"
    assert detect_encoding(df(y=[1.5, 2.5]), df(y=[1.2, 2.2])) is None


def test_halves_are_stable_and_balanced():
    ids = [f"id{i}" for i in range(2000)]
    halves = [half_of(i) for i in ids]
    assert halves == [half_of(i) for i in ids]
    assert 900 < sum(halves) < 1100


def test_assessment_notes_missing_val_files(tmp_path):
    ctx = make_ctx(tmp_path)
    node = mk(ctx.tree, "draft", ctx.tree.root, score=0.5)
    out = ctx.executor.prepare_run_dir(node.id, "print(1)") / "out"
    pd.read_csv(ctx.sample_path).to_csv(out / "predictions.csv", index=False)
    info = {}
    _check_val_files(ctx, node, out / "predictions.csv", info)
    assert node.val_files_ok is False and "missing" in info["val_files"]
    assert "cannot take part in validated ensembles" in node.output
    sample = pd.read_csv(ctx.sample_path)
    sample.iloc[:20].to_csv(out / "val_predictions.csv", index=False)
    sample.iloc[:20].to_csv(out / "val_labels.csv", index=False)
    info = {}
    _check_val_files(ctx, node, out / "predictions.csv", info)
    assert node.val_files_ok is True and "val_files" not in info


def test_source_json_is_complete(tmp_path):
    cands = _multi_members(tmp_path)
    build_final(tmp_path / "run", cands, False, CFG)
    src = json.loads((tmp_path / "run" / "final" / "source.json").read_text())
    for key in ("kind", "method", "metric", "validation", "candidates", "members", "format", "reason"):
        assert key in src
    assert abs(sum(m["weight"] for m in src["members"]) - 1.0) < 1e-6
