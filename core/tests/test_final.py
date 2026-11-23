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


def test_identify_metric_prefers_hinted_and_exact_accuracy_over_noisy_precision_macro(tmp_path):
    from superagent.metrics import extract_metric_hints

    assert "accuracy" in extract_metric_hints("Metric: accuracy = correctly classified rows / evaluated rows")
    assert "accuracy" not in extract_metric_hints("Evaluation metric: balanced accuracy")

    ids = [f"r{i}" for i in range(200)]
    y = ([0] * 110) + ([1] * 90)
    # Construct two members: one exact accuracy match, one slightly shifted reported OOF score
    p1 = y[:190] + [1 - v for v in y[190:]]
    p2 = ([1 - v for v in y[:8]]) + y[8:196] + ([1 - v for v in y[196:]])
    members = []
    for idx, (pred, rep) in enumerate([(p1, sum(a == b for a, b in zip(y, p1)) / len(y)), (p2, 0.943)]):
        out = tmp_path / f"m{idx}"
        out.mkdir()
        pd.DataFrame({"id": ids, "satisfaction": pred}).to_csv(out / "val_predictions.csv", index=False)
        pd.DataFrame({"id": ids, "satisfaction": y}).to_csv(out / "val_labels.csv", index=False)
        members.append((load_val_files(out, ["id", "satisfaction"]), rep))
    ident = identify_metric(members, True, task_desc="Metric: accuracy, higher is better.")
    assert ident.metric is not None and ident.metric.name == "accuracy"


def test_greedy_weights_escapes_tie_trap_via_pair_step_on_discrete_votes(tmp_path):
    from superagent.final import Scorer, greedy_weights
    from superagent.ensemble import detect_format
    from superagent.metrics import metric_library

    ids = [f"v{i}" for i in range(60)]
    y = [str(i % 3) for i in range(60)]
    # Member 0 gets 48/60 (misses indices 0..11)
    # Member 1 gets 46/60 (correct on 0..11, misses 12..25)
    # Member 2 gets 46/60 (correct on 0..11, misses 26..39)
    # Any 2-member vote [0.5, 0.5] ties and picks Member 0 (48/60),
    # whereas 3-member majority [1/3, 1/3, 1/3] gets ALL 60/60 (1.0)!
    p0 = [str((i + 1) % 3) if i < 12 else y[i] for i in range(60)]
    p1 = [str((i + 1) % 3) if 12 <= i < 26 else y[i] for i in range(60)]
    p2 = [str((i + 1) % 3) if 26 <= i < 40 else y[i] for i in range(60)]
    vals = []
    for k, p in enumerate((p0, p1, p2)):
        d = tmp_path / f"v{k}"
        d.mkdir()
        pd.DataFrame({"id": ids, "label": p}).to_csv(d / "val_predictions.csv", index=False)
        pd.DataFrame({"id": ids, "label": y}).to_csv(d / "val_labels.csv", index=False)
        vals.append(load_val_files(d, ["id", "label"]))
    fmt = detect_format(pd.DataFrame({"id": ids, "label": p0}))
    acc = next(m for m in metric_library() if m.encoding == "label" and m.name == "accuracy")
    scorer = Scorer(vals, vals[0].true, acc, fmt, "id")
    w = greedy_weights(scorer, 3, ids, 6, True)
    assert w[1] > 0 and w[2] > 0
    assert scorer.score(w, ids) == 1.0


def test_probability_sidecar_ensemble_and_combine_replay_multiclass_and_course(tmp_path):
    import subprocess
    import sys

    ids = [f"r{i}" for i in range(200)]
    t_ids = [f"t{i}" for i in range(40)]
    classes = ["buildings", "forest", "sea"]
    y_idx = [i % 3 for i in range(200)]
    y = [classes[i] for i in y_idx]

    # Member 0 and Member 1: calibrated probabilities where probability average beats both on both halves!
    cands = []
    for k in range(2):
        d = tmp_path / f"m{k}"
        d.mkdir()
        probs = np.full((200, 3), 0.1)
        for i, yi in enumerate(y_idx):
            # Member 0 errs softly (0.45 true vs 0.48 wrong) on even blocks; Member 1 errs softly on odd blocks,
            # while the other member is confident (0.85 true)! Average probability is 0.65 true -> 100% accuracy!
            if (i % 10 < 3 and k == 0) or (3 <= i % 10 < 6 and k == 1):
                wrong = (yi + 1) % 3
                probs[i] = [0.07, 0.07, 0.07]
                probs[i, yi] = 0.45
                probs[i, wrong] = 0.48
            else:
                probs[i] = [0.075, 0.075, 0.075]
                probs[i, yi] = 0.85
        pred_lbl = [classes[int(j)] for j in probs.argmax(axis=1)]
        sc = float(np.mean([a == b for a, b in zip(y, pred_lbl)]))
        pd.DataFrame({"id": ids, "label": pred_lbl}).to_csv(d / "val_predictions.csv", index=False)
        pd.DataFrame({"id": ids, "label": y}).to_csv(d / "val_labels.csv", index=False)
        pd.DataFrame({"id": ids, **{c: probs[:, j] for j, c in enumerate(classes)}}).to_csv(d / "val_probabilities.csv", index=False)

        t_probs = np.tile([0.2, 0.5, 0.3] if k == 0 else [0.1, 0.6, 0.3], (40, 1))
        t_lbl = [classes[int(j)] for j in t_probs.argmax(axis=1)]
        pd.DataFrame({"id": t_ids, "label": t_lbl}).to_csv(d / "predictions.csv", index=False)
        pd.DataFrame({"id": t_ids, **{c: t_probs[:, j] for j, c in enumerate(classes)}}).to_csv(d / "test_probabilities.csv", index=False)
        cands.append(Candidate(rank=k + 1, node_id=f"node{k}", score=sc, out_dir=d, code="print(1)"))

    run_dir = tmp_path / "run_out"
    info = build_final(run_dir, cands, True, CFG, task_desc="Metric: accuracy")
    assert info["kind"] == "ensemble"
    assert info["format"]["strategy"] == "prob_argmax"
    assert info["validation"]["score"] == 1.0

    # Verify standalone final/combine.py reproduces final/predictions.csv byte-for-byte
    orig_bytes = (run_dir / "final/predictions.csv").read_bytes()
    (run_dir / "final/predictions.csv").unlink()
    subprocess.run([sys.executable, str(run_dir / "final/combine.py")], check=True)
    assert (run_dir / "final/predictions.csv").read_bytes() == orig_bytes
