"""Equal-weight votes, the log-loss tie-break, the synthesis time check and failure counting."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from helpers import cfg, make_ctx, mk, tree
from superagent.control import StepTiming
from superagent.ensemble import weights
from superagent.final import Candidate, build_final
from superagent.scheduling import synthesis_blocker, synthesis_conditions
from superagent.search import SearchController

CFG = {"enabled": True, "allow_ensemble": True, "min_validation_rows": 50, "member_overlap": 0.9, "min_gain": 0.0001,
       "greedy_rounds": 20, "tie_break": True, "tie_break_min_gain": 0.001, "small_validation_rows": 200}


def _write(out, ids, col, val_pred, val_true, test_pred):
    out.mkdir(parents=True)
    pd.DataFrame({"id": ids, col: val_pred}).to_csv(out / "val_predictions.csv", index=False)
    pd.DataFrame({"id": ids, col: val_true}).to_csv(out / "val_labels.csv", index=False)
    pd.DataFrame({"id": [f"t{i}" for i in range(len(test_pred))], col: test_pred}).to_csv(out / "predictions.csv", index=False)


def test_vote_weights_are_equal_and_average_weights_fall_with_rank():
    assert weights([0.9, 0.8, 0.7], True, "string_vote") == [1 / 3] * 3
    assert weights([0.9, 0.8, 0.7], True, "vote") == [1 / 3] * 3
    w = weights([0.9, 0.8, 0.7], True, "average")
    assert w[0] > w[1] > w[2] and abs(sum(w) - 1) < 1e-9
    assert weights([0.9, 0.8, 0.7], True) == w


def _label_members(tmp_path, n=3, rows=600, err=0.2, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.array(["a", "b", "c"])
    ids = [f"v{i}" for i in range(rows)]
    y = labels[rng.integers(0, 3, rows)]
    cands = []
    for k in range(n):
        pred = y.copy()
        wrong = rng.random(rows) < err + 0.01 * k
        pred[wrong] = [labels[(list(labels).index(v) + 1 + rng.integers(0, 2)) % 3] for v in y[wrong]]
        out = tmp_path / f"m{k}" / "out"
        _write(out, ids, "label", pred, y, labels[rng.integers(0, 3, 40)])
        cands.append((out, accuracy_score(y, pred)))
    cands.sort(key=lambda t: -t[1])
    return [Candidate(rank=i, node_id=f"n{i}", score=s, out_dir=o) for i, (o, s) in enumerate(cands, 1)]


def test_majority_vote_of_three_label_solutions_can_be_chosen(tmp_path):
    cands = _label_members(tmp_path)
    info = build_final(tmp_path / "run", cands, True, CFG)
    assert info["encoding"] == "label" and info["metric"] == "accuracy"
    assert info["kind"] == "ensemble", info["reason"]
    assert info["validation"]["score"] > info["validation"]["single_full_score"]
    top3 = next(c for c in info["candidates"] if c["method"] == "top3")
    assert top3["weights"] == [round(1 / 3, 6)] * 3
    assert top3["half_scores"] != info["validation"]["single_half_scores"]


def _saturated_binary(tmp_path, n=3, rows=400, seed=1):
    """Every member ranks the samples perfectly (AUC 1.0) but with different, noisy probabilities."""
    rng = np.random.default_rng(seed)
    ids = [f"v{i}" for i in range(rows)]
    y = rng.integers(0, 2, rows)
    cands = []
    for k in range(n):
        p = np.where(y == 1, rng.uniform(0.55, 0.95, rows), rng.uniform(0.05, 0.45, rows))
        out = tmp_path / f"b{k}" / "out"
        _write(out, ids, "target", np.round(p, 6), y, np.round(rng.random(30), 6))
        cands.append(Candidate(rank=k + 1, node_id=f"b{k}", score=1.0, out_dir=out))
    return cands


def test_tie_break_by_log_loss_when_auc_is_saturated(tmp_path):
    cands = _saturated_binary(tmp_path)
    info = build_final(tmp_path / "run", cands, True, CFG)
    assert info["metric"] == "roc_auc"
    assert info["validation"]["single_half_scores"] == [1.0, 1.0]
    assert info["kind"] == "ensemble" and info["decided_by_tie_break"] is True
    tb = info["validation"]["tie_break"]
    assert tb["metric"] == "log_loss" and tb["enabled"] is True
    chosen = next(c for c in info["candidates"] if c["method"] == info["method"])
    assert all(v in ("metric", "tie_break") for v in chosen["half_verdicts"])
    assert chosen["full_tie_break_score"] < tb["single_full_score"]


def test_tie_break_disabled_keeps_single_and_explains(tmp_path):
    cands = _saturated_binary(tmp_path)
    info = build_final(tmp_path / "run", cands, True, {**CFG, "tie_break": False})
    assert info["kind"] == "single"
    assert "tied" in info["reason"] and "no second measure" in info["reason"]


def test_small_validation_set_warning(tmp_path):
    cands = _saturated_binary(tmp_path, rows=120)
    info = build_final(tmp_path / "run", cands, True, CFG)
    assert "warning" in info["validation"]


def test_synthesis_blocked_when_time_is_short():
    t = tree()
    c = cfg(operators__synthesis__min_branches=1, operators__synthesis__min_ok_per_branch=1)
    d = mk(t, "draft", t.root, score=0.5)
    mk(t, "refine", d, score=0.4)
    t.synthesis_time_ok = False
    assert synthesis_blocker(t, c, 0.6) == "not_enough_time"
    assert synthesis_conditions(t, c, 0.6) is False


def test_synthesis_time_check_uses_measured_step_times(tmp_path):
    ctx = make_ctx(tmp_path)
    holder = SimpleNamespace(ctx=ctx)
    scfg = ctx.cfg
    assert SearchController._synthesis_time_ok(holder, scfg) is True     # nothing measured yet
    # Typical step: 290 s outside the script, 125 s script -> a synthesis step needs about 415 s.
    for i in range(4):
        ctx.record_step(StepTiming(i + 1, f"n{i}", "refine", True, 415.0, 125.0))
    ctx.cfg["run"]["finalize_reserve_seconds"] = 300
    ctx.time_left = lambda: 300.0 + 307.0                                  # 307 s usable, as in the recorded run
    assert SearchController._synthesis_time_ok(holder, scfg) is False
    ctx.time_left = lambda: 300.0 + 600.0
    assert SearchController._synthesis_time_ok(holder, scfg) is True


def test_failed_node_counts_once(tmp_path):
    from superagent.report import build_summary

    ctx = make_ctx(tmp_path)
    t = ctx.tree
    d = mk(t, "draft", t.root, score=0.5)
    bad = mk(t, "refine", d, failed=True)
    bad.failure_reasons = ["reported_error", "runtime_error", "no_score", "no_predictions"]
    t.best = d
    f = build_summary(ctx)["failures"]
    assert f["failed_nodes"] == 1 and f["by_reason"] == {"reported_error": 1}
    assert f["all_reasons"]["no_score"] == 1 and f["examples"]["reported_error"]["all_reasons"][-1] == "no_predictions"
