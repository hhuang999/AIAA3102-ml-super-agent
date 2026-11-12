"""Assessment rules. The log reader's answer (and, where needed, the column and leak answers)
are given directly as inputs; no LLM call is made."""

import pandas as pd

from helpers import make_ctx, mk
from superagent.assessment import ReaderResult, apply_assessment, apply_column_answer, fallback_reader, leak_decision, leak_triggered
from superagent.executor import ExecResult


def reader(has_error=False, score=0.9, goal="maximize"):
    return ReaderResult(has_error, "ok", score, goal, "m", False)


def run_node(ctx, output, score_rows=100, values=None, write_pred=True):
    n = mk(ctx.tree, "draft", ctx.tree.root, failed=None)
    n.code = "print(1)"
    n.output = output
    d = ctx.executor.prepare_run_dir(n.id, n.code)
    if write_pred:
        vals = values if values is not None else [i % 2 for i in range(score_rows)]
        pd.DataFrame({"id": [f"r{i}" for i in range(score_rows)], "label": vals}).to_csv(d / "out" / "predictions.csv", index=False)
    return n, ExecResult(output, 1.0, None, None, False, 0, 100)


def test_reader_failure_fallback(tmp_path):
    ctx = make_ctx(tmp_path)
    n, res = run_node(ctx, "VALIDATION_SCORE: 0.9\n")
    fb = fallback_reader(n, res, True)
    assert fb.fallback and fb.score is None and fb.goal == "maximize" and not fb.has_error
    apply_assessment(ctx, n, res, fb)
    assert n.ok and n.score == 0.9 and n.score_source == "marker"


def test_no_score(tmp_path):
    ctx = make_ctx(tmp_path)
    n, res = run_node(ctx, "nothing\n")
    apply_assessment(ctx, n, res, reader(score=None))
    assert n.is_failed and "no_score" in n.failure_reasons


def test_marker_overrides_reader_score(tmp_path):
    ctx = make_ctx(tmp_path)
    n, res = run_node(ctx, "VALIDATION_SCORE: 0.873\n")
    info = apply_assessment(ctx, n, res, reader(score=0.91))
    assert n.ok and n.score == 0.873 and n.score_source == "marker" and info["score_mismatch"]


def test_zero_score(tmp_path):
    ctx = make_ctx(tmp_path)
    n, res = run_node(ctx, "VALIDATION_SCORE: 0\n")
    apply_assessment(ctx, n, res, reader(score=0.0))
    assert n.is_failed and n.failure_reasons == ["zero_score"]


def test_zero_score_lower_is_better_triggers_leak_check(tmp_path):
    ctx = make_ctx(tmp_path, higher=False)
    n, res = run_node(ctx, "VALIDATION_SCORE: 0\n")
    asked = []

    def leak_answer():
        asked.append(1)
        return {"leak_found": False, "evidence": "fine", "certainty": "low"}

    info = apply_assessment(ctx, n, res, reader(score=0.0, goal="minimize"), leak_answer=leak_answer)
    assert n.ok and n.score == 0.0
    assert asked == [1] and info["leak_check_triggered"]


def test_leak_thresholds():
    assert leak_triggered(0.999, True, 0.999, 1e-6)
    assert not leak_triggered(0.9989, True, 0.999, 1e-6)
    assert leak_triggered(0.0, False, 0.999, 1e-6)
    assert not leak_decision(True, "low")
    assert leak_decision(True, "medium")


def test_leak_not_asked_for_ordinary_score(tmp_path):
    ctx = make_ctx(tmp_path)
    n, res = run_node(ctx, "VALIDATION_SCORE: 0.9\n")
    asked = []
    apply_assessment(ctx, n, res, reader(), leak_answer=lambda: asked.append(1))
    assert n.ok and not asked


def test_leak_medium_fails(tmp_path):
    ctx = make_ctx(tmp_path)
    n, res = run_node(ctx, "VALIDATION_SCORE: 0.9995\n")
    verdict = {"leak_found": True, "evidence": "scaler fit on all data", "certainty": "medium"}
    apply_assessment(ctx, n, res, reader(score=0.9995), leak_answer=lambda: verdict)
    assert n.is_failed and n.failure_reasons == ["data_leak"] and n.summary.startswith("DATA_LEAK:")


def test_leak_low_keeps_node(tmp_path):
    ctx = make_ctx(tmp_path)
    n, res = run_node(ctx, "VALIDATION_SCORE: 0.9995\n")
    verdict = {"leak_found": True, "evidence": "unsure", "certainty": "low"}
    apply_assessment(ctx, n, res, reader(score=0.9995), leak_answer=lambda: verdict)
    assert n.ok


def test_leak_answer_unavailable_keeps_node(tmp_path):
    ctx = make_ctx(tmp_path)
    n, res = run_node(ctx, "VALIDATION_SCORE: 0.9995\n")
    apply_assessment(ctx, n, res, reader(score=0.9995), leak_answer=lambda: None)
    assert n.ok


def test_direction_mismatch(tmp_path):
    ctx = make_ctx(tmp_path)
    n, res = run_node(ctx, "VALIDATION_SCORE: 0.9\n")
    apply_assessment(ctx, n, res, reader(goal="minimize"))
    assert n.failure_reasons == ["direction_mismatch"]


def test_constant_predictions(tmp_path):
    ctx = make_ctx(tmp_path)
    n, res = run_node(ctx, "VALIDATION_SCORE: 0.9\n", values=["1"] * 100)
    apply_assessment(ctx, n, res, reader())
    assert n.failure_reasons == ["content_error"] and n.predictions_ok is False and n.summary.startswith("CONTENT_ERROR")


def test_missing_predictions(tmp_path):
    ctx = make_ctx(tmp_path)
    n, res = run_node(ctx, "VALIDATION_SCORE: 0.9\n", write_pred=False)
    apply_assessment(ctx, n, res, reader())
    assert "no_predictions" in n.failure_reasons


def test_column_names_fixed_with_given_answer(tmp_path):
    ctx = make_ctx(tmp_path)
    n, res = run_node(ctx, "VALIDATION_SCORE: 0.9\n")
    p = ctx.executor.run_dir(n.id) / "out" / "predictions.csv"
    pd.read_csv(p).rename(columns={"label": "prediction"}).to_csv(p, index=False)
    apply_assessment(ctx, n, res, reader(), column_answer=lambda pred, sample, scols, pcols: ["id", "label"])
    assert n.ok and list(pd.read_csv(p).columns) == ["id", "label"]


def test_column_names_without_answer_fail(tmp_path):
    ctx = make_ctx(tmp_path)
    n, res = run_node(ctx, "VALIDATION_SCORE: 0.9\n")
    p = ctx.executor.run_dir(n.id) / "out" / "predictions.csv"
    pd.read_csv(p).rename(columns={"label": "prediction"}).to_csv(p, index=False)
    apply_assessment(ctx, n, res, reader())
    assert n.failure_reasons == ["format_error"] and n.summary.startswith("FORMAT_ERROR")


def test_apply_column_answer_rejects_wrong_length(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text("id,prediction\nr0,1\n")
    assert not apply_column_answer(p, ["id"], ["id", "prediction"])
    assert not apply_column_answer(p, None, ["id", "prediction"])
    assert apply_column_answer(p, ["id", "label"], ["id", "prediction"])
    assert p.read_text().splitlines()[0] == "id,label"
