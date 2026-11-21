"""Result assessment: LLM log reading followed by independent code checks."""

from __future__ import annotations

import logging
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .checks import check_content, check_format, head_text, read_str_csv, reconcile_score, rename_columns
from .context import RunContext
from .errors import LLMUnavailableError, StructuredOutputError
from .executor import ExecResult
from .llm import Field
from .metrics import load_val_files
from .prompts import builders
from .textfmt import trim_with
from .tree import Node
from .utils import now_iso

log = logging.getLogger("superagent")

READER_SCHEMA = {
    "has_error": Field("boolean", "true if the run failed or hit an error"),
    "observations": Field("string", "2-3 sentences of observations"),
    "score": Field("number", "validation score, or null", nullable=True),
    "goal": Field("enum", "whether the reported score is better lower or higher", values=("minimize", "maximize")),
    "method_summary": Field("string", "summary of the method per stage"),
}
LEAK_SCHEMA = {
    "leak_found": Field("boolean", "whether a leak exists"),
    "evidence": Field("string", "code construct and location, or why the score is plausible"),
    "certainty": Field("enum", "how certain", values=("high", "medium", "low")),
}
FIXER_SCHEMA = {"columns": Field("string_list", "sample column name for each current column")}

ZERO_SCORE_SUMMARY = (
    "ZERO_SCORE: the validation score is exactly 0, which means complete failure. Check: the model architecture may be "
    "wrong or not learning; data preprocessing may be broken; the loss or metric computation may be wrong; the training "
    "loop may not update the weights; the input data may not be read correctly."
)
LEAK_FIX_HINT = (
    "Fix hints: check the train/validation split logic; make sure no validation or test data is used in training; "
    "compute feature-engineering statistics on the training part only; never let augmentation copy validation samples "
    "into training; split by time or group when needed."
)
CONTENT_FIX_HINT = (
    "Every prediction must come from real model inference: read each test sample, apply the same preprocessing as in "
    "training, pass it through the trained model, and use its output as the prediction."
)
METHOD_SUMMARY_FALLBACK_CHARS = 500


@dataclass
class ReaderResult:
    has_error: bool
    observations: str
    score: float | None
    goal: str
    method_summary: str
    fallback: bool


def _fail(node: Node, reason: str) -> None:
    node.failed = True
    node.score = None
    if reason not in node.failure_reasons:
        node.failure_reasons.append(reason)


def read_results(ctx: RunContext, node: Node, res: ExecResult) -> ReaderResult:
    system, user = builders.result_reader(node.code, trim_with(ctx.cfg, node.output))
    try:
        obj = ctx.gateway.structured("result_reader", system, user, READER_SCHEMA, node_id=node.id)
        return ReaderResult(bool(obj["has_error"]), str(obj["observations"]), obj.get("score"), str(obj["goal"]),
                            str(obj.get("method_summary") or ""), False)
    except (StructuredOutputError, LLMUnavailableError) as exc:
        log.warning("node %s: could not read the run log with the LLM (%s); using code-only fallback", node.id, exc)
        return fallback_reader(node, res, ctx.direction.higher_is_better)


def fallback_reader(node: Node, res: ExecResult, higher_is_better: bool) -> ReaderResult:
    """What is used in place of the log reader's answer when that call failed."""
    return ReaderResult(res.error_type is not None, "Could not read the run log (LLM call failed).", None,
                        "maximize" if higher_is_better else "minimize", node.plan[:METHOD_SUMMARY_FALLBACK_CHARS], True)


def _pred_path(ctx: RunContext, node: Node) -> Path:
    run_dir = ctx.executor.run_dir(node.id)
    std = run_dir / "out" / "predictions.csv"
    alt = run_dir / "predictions.csv"
    if not std.exists() and alt.exists():
        std.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(alt), str(std))
    return std


def _ask_columns(ctx: RunContext, node: Node, pred: Path, sample: Path, scols: list[str], pcols: list[str]) -> list[str] | None:
    """Ask the LLM which sample column each prediction column corresponds to."""
    system, user = builders.format_fixer(head_text(sample), head_text(pred), scols, pcols)
    try:
        obj = ctx.gateway.structured("format_fixer", system, user, FIXER_SCHEMA, node_id=node.id)
    except (StructuredOutputError, LLMUnavailableError):
        return None
    return [str(c) for c in (obj.get("columns") or [])]


def _ask_leak(ctx: RunContext, node: Node) -> dict[str, object] | None:
    """Ask the LLM whether the code leaks data. None when the call failed."""
    assert node.score is not None
    system, user = builders.leak_checker(ctx.task_desc, node.code, trim_with(ctx.cfg, node.output), node.score,
                                         ctx.direction.higher_is_better)
    try:
        return ctx.gateway.structured("leak_checker", system, user, LEAK_SCHEMA, node_id=node.id)
    except (StructuredOutputError, LLMUnavailableError) as exc:
        log.warning("node %s: leak check failed (%s); treated as no leak", node.id, exc)
        return None


def apply_column_answer(pred: Path, cols: list[str] | None, pcols: list[str]) -> bool:
    """Rename the prediction columns to the given names. False when the answer is unusable."""
    if not cols or len(cols) != len(pcols):
        return False
    try:
        rename_columns(pred, [str(c) for c in cols])
    except Exception:  # noqa: BLE001
        return False
    return True


VAL_FILES_NOTE = (
    "The solution still counts, but without these files it cannot take part in validated ensembles. "
    "Write ./out/val_predictions.csv and ./out/val_labels.csv as described in the implementation rules."
)


def _check_val_files(ctx: RunContext, node: Node, pred: Path, info: dict[str, object]) -> None:
    """Check the validation files (they never fail the node)."""
    try:
        cols = [str(c) for c in read_str_csv(ctx.sample_path if ctx.sample_path is not None else pred).columns]
        res = load_val_files(pred.parent, cols)
    except Exception as exc:  # noqa: BLE001
        node.val_files_ok = False
        info["val_files"] = f"check failed: {type(exc).__name__}"
        return
    node.val_files_ok = res.ok
    if not res.ok:
        info["val_files"] = res.message
        node.output += f"\n[superagent checker] Validation files: {res.message}. {VAL_FILES_NOTE}\n"


def leak_decision(leak_found: bool, certainty: str) -> bool:
    """True when the leak check result must fail the node."""
    return bool(leak_found) and certainty in ("high", "medium")


def leak_triggered(score: float, higher_is_better: bool, high: float, low: float) -> bool:
    return score >= high if higher_is_better else score <= low


def jump_triggered(score: float, previous_best: float | None, higher_is_better: bool, ratio: float) -> bool:
    """True when ``score`` improves the previous global best by at least ``ratio`` of its size."""
    if ratio <= 0 or previous_best is None:
        return False
    gain = score - previous_best if higher_is_better else previous_best - score
    return gain > 0 and gain >= ratio * max(abs(previous_best), 1e-12)


ColumnAnswer = Callable[[Path, Path, list[str], list[str]], "list[str] | None"]
LeakAnswer = Callable[[], "dict[str, object] | None"]


def _classification_diagnostic(ctx: RunContext, node: Node, output: Path, info: dict[str, object]) -> None:
    spec = ctx.task_spec
    assert spec is not None

    def labels(folder: Path):
        predictions = spec.predictions(folder / "val_predictions.csv", spec.truth.index, ordered=False)
        return np.asarray(spec.classes)[predictions.argmax(1)] if spec.probability else predictions

    new = labels(output)
    truth = spec.truth.to_numpy()
    counts = Counter(new)
    largest = counts.most_common(1)[0][1]
    majority = Counter(truth).most_common(1)[0][1]
    node.output += (
        f"\n[superagent checker] Model predicted {len(counts)} of {len(spec.classes)} declared classes. "
        f"The most frequent class accounts for {largest}/{len(new)} validation predictions. "
        f"The majority-label validation baseline is {majority / len(truth):.6f}.\n"
    )
    info["validation_class_coverage"] = {"predicted": len(counts), "declared": len(spec.classes),
                                         "dominant_count": largest, "total": len(new), "truth_classes": len(set(truth))}
    best = ctx.tree.best
    if best:
        old = labels(ctx.executor.run_dir(best.id) / "out")
        node.complementarity = float(np.mean((old != truth) & (new == truth)))


def assess(ctx: RunContext, node: Node, res: ExecResult) -> dict[str, object]:
    """Fill in the node's result fields. Returns extra details for the decision log."""
    if ctx.task_spec is not None:
        info: dict[str, object] = {"evaluator": "task_contract"}
        node.finished_at = now_iso()
        node.output = res.output
        output = ctx.executor.run_dir(node.id) / "out"
        if res.error_type or res.exit_code != 0:
            _fail(node, "continuation_no_progress" if res.error_type=="ContinuationError" else "artifact_contract" if res.error_type=="ArtifactError" else "timeout" if res.timed_out else "runtime_error")
            node.summary = res.error_message or "script failed"
            if not node.method_summary and node.plan:
                node.method_summary = node.plan[:500]
        else:
            try:
                node.score = ctx.task_spec.grade(output)
                if not ctx.flags.get("formal"):
                    truth = ctx.task_spec.truth.rename_axis(ctx.task_spec.columns[0])
                    import pandas as pd

                    if not ctx.task_spec.probability:
                        truth.rename(ctx.task_spec.columns[1]).to_csv(output / "val_labels.csv")
                    else:
                        pd.get_dummies(truth).reindex(columns=ctx.task_spec.classes, fill_value=0).astype(int).to_csv(output / "val_labels.csv")
                node.val_files_ok = True
                node.predictions_ok = True
                node.score_source = "task_contract"
                node.failed = False
                node.summary = f"Independent {ctx.task_spec.metric}: {node.score:.6f}"
                node.method_summary = node.plan[:500]
            except (OSError, ValueError) as exc:
                node.predictions_ok = False
                _fail(node, "task_contract")
                node.summary = str(exc)
                if not node.method_summary and node.plan:
                    node.method_summary = node.plan[:500]
            else:
                if ctx.flags.get("formal") or ctx.task_spec.metric == "accuracy":
                    try:
                        _classification_diagnostic(ctx, node, output, info)
                    except (OSError, ValueError) as exc:
                        info["comparison_unavailable"] = str(exc)
        if ctx.flags.get("formal"):
            from .utils import atomic_write_json

            atomic_write_json(ctx.executor.run_dir(node.id) / "score.json", {
                "schema_version": 1, "split_id": ctx.task_spec.signature,
                "metric": ctx.task_spec.metric, "score": node.score, "score_verified": node.ok,
                "diagnostics": info, "error": node.summary if node.is_failed else None,
            })
        return _finish(ctx, node, info)
    reader = read_results(ctx, node, res)
    return apply_assessment(
        ctx, node, res, reader,
        column_answer=lambda pred, sample, scols, pcols: _ask_columns(ctx, node, pred, sample, scols, pcols),
        leak_answer=lambda: _ask_leak(ctx, node),
    )


def apply_assessment(
    ctx: RunContext,
    node: Node,
    res: ExecResult,
    reader: ReaderResult,
    column_answer: ColumnAnswer | None = None,
    leak_answer: LeakAnswer | None = None,
) -> dict[str, object]:
    """All assessment rules, given the log reader's answer. Makes no LLM call itself.

    ``column_answer`` and ``leak_answer`` supply the two answers that are only needed in
    some cases (column names for a mismatching prediction file; the leak verdict for a
    suspicious score). They are asked lazily; None means "no answer available".
    """
    cfg = ctx.cfg["assessment"]
    info: dict[str, object] = {}
    node.summary = reader.observations
    node.method_summary = reader.method_summary or node.plan[:METHOD_SUMMARY_FALLBACK_CHARS]
    info["reader_fallback"] = reader.fallback

    # 22.3 score.
    dec = reconcile_score(res.output, reader.score, float(cfg["score_tolerance"]))
    node.score = dec.score
    node.score_source = dec.source
    if dec.mismatch and dec.score is not None:
        log.info("node %s: marker score %s differs from the LLM's reading %s; using the marker", node.id, dec.score, reader.score)
        info["score_mismatch"] = {"marker": dec.score, "llm": reader.score}

    # 22.4 basic errors.
    pred = _pred_path(ctx, node)
    reasons = []
    if reader.has_error:
        reasons.append("reported_error")
    if res.error_type is not None:
        reasons.append("timeout" if res.error_type == "TimeoutError" else "runtime_error")
    if node.score is None:
        reasons.append("no_score")
    if not pred.exists():
        reasons.append("no_predictions")
    if reasons:
        for r in reasons:
            _fail(node, r)
        return _finish(ctx, node, info)

    # 22.5 format.
    if ctx.sample_path is not None:
        fr = check_format(pred, ctx.sample_path)
        if not fr.ok and fr.rule == 1 and fr.fixable_columns:
            info["format_fixer_used"] = True
            cols = column_answer(pred, ctx.sample_path, fr.sample_columns, fr.pred_columns) if column_answer else None
            if apply_column_answer(pred, cols, fr.pred_columns):
                fr = check_format(pred, ctx.sample_path)
        if not fr.ok:
            node.predictions_ok = False
            _fail(node, "format_error")
            detail = f"Format check failed (rule {fr.rule}): {fr.message}"
            node.output += f"\n[superagent checker] The script ran, but ./out/predictions.csv failed the format check. {detail}\n"
            node.summary = f"FORMAT_ERROR: the run succeeded, but the prediction file did not pass the format check. {detail}"
            return _finish(ctx, node, info)

    # 22.6 content.
    cr = check_content(pred, ctx.sample_path, float(cfg["constant_ratio_threshold"]))
    if not cr.ok:
        node.predictions_ok = False
        _fail(node, "content_error")
        node.output += f"\n[superagent checker] The prediction file looks like constant or placeholder values: {cr.message}. {CONTENT_FIX_HINT}\n"
        node.summary = f"CONTENT_ERROR: {cr.message}. {CONTENT_FIX_HINT}"
        return _finish(ctx, node, info)
    node.predictions_ok = True
    _check_val_files(ctx, node, pred, info)

    # 22.7 direction and zero score.
    sys_goal = "maximize" if ctx.direction.higher_is_better else "minimize"
    if not reader.fallback and reader.goal != sys_goal:
        _fail(node, "direction_mismatch")
        node.summary += (f" DIRECTION_MISMATCH: the script's score is reported as '{reader.goal}', but the task metric is "
                         f"'{sys_goal}'. This usually means the script prints a metric other than the one the task requires.")
        node.output += f"\n[superagent checker] The printed score is a '{reader.goal}' metric, but the task metric must be '{sys_goal}'d. Print the official task metric.\n"
        return _finish(ctx, node, info)
    if ctx.direction.higher_is_better and node.score == 0:
        _fail(node, "zero_score")
        node.summary = ZERO_SCORE_SUMMARY
        node.output += "\n[superagent checker] " + ZERO_SCORE_SUMMARY + "\n"
        return _finish(ctx, node, info)

    # 22.8 leak check.
    lc = cfg["leak_check"]
    assert node.score is not None
    with ctx.tree.lock:
        previous_best = ctx.tree.best.score if ctx.tree.best is not None else None
    near_perfect = leak_triggered(node.score, ctx.direction.higher_is_better, float(lc["high_threshold"]), float(lc["low_threshold"]))
    jump = jump_triggered(node.score, previous_best, ctx.direction.higher_is_better, float(lc["jump_ratio"]))
    if lc["enabled"] and (near_perfect or jump):
        info["leak_check_triggered"] = True
        if jump and not near_perfect:
            info["leak_check_reason"] = "score_jump"
        obj = leak_answer() if leak_answer else None
        if obj is not None:
            info["leak_check"] = {"leak_found": obj["leak_found"], "certainty": obj["certainty"]}
            if leak_decision(bool(obj["leak_found"]), str(obj["certainty"])):
                score = node.score
                _fail(node, "data_leak")
                node.summary = (f"DATA_LEAK: certainty {obj['certainty']}. Evidence: {obj['evidence']} "
                                f"(validation score was {score}). {LEAK_FIX_HINT}")
                node.output += "\n[superagent checker] " + node.summary + "\n"
                ctx.add_event("data_leak_detected", node.id, str(obj["evidence"])[:300])
                return _finish(ctx, node, info)
            log.info("node %s: leak check passed (%s): %s", node.id, obj["certainty"], str(obj["evidence"])[:200])
    node.failed = False
    return _finish(ctx, node, info)


def _finish(ctx: RunContext, node: Node, info: dict[str, object]) -> dict[str, object]:
    node.finished_at = now_iso()
    if node.failed is None:
        node.failed = False
    if ctx.flags.get("formal"):
        return info
    if node.ok:
        ctx.tree.register_ok(node)
    if ctx.experience is not None:
        from .bookkeeping import save_experience
        try:
            save_experience(ctx, node)
        except OSError as exc:
            log.warning("node %s: could not save the experience record: %s", node.id, exc)
    return info
