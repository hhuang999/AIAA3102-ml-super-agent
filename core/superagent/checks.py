"""Code-only checks on a finished run: score marker, prediction file format and content."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

MARKER = "VALIDATION_SCORE:"
_MARKER_RE = re.compile(r"^\s*VALIDATION_SCORE:\s*(\S+)\s*$")
_ZERO_SEQ_RE = re.compile(r"^0(?:\s+0)+$")
_ONE_SEQ_RE = re.compile(r"^1(?:\s+1)+$")


def marker_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if _MARKER_RE.match(ln)]


def extract_marker_score(output: str) -> tuple[bool, float | None]:
    """Find the last score marker line. Returns (found, finite value or None)."""
    last: str | None = None
    for line in output.splitlines():
        m = _MARKER_RE.match(line)
        if m:
            last = m.group(1)
    if last is None:
        return False, None
    try:
        value = float(last)
    except ValueError:
        return True, None
    if not math.isfinite(value):
        return True, None
    return True, value


@dataclass
class ScoreDecision:
    score: float | None
    source: str | None
    mismatch: bool


def reconcile_score(output: str, llm_score: float | None, tolerance: float) -> ScoreDecision:
    _found, marker = extract_marker_score(output)
    llm_ok = llm_score is not None and isinstance(llm_score, (int, float)) and math.isfinite(float(llm_score))
    if marker is not None:
        mismatch = not llm_ok or abs(float(llm_score) - marker) > tolerance  # type: ignore[arg-type]
        return ScoreDecision(marker, "marker", mismatch)
    if llm_ok:
        return ScoreDecision(float(llm_score), "llm", False)  # type: ignore[arg-type]
    return ScoreDecision(None, None, False)


# --------------------------------------------------------------------------- format check

@dataclass
class FormatResult:
    ok: bool
    rule: int | None = None           # first failed rule (1-4)
    message: str = ""
    fixable_columns: bool = False     # rule 1 failed, same column count, different names
    reordered: bool = False
    sample_columns: list[str] = field(default_factory=list)
    pred_columns: list[str] = field(default_factory=list)


def read_str_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def _check_once(pred: pd.DataFrame, sample: pd.DataFrame) -> FormatResult:
    scols = [str(c) for c in sample.columns]
    pcols = [str(c) for c in pred.columns]
    if pcols != scols:
        fixable = len(pcols) == len(scols) and set(pcols) != set(scols)
        return FormatResult(False, 1, f"column names/order differ: expected {scols}, got {pcols}", fixable, False, scols, pcols)
    if len(pred) != len(sample):
        return FormatResult(False, 2, f"row count differs: expected {len(sample)} rows, got {len(pred)}", False, False, scols, pcols)
    pid = pred.iloc[:, 0].astype(str).str.strip()
    sid = sample.iloc[:, 0].astype(str).str.strip()
    dups = pid[pid.duplicated()].unique().tolist()
    if dups:
        return FormatResult(False, 3, f"duplicate ids in the first column, e.g. {dups[:5]}", False, False, scols, pcols)
    pset, sset = set(pid), set(sid)
    if pset != sset:
        missing = sorted(sset - pset)[:5]
        extra = sorted(pset - sset)[:5]
        return FormatResult(False, 3, f"id set differs from the sample: missing {missing}, unexpected {extra}", False, False, scols, pcols)
    for c in pcols[1:]:
        empty = pred[c].astype(str).str.strip() == ""
        if empty.any():
            n = int(empty.sum())
            return FormatResult(False, 4, f"column '{c}' has {n} empty cells", False, False, scols, pcols)
    return FormatResult(True, None, "ok", False, False, scols, pcols)


def check_format(pred_path: Path, sample_path: Path) -> FormatResult:
    """Check the prediction file against the sample. Reorders columns in place when only the order differs."""
    try:
        sample = read_str_csv(sample_path)
        pred = read_str_csv(pred_path)
    except Exception as exc:  # noqa: BLE001
        return FormatResult(False, 1, f"prediction file could not be read as CSV: {type(exc).__name__}: {exc}")
    res = _check_once(pred, sample)
    if not res.ok and res.rule == 1 and set(res.pred_columns) == set(res.sample_columns) and len(res.pred_columns) == len(res.sample_columns):
        pred = pred[res.sample_columns]
        pred.to_csv(pred_path, index=False)
        res = _check_once(pred, sample)
        res.reordered = True
    return res


def rename_columns(pred_path: Path, new_columns: list[str]) -> None:
    pred = read_str_csv(pred_path)
    pred.columns = new_columns
    pred.to_csv(pred_path, index=False)


def head_text(path: Path, n: int = 20) -> str:
    lines: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= n + 1:
                break
            lines.append(line.rstrip("\n"))
    return "\n".join(lines)


# --------------------------------------------------------------------------- content check

@dataclass
class ContentResult:
    ok: bool
    message: str = ""


def check_content(pred_path: Path, sample_path: Path | None, threshold: float) -> ContentResult:
    try:
        pred = read_str_csv(pred_path)
        if sample_path is not None and sample_path.exists():
            sample_cols = [str(c) for c in read_str_csv(sample_path).columns]
            cols = [c for c in sample_cols[1:] if c in pred.columns]
        else:
            if pred.shape[1] <= 1:
                return ContentResult(True, "single column; content check skipped")
            cols = [pred.columns[-1]]
        n = len(pred)
        for c in cols:
            vals = pred[c].astype(str)
            stripped = vals.str.strip()
            if stripped.nunique() == 2:
                continue
            if n == 0:
                continue
            empty_ratio = float((stripped == "").mean())
            if empty_ratio > threshold:
                return ContentResult(False, f"column '{c}': {empty_ratio:.1%} of the values are empty")
            counts = stripped.value_counts()
            top_val, top_count = counts.index[0], int(counts.iloc[0])
            top_ratio = top_count / n
            if n > 2 and top_ratio > threshold:
                return ContentResult(False, f"column '{c}': the value {top_val!r} makes up {top_ratio:.1%} of all predictions")
            seq = stripped.str.match(_ZERO_SEQ_RE) | stripped.str.match(_ONE_SEQ_RE)
            if float(seq.mean()) > 0.5:
                return ContentResult(False, f"column '{c}': most values are space-separated runs of 0 or 1 (placeholder masks)")
        return ContentResult(True, "ok")
    except Exception as exc:  # noqa: BLE001 - a failing check counts as passed
        return ContentResult(True, f"content check skipped after an internal error: {type(exc).__name__}")
