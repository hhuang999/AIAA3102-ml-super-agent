"""Validation files of a solution and identification of the task metric from them.

Every solution writes ``val_predictions.csv`` and ``val_labels.csv``: the same columns as the
prediction file, one row per validation sample, the labels in the same encoding as the predictions.
The metric is identified by recomputing each solution's reported validation score from its files
with a library of common classification metrics; the metric that reproduces the reported scores is
used to score ensembles on validation data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

VAL_PRED_NAME = "val_predictions.csv"
VAL_LABEL_NAME = "val_labels.csv"
MIN_VAL_ROWS = 2
ABS_TOLERANCE = 1e-4
REL_TOLERANCE = 0.01
PROB_SUM_TOLERANCE = 0.02
LOG_EPS = 1e-15


# --------------------------------------------------------------------------- validation files

@dataclass
class ValFiles:
    ok: bool
    message: str = ""
    ids: list[str] = field(default_factory=list)
    pred: pd.DataFrame | None = None     # prediction columns only, indexed by id
    true: pd.DataFrame | None = None     # same columns and index as ``pred``


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def _canon(v: str) -> str:
    s = str(v).strip()
    try:
        f = float(s)
    except ValueError:
        return s
    if math.isfinite(f) and f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return s


def load_val_files(out_dir: Path, pred_columns: Sequence[str]) -> ValFiles:
    """Read and check a solution's validation files against the prediction columns (id column first)."""
    pp, lp = Path(out_dir) / VAL_PRED_NAME, Path(out_dir) / VAL_LABEL_NAME
    missing = [p.name for p in (pp, lp) if not p.exists()]
    if missing:
        return ValFiles(False, f"missing {', '.join(missing)}")
    try:
        pred, true = _read(pp), _read(lp)
    except Exception as exc:  # noqa: BLE001
        return ValFiles(False, f"could not be read as CSV: {type(exc).__name__}: {exc}")
    cols = [str(c) for c in pred_columns]
    for name, df in ((VAL_PRED_NAME, pred), (VAL_LABEL_NAME, true)):
        if [str(c) for c in df.columns] != cols:
            return ValFiles(False, f"{name}: columns must be {cols}, got {[str(c) for c in df.columns]}")
        if len(df) < MIN_VAL_ROWS:
            return ValFiles(False, f"{name}: fewer than {MIN_VAL_ROWS} rows")
        ids = df.iloc[:, 0].astype(str).str.strip()
        if ids.duplicated().any():
            return ValFiles(False, f"{name}: duplicate ids in the first column")
        empty = df.iloc[:, 1:].apply(lambda c: c.astype(str).str.strip() == "").any()
        if bool(empty.any()):
            return ValFiles(False, f"{name}: empty cells in column(s) {list(empty[empty].index)}")
    pid = pred.iloc[:, 0].astype(str).str.strip()
    tid = true.iloc[:, 0].astype(str).str.strip()
    if set(pid) != set(tid):
        return ValFiles(False, "the two files do not contain the same ids")
    p = pred.iloc[:, 1:].copy()
    p.index = pid
    t = true.iloc[:, 1:].copy()
    t.index = tid
    t = t.loc[p.index]
    return ValFiles(True, "ok", list(p.index), p, t)


# --------------------------------------------------------------------------- encodings

def _numeric(df: pd.DataFrame) -> np.ndarray | None:
    try:
        arr = df.apply(lambda c: pd.to_numeric(c.str.strip(), errors="raise")).to_numpy(dtype=float)
    except (ValueError, TypeError):
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def detect_encoding(true: pd.DataFrame, pred: pd.DataFrame) -> str | None:
    """label | binary_prob | multi_prob | multilabel_prob, or None when no metric of the library applies."""
    tn, pn = _numeric(true), _numeric(pred)
    k = pred.shape[1]
    if k == 1:
        if pn is not None and tn is not None:
            tv = set(np.unique(tn).tolist())
            is_int_pred = bool(np.all(pn == np.round(pn)))
            if tv <= {0.0, 1.0} and not is_int_pred and pn.min() >= 0 and pn.max() <= 1:
                return "binary_prob"
            if is_int_pred and bool(np.all(tn == np.round(tn))):
                return "label"
            return None
        return "label"
    if pn is None or tn is None:
        return None
    if not set(np.unique(tn).tolist()) <= {0.0, 1.0} or pn.min() < 0:
        return None
    if bool(np.all(tn.sum(axis=1) == 1)):
        return "multi_prob"
    return "multilabel_prob"


# --------------------------------------------------------------------------- metric library

@dataclass(frozen=True)
class Metric:
    name: str
    encoding: str
    higher_is_better: bool
    fn: Callable[[pd.DataFrame, pd.DataFrame], float]


def _labels(df: pd.DataFrame) -> list[str]:
    return [_canon(v) for v in df.iloc[:, 0].tolist()]


def _ints(df: pd.DataFrame) -> np.ndarray:
    return np.asarray([int(float(v)) for v in df.iloc[:, 0].tolist()])


def _norm_rows(p: np.ndarray) -> np.ndarray:
    s = p.sum(axis=1, keepdims=True)
    return p / np.where(s > 0, s, 1.0)


def _library() -> list[Metric]:
    from sklearn import metrics as m

    def lab(f: Callable[..., float], **kw: object) -> Callable[[pd.DataFrame, pd.DataFrame], float]:
        return lambda t, p: float(f(_labels(t), _labels(p), **kw))

    def bin_(f: Callable[..., float], threshold: bool = False, **kw: object) -> Callable[[pd.DataFrame, pd.DataFrame], float]:
        def go(t: pd.DataFrame, p: pd.DataFrame) -> float:
            y = _numeric(t)[:, 0]  # type: ignore[index]
            s = _numeric(p)[:, 0]  # type: ignore[index]
            return float(f(y, (s >= 0.5).astype(int) if threshold else s, **kw))
        return go

    def multi(f: Callable[..., float], argmax: bool = False, **kw: object) -> Callable[[pd.DataFrame, pd.DataFrame], float]:
        def go(t: pd.DataFrame, p: pd.DataFrame) -> float:
            y = _numeric(t).argmax(axis=1)  # type: ignore[union-attr]
            s = _numeric(p)
            assert s is not None
            if argmax:
                return float(f(y, s.argmax(axis=1), **kw))
            return float(f(y, _norm_rows(s), **kw))
        return go

    def mlab(f: Callable[..., float]) -> Callable[[pd.DataFrame, pd.DataFrame], float]:
        def go(t: pd.DataFrame, p: pd.DataFrame) -> float:
            y, s = _numeric(t), _numeric(p)
            assert y is not None and s is not None
            return float(np.mean([f(y[:, j], s[:, j]) for j in range(y.shape[1])]))
        return go

    def logloss_multi(t: pd.DataFrame, p: pd.DataFrame) -> float:
        y = _numeric(t)
        s = _numeric(p)
        assert y is not None and s is not None
        s = np.clip(_norm_rows(s), LOG_EPS, 1 - LOG_EPS)
        s = _norm_rows(s)
        return float(-np.mean(np.log(s[np.arange(len(s)), y.argmax(axis=1)])))

    def logloss_bin(t: pd.DataFrame, p: pd.DataFrame) -> float:
        y = _numeric(t)[:, 0]  # type: ignore[index]
        s = np.clip(_numeric(p)[:, 0], LOG_EPS, 1 - LOG_EPS)  # type: ignore[index]
        return float(-np.mean(y * np.log(s) + (1 - y) * np.log(1 - s)))

    def kappa(weights: str | None) -> Callable[[pd.DataFrame, pd.DataFrame], float]:
        return lambda t, p: float(m.cohen_kappa_score(_ints(t), _ints(p), weights=weights))

    def gini(t: pd.DataFrame, p: pd.DataFrame) -> float:
        return 2 * bin_(m.roc_auc_score)(t, p) - 1

    def apk3(t: pd.DataFrame, p: pd.DataFrame) -> float:
        y = _numeric(t).argmax(axis=1)  # type: ignore[union-attr]
        s = _numeric(p)
        assert s is not None
        top = np.argsort(-s, axis=1)[:, :3]
        return float(np.mean([1.0 / (list(row).index(yi) + 1) if yi in row else 0.0 for row, yi in zip(top, y)]))

    return [
        Metric("accuracy", "label", True, lab(m.accuracy_score)),
        Metric("balanced_accuracy", "label", True, lab(m.balanced_accuracy_score)),
        Metric("f1_macro", "label", True, lab(m.f1_score, average="macro")),
        Metric("f1_weighted", "label", True, lab(m.f1_score, average="weighted")),
        Metric("f1_binary", "label", True, lambda t, p: float(m.f1_score(_ints(t), _ints(p)))),
        Metric("precision_macro", "label", True, lab(m.precision_score, average="macro", zero_division=0)),
        Metric("recall_macro", "label", True, lab(m.recall_score, average="macro", zero_division=0)),
        Metric("mcc", "label", True, lab(m.matthews_corrcoef)),
        Metric("kappa", "label", True, kappa(None)),
        Metric("kappa_linear", "label", True, kappa("linear")),
        Metric("kappa_quadratic", "label", True, kappa("quadratic")),
        Metric("roc_auc", "binary_prob", True, bin_(m.roc_auc_score)),
        Metric("gini", "binary_prob", True, gini),
        Metric("average_precision", "binary_prob", True, bin_(m.average_precision_score)),
        Metric("log_loss", "binary_prob", False, logloss_bin),
        Metric("brier", "binary_prob", False, bin_(m.brier_score_loss)),
        Metric("accuracy_at_0.5", "binary_prob", True, bin_(m.accuracy_score, threshold=True)),
        Metric("f1_at_0.5", "binary_prob", True, bin_(m.f1_score, threshold=True)),
        Metric("log_loss", "multi_prob", False, logloss_multi),
        Metric("accuracy", "multi_prob", True, multi(m.accuracy_score, argmax=True)),
        Metric("f1_macro", "multi_prob", True, multi(m.f1_score, argmax=True, average="macro")),
        Metric("roc_auc_ovr_macro", "multi_prob", True, multi(m.roc_auc_score, multi_class="ovr", average="macro")),
        Metric("roc_auc_ovr_weighted", "multi_prob", True, multi(m.roc_auc_score, multi_class="ovr", average="weighted")),
        Metric("map_at_3", "multi_prob", True, apk3),
        Metric("mean_column_roc_auc", "multilabel_prob", True, mlab(m.roc_auc_score)),
        Metric("mean_column_log_loss", "multilabel_prob", False,
               mlab(lambda y, s: m.log_loss(y, np.clip(s, LOG_EPS, 1 - LOG_EPS), labels=[0, 1]))),
    ]


def metric_library() -> list[Metric]:
    return _library()


def safe_score(metric: Metric, true: pd.DataFrame, pred: pd.DataFrame) -> float | None:
    try:
        v = metric.fn(true, pred)
    except Exception:  # noqa: BLE001 - a metric that does not apply to this data gives no value
        return None
    return v if math.isfinite(v) else None


def matches(value: float | None, reported: float | None) -> bool:
    if value is None or reported is None:
        return False
    return abs(value - reported) <= max(ABS_TOLERANCE, REL_TOLERANCE * abs(reported))


# --------------------------------------------------------------------------- identification

@dataclass
class Identification:
    metric: Metric | None
    encoding: str | None
    matched: int = 0
    checked: int = 0
    reason: str = ""
    per_metric: dict[str, int] = field(default_factory=dict)


PRIMARY_METRICS = frozenset({
    "accuracy", "roc_auc", "log_loss", "f1_macro", "roc_auc_ovr_macro",
    "mean_column_roc_auc", "mean_column_log_loss", "kappa_quadratic", "average_precision", "mcc",
})


def extract_metric_hints(task_desc: str) -> set[str]:
    """Extract canonical metric names explicitly mentioned in ``task_desc``."""
    import re
    text = (task_desc or "").lower()
    if not text:
        return set()
    hints: set[str] = set()
    patterns: list[tuple[str, set[str]]] = [
        (r"\bbalanced[_\s-]+accuracy\b", {"balanced_accuracy"}),
        (r"\b(?:macro[_\s-]+f1|f1[_\s-]+macro)\b", {"f1_macro"}),
        (r"\b(?:weighted[_\s-]+f1|f1[_\s-]+weighted)\b", {"f1_weighted"}),
        (r"\b(?:binary[_\s-]+f1|f1[_\s-]+binary|f1[_\s-]score)\b", {"f1_binary", "f1_at_0.5"}),
        (r"\b(?:macro[_\s-]+precision|precision[_\s-]+macro)\b", {"precision_macro"}),
        (r"\b(?:macro[_\s-]+recall|recall[_\s-]+macro)\b", {"recall_macro"}),
        (r"\b(?:quadratic[_\s-]+weighted[_\s-]+kappa|qwk)\b", {"kappa_quadratic"}),
        (r"\b(?:cohen'?s?[_\s-]+kappa)\b", {"kappa", "kappa_linear", "kappa_quadratic"}),
        (r"\b(?:matthews|mcc)\b", {"mcc"}),
        (r"\b(?:roc[_\s-]*auc|area\s+under\s+the\s+roc|auroc)\b",
         {"roc_auc", "roc_auc_ovr_macro", "roc_auc_ovr_weighted", "mean_column_roc_auc"}),
        (r"\b(?:average[_\s-]+precision|pr[_\s-]*auc|auprc)\b", {"average_precision"}),
        (r"\b(?:log[_\s-]*loss|cross[_\s-]*entropy)\b", {"log_loss", "mean_column_log_loss"}),
        (r"\bbrier\b", {"brier"}),
        (r"\bgini\b", {"gini"}),
        (r"\bmap\s*@\s*3|map_at_3\b", {"map_at_3"}),
    ]
    for pat, names in patterns:
        if re.search(pat, text):
            hints |= names
    cleaned = re.sub(r"\bbalanced[_\s-]+accuracy\b", " ", text)
    if re.search(r"\baccuracy\b", cleaned):
        hints |= {"accuracy", "accuracy_at_0.5"}
    return hints


def identify_metric(
    members: Sequence[tuple[ValFiles, float]],
    higher_is_better: bool,
    min_members: int = 2,
    *,
    task_desc: str = "",
) -> Identification:
    """``members`` are (validation files, reported score). The metric must reproduce most reported scores."""
    usable = [(v, s) for v, s in members if v.ok and v.true is not None and v.pred is not None]
    if len(usable) < min_members:
        return Identification(None, None, reason=f"fewer than {min_members} solutions with usable validation files")
    encodings = {detect_encoding(v.true, v.pred) for v, _ in usable}  # type: ignore[arg-type]
    encodings.discard(None)
    if len(encodings) != 1:
        return Identification(None, None, checked=len(usable), reason=f"validation files use different or unknown encodings: {sorted(map(str, encodings))}")
    enc = encodings.pop()
    hints = extract_metric_hints(task_desc)
    needed = max(min_members, math.ceil(0.6 * len(usable)))
    max_hits = 0
    best_valid: tuple[tuple[float, ...], Metric, int] | None = None
    per: dict[str, int] = {}
    for metric in metric_library():
        if metric.encoding != enc or metric.higher_is_better != higher_is_better:
            continue
        hits = 0
        exact_hits = 0
        errs = []
        for v, reported in usable:
            val = safe_score(metric, v.true, v.pred)  # type: ignore[arg-type]
            if matches(val, reported):
                hits += 1
            if val is not None:
                diff = abs(val - reported)
                errs.append(diff)
                if diff <= ABS_TOLERANCE:
                    exact_hits += 1
        per[metric.name] = hits
        max_hits = max(max_hits, hits)
        if hits < needed:
            continue
        err = float(np.median(errs)) if errs else float("inf")
        hinted = 1 if (metric.name in hints and err <= max(ABS_TOLERANCE, REL_TOLERANCE)) else 0
        primary = 1 if metric.name in PRIMARY_METRICS else 0
        err_band = -round(err / 2e-3) if not hinted else 0
        key = (float(hits), float(hinted), float(exact_hits), float(err_band), float(primary), -err)
        if best_valid is None or key > best_valid[0]:
            best_valid = (key, metric, hits)
    if best_valid is None:
        return Identification(None, enc, max_hits, len(usable),
                              f"no metric of the library reproduces at least {needed} of {len(usable)} reported scores", per)
    return Identification(best_valid[1], enc, best_valid[2], len(usable), "ok", per)
