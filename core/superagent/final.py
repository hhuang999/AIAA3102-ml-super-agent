"""The primary result in ``final/``: the single best solution or a validated ensemble of top solutions.

An ensemble becomes the primary result only when all of the following hold:
- the task metric could be identified from the solutions' validation files (see ``metrics``);
- the solutions share enough validation samples with the best solution;
- the ensemble beats the single best solution on both halves of the shared validation samples,
  where each half is scored with weights chosen on the other half.

On a half where the task metric cannot tell the two apart (for example both reach the metric's
best value), a second measure decides when the predictions are probabilities: the log loss of the
same probabilities must be lower by at least ``final.tie_break_min_gain``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from .ensemble import SCRIPT_HELPERS, combine, detect_format, member_sizes, weights
from .metrics import (VAL_LABEL_NAME, VAL_PRED_NAME, Identification, Metric, ValFiles, identify_metric, load_val_files,
                      metric_library, safe_score)
from .utils import atomic_write_json, atomic_write_text

log = logging.getLogger("superagent")


@dataclass
class Candidate:
    rank: int
    node_id: str
    score: float
    out_dir: Path            # the solution's ./out directory (predictions, validation files, model/)
    code: str = ""


@dataclass
class Plan:
    method: str
    weights: list[float]


TIE_EPS = 1e-12


def _better(a: float, b: float, hib: bool) -> bool:
    return a > b if hib else a < b


def tie_break_metric(metric: Metric) -> Metric | None:
    """The second measure used when the task metric ties: log loss on probability predictions."""
    if metric.encoding not in ("binary_prob", "multi_prob") or metric.name == "log_loss":
        return None
    for m in metric_library():
        if m.name == "log_loss" and m.encoding == metric.encoding:
            return m
    return None


def half_of(sample_id: str) -> int:
    return int(hashlib.md5(sample_id.encode("utf-8")).hexdigest(), 16) % 2


def stratified_halves(rows: list[str], truth: pd.DataFrame) -> list[list[str]]:
    if not rows:
        return [[], []]
    sub = truth.loc[rows]
    groups: dict[tuple[str, ...], list[str]] = {}
    for r, vals in zip(rows, sub.itertuples(index=False, name=None)):
        key = tuple(str(v).strip() for v in vals)
        groups.setdefault(key, []).append(r)
    halves: list[list[str]] = [[], []]
    singletons: list[str] = []
    for key in sorted(groups):
        items = sorted(groups[key], key=lambda s: (hashlib.md5(s.encode("utf-8")).hexdigest(), s))
        if len(items) == 1:
            singletons.append(items[0])
            continue
        for idx, item in enumerate(items):
            halves[idx % 2].append(item)
    for item in sorted(singletons, key=lambda s: (hashlib.md5(s.encode("utf-8")).hexdigest(), s)):
        h = 0 if len(halves[0]) <= len(halves[1]) else 1
        halves[h].append(item)
    order = {r: i for i, r in enumerate(rows)}
    return [sorted(h, key=order.__getitem__) for h in halves]


def _spec_encoding(spec: Any) -> str:
    if not spec.probability:
        return "label"
    return "binary_prob" if len(spec.columns) == 2 else "multi_prob"


def _frames(vals: Sequence[ValFiles], rows: list[str], id_col: str) -> list[pd.DataFrame]:
    out = []
    for v in vals:
        assert v.pred is not None
        df = v.pred.loc[rows].reset_index(drop=True)
        df.insert(0, id_col, rows)
        out.append(df)
    return out


class Scorer:
    """Scores weighted combinations of the members' validation predictions on a set of rows."""

    def __init__(self, vals: Sequence[ValFiles], truth: pd.DataFrame, metric: Metric, fmt: Mapping[str, Any], id_col: str,
                 secondary: Metric | None = None) -> None:
        self.vals = vals
        self.truth = truth
        self.metric = metric
        self.secondary = secondary
        self.fmt = dict(fmt)
        self.id_col = id_col

    def _pred(self, w: Sequence[float], rows: list[str]) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        idx = [i for i, x in enumerate(w) if x > 0]
        if not idx:
            return None
        frames = _frames([self.vals[i] for i in idx], rows, self.id_col)
        comb = combine(frames, [w[i] for i in idx], self.fmt)
        pred = comb[self.fmt["pred_cols"]].astype(str)
        pred.index = rows
        return self.truth.loc[rows][self.fmt["pred_cols"]], pred

    def score(self, w: Sequence[float], rows: list[str]) -> float | None:
        tp = self._pred(w, rows)
        return None if tp is None else safe_score(self.metric, *tp)

    def pair(self, w: Sequence[float], rows: list[str]) -> tuple[float | None, float | None]:
        """Task metric and second measure (lower is better; None without a second measure)."""
        tp = self._pred(w, rows)
        if tp is None:
            return None, None
        sec = safe_score(self.secondary, *tp) if self.secondary is not None else None
        return safe_score(self.metric, *tp), sec


def _pair_better(a: tuple[float | None, float | None], b: tuple[float | None, float | None], hib: bool) -> bool:
    """Better on the task metric, or equal on it and lower on the second measure."""
    if a[0] is None:
        return False
    if b[0] is None:
        return True
    if abs(a[0] - b[0]) > TIE_EPS:
        return _better(a[0], b[0], hib)
    return a[1] is not None and b[1] is not None and a[1] < b[1] - TIE_EPS


def greedy_weights(scorer: Scorer, n: int, rows: list[str], rounds: int, hib: bool) -> list[float]:
    """Forward selection with replacement, starting from the best member (index 0)."""
    counts = [0] * n
    counts[0] = 1
    current = scorer.pair(counts, rows)
    if current[0] is None:
        return [1.0] + [0.0] * (n - 1)
    for _ in range(rounds):
        best_j: int | None = None
        best_s = current
        for j in range(n):
            trial = list(counts)
            trial[j] += 1
            s = scorer.pair(trial, rows)
            if _pair_better(s, best_s, hib):
                best_j, best_s = j, s
        if best_j is None:
            break
        counts[best_j] += 1
        current = best_s
    total = sum(counts)
    return [c / total for c in counts]


def _single(n: int) -> list[float]:
    return [1.0] + [0.0] * (n - 1)


def _fixed_plans(scores: Sequence[float], hib: bool, strategy: str | None = None) -> list[Plan]:
    n = len(scores)
    plans = []
    for k in member_sizes(n):
        w = weights(list(scores[:k]), hib, strategy)
        plans.append(Plan(f"top{k}", list(w) + [0.0] * (n - k)))
    return plans


def _half_verdict(s: tuple[float | None, float | None], b: tuple[float | None, float | None], hib: bool, min_gain: float,
                  tie_gain: float | None) -> str | None:
    """'metric' if the ensemble wins the half on the task metric, 'tie_break' if it ties there and wins on the
    second measure, otherwise None."""
    if s[0] is None or b[0] is None:
        return None
    if _better(s[0], b[0], hib) and abs(s[0] - b[0]) > min_gain:
        return "metric"
    if tie_gain is not None and abs(s[0] - b[0]) <= min_gain and s[1] is not None and b[1] is not None \
            and b[1] - s[1] > tie_gain:
        return "tie_break"
    return None


def evaluate(scorer: Scorer, scores: Sequence[float], rows: list[str], hib: bool, rounds: int, min_gain: float,
             tie_gain: float | None = None) -> dict[str, Any]:
    """Score the single best and every ensemble plan on both halves of ``rows``.

    ``tie_gain`` enables the second measure (see the module docstring); None disables it.
    """
    n = len(scores)
    if scorer.secondary is None:
        tie_gain = None
    halves = stratified_halves(rows, scorer.truth)
    single = [scorer.pair(_single(n), h) for h in halves]
    results: list[dict[str, Any]] = []
    plans: list[tuple[str, Callable[[list[str]], list[float]]]] = [
        (p.method, (lambda rws, w=p.weights: w)) for p in _fixed_plans(scores, hib, scorer.fmt.get("strategy"))
    ]
    plans.append(("greedy", lambda rws: greedy_weights(scorer, n, rws, rounds, hib)))
    for method, fit in plans:
        pairs = []
        for h in (0, 1):
            w = fit(halves[1 - h])
            pairs.append(scorer.pair(w, halves[h]))
        verdicts = [_half_verdict(s, b, hib, min_gain, tie_gain) for s, b in zip(pairs, single)]
        accepted = all(v is not None for v in verdicts)
        full_w = fit(rows)
        full = scorer.pair(full_w, rows)
        row: dict[str, Any] = {"method": method, "weights": [round(x, 6) for x in full_w], "half_scores": [p[0] for p in pairs],
                               "full_score": full[0], "accepted": accepted, "half_verdicts": verdicts}
        if scorer.secondary is not None:
            row["half_tie_break_scores"] = [p[1] for p in pairs]
            row["full_tie_break_score"] = full[1]
        results.append(row)
    single_full = scorer.pair(_single(n), rows)
    out: dict[str, Any] = {"single_half_scores": [p[0] for p in single], "single_full_score": single_full[0],
                           "candidates": results, "half_sizes": [len(h) for h in halves]}
    if scorer.secondary is not None:
        out["tie_break"] = {"metric": scorer.secondary.name, "enabled": tie_gain is not None, "min_gain": tie_gain,
                            "single_half_scores": [p[1] for p in single], "single_full_score": single_full[1]}
    return out


def _mean(xs: Sequence[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if len(vals) == len(xs) and vals else None


FINAL_SCRIPT = '''"""Recompute final/predictions.csv from the member solutions' predictions.

Usage (from the output directory):  python final/combine.py
Only pandas is required. The members' prediction files are in final/members/*/predictions.csv; each
member's solution.py, run in a directory that contains the task data as ./data, writes that file again.
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


''' + SCRIPT_HELPERS + '''

def main():
    with open(os.path.join(HERE, "source.json")) as fh:
        meta = json.load(fh)
    frames, ws = [], []
    for m in meta["members"]:
        path = os.path.join(HERE, "members", m["dir"], "predictions.csv")
        frames.append(pd.read_csv(path, dtype=str, keep_default_na=False))
        ws.append(m["weight"])
    if len(frames) == 1:
        res = frames[0]
    else:
        res = combine(frames, ws, meta["format"])
    res.to_csv(os.path.join(HERE, "predictions.csv"), index=False)


if __name__ == "__main__":
    main()
'''


def _member_dir(rank: int, node_id: str) -> str:
    return f"rank_{rank:02d}_{node_id}"


def _write_member(final_dir: Path, c: Candidate) -> str:
    name = _member_dir(c.rank, c.node_id)
    d = final_dir / "members" / name
    d.mkdir(parents=True, exist_ok=True)
    atomic_write_text(d / "solution.py", c.code)
    for fname in ("predictions.csv", VAL_PRED_NAME, VAL_LABEL_NAME):
        src = c.out_dir / fname
        if src.exists():
            shutil.copyfile(src, d / fname)
    md = c.out_dir / "model"
    if md.is_dir() and any(md.iterdir()):
        link = d / "artifacts"
        if not link.exists() and not link.is_symlink():
            os.symlink(os.path.relpath(md, d), link)
    atomic_write_json(d / "score.json", {"node_id": c.node_id, "rank": c.rank, "score": c.score})
    return name


def build_final(output_dir: Path, cands: Sequence[Candidate], higher_is_better: bool, cfg: Mapping[str, Any], *, spec=None, deadline=None) -> dict[str, Any]:
    """Write ``final/`` and return its description (also saved as ``final/source.json``)."""
    from .publication import publish, version
    cands = [c for c in cands if (c.out_dir / "predictions.csv").exists()]
    if not cands:
        return {"available": False, "kind": None, "reason": "no solution with a prediction file"}
    final_dir = version(output_dir)
    best = cands[0]
    best_pred = pd.read_csv(best.out_dir / "predictions.csv", dtype=str, keep_default_na=False)
    columns = [str(c) for c in best_pred.columns]
    fmt = detect_format(best_pred)
    info: dict[str, Any] = {"available": True, "kind": "single", "method": "single", "metric": None, "encoding": None,
                            "validation": None, "candidates": [], "reason": ""}
    chosen: list[tuple[Candidate, float]] = [(best, 1.0)]
    try:
        chosen, info = _choose(cands, columns, fmt, higher_is_better, cfg, info, spec=spec)
    except Exception as exc:  # noqa: BLE001 - any problem keeps the single best solution
        log.warning("validated ensembling failed: %s", exc)
        info["reason"] = f"ensemble evaluation failed: {type(exc).__name__}: {exc}"
        chosen = [(best, 1.0)]
        info["kind"], info["method"] = "single", "single"
    members = []
    for c, w in chosen:
        members.append({"dir": _write_member(final_dir, c), "rank": c.rank, "node_id": c.node_id, "score": c.score, "weight": w})
    if len(chosen) == 1:
        shutil.copyfile(chosen[0][0].out_dir / "predictions.csv", final_dir / "predictions.csv")
    else:
        frames = [pd.read_csv(c.out_dir / "predictions.csv", dtype=str, keep_default_na=False) for c, _ in chosen]
        combine(frames, [w for _, w in chosen], fmt).to_csv(final_dir / "predictions.csv", index=False)
    if info["kind"] == "single":
        info["node_id"] = chosen[0][0].node_id
        if spec:
            info["metric"] = spec.metric
            info["encoding"] = _spec_encoding(spec)
            if info["validation"] is None:
                info["validation"] = {}
            info["validation"].update(score=chosen[0][0].score, rows=len(spec.truth))
    info["members"] = members
    info["format"] = {k: fmt[k] for k in ("id_col", "pred_cols", "copy_cols", "strategy", "normalize", "columns")}
    info["columns"] = columns
    atomic_write_json(final_dir / "source.json", info)
    atomic_write_text(final_dir / "combine.py", FINAL_SCRIPT)
    if spec:
        spec.predictions(final_dir / "predictions.csv", spec.test_ids, ordered=True)
    if deadline is not None and __import__("time").monotonic() >= deadline:
        return {"available":False,"reason":"publication deadline"}
    publish(output_dir, final_dir)
    return info


def _choose(cands: Sequence[Candidate], columns: list[str], fmt: Mapping[str, Any], hib: bool, cfg: Mapping[str, Any],
            info: dict[str, Any], *, spec=None) -> tuple[list[tuple[Candidate, float]], dict[str, Any]]:
    best = cands[0]
    single = [(best, 1.0)]
    if not cfg["allow_ensemble"]:
        info["reason"] = "ensembles are not allowed as the primary result (final.allow_ensemble: false)"
        return single, info
    if len(cands) < 2:
        info["reason"] = "only one solution available"
        return single, info
    vals = [load_val_files(c.out_dir, columns) for c in cands]
    ident: Identification = identify_metric([(v, c.score) for v, c in zip(vals, cands)], hib)
    if spec:
        from .metrics import metric_library
        encoding = _spec_encoding(spec)
        metric = next((m for m in metric_library() if m.encoding == encoding and m.name == spec.metric), None)
        from dataclasses import replace
        ident = replace(ident, metric=metric, encoding=encoding, reason="task_contract")
    info["metric"] = ident.metric.name if ident.metric else None
    info["encoding"] = ident.encoding
    info["metric_identification"] = {"matched": ident.matched, "checked": ident.checked, "reason": ident.reason,
                                     "per_metric": ident.per_metric}
    if ident.metric is None:
        info["reason"] = f"metric not identified: {ident.reason}"
        return single, info
    if not vals[0].ok:
        info["reason"] = f"the best solution has no usable validation files: {vals[0].message}"
        return single, info
    base_ids = set(vals[0].ids)
    keep = [0]
    for i in range(1, len(cands)):
        if vals[i].ok and len(base_ids & set(vals[i].ids)) >= float(cfg["member_overlap"]) * len(base_ids):
            keep.append(i)
    if len(keep) < 2:
        info["reason"] = "no other solution shares enough validation samples with the best solution"
        return single, info
    id_sets = [set(vals[i].ids) for i in keep[1:]]
    rows = [r for r in vals[0].ids if all(r in ids for ids in id_sets)]
    if len(rows) < int(cfg["min_validation_rows"]):
        info["reason"] = f"only {len(rows)} shared validation samples (minimum {cfg['min_validation_rows']})"
        return single, info
    kc = [cands[i] for i in keep]
    kv = [vals[i] for i in keep]
    assert kv[0].true is not None
    use_tie = bool(cfg.get("tie_break", True))
    secondary = tie_break_metric(ident.metric) if use_tie else None
    scorer = Scorer(kv, kv[0].true, ident.metric, fmt, columns[0], secondary=secondary)
    tie_gain = float(cfg.get("tie_break_min_gain", 0.001)) if secondary is not None else None
    ev = evaluate(scorer, [c.score for c in kc], rows, hib, int(cfg["greedy_rounds"]), float(cfg["min_gain"]), tie_gain)
    info["validation"] = {"rows": len(rows), "half_sizes": ev["half_sizes"], "single_full_score": ev["single_full_score"],
                          "single_half_scores": ev["single_half_scores"], "members_considered": [c.node_id for c in kc]}
    if "tie_break" in ev:
        info["validation"]["tie_break"] = ev["tie_break"]
    small = int(cfg.get("small_validation_rows", 200))
    if len(rows) < small:
        info["validation"]["warning"] = (f"only {len(rows)} shared validation samples (below {small}); "
                                         "the validation score may not separate good solutions")
    info["candidates"] = ev["candidates"]
    accepted = [r for r in ev["candidates"] if r["accepted"]]
    if not accepted:
        info["reason"] = "no ensemble beat the single best solution on both validation halves"
        mg = float(cfg["min_gain"])
        if any(all(s is not None and b is not None and abs(s - b) <= mg for s, b in zip(r["half_scores"], ev["single_half_scores"]))
               for r in ev["candidates"] if r["method"] != "greedy"):
            info["reason"] += " (the ensembles tied with it on the task metric"
            info["reason"] += "; no second measure for this encoding)" if secondary is None else "; the second measure did not decide)"
        info["validation"]["score"] = ev["single_full_score"]
        return single, info

    def key(r: dict[str, Any]) -> tuple[float, float]:
        m = _mean(r["half_scores"])
        assert m is not None
        t = _mean(r.get("half_tie_break_scores") or [None])
        return (-m if hib else m, t if t is not None else 0.0)

    pick = min(accepted, key=key)
    chosen = [(c, w) for c, w in zip(kc, pick["weights"]) if w > 0]
    if len(chosen)>1:
        marginal={}
        for i,candidate in enumerate(kc):
            if pick["weights"][i]<=0:
                continue
            w=list(pick["weights"])
            w[i]=0
            total=sum(w)
            without=scorer.score([v/total for v in w],rows) if total else None
            marginal[candidate.node_id]=None if without is None else (pick["full_score"]-without)*(1 if hib else -1)
        info["validation"]["leave_one_out_gain"]=marginal
    by_tie = "tie_break" in pick["half_verdicts"]
    reason = ("ensemble beat the single best solution on both validation halves"
              + (f" (on a tied half by {secondary.name if secondary else 'the second measure'})" if by_tie else ""))
    info.update({"kind": "ensemble", "method": pick["method"], "reason": reason, "decided_by_tie_break": by_tie})
    info["validation"]["score"] = pick["full_score"]
    info["validation"]["mean_half_score"] = _mean(pick["half_scores"])
    return chosen, info


__all__ = ["Candidate", "FINAL_SCRIPT", "Scorer", "build_final", "evaluate", "greedy_weights", "half_of", "stratified_halves", "tie_break_metric"]
