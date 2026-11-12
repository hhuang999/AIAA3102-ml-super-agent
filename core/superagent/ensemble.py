"""Post-search ensembling of the top solutions' test predictions (code only)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .utils import atomic_write_json

log = logging.getLogger("superagent")

ID_UNIQUE_RATIO = 0.5
LONG_TEXT_MEAN_CHARS = 200
ROW_SUM_TOLERANCE = 0.05
EPS = 1e-6


def member_sizes(n: int) -> list[int]:
    if n < 2:
        return []
    if n <= 4:
        return list(range(2, n + 1))
    return [k for k in (2, 3, 4, 6) if k <= n]


VOTE_STRATEGIES = ("vote", "string_vote", "tokenwise_vote")


def weights(scores: Sequence[float], higher_is_better: bool, strategy: str | None = None) -> list[float]:
    """Member weights for a top-k ensemble.

    Averaged predictions use weights that fall with rank. Votes use equal weights: with falling
    weights the best member alone would outvote all the others and the vote would only repeat it.
    Ties in a vote go to the higher-ranked member.
    """
    k = len(scores)
    if strategy in VOTE_STRATEGIES:
        return [1.0 / k] * k
    base = [s if higher_is_better else 1.0 / (s + EPS) for s in scores]
    raw = [b * (k - i) for i, b in enumerate(base)]
    total = sum(raw)
    if total == 0:
        return [1.0 / k] * k
    return [r / total for r in raw]


def _is_num(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def _is_int(v: str) -> bool:
    try:
        int(v)
        return True
    except ValueError:
        return False


def detect_format(df: pd.DataFrame) -> dict[str, Any]:
    cols = list(df.columns)
    first = df[cols[0]].astype(str)
    has_id = len(df) > 0 and first.nunique() / len(df) > ID_UNIQUE_RATIO
    rest = cols[1:] if has_id else cols
    copy_cols = [c for c in rest if df[c].astype(str).str.len().mean() > LONG_TEXT_MEAN_CHARS]
    pred_cols = [c for c in rest if c not in copy_cols]
    if not pred_cols:
        pred_cols = [cols[-1]]
        copy_cols = [c for c in copy_cols if c != cols[-1]]
    vals = [v for v in df[pred_cols[0]].astype(str).tolist() if v != ""]
    if any(" " in v.strip() for v in vals):
        tokens = [v.split() for v in vals]
        same_len = len({len(t) for t in tokens}) == 1
        first_numeric = bool(tokens) and all(_is_num(t) for t in tokens[0])
        strategy = "tokenwise_vote" if same_len and first_numeric else "string_vote"
    elif any(not _is_num(v) for v in vals):
        strategy = "string_vote"
    elif all(_is_int(v) for v in vals):
        strategy = "vote"
    else:
        strategy = "average"
    normalize = False
    if len(pred_cols) > 1 and strategy == "average":
        try:
            sums = df[pred_cols].astype(float).sum(axis=1)
            normalize = bool(((sums - 1.0).abs() < ROW_SUM_TOLERANCE).all())
        except ValueError:
            normalize = False
    return {"id_col": cols[0] if has_id else None, "pred_cols": pred_cols, "copy_cols": copy_cols,
            "strategy": strategy, "normalize": normalize, "columns": cols}


def _align(frames: list[pd.DataFrame], id_col: str | None) -> list[pd.DataFrame]:
    first = frames[0]
    out = [first.reset_index(drop=True)]
    for f in frames[1:]:
        aligned = None
        if id_col is not None and id_col in f.columns and not f[id_col].duplicated().any():
            try:
                g = f.set_index(f[id_col].astype(str).str.strip())
                keys = first[id_col].astype(str).str.strip()
                if set(keys) <= set(g.index):
                    aligned = g.loc[keys.tolist()].reset_index(drop=True)
            except Exception:  # noqa: BLE001
                aligned = None
        if aligned is None:
            aligned = f.reset_index(drop=True).reindex(range(len(first)))
        out.append(aligned)
    return out


def _vote(values: list[str], w: Sequence[float]) -> str:
    acc: dict[str, float] = {}
    order: dict[str, int] = {}
    for i, (v, wt) in enumerate(zip(values, w)):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        v = str(v)
        acc[v] = acc.get(v, 0.0) + wt
        order.setdefault(v, i)
    if not acc:
        return ""
    return max(acc, key=lambda k: (acc[k], -order[k]))


def combine(frames: list[pd.DataFrame], w: Sequence[float], fmt: dict[str, Any]) -> pd.DataFrame:
    frames = _align(frames, fmt["id_col"])
    out = frames[0].copy()
    strat = fmt["strategy"]
    for c in fmt["pred_cols"]:
        cols = [f[c] if c in f.columns else pd.Series([None] * len(out)) for f in frames]
        if strat == "average":
            nums = [pd.to_numeric(col, errors="coerce") for col in cols]
            total = sum(n.fillna(0) * wt for n, wt in zip(nums, w))
            wsum = sum((~n.isna()).astype(float) * wt for n, wt in zip(nums, w))
            out[c] = (total / wsum.where(wsum > 0, 1.0))
        elif strat == "tokenwise_vote":
            merged = []
            for i in range(len(out)):
                toks = [str(col.iloc[i]).split() if pd.notna(col.iloc[i]) else None for col in cols]
                ref = toks[0] or []
                res = []
                for j in range(len(ref)):
                    res.append(_vote([t[j] if t is not None and j < len(t) else None for t in toks], w))  # type: ignore[misc]
                merged.append(" ".join(res))
            out[c] = merged
        else:
            out[c] = [_vote([col.iloc[i] for col in cols], w) for i in range(len(out))]
    if fmt["normalize"]:
        sub = out[fmt["pred_cols"]].astype(float)
        sums = sub.sum(axis=1)
        safe = sums.where(sums != 0, 1.0)
        out[fmt["pred_cols"]] = sub.div(safe, axis=0)
    return out[fmt["columns"]]


SCRIPT_HELPERS = '''def vote(values, weights):
    acc, order = {}, {}
    for i, (v, w) in enumerate(zip(values, weights)):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        v = str(v)
        acc[v] = acc.get(v, 0.0) + w
        order.setdefault(v, i)
    if not acc:
        return ""
    return max(acc, key=lambda k: (acc[k], -order[k]))


def align(frames, id_col):
    first = frames[0]
    out = [first.reset_index(drop=True)]
    for f in frames[1:]:
        aligned = None
        if id_col is not None and id_col in f.columns and not f[id_col].duplicated().any():
            g = f.set_index(f[id_col].astype(str).str.strip())
            keys = first[id_col].astype(str).str.strip()
            if set(keys) <= set(g.index):
                aligned = g.loc[keys.tolist()].reset_index(drop=True)
        if aligned is None:
            aligned = f.reset_index(drop=True).reindex(range(len(first)))
        out.append(aligned)
    return out


def combine(frames, weights, fmt):
    frames = align(frames, fmt["id_col"])
    out = frames[0].copy()
    for c in fmt["pred_cols"]:
        cols = [f[c] if c in f.columns else pd.Series([None] * len(out)) for f in frames]
        if fmt["strategy"] == "average":
            nums = [pd.to_numeric(col, errors="coerce") for col in cols]
            total = sum(n.fillna(0) * w for n, w in zip(nums, weights))
            wsum = sum((~n.isna()).astype(float) * w for n, w in zip(nums, weights))
            out[c] = total / wsum.where(wsum > 0, 1.0)
        elif fmt["strategy"] == "tokenwise_vote":
            merged = []
            for i in range(len(out)):
                toks = [str(col.iloc[i]).split() if pd.notna(col.iloc[i]) else None for col in cols]
                ref = toks[0] or []
                merged.append(" ".join(vote([t[j] if t is not None and j < len(t) else None for t in toks], weights)
                                       for j in range(len(ref))))
            out[c] = merged
        else:
            out[c] = [vote([col.iloc[i] for col in cols], weights) for i in range(len(out))]
    if fmt["normalize"]:
        sub = out[fmt["pred_cols"]].astype(float)
        sums = sub.sum(axis=1)
        out[fmt["pred_cols"]] = sub.div(sums.where(sums != 0, 1.0), axis=0)
    return out[fmt["columns"]]
'''

ENSEMBLE_SCRIPT = '''"""Recompute ensemble/predictions*.csv from the top solutions' predictions.

Usage (from the output directory):  python ensemble/ensemble.py
Only pandas is required.
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


''' + SCRIPT_HELPERS + '''

def main():
    with open(os.path.join(HERE, "members.json")) as fh:
        meta = json.load(fh)
    fmt = meta["format"]
    for size, group in meta["groups"].items():
        frames = [pd.read_csv(os.path.join(ROOT, "top", f"rank_{m['rank']:02d}", "predictions.csv"), dtype=str,
                              keep_default_na=False) for m in group]
        res = combine(frames, [m["weight"] for m in group], fmt)
        res.to_csv(os.path.join(HERE, f"predictions_top{size}.csv"), index=False)
        if int(size) == meta["primary_size"]:
            res.to_csv(os.path.join(HERE, "predictions.csv"), index=False)


if __name__ == "__main__":
    main()
'''


def run_ensemble(output_dir: Path, top: list[dict[str, Any]], higher_is_better: bool, cfg: dict[str, Any]) -> dict[str, Any]:
    """``top`` is a list of {rank, node_id, score} in top-list order. Returns the summary block."""
    out_dir = Path(output_dir) / "ensemble"
    cands = []
    for t in top:
        pred = Path(output_dir) / "top" / f"rank_{t['rank']:02d}" / "predictions.csv"
        if not pred.exists():
            continue
        if not higher_is_better and t["score"] is not None and t["score"] < 0:
            continue
        cands.append({**t, "path": pred})
        if len(cands) >= int(cfg["max_candidates"]):
            break
    n = len(cands)
    sizes = member_sizes(n)
    if not sizes:
        return {"available": False, "primary_size": None, "strategy": None, "members": []}
    frames = [pd.read_csv(c["path"], dtype=str, keep_default_na=False) for c in cands]
    fmt = detect_format(frames[0])
    primary = min(int(cfg["primary_size"]), n)
    if primary not in sizes:
        primary = max(k for k in sizes if k <= primary) if any(k <= primary for k in sizes) else sizes[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict[str, Any]]] = {}
    for k in sizes:
        w = weights([float(c["score"]) for c in cands[:k]], higher_is_better, fmt["strategy"])
        res = combine(frames[:k], w, fmt)
        res.to_csv(out_dir / f"predictions_top{k}.csv", index=False)
        if k == primary:
            shutil.copyfile(out_dir / f"predictions_top{k}.csv", out_dir / "predictions.csv")
        groups[str(k)] = [{"rank": c["rank"], "node_id": c["node_id"], "score": c["score"], "weight": wt}
                          for c, wt in zip(cands[:k], w)]
    meta = {"primary_size": primary, "strategy": fmt["strategy"], "row_normalized": fmt["normalize"],
            "format": {k: fmt[k] for k in ("id_col", "pred_cols", "copy_cols", "strategy", "normalize", "columns")},
            "groups": groups}
    atomic_write_json(out_dir / "members.json", meta)
    (out_dir / "ensemble.py").write_text(ENSEMBLE_SCRIPT, encoding="utf-8")
    return {"available": True, "primary_size": primary, "strategy": fmt["strategy"], "members": groups[str(primary)]}


__all__ = ["run_ensemble", "member_sizes", "weights", "VOTE_STRATEGIES", "detect_format", "combine", "SCRIPT_HELPERS", "ENSEMBLE_SCRIPT"]
