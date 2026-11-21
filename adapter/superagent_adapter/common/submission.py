"""Prediction files of a Core output directory, and their conversion into submission files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def make_submission(pred_csv: Path, sample_csv: Path, out_csv: Path) -> Path:
    """Export predictions in the sample's column and ID order."""
    # Identifiers and label strings may be numeric-looking (e.g. 0007 or NA).
    pred = pd.read_csv(pred_csv, dtype=str, keep_default_na=False)
    sample = pd.read_csv(sample_csv, dtype=str, keep_default_na=False)
    cols = list(sample.columns)
    missing = [c for c in cols if c not in pred.columns]
    if missing:
        raise ValueError(f"{pred_csv} lacks submission columns {missing}")
    id_col = cols[0]
    expected_raw = sample[id_col].astype(str)
    expected = expected_raw.str.strip()
    actual = pred[id_col].astype(str).str.strip()
    if (expected.empty or expected.duplicated().any() or actual.duplicated().any()
            or "" in set(expected) or set(actual) != set(expected)):
        raise ValueError(f"{pred_csv} must have one prediction per nonempty sample ID")
    for c in cols[1:]:
        if (pred[c].astype(str).str.strip() == "").any():
            raise ValueError(f"{pred_csv} has empty prediction cells in column '{c}'")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    aligned = pred.copy()
    aligned["_join_id"] = actual
    aligned = aligned.set_index("_join_id").loc[expected].reset_index(drop=True)
    aligned[id_col] = expected_raw.values
    aligned[cols].to_csv(out_csv, index=False)
    return out_csv


def primary_name(core_out: Path) -> str:
    """'final' when the Core wrote final/predictions.csv (interface 1.1), otherwise 'best'."""
    root = Path(core_out)
    return "final" if (root / "final/predictions.csv").exists() or (root / "logs/task_facts.json").is_file() else "best"


def publication(core_out):
    """Resolve the pointer once so the prediction and its provenance stay together."""
    root = Path(core_out)
    directory = (root / "final").resolve()
    if not (directory / "predictions.csv").is_file():
        return None
    ref = directory / "source.json"
    return directory, json.loads(ref.read_text(encoding="utf-8")) if ref.is_file() else {}


def candidate_files(core_out: Path, include_top: bool = False) -> dict[str, Path]:
    """Prediction files to grade, the primary result first. The others are for comparison only."""
    core_out = Path(core_out)
    files: dict[str, Path] = {}
    published = publication(core_out)
    if published:
        files["final"] = published[0] / "predictions.csv"
    elif primary_name(core_out) == "final":
        return {}
    files["best"] = core_out / "best" / "predictions.csv"
    ens = core_out / "ensemble"
    if (ens / "predictions.csv").exists():
        files["ensemble_primary"] = ens / "predictions.csv"
        for p in sorted(ens.glob("predictions_top*.csv")):
            files[f"ensemble_{p.stem.replace('predictions_', '')}"] = p
    if include_top:
        for d in sorted((core_out / "top").glob("rank_*")):
            if (d / "predictions.csv").exists():
                files[f"top_{d.name}"] = d / "predictions.csv"
    return {k: v for k, v in files.items() if v.exists()}
