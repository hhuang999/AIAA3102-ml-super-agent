"""Prediction files of a Core output directory, and their conversion into submission files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def make_submission(pred_csv: Path, sample_csv: Path, out_csv: Path) -> Path:
    """Copy a prediction file into the submission format (the sample's columns and column order)."""
    pred = pd.read_csv(pred_csv)
    cols = list(pd.read_csv(sample_csv, nrows=1).columns)
    missing = [c for c in cols if c not in pred.columns]
    if missing:
        raise ValueError(f"{pred_csv} lacks submission columns {missing}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pred[cols].to_csv(out_csv, index=False)
    return out_csv


def primary_name(core_out: Path) -> str:
    """'final' when the Core wrote final/predictions.csv (interface 1.1), otherwise 'best'."""
    return "final" if (Path(core_out) / "final" / "predictions.csv").exists() else "best"


def candidate_files(core_out: Path, include_top: bool = False) -> dict[str, Path]:
    """Prediction files to grade, the primary result first. The others are for comparison only."""
    core_out = Path(core_out)
    files: dict[str, Path] = {}
    if (core_out / "final" / "predictions.csv").exists():
        files["final"] = core_out / "final" / "predictions.csv"
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
