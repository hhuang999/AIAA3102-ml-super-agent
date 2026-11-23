"""Prediction files of a Core output directory, and their conversion into submission files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def make_submission(pred_csv: Path, sample_csv: Path, out_csv: Path) -> Path:
    """Copy a prediction file into the submission format (the sample's columns, aligned IDs and non-empty cells)."""
    pred = pd.read_csv(pred_csv, dtype=str, keep_default_na=False)
    sample = pd.read_csv(sample_csv, dtype=str, keep_default_na=False)
    cols = list(sample.columns)
    missing = [c for c in cols if c not in pred.columns]
    if missing:
        raise ValueError(f"{pred_csv} lacks submission columns {missing}")
    id_col = cols[0]
    p_ids = pred[id_col].astype(str).str.strip()
    s_ids = sample[id_col].astype(str).str.strip()
    if p_ids.duplicated().any():
        raise ValueError(f"{pred_csv} contains duplicate sample IDs in {id_col}")
    if set(p_ids) != set(s_ids) or len(p_ids) != len(s_ids):
        raise ValueError(f"{pred_csv} ID set does not match {sample_csv}")
    aligned = pred.set_index(p_ids).loc[s_ids.tolist()].reset_index(drop=True)
    aligned[id_col] = sample[id_col].values
    for c in cols[1:]:
        if (aligned[c].astype(str).str.strip() == "").any():
            raise ValueError(f"{pred_csv} contains empty prediction cells in column '{c}'")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    aligned[cols].to_csv(out_csv, index=False)
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
