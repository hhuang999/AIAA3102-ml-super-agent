"""Score a prediction file against held-out test labels (for the example tasks).

    python examples/score_predictions.py PREDICTIONS.csv TEST_LABELS.csv --metric accuracy|roc_auc|macro_f1
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def score(pred_path: str, labels_path: str, metric: str) -> float:
    pred = pd.read_csv(pred_path)
    truth = pd.read_csv(labels_path)
    id_col, label = truth.columns[0], truth.columns[1]
    merged = truth.merge(pred, on=id_col, suffixes=("_true", "_pred"))
    y_true, y_pred = merged[f"{label}_true"], merged[f"{label}_pred"]
    if metric == "accuracy":
        return float(accuracy_score(y_true.astype(str), y_pred.astype(str)))
    if metric == "macro_f1":
        return float(f1_score(y_true.astype(str), y_pred.astype(str), average="macro"))
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, y_pred.astype(float)))
    raise ValueError(metric)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("predictions")
    ap.add_argument("labels")
    ap.add_argument("--metric", required=True, choices=["accuracy", "roc_auc", "macro_f1"])
    a = ap.parse_args()
    print(f"{a.metric}: {score(a.predictions, a.labels, a.metric):.6f}")


if __name__ == "__main__":
    main()
