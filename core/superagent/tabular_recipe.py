"""Source template for tab_linear@1 and its standalone inference entry point."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TASK = None


def read_split(data, name):
    binding = TASK["input_manifest"]["splits"][name]
    return pd.concat(
        [pd.read_csv(data / path, dtype=str, keep_default_na=False) for path in binding["paths"]],
        ignore_index=True,
    )


def matrix(table, numeric, categorical):
    result = pd.DataFrame(index=table.index)
    for name in numeric:
        result[name] = pd.to_numeric(table[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for name in categorical:
        result[name] = table[name].astype(object).replace("", np.nan)
    return result


def predict_only(model_dir, data, output):
    state = joblib.load(model_dir / "pipeline.joblib")
    model = state["pipeline"]
    columns = TASK["submission_columns"]
    output.mkdir(parents=True, exist_ok=True)
    for split, destination in (("validation", "val_predictions.csv"), ("test", "predictions.csv")):
        table = read_split(data, split)
        inputs = matrix(table, state["numeric"], state["categorical"])
        key = TASK["input_manifest"]["splits"][split]["id"]
        rows = pd.DataFrame({columns[0]: table[key]})
        if TASK["probability"]:
            raw = model.predict_proba(inputs)
            indices = {label: i for i, label in enumerate(model.classes_)}
            for label in TASK["classes"]:
                rows[label] = raw[:, indices[label]] if label in indices else 0.0
        else:
            rows[columns[1]] = model.predict(inputs)
        if split == "test":
            sample = pd.read_csv(data / TASK["input_manifest"]["sample"], dtype=str, keep_default_na=False)
            rows = rows.set_index(columns[0]).loc[sample[columns[0]]].reset_index()
        rows.to_csv(output / destination, index=False)


def fit(data=Path("data"), output=Path("out")):
    binding = TASK["input_manifest"]["splits"]["train"]
    training = read_split(data, "train")
    numeric = []
    for name in binding["features"]:
        populated = training.loc[training[name] != "", name]
        if not len(populated) or pd.to_numeric(populated, errors="coerce").notna().all():
            numeric.append(name)
    categorical = [name for name in binding["features"] if name not in numeric]
    transforms = []
    if numeric:
        transforms.append(("numeric", make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
        ), numeric))
    if categorical:
        transforms.append(("categorical", make_pipeline(
            SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True),
            OneHotEncoder(handle_unknown="ignore"),
        ), categorical))
    model = make_pipeline(ColumnTransformer(transforms), LogisticRegression(max_iter=2000, C=1, random_state=TASK["seed"]))
    model.fit(matrix(training, numeric, categorical), training[binding["label"]])
    model_dir = output / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "numeric": numeric, "categorical": categorical}, model_dir / "pipeline.joblib")
    (model_dir / "inference.json").write_text(
        json.dumps({"version": 1, "weight_files": ["pipeline.joblib"], "recipe_id": "tab_linear@1"}), encoding="utf-8",
    )
    predict_only(model_dir, data, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.predict_only:
        predict_only(args.model_dir, args.data_dir, args.output_dir)
    else:
        fit()
