"""Generate small example task directories from datasets that ship with common ML libraries.

    python examples/make_example_tasks.py --out example_tasks [--tasks tabular_small tabular image text]

Each task gets ``<name>/task/`` (description.md + data/) and ``<name>/test_labels.csv``, which is kept OUTSIDE
the task directory so the agent never sees it. ``score_predictions.py`` scores a prediction file against it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

SEED = 0


def _write_task(root: Path, description: str, train: pd.DataFrame, test: pd.DataFrame, test_labels: pd.DataFrame,
                sample: pd.DataFrame, extra: dict[str, bytes] | None = None) -> None:
    data = root / "task" / "data"
    data.mkdir(parents=True, exist_ok=True)
    (root / "task" / "description.md").write_text(description.strip() + "\n", encoding="utf-8")
    train.to_csv(data / "train.csv", index=False)
    test.to_csv(data / "test.csv", index=False)
    sample.to_csv(data / "sample_predictions.csv", index=False)
    test_labels.to_csv(root / "test_labels.csv", index=False)
    for rel, content in (extra or {}).items():
        p = data / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)


def _split(df: pd.DataFrame, label: str, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr, te = train_test_split(df, test_size=test_size, random_state=SEED, stratify=df[label])
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def make_tabular(out: Path, small: bool, contract: bool = False) -> None:
    from sklearn.datasets import load_breast_cancer, load_iris

    if small:
        ds = load_iris(as_frame=True)
        df = ds.frame.rename(columns=lambda c: c.replace(" (cm)", "").replace(" ", "_"))
        df["species"] = df.pop("target").map(dict(enumerate(ds.target_names)))
        label, metric = "species", "accuracy"
        name = "tabular_small"
        goal = "Predict the species of each flower from its four measurements."
    else:
        ds = load_breast_cancer(as_frame=True)
        df = ds.frame.rename(columns=lambda c: c.replace(" ", "_"))
        df["diagnosis"] = df.pop("target")
        label, metric = "diagnosis", "ROC AUC"
        name = "tabular"
        goal = "Predict whether each tumour is benign (1) or malignant (0) from 30 numeric features computed from a digitised image."
    df.insert(0, "id", [f"s{i:04d}" for i in range(len(df))])
    tr, te = _split(df, label, 0.3)
    extra = {}
    validation = ""
    if contract:
        tr, val = _split(tr, label, 0.25)
        extra["val.csv"] = val.drop(columns=[label]).to_csv(index=False).encode()
        extra["val_labels.csv"] = val[["id", label]].to_csv(index=False).encode()
        validation = "- `val.csv`: fixed validation features. Core scores these predictions using `val_labels.csv`.\n"
    if metric == "accuracy":
        sample = pd.DataFrame({"id": te["id"], label: tr[label].iloc[0]})
        fmt = f"a CSV with columns `id` and `{label}` (the predicted class name)"
        metric_txt = "Accuracy (fraction of correctly classified test samples). Higher is better."
    else:
        sample = pd.DataFrame({"id": te["id"], label: 0.5})
        fmt = f"a CSV with columns `id` and `{label}` holding the predicted probability that `{label}` is 1"
        metric_txt = "Area under the ROC curve (ROC AUC) of the predicted probabilities. Higher is better."
    desc = f"""# Task
{goal}

## Data
- `train.csv`: training samples with an `id` column, the feature columns and the target column `{label}`.
- `test.csv`: test samples with the same columns except `{label}`.
{validation}- `sample_predictions.csv`: an example of the required submission format.

## Evaluation
{metric_txt}

## Submission
Predict every row of `test.csv` and write {fmt}.
"""
    root = out / name
    _write_task(root, desc, tr, te.drop(columns=[label]), te[["id", label]], sample, extra)
    if contract:
        spec = {
            "version": 1,
            "metric": "accuracy",
            "classes": sorted(map(str, tr[label].unique())),
            "sample": "sample_predictions.csv",
            "inference_inputs": ["val.csv", "test.csv", "sample_predictions.csv"],
            "validation": {"format": "csv", "path": "val_labels.csv", "id": "id", "label": label},
        }
        (root / "task/task.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")


def make_image(out: Path) -> None:
    import io

    from PIL import Image
    from sklearn.datasets import load_digits

    ds = load_digits()
    imgs = (ds.images / 16.0 * 255).astype(np.uint8)
    ids = [f"img_{i:05d}" for i in range(len(imgs))]
    df = pd.DataFrame({"id": ids, "label": ds.target})
    tr, te = _split(df, "label", 0.3)
    files: dict[str, bytes] = {}
    for i, img in zip(ids, imgs):
        buf = io.BytesIO()
        Image.fromarray(img, mode="L").resize((32, 32), Image.NEAREST).save(buf, format="PNG")
        files[f"images/{i}.png"] = buf.getvalue()
    train = pd.DataFrame({"id": tr["id"], "file": "images/" + tr["id"] + ".png", "label": tr["label"]})
    test = pd.DataFrame({"id": te["id"], "file": "images/" + te["id"] + ".png"})
    sample = pd.DataFrame({"id": te["id"], "label": 0})
    desc = """# Task
Classify small grayscale images of handwritten digits into the ten classes 0-9.

## Data
- `images/`: one 32x32 grayscale PNG file per sample.
- `train.csv`: columns `id`, `file` (path of the image relative to the data directory) and `label` (0-9).
- `test.csv`: columns `id` and `file`.
- `sample_predictions.csv`: an example of the required submission format.

## Evaluation
Accuracy on the test images. Higher is better.

## Submission
A CSV with columns `id` and `label` (the predicted digit) for every test image.
"""
    _write_task(out / "image", desc, train, test, te[["id", "label"]], sample, files)


def make_text(out: Path) -> None:
    from sklearn.datasets import fetch_20newsgroups

    cats = ["rec.autos", "sci.med", "comp.graphics", "talk.politics.guns"]
    ds = fetch_20newsgroups(subset="all", categories=cats, remove=("headers", "footers", "quotes"), random_state=SEED)
    df = pd.DataFrame({"text": ds.data, "topic": [ds.target_names[t] for t in ds.target]})
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len() > 20].reset_index(drop=True)
    df = df.sample(n=min(len(df), 2000), random_state=SEED).reset_index(drop=True)
    df.insert(0, "id", [f"doc{i:05d}" for i in range(len(df))])
    tr, te = _split(df, "topic", 0.3)
    sample = pd.DataFrame({"id": te["id"], "topic": cats[0]})
    desc = f"""# Task
Assign each English newsgroup post to its topic. There are four topics: {", ".join(cats)}.

## Data
- `train.csv`: columns `id`, `text` (the post body) and `topic`.
- `test.csv`: columns `id` and `text`.
- `sample_predictions.csv`: an example of the required submission format.

## Evaluation
Macro-averaged F1 score over the four topics. Higher is better.

## Submission
A CSV with columns `id` and `topic` (one of the four topic names) for every test post.
"""
    _write_task(out / "text", desc, tr, te.drop(columns=["topic"]), te[["id", "topic"]], sample)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="example_tasks")
    ap.add_argument("--tasks", nargs="+", default=["tabular_small", "tabular", "image", "text"],
                    choices=["tabular_small", "tabular", "image", "text"])
    ap.add_argument("--contract", action="store_true", help="Add a fixed, independently scored split to tabular_small")
    args = ap.parse_args()
    out = Path(args.out)
    for t in args.tasks:
        if t == "tabular_small":
            make_tabular(out, small=True, contract=args.contract)
        elif t == "tabular":
            make_tabular(out, small=False)
        elif t == "image":
            make_image(out)
        elif t == "text":
            make_text(out)
        print(f"created {out / t}")


if __name__ == "__main__":
    main()
