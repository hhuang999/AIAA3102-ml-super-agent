"""Deterministic binary-classification script served by the acceptance HTTP fixture."""

import argparse
import csv
import json
import os
from pathlib import Path

from superagent.training import resume_path, save_checkpoint


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def predict(model, data, output):
    output.mkdir(parents=True, exist_ok=True)
    threshold = sum(model["means"].values()) / 2
    for split, name in (("val", "val_predictions.csv"), ("test", "predictions.csv")):
        values = rows(data / (split + ".csv"))
        if split == "test":
            by_id = {r["id"]: r for r in values}
            values = [by_id[r["id"]] for r in rows(data / "sample_predictions.csv")]
        with (output / name).open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label"])
            writer.writerows((r["id"], "01" if float(r["x"]) < threshold else "NA") for r in values)


def train():
    previous = resume_path()
    old = json.loads(previous.read_text()) if previous else {"means": {"01": 0.0, "NA": 0.0}, "steps": 0, "epoch": 0}
    data = Path("data")
    original = rows(data / "train.csv")
    updated = dict(old["means"])
    for row in original:
        key = row["label"]
        updated[key] += 0.4 * (float(row["x"]) - updated[key])
    state = {"means": updated, "steps": old["steps"] + len(original), "epoch": old["epoch"] + 1}
    predict(state, data, Path("out"))
    model = Path("out/model")
    model.mkdir(exist_ok=True)
    (model / "weights.json").write_text(json.dumps(state))
    (model / "inference.json").write_text('{"version":1,"weight_files":["weights.json"]}')
    save_checkpoint(model / "weights.json", epoch=state["epoch"], next_seconds=1, delivery_seconds=2,
                    can_continue=os.getenv("SUPERAGENT_TEST_NO_CONTINUE") != "1",
                    optimizer_steps=state["steps"], model_state=updated, optimizer_state={"steps": state["steps"]})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predict-only", action="store_true")
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    args = p.parse_args()
    if args.predict_only:
        predict(json.loads((args.model_dir / "weights.json").read_text()), args.data_dir, args.output_dir)
    else:
        train()
