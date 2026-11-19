"""Small deterministic regression baseline, not a competition solution."""

import io
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from superagent.contract import records
from superagent.training import save_checkpoint, resume_path


def examples(split, data=Path("data")):
    for path in sorted((data / split).glob("*.tfrec")):
        for i, f in enumerate(records(path)):
            v = Image.open(io.BytesIO(f["image"].bytes_list.value[0])).convert("RGB")
            x = np.r_[np.asarray(v).mean(axis=(0, 1)) / 255.0, 1]
            key = (
                f["id"].bytes_list.value[0].decode()
                if split == "test"
                else f"{path.relative_to(data).as_posix()}:{i}"
            )
            yield key, x, f["class"].int64_list.value[0] if split == "train" else None


def predict(w, data, output):
    output.mkdir(exist_ok=True, parents=True)
    for split, dest in (("val", "val_predictions.csv"), ("test", "predictions.csv")):
        rows = [(key, int((x @ w).argmax())) for key, x, _ in examples(split, data)]
        df = pd.DataFrame(rows, columns=["id", "label"])
        if split == "test":
            order = pd.read_csv(data / "sample_submission.csv", dtype=str).id
            df = df.set_index("id").loc[order].reset_index()
        df.to_csv(output / dest, index=False)


def main():
    resumed = resume_path()
    if resumed:
        with np.load(resumed, allow_pickle=False) as data:
            w, epoch, steps = data["weights"], int(data["epoch"]), int(data["steps"])
    else:
        w, epoch, steps = np.zeros((4, 104)), 0, 0
    t = time.monotonic()
    for _, x, y in examples("train"):
        logits = x @ w
        p = np.exp(logits - logits.max())
        p /= p.sum()
        p[y] -= 1
        w -= 0.3 * x[:, None] * p
        steps += 1
    train_s = time.monotonic() - t
    t = time.monotonic()
    predict(w, Path("data"), Path("out"))
    model = Path("out/model")
    model.mkdir(exist_ok=True)
    np.savez(model / "weights.npz", weights=w)
    (model / "inference.json").write_text(json.dumps({"version":1, "weight_files":["weights.npz"]}))
    np.savez("work/head.npz", weights=w, epoch=epoch + 1, steps=steps)
    save_checkpoint(
        "work/head.npz",
        epoch=epoch + 1,
        next_seconds=max(1, train_s),
        delivery_seconds=max(1, time.monotonic() - t),
        can_continue=epoch < 3,
        optimizer_steps=steps, model_state=w, optimizer_state={"steps":steps,"lr":.3},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.predict_only:
        with np.load(args.model_dir / "weights.npz", allow_pickle=False) as state:
            predict(state["weights"], args.data_dir, args.output_dir)
    else:
        main()
