import json
from pathlib import Path

import numpy as np
import pandas as pd

from superagent.config import load_defaults
from superagent.profiler import detect_modalities, profile

PCFG = load_defaults()["profiler"]


def test_wide_csv_switches_to_brief(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"column_number_{i}": rng.choice([f"some_long_category_value_{j}" for j in range(40)], 300) for i in range(40)})
    df.to_csv(tmp_path / "train.csv", index=False)
    res = profile(tmp_path, PCFG)
    assert "300 rows, 40 columns" in res.preview
    assert "column_number_14" in res.preview
    assert "column_number_15" not in res.preview
    assert "(25 more)" in res.preview


def test_sample_predictions_annotated(tmp_path):
    pd.DataFrame({"id": [1, 2], "label": [0, 1]}).to_csv(tmp_path / "sample_predictions.csv", index=False)
    res = profile(tmp_path, PCFG)
    assert "REQUIRED FORMAT" in res.preview
    assert "['id', 'label']" in res.preview


def test_detect_tabular(tmp_path):
    rng = np.random.default_rng(0)
    pd.DataFrame({"id": range(100), "x1": rng.random(100), "x2": rng.random(100), "y": rng.integers(0, 2, 100)}).to_csv(tmp_path / "train.csv", index=False)
    assert detect_modalities(tmp_path, PCFG) == ["tabular"]


def test_detect_text(tmp_path):
    rng = np.random.default_rng(0)
    words = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit", "sed", "do"]
    texts = [" ".join(rng.choice(words, 35)) + f" doc{i}" for i in range(100)]
    pd.DataFrame({"id": range(100), "text": texts, "label": rng.integers(0, 2, 100)}).to_csv(tmp_path / "train.csv", index=False)
    assert detect_modalities(tmp_path, PCFG) == ["text"]


def test_detect_image(tmp_path):
    from PIL import Image

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    names = []
    for i in range(500):
        n = f"img_{i}.png"
        Image.new("L", (8, 8), color=i % 255).save(img_dir / n)
        names.append(n)
    pd.DataFrame({"filename": names, "label": [i % 3 for i in range(500)]}).to_csv(tmp_path / "train.csv", index=False)
    assert detect_modalities(tmp_path, PCFG) == ["image"]


def test_preview_text_and_json(tmp_path):
    (tmp_path / "notes.txt").write_text("\n".join(f"line {i}" for i in range(50)))
    (tmp_path / "meta.json").write_text(json.dumps({"a": [1, 2], "b": {"c": "x"}}))
    res = profile(tmp_path, PCFG)
    assert "first 15 of 50 lines" in res.preview
    assert "JSON structure" in res.preview
