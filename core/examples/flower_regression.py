"""Create a small synthetic flower-shaped TFRecord task for offline regression."""

import argparse
import io
from pathlib import Path

import yaml
from PIL import Image
from tfrecord import TFRecordWriter


def create(root):
    root = Path(root)
    root.mkdir(exist_ok=False)
    (root / "baseline.py").write_bytes(Path(__file__).with_name("flower_baseline.py").read_bytes())
    (root / "description.md").write_text(
        "Classify JPEG flower images into 104 species. Use the supplied validation split."
    )
    spec = {
        "version": 1,
        "metric": "accuracy",
        "classes": list(range(104)),
        "sample": "sample_submission.csv",
        "inference_inputs": ["val", "test", "sample_submission.csv"],
        "validation": {"format": "tfrecord", "glob": "val/*.tfrec", "label": "class"},
    }
    (root / "task.yaml").write_text(yaml.safe_dump(spec))
    for split in ("train", "val", "test"):
        folder = root / "data" / split
        folder.mkdir(parents=True)
        w = TFRecordWriter(str(folder / "00.tfrec"))
        try:
            for i in range(6):
                im = io.BytesIO()
                Image.new("RGB", (16, 16), (i * 30, 50, 120)).save(im, "JPEG")
                w.write(
                    {
                        "image": (im.getvalue(), "byte"),
                        **(
                            {"id": (f"f{i}".encode(), "byte")}
                            if split == "test"
                            else {"class": (i % 3, "int")}
                        ),
                    }
                )
        finally:
            w.close()
    (root / "data/sample_submission.csv").write_text(
        "id,label\n" + "".join(f"f{i},0\n" for i in range(6))
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("destination")
    create(p.parse_args().destination)
