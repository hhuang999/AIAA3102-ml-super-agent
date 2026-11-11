"""Report local readiness; optional checks inspect train/val and CUDA only."""

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys


PACKAGES = (
    "numpy", "pandas", "scikit-learn", "torch", "catboost", "lightgbm",
    "xgboost", "transformers", "sentence-transformers", "openai",
    "pydantic", "paramiko", "httpx",
)


def package_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def check_data(root):
    import numpy as np

    tasks = root / "final_project_tasks"
    result = {}
    for split in ("train", "val"):
        headlines = tasks / "headlines_ctr_clean" / f"{split}.csv"
        with headlines.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            expected = ["headline_A", "headline_B", "label_pairwise"]
            if columns != expected:
                raise ValueError(f"{headlines}: unexpected columns {columns}")
            rows = sum(1 for _ in reader)
        coupon = tasks / "in_vehicle_coupon_recommendation"
        # The trusted course feature files contain pickled object arrays.
        features = np.load(coupon / f"C_{split}.npy", allow_pickle=True)
        labels = np.load(coupon / f"y_{split}.npy")
        if features.ndim != 2 or features.shape[1] != 21:
            raise ValueError(f"coupon {split}: unexpected shape {features.shape}")
        if labels.shape != (features.shape[0],):
            raise ValueError(f"coupon {split}: feature/label row mismatch")
        result[split] = {
            "headline_rows": rows,
            "coupon_features_shape": list(features.shape),
            "coupon_labels_shape": list(labels.shape),
        }
    return result


def check_cuda():
    import torch

    result = {"available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
    if result["available"]:
        device = torch.cuda.get_device_properties(0)
        values = torch.tensor([1.0, 2.0], device="cuda")
        value = (values * values).sum().item()
        if value != 5.0:
            raise RuntimeError(f"CUDA computation returned {value}, expected 5.0")
        result.update(
            device=device.name,
            memory_gib=round(device.total_memory / 2**30, 2),
            compute_check="passed",
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-cuda", action="store_true")
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "workspace": str(root),
        "workspace_free_gib": round(shutil.disk_usage(root).free / 2**30, 2),
        "packages": package_versions(),
    }
    if args.check_cuda:
        report["cuda"] = check_cuda()
    if args.check_data:
        report["data"] = check_data(root)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
