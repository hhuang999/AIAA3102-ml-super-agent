"""Validate one allocated CUDA GPU, training, and checkpoint round trips."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18


def classifier():
    return nn.Sequential(
        nn.Linear(32, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 2),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"PyTorch {torch.__version__}, CUDA runtime {torch.version.cuda}", flush=True)
    # Explicit initialization preserves the driver's diagnostic on failure.
    torch.cuda.init()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one allocated GPU, got {torch.cuda.device_count()}")

    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))))
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    print(f"Allocated GPU: {properties.name}, {properties.total_memory / 2**30:.2f} GiB", flush=True)

    # Compare actual CUDA output with a CPU reference, not just device discovery.
    matrix_a = torch.randn(512, 512)
    matrix_b = torch.randn(512, 512)
    expected = matrix_a @ matrix_b
    actual = (matrix_a.to(device) @ matrix_b.to(device)).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    print("CUDA matrix multiplication: passed", flush=True)

    # Exercise torchvision, cuDNN convolutions, and gradients without downloads.
    image_model = resnet18(weights=None, num_classes=2).to(device)
    image_loss = nn.functional.cross_entropy(
        image_model(torch.randn(4, 3, 64, 64, device=device)),
        torch.tensor([0, 1, 0, 1], device=device),
    )
    image_loss.backward()
    if not torch.isfinite(image_loss) or not all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in image_model.parameters()
    ):
        raise RuntimeError("Convolution forward/backward produced invalid gradients")
    del image_model, image_loss
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("ResNet18 forward/backward: passed", flush=True)

    features, labels = make_classification(
        n_samples=10000, n_features=32, n_informative=20, n_redundant=4,
        class_sep=2.5, flip_y=0.01, random_state=2026,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=2026,
    )
    scaler = StandardScaler()
    x_train = torch.from_numpy(scaler.fit_transform(x_train).astype(np.float32))
    x_val = torch.from_numpy(scaler.transform(x_val).astype(np.float32)).to(device)
    y_train = torch.from_numpy(y_train)
    y_val = torch.from_numpy(y_val).to(device)
    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=256, shuffle=True,
        num_workers=2, pin_memory=True, persistent_workers=True,
        generator=torch.Generator().manual_seed(2026),
    )
    model = classifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_function = nn.CrossEntropyLoss()
    history = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for epoch in range(10):
        model.train()
        loss_sum = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_x), batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss became non-finite")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch_x)
        model.eval()
        with torch.inference_mode():
            val_accuracy = (model(x_val).argmax(dim=1) == y_val).float().mean().item()
        metrics = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / len(x_train),
            "val_accuracy": val_accuracy,
        }
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if history[-1]["train_loss"] >= history[0]["train_loss"] * 0.6:
        raise RuntimeError("Training did not sufficiently reduce loss")
    if history[-1]["val_accuracy"] < 0.90:
        raise RuntimeError("Synthetic validation accuracy is below 90%")

    with torch.inference_mode():
        before_reload = model(x_val).cpu()
    checkpoint = args.output_dir / "classifier.pt"
    torch.save({
        "model": model.state_dict(),
        "scaler_mean": torch.from_numpy(scaler.mean_),
        "scaler_scale": torch.from_numpy(scaler.scale_),
    }, checkpoint)
    restored = classifier().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    restored.load_state_dict(state["model"])
    restored.eval()
    with torch.inference_mode():
        after_reload = restored(x_val).cpu()
    torch.testing.assert_close(before_reload, after_reload, rtol=0, atol=0)

    packages = (
        "torch", "torchvision", "numpy", "pandas", "scikit-learn",
        "catboost", "lightgbm", "xgboost", "transformers",
    )
    # Import the installed training libraries too; metadata alone misses linker errors.
    import catboost
    import lightgbm
    import pandas
    import transformers
    import xgboost

    result = {
        "status": "passed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "gpu": properties.name,
        "gpu_memory_gib": properties.total_memory / 2**30,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "scheduler_cuda_visible_devices": os.environ.get("SCHEDULER_CUDA_VISIBLE_DEVICES"),
        "torch_cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "versions": {name: importlib.metadata.version(name) for name in packages},
        "checks": {
            "cuda_matmul": "passed", "resnet18_backward": "passed",
            "synthetic_training": "passed", "checkpoint_reload": "passed",
            "training_library_imports": "passed",
        },
        "train_samples": len(x_train), "validation_samples": len(x_val),
        "history": history,
        "training_seconds": elapsed,
        "peak_training_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        "checkpoint": str(checkpoint),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
