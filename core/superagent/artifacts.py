"""Verify a persistent inference package inside the execution deadline and slot."""

import json
import os
import hashlib
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import pandas as pd

from .training import digest


PROB_ATOL = 1e-5
PROB_RTOL = 1e-4


WRAPPER = """import runpy
try:
    import torch
except ImportError:
    torch = None
def forbidden(*a, **kw):
    raise RuntimeError('inference-only verification attempted training')
if torch is not None:
    torch.Tensor.backward = forbidden
    torch.autograd.backward = forbidden
    torch.optim.Optimizer.__init__ = forbidden
runpy.run_path('solution.py', run_name='__main__')
"""


def payload_hash(path, deadline=None):
    h = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            if deadline is not None and time.monotonic() >= deadline:
                raise ValueError("deadline exhausted checking model payload")
            h.update(chunk)
    return h.hexdigest()


def package_files(root, deadline=None):
    if not root.is_dir() or root.is_symlink():
        raise ValueError("persistent out/model/ package is missing or symlinked")
    files = {}
    for p in root.rglob("*"):
        if p.is_symlink():
            raise ValueError("model package must contain its own files, not symlinks")
        if p.is_file():
            files[p.relative_to(root).as_posix()] = payload_hash(p, deadline)
    manifest = json.loads((root / "inference.json").read_text(encoding="utf8"))
    if not isinstance(manifest, dict):
        raise ValueError("inference.json must be an object")
    weights = manifest.get("weight_files")
    if manifest.get("version") != 1 or not isinstance(weights, list) or not weights:
        raise ValueError("inference.json requires version=1 and nonempty weight_files")
    for name in weights:
        if not isinstance(name, str) or name == "inference.json" or name not in files:
            raise ValueError(
                "weight_files must name actual payload files inside model/"
            )
    return files, weights


def same_predictions(original, replay, probability, *, atol=PROB_ATOL, rtol=PROB_RTOL):
    for name in ("predictions.csv", "val_predictions.csv"):
        if any((folder / name).is_symlink() or not (folder / name).is_file() for folder in (original, replay)):
            return False
        a = pd.read_csv(original / name, dtype=str, keep_default_na=False)
        b = pd.read_csv(replay / name, dtype=str, keep_default_na=False)
        if (
            list(a) != list(b)
            or len(a) != len(b)
            or a.iloc[:, 0].duplicated().any()
            or b.iloc[:, 0].duplicated().any()
        ):
            return False
        if name == "predictions.csv" and a.iloc[:, 0].tolist() != b.iloc[:, 0].tolist():
            return False
        a, b = (
            a.set_index(a.columns[0]).sort_index(),
            b.set_index(b.columns[0]).sort_index(),
        )
        if not a.index.equals(b.index):
            return False
        if probability:
            if not np.allclose(
                a.to_numpy(float), b.to_numpy(float), atol=atol, rtol=rtol
            ):
                return False
        elif not a.equals(b):
            return False
    return True


def verify(executor, run_dir, slot, deadline, abort, inputs, probability, *, spec=None):
    started = time.monotonic()
    receipt = {"passed": False}
    package = run_dir / "out/model"
    try:
        isolated = getattr(executor, "isolation", None) is not None
        if not inputs and not isolated:
            raise ValueError(
                "task.yaml must declare inference_inputs without training data"
            )
        inventory, weights = package_files(package, deadline)
        receipt["package_sha256"] = inventory
        with tempfile.TemporaryDirectory(prefix="inference-", dir=run_dir) as temp:
            fresh = Path(temp)
            shutil.copyfile(run_dir / "solution.py", fresh / "solution.py")
            for name in inventory:
                target = fresh / "model" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with (package / name).open("rb") as source, target.open("wb") as dest:
                    while chunk := source.read(1024 * 1024):
                        if time.monotonic() >= deadline:
                            raise ValueError(
                                "deadline exhausted copying inference package"
                            )
                        dest.write(chunk)
            (fresh / "verify_inference.py").write_text(WRAPPER, encoding="utf8")
            data, output = fresh / "data", fresh / "out"
            data.mkdir()
            output.mkdir()
            for name in (() if isolated else inputs):
                target = data / name
                target.parent.mkdir(parents=True, exist_ok=True)
                source = executor.prepared_dir / name
                target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
            env = {
                "PYTHONPATH": os.pathsep.join(
                    filter(
                        None,
                        [
                            str(Path(__file__).resolve().parents[1]),
                            os.environ.get("PYTHONPATH"),
                        ],
                    )
                )
            }
            argv = [
                "verify_inference.py",
                "--predict-only",
                "--model-dir",
                "model" if isolated else str(fresh / "model"),
                "--data-dir",
                "data" if isolated else str(data),
                "--output-dir",
                "out" if isolated else str(output),
            ]
            left = deadline - time.monotonic()
            if left <= 0:
                raise ValueError("deadline exhausted before inference verification")
            positive = executor._run_in_slot(
                fresh, slot, left * 0.65, abort, env, argv=argv
            )
            shutil.copyfile(fresh / "stdout.log", run_dir / "inference_stdout.log")
            if positive.error_type or positive.exit_code != 0:
                raise ValueError(
                    "inference-only process failed: " + positive.output[-1200:]
                )
            if not same_predictions(run_dir / "out", output, probability):
                raise ValueError(
                    "reloaded model predictions differ from training outputs"
                )
            receipt["reloaded_predictions_match"] = True
            if spec is not None:
                receipt["score"] = spec.grade(output)
                if not np.isclose(receipt["score"], spec.grade(run_dir / "out"), atol=PROB_ATOL, rtol=PROB_RTOL):
                    raise ValueError("reloaded predictions do not reproduce the independent score")
            for name in weights:
                (fresh / "model" / name).write_bytes(
                    b"invalid evaluator-only weight payload"
                )
            shutil.rmtree(output)
            output.mkdir()
            left = deadline - time.monotonic()
            if left <= 0:
                raise ValueError("deadline exhausted before corrupt-weight check")
            negative = executor._run_in_slot(fresh, slot, left, abort, env, argv=argv)
            shutil.copyfile(fresh / "stdout.log", run_dir / "corrupt_weight_stdout.log")
            if (
                negative.timed_out
                or negative.error_message
                and "cancelled" in negative.error_message
            ):
                raise ValueError("corrupt-weight verification did not finish")
            if negative.exit_code in (None, 0):
                raise ValueError("inference accepted corrupt declared weights")
            receipt.update(
                passed=True,
                corrupt_weights_rejected=True,
                code_sha256=digest(run_dir / "solution.py"),
                package_sha256=inventory,
                predictions_sha256={
                    name: digest(run_dir / "out" / name)
                    for name in ("val_predictions.csv", "predictions.csv")
                },
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        receipt["error"] = str(exc)
    receipt["seconds"] = time.monotonic() - started
    return receipt
