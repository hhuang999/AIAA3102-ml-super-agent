"""Checkpoint protocol shared by solution subprocesses and the search scheduler.

The agent only verifies bytes and JSON metadata. Framework state is deserialized
inside the training process, never in the controller.
"""

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

from .utils import atomic_write_json


def state_fingerprint(value):
    """Stable hashes of arrays/tensors and framework state dictionaries, on CPU."""
    import numpy as np

    h = hashlib.sha256()

    def add(v):
        if isinstance(v, dict):
            h.update(b"dict")
            for k in sorted(v, key=repr):
                add(k)
                add(v[k])
            h.update(b"}")
        elif isinstance(v, (tuple, list)):
            h.update(type(v).__name__.encode())
            for item in v:
                add(item)
            h.update(b"]")
        elif isinstance(v, np.ndarray):
            if v.dtype.hasobject:
                raise TypeError("object arrays are not reproducible training state")
            h.update(str((v.dtype.str, v.shape)).encode())
            h.update(v.tobytes())
        elif type(v).__module__.split(".")[0] == "torch":
            import torch

            t = v.detach().cpu().contiguous()
            h.update(str((t.dtype, tuple(t.shape))).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, np.generic):
            add(v.item())
        elif v is None or isinstance(v, (str, int, float, bool)):
            h.update(json.dumps([type(v).__name__, v], allow_nan=False).encode())
        else:
            raise TypeError(f"unsupported training state: {type(v)}")

    add(value)
    return h.hexdigest()


_RESTORED = {}


def _context():
    return (
        os.environ["SUPERAGENT_TRIAL_SIGNATURE"],
        os.environ["SUPERAGENT_CHECKPOINT_DIR"],
    )


def progress_available(record):
    p = (record or {}).get("progress")
    return bool(
        isinstance(p, dict)
        and type(p.get("optimizer_steps")) is int
        and p["optimizer_steps"] > 0
        and all(
            isinstance(p.get(k), str)
            and len(p[k]) == 64
            and all(c in "0123456789abcdef" for c in p[k])
            for k in ("model_sha256", "optimizer_sha256")
        )
    )


def verify_continuation(previous, current):
    """Only compare independently committed states; never deserialize in the controller."""
    if not progress_available(previous) or not progress_available(current):
        raise ValueError("continuation has no verifiable optimizer/model progress")
    a, b = previous["progress"], current["progress"]
    if b.get("restored_from") != previous["sha256"]:
        raise ValueError("continuation did not restore the selected parent checkpoint")
    if (
        current["epoch"] <= previous["epoch"]
        or b["optimizer_steps"] <= a["optimizer_steps"]
    ):
        raise ValueError("continuation did not advance epochs and optimizer steps")
    if (
        any(a[k] == b[k] for k in ("model_sha256", "optimizer_sha256"))
        or previous["sha256"] == current["sha256"]
    ):
        raise ValueError("continuation repeated unchanged model or optimizer state")
    return dict(
        passed=True,
        from_epoch=previous["epoch"],
        to_epoch=current["epoch"],
        from_steps=a["optimizer_steps"],
        to_steps=b["optimizer_steps"],
        restored_from=previous["sha256"],
    )


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def checkpoint(directory, signature, *, verify=True):
    meta = Path(directory) / "checkpoint.json"
    if not meta.exists():
        return None
    if meta.is_symlink() or not meta.is_file():
        raise ValueError("checkpoint metadata must be a regular file")
    record = json.loads(meta.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not isinstance(record.get("signature"), str):
        raise ValueError("invalid checkpoint metadata")
    if (
        type(record.get("epoch")) is not int
        or record["epoch"] < 1
        or type(record.get("can_continue")) is not bool
    ):
        raise ValueError("invalid checkpoint epoch or continuation flag")
    if type(record.get("rung")) is not int or record["rung"] < 0:
        raise ValueError("invalid checkpoint rung")
    if any(
        type(record.get(k)) not in (int, float)
        or not math.isfinite(record[k])
        or record[k] <= 0
        for k in ("next_seconds", "delivery_seconds")
    ):
        raise ValueError("invalid checkpoint training/delivery duration")
    if record["signature"] != signature:
        raise ValueError("checkpoint does not match task, code, seed or weights")
    name = record.get("sha256")
    if (
        not isinstance(name, str)
        or len(name) != 64
        or any(c not in "0123456789abcdef" for c in name)
    ):
        raise ValueError("invalid checkpoint hash")
    path = meta.parent / name
    if path.is_symlink() or not path.is_file() or (verify and digest(path) != name):
        raise ValueError("checkpoint payload missing or changed")
    return record


def remaining_train_seconds():
    """Leave the controller's allowance for validation and full test inference."""
    return max(0.0, float(os.environ["SUPERAGENT_TRAIN_UNTIL"]) - time.monotonic())


def save_checkpoint(
    payload,
    *,
    epoch,
    next_seconds,
    delivery_seconds,
    can_continue=True,
    optimizer_steps=None,
    model_state=None,
    optimizer_state=None,
):
    """Commit an opaque state file after an epoch. Timings are seconds, not scores."""
    if os.getenv("SUPERAGENT_SMOKE") == "1":
        return
    if (
        type(epoch) is not int
        or epoch < 1
        or not all(math.isfinite(x) and x > 0 for x in (next_seconds, delivery_seconds))
    ):
        raise ValueError("positive epoch and finite training/delivery timings required")
    progress = None
    if optimizer_steps is not None:
        if (
            type(optimizer_steps) is not int
            or optimizer_steps < 1
            or model_state is None
            or optimizer_state is None
        ):
            raise ValueError(
                "progress requires positive integer optimizer_steps and actual model/optimizer state"
            )
        progress = dict(
            optimizer_steps=optimizer_steps,
            model_sha256=state_fingerprint(model_state),
            optimizer_sha256=state_fingerprint(
                {"state": optimizer_state, "steps": optimizer_steps}
            ),
            restored_from=_RESTORED.get(_context()),
        )
    directory = Path(os.environ["SUPERAGENT_CHECKPOINT_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    name = digest(payload)
    dest = directory / name
    if not dest.exists():
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as tmp:
            with Path(payload).open("rb") as f:
                shutil.copyfileobj(f, tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp.name, dest)
    atomic_write_json(
        directory / "checkpoint.json",
        dict(
            signature=os.environ["SUPERAGENT_TRIAL_SIGNATURE"],
            sha256=name,
            epoch=epoch,
            next_seconds=next_seconds,
            delivery_seconds=delivery_seconds,
            can_continue=bool(can_continue),
            rung=int(os.environ["SUPERAGENT_RUNG"]),
            progress=progress,
        ),
    )


def resume_path():
    path = os.getenv("SUPERAGENT_RESUME_DIR")
    if not path:
        _RESTORED.pop(_context(), None)
        return None
    record = checkpoint(path, os.environ["SUPERAGENT_TRIAL_SIGNATURE"])
    if record is None:
        raise ValueError("resume requested without a committed checkpoint")
    _RESTORED[_context()] = record["sha256"]
    return Path(path) / record["sha256"]


def _optimizer_steps(optimizer):
    if hasattr(optimizer, "_superagent_steps"):
        return optimizer._superagent_steps
    steps = [int(s["step"]) for s in optimizer.state.values() if "step" in s]
    return max(steps) if steps else None


def _observe_steps(optimizer, steps):
    if not hasattr(optimizer, "_superagent_steps"):

        def updated(opt, args, kwargs):
            opt._superagent_steps += 1

        optimizer.register_step_post_hook(updated)
    optimizer._superagent_steps = steps


def save_torch(
    model,
    optimizer,
    *,
    epoch,
    next_seconds,
    delivery_seconds,
    scheduler=None,
    scaler=None,
    generator=None,
    sampler_state=None,
    can_continue=True,
):
    import torch
    import numpy as np

    if os.getenv("SUPERAGENT_SMOKE") == "1":
        return
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    steps = _optimizer_steps(optimizer)
    state = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        epoch=epoch,
        rng=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        python=random.getstate(),
        numpy=(kind, torch.tensor(keys.astype("int64")), pos, has_gauss, cached),
        scheduler=scheduler.state_dict() if scheduler else None,
        scaler=scaler.state_dict() if scaler else None,
        generator=generator.get_state() if generator is not None else None,
        sampler=sampler_state,
        optimizer_steps=steps,
    )
    with tempfile.TemporaryDirectory(dir=".") as folder:
        file = Path(folder) / "state.pt"
        torch.save(state, file)
        save_checkpoint(
            file,
            epoch=epoch,
            next_seconds=next_seconds,
            delivery_seconds=delivery_seconds,
            can_continue=can_continue,
            optimizer_steps=steps,
            model_state=state["model"],
            optimizer_state=state["optimizer"],
        )


def restore_torch(model, optimizer, *, scheduler=None, scaler=None, generator=None):
    import torch
    import numpy as np

    path = resume_path()
    if path is None:
        _observe_steps(optimizer, 0)
        return 0, None
    s = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(s["model"])
    optimizer.load_state_dict(s["optimizer"])
    _observe_steps(
        optimizer, s.get("optimizer_steps") or _optimizer_steps(optimizer) or 0
    )
    for obj, key in (
        (scheduler, "scheduler"),
        (scaler, "scaler"),
        (generator, "generator"),
    ):
        if (obj is None) != (s[key] is None):
            raise ValueError(f"checkpoint {key} configuration changed")
        if obj is not None:
            obj.set_state(s[key]) if key == "generator" else obj.load_state_dict(s[key])
    random.setstate(s["python"])
    kind, keys, pos, has_gauss, cached = s["numpy"]
    np.random.set_state((kind, keys.numpy().astype("uint32"), pos, has_gauss, cached))
    if s["cuda"]:
        torch.cuda.set_rng_state_all(s["cuda"])
    torch.set_rng_state(s["rng"])
    return s["epoch"], s["sampler"]
