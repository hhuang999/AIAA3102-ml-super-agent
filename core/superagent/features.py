"""Optional cache for deterministic, frozen features within one run."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .training import digest


def feature_key(*, split, row_ids, weights_sha256, transform, extractor):
    """Use ordered, unique data IDs and an explicit preprocessing/extractor revision."""
    ids = [str(x) for x in row_ids]
    if not ids or len(ids) != len(set(ids)) or not all((split, transform, extractor)):
        raise ValueError(
            "cache requires split, unique ordered IDs, transform and extractor revision"
        )
    if len(weights_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in weights_sha256
    ):
        raise ValueError("cache requires verified weight SHA256")
    identity = [
        os.environ["SUPERAGENT_FEATURE_NAMESPACE"],
        split,
        ids,
        weights_sha256,
        transform,
        extractor,
    ]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def frozen_features(*, split, row_ids, weights_sha256, transform, extractor, compute):
    """Return a read-only array. Do not cache random augmentation or trainable encoders."""
    ids = list(row_ids)
    if os.getenv("SUPERAGENT_SMOKE") == "1":
        return compute()
    key = feature_key(
        split=split,
        row_ids=ids,
        weights_sha256=weights_sha256,
        transform=transform,
        extractor=extractor,
    )
    root = Path(os.environ["SUPERAGENT_FEATURE_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    entry = root / key
    if not entry.exists():
        temp = Path(tempfile.mkdtemp(dir=root))
        try:
            values = np.asarray(compute())
            if values.ndim < 1 or len(values) != len(ids) or values.dtype.hasobject:
                raise ValueError("features must have one numeric row per stable ID")
            np.save(temp / "features.npy", values, allow_pickle=False)
            (temp / "sha256").write_text(digest(temp / "features.npy"))
            try:
                temp.rename(entry)
            except OSError:
                if not entry.is_dir():
                    raise
        finally:
            shutil.rmtree(temp, ignore_errors=True)
    if digest(entry / "features.npy") != (entry / "sha256").read_text():
        raise ValueError("cached feature payload changed")
    return np.load(entry / "features.npy", mmap_mode="r", allow_pickle=False)
