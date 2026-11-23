import numpy as np
import pytest
from superagent.features import frozen_features


def test_cache_reuses_only_same_frozen_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPERAGENT_FEATURE_DIR", str(tmp_path))
    monkeypatch.setenv("SUPERAGENT_FEATURE_NAMESPACE", "dataset-a")
    calls = []

    def run(**kw):
        calls.append(kw)
        return np.eye(2)

    cfg = dict(
        split="validation",
        row_ids=["01", "02"],
        weights_sha256="a" * 64,
        transform="rgb-v1",
        extractor="codehash",
    )

    def fetch(**update):
        return frozen_features(**{**cfg, **update}, compute=run)

    assert np.array_equal(fetch(), fetch()) and len(calls) == 1
    fetch(split="test")
    fetch(row_ids=["02", "01"])
    fetch(weights_sha256="b" * 64)
    fetch(transform="rgb-v2")
    fetch(extractor="new-code")
    monkeypatch.setenv("SUPERAGENT_FEATURE_NAMESPACE", "dataset-b")
    fetch()
    assert len(calls) == 7
    for p in tmp_path.glob("*/features.npy"):
        p.write_bytes(b"truncated")
    with pytest.raises(ValueError, match="changed"):
        fetch()
