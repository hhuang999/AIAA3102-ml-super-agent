"""Task-owned classification schema, validation truth and deterministic grading."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .errors import InvalidInputError


def records(path: Path):
    """Read Example records, validating both checksums before parsing."""
    from tfrecord import TFRecordWriter, example_pb2

    with path.open("rb") as source:
        while header := source.read(12):
            if len(header) != 12 or header[8:] != TFRecordWriter.masked_crc(header[:8]):
                raise ValueError(f"{path}: invalid TFRecord length header")
            (length,) = struct.unpack("<Q", header[:8])
            if length > 64 * 1024**2:
                raise ValueError(f"{path}: record exceeds 64 MiB")
            body = source.read(length)
            if len(body) != length or source.read(4) != TFRecordWriter.masked_crc(body):
                raise ValueError(f"{path}: truncated record or bad payload CRC")
            yield example_pb2.Example.FromString(body).features.feature


def image_features(features):
    return sorted(
        k
        for k, v in features.items()
        if v.bytes_list.value
        and v.bytes_list.value[0].startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n"))
    )


def scalar(features, name):
    if name not in features:
        raise ValueError(f"missing TFRecord feature {name}")
    field = features[name]
    items = getattr(field, field.WhichOneof("kind")).value
    if len(items) != 1:
        raise ValueError(f"expected one value for {name}")
    return items[0].decode() if isinstance(items[0], bytes) else str(items[0])


def relative(value):
    path = Path(value)
    if path.is_absolute() or path.drive or ".." in path.parts:
        raise ValueError("contract paths must be relative to prepared data")
    return path


def frame(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


@dataclass
class TaskSpec:
    metric: str
    classes: list[str]
    columns: list[str]
    truth: pd.Series
    test_ids: list[str]
    prompt: str
    signature: str
    probability: bool
    inference_inputs: tuple[str, ...] = ()

    @property
    def maximize(self):
        return self.metric == "accuracy"

    def predictions(self, path, ids, *, ordered):
        f = frame(path)
        if list(f) != self.columns:
            raise ValueError(f"{path.name}: expected columns {self.columns}")
        got = f.iloc[:, 0]
        if (
            got.duplicated().any()
            or set(got) != set(ids)
            or (ordered and got.tolist() != list(ids))
        ):
            raise ValueError(f"{path.name}: incorrect IDs or submission order")
        values = f.set_index(self.columns[0]).loc[ids]
        if self.probability:
            p = values.to_numpy(float)
            if (
                not np.isfinite(p).all()
                or (p < 0).any()
                or (p > 1).any()
                or not np.allclose(p.sum(1), 1, atol=1e-5)
            ):
                raise ValueError(f"{path.name}: invalid class probabilities")
            return p
        labels = values.iloc[:, 0]
        if not labels.isin(self.classes).all():
            raise ValueError(f"{path.name}: predictions outside class domain")
        return labels.to_numpy()

    def grade(self, output):
        self.predictions(output / "predictions.csv", self.test_ids, ordered=True)
        val = self.predictions(
            output / "val_predictions.csv", self.truth.index, ordered=False
        )
        truth = self.truth.to_numpy()
        if self.metric == "accuracy":
            labels = (
                np.asarray(self.classes)[val.argmax(1)] if self.probability else val
            )
            return float(np.mean(labels == truth))
        indices = pd.Index(self.classes).get_indexer(truth)
        return float(
            -np.log(np.clip(val[np.arange(len(indices)), indices], 1e-15, 1)).mean()
        )


def load_spec(task: Path, prepared: Path):
    config = task / "task.yaml"
    if not config.exists():
        return None
    try:
        spec = yaml.safe_load(config.read_text(encoding="utf-8"))
        if spec["version"] != 1 or spec["metric"] not in ("accuracy", "log_loss"):
            raise ValueError("expected version 1 and accuracy or log_loss")
        if not isinstance(spec["classes"], list) or any(
            not isinstance(v, (int, str)) for v in spec["classes"]
        ):
            raise ValueError("classes must be a list of labels")
        classes = list(map(str, spec["classes"]))
        if len(set(classes)) != len(classes) or len(classes) < 2:
            raise ValueError("classes must contain distinct labels")
        sample = frame(prepared / relative(spec["sample"]))
        cols = list(sample)
        prob = spec.get("encoding", "label") == "probability"
        if (
            spec.get("encoding", "label") not in ("label", "probability")
            or (prob and cols[1:] != classes)
            or (not prob and (len(cols) != 2 or spec["metric"] == "log_loss"))
        ):
            raise ValueError("sample columns incompatible with prediction encoding")
        val = spec["validation"]
        if val["format"] == "csv":
            labels = frame(prepared / relative(val["path"])).set_index(val["id"])[val["label"]]
        elif val["format"] == "tfrecord":
            pairs = []
            files = sorted(prepared.glob(str(relative(val["glob"]))))
            if not files:
                raise ValueError("validation glob matched no shards")
            for path in files:
                for i, features in enumerate(records(path)):
                    key = (
                        scalar(features, val["id"])
                        if val.get("id")
                        else f"{path.relative_to(prepared).as_posix()}:{i}"
                    )
                    pairs.append((key, scalar(features, val["label"])))
            labels = pd.Series(
                [v for _, v in pairs], index=[k for k, _ in pairs], dtype=str
            )
        else:
            raise ValueError("unsupported validation format")
        ids = sample.iloc[:, 0].tolist()
        if (
            not ids
            or len(ids) != len(set(ids))
            or "" in ids
            or labels.empty
            or not labels.index.is_unique
            or "" in labels.index
            or not labels.isin(classes).all()
        ):
            raise ValueError("empty/duplicate IDs or invalid task-owned labels")
        inputs = spec.get("inference_inputs", [])
        if not isinstance(inputs,list) or any(not isinstance(x,str) or str(relative(x))=="." for x in inputs):
            raise ValueError("inference_inputs must name relative input files/directories")
        for name in inputs:
            if not (prepared/name).exists():
                raise ValueError(f"inference input does not exist: {name}")
        paths=[relative(x) for x in inputs]
        if any(a==b or a in b.parents or b in a.parents for i,a in enumerate(paths) for b in paths[i+1:]):
            raise ValueError("inference_inputs overlap")
        prompt = (
            "Authoritative classification contract:\n"
            + json.dumps(spec, ensure_ascii=False)
            + "\nEmit out/val_predictions.csv and out/predictions.csv with sample columns. "
            "Test IDs must follow sample order. For records without ID use relative/shard:zero_based_index."
            "\nThe agent scores validation predictions. In --predict-only mode read only the model package and "
            "the paths listed in inference_inputs. Produce both prediction files without using validation labels "
            "or reporting a validation score."
        )
        sig = hashlib.sha256(
            (prompt + labels.to_json() + json.dumps(ids)).encode()
        ).hexdigest()
        return TaskSpec(spec["metric"], classes, cols, labels, ids, prompt, sig, prob, tuple(inputs))
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise InvalidInputError(f"{config}: {exc}") from exc
