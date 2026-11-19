"""Build the script-visible views from a task-owned classification contract."""

from __future__ import annotations

import shutil
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .contract import frame, image_features, records, relative, scalar
from .errors import InvalidInputError
from .utils import atomic_write_json


@dataclass(frozen=True)
class Binding:
    format: str
    paths: tuple[str, ...]
    id: str | None
    label: str | None
    features: tuple[str, ...]
    file_features: tuple[str, ...]


@dataclass(frozen=True)
class InputManifest:
    schema_version: int
    sample: str
    splits: dict[str, Binding]

    def payload(self, *, inference=False):
        return {
            "schema_version": self.schema_version,
            "sample": self.sample,
            "splits": {
                name: asdict(binding)
                for name, binding in self.splits.items()
                if not inference or name != "train"
            },
        }


def _implicit(raw, prepared):
    val = raw["validation"]
    if val["format"] == "csv" and all(
        (prepared / name).is_file() for name in ("train.csv", "val.csv", "test.csv")
    ):
        features = [c for c in frame(prepared / "train.csv").columns if c not in (val["id"], val["label"])]
        if "file" in features:
            raise ValueError("input_bindings must identify file_features for CSV asset inputs")
        return {
            split: {
                "format": "csv", "path": filename, "id": val["id"], "features": features,
                **({"label": val["label"]} if split == "train" else {}),
            }
            for split, filename in (("train", "train.csv"), ("validation", "val.csv"), ("test", "test.csv"))
        }
    if val["format"] == "tfrecord" and set(raw.get("inference_inputs", ())) >= {"val", "test"}:
        shards = sorted(prepared.glob(str(relative(val["glob"]))))
        if shards:
            features = image_features(next(records(shards[0]), {}))
            if features:
                return {
                    "train": {"format": "tfrecord", "glob": "train/*.tfrec", "label": val["label"], "features": features},
                    "validation": {"format": "tfrecord", "glob": val["glob"], "id": val.get("id"), "features": features},
                    "test": {"format": "tfrecord", "glob": "test/*.tfrec", "id": raw.get("test_id", "id"), "features": features},
                }
    raise ValueError("input_bindings must declare train, validation and test with format, path or glob, ID and features")


def _binding(split, raw, prepared, validation_label):
    fmt = raw["format"]
    if fmt not in ("csv", "tfrecord") or ("path" in raw) == ("glob" in raw):
        raise ValueError(f"input_bindings.{split} requires csv/tfrecord and exactly one of path or glob")
    if "path" in raw:
        paths = (relative(raw["path"]).as_posix(),)
    else:
        paths = tuple(p.relative_to(prepared).as_posix() for p in sorted(prepared.glob(str(relative(raw["glob"])))))
    if not paths or any(not (prepared / path).is_file() for path in paths):
        raise ValueError(f"input_bindings.{split} matched no valid input file")
    ident, label, features = raw.get("id"), raw.get("label"), raw["features"]
    attached_files = raw.get("file_features", [])
    if not isinstance(features, list) or not features or any(not isinstance(x, str) or not x for x in features):
        raise ValueError(f"input_bindings.{split}.features requires distinct field names")
    if len(set(features)) != len(features) or set(features) & {ident, label, validation_label}:
        raise ValueError(f"input_bindings.{split}.features includes a duplicate, ID or label")
    if split == "train":
        if not isinstance(label, str) or not label or label == ident:
            raise ValueError("input_bindings.train.label is required and must differ from its ID")
    elif label is not None:
        raise ValueError(f"input_bindings.{split} is a feature view and must omit label")
    if ident is None and (fmt == "csv" or split == "test"):
        raise ValueError(f"input_bindings.{split}.id is required")
    if not isinstance(attached_files, list) or not set(attached_files) <= set(features) or (attached_files and fmt != "csv"):
        raise ValueError(f"input_bindings.{split}.file_features must be a subset of CSV features")
    return Binding(fmt, paths, ident, label, tuple(features), tuple(attached_files))


def _project(prepared, view, binding, classes, blocked):
    ids = []
    field_types = {}
    fields = [field for field in (binding.id, *binding.features, binding.label) if field is not None]
    for name in binding.paths:
        dest = view / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if binding.format == "csv":
            table = frame(prepared / name)
            missing = set(fields) - set(table.columns)
            if missing:
                raise ValueError(f"{name}: missing fields {sorted(missing)}")
            ids.extend(table[binding.id].tolist())
            if binding.label and not table[binding.label].isin(classes).all():
                raise ValueError(f"{name}: unknown training class in {binding.label}")
            for col in binding.file_features:
                for value in table[col].unique():
                    asset = relative(value)
                    source = prepared / asset
                    if source.resolve() in blocked or not source.is_file():
                        raise ValueError(f"{name}: invalid asset for {col}: {value}")
                    target = view / asset
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        shutil.copyfile(source, target)
            table.loc[:, [c for c in table.columns if c in fields]].to_csv(dest, index=False)
        else:
            from tfrecord import TFRecordWriter, example_pb2

            with dest.open("wb") as output:
                for index, original in enumerate(records(prepared / name)):
                    example = example_pb2.Example()
                    for key in fields:
                        if key not in original or not (kind := original[key].WhichOneof("kind")):
                            raise ValueError(f"{name}:{index}: missing feature {key}")
                        if field_types.setdefault(key, kind) != kind:
                            raise ValueError(f"{name}:{index}: inconsistent feature type for {key}")
                        example.features.feature[key].CopyFrom(original[key])
                    ids.append(scalar(original, binding.id) if binding.id else f"{name}:{index}")
                    if binding.label and scalar(original, binding.label) not in classes:
                        raise ValueError(f"{name}:{index}: unknown training class")
                    body = example.SerializeToString()
                    header = struct.pack("<Q", len(body))
                    output.write(header + TFRecordWriter.masked_crc(header) + body + TFRecordWriter.masked_crc(body))
    if not ids or "" in ids or len(ids) != len(set(ids)):
        raise ValueError(f"{binding.paths}: empty or duplicated IDs")
    return ids, {k: field_types[k] for k in binding.features if k in field_types}


def prepare_inputs(task, prepared, spec, root):
    """Retain original truth exclusively in the trusted specification."""
    try:
        raw = yaml.safe_load((task / "task.yaml").read_text(encoding="utf-8"))
        sample = relative(raw["sample"]).as_posix()
        bindings = raw.get("input_bindings")
        if bindings is None:
            bindings = _implicit(raw, prepared)
        splits = {
            name: _binding(name, bindings[name], prepared, raw["validation"]["label"])
            for name in ("train", "validation", "test")
        }
        val = raw["validation"]
        truth_sources = [prepared / relative(val["path"])] if val["format"] == "csv" else list(prepared.glob(str(relative(val["glob"]))))
        reserved = [sample, "input_manifest.json"] + [path for bind in splits.values() for path in bind.paths]
        if len(reserved) != len(set(reserved)):
            raise ValueError("split and sample paths must be distinct")
        blocked = {path.resolve() for path in truth_sources}
        if any((prepared / path).resolve() in blocked for name in ("train", "test") for path in splits[name].paths):
            raise ValueError("train or test input aliases the validation truth source")
        real_paths = [(prepared / path).resolve() for path in reserved if path != "input_manifest.json"]
        if len(real_paths) != len(set(real_paths)):
            raise ValueError("split or sample paths alias the same source")
        blocked.update((prepared / path).resolve() for path in reserved)
        inference, training = root / "inference", root / "training"
        inference.mkdir(parents=True)
        training.mkdir()
        train_ids, train_types = _project(prepared, training, splits["train"], spec.classes, blocked)
        val_ids, val_types = _project(prepared, inference, splits["validation"], spec.classes, blocked)
        test_ids, test_types = _project(prepared, inference, splits["test"], spec.classes, blocked)
        if set(val_ids) != set(spec.truth.index) or set(test_ids) != set(spec.test_ids):
            raise ValueError("validation or test input IDs differ from the truth or submission IDs")
        if raw.get("shared_id_domain"):
            train_set, val_set, test_set = map(set, (train_ids, val_ids, test_ids))
            for first, second, left, right in (
                ("training", "validation", train_set, val_set),
                ("training", "test", train_set, test_set),
                ("validation", "test", val_set, test_set),
            ):
                if not left.isdisjoint(right):
                    raise ValueError(f"{first} and {second} IDs overlap in the declared shared ID domain")
        for types in (val_types, test_types):
            if any(train_types.get(k, v) != v for k, v in types.items()):
                raise ValueError("TFRecord feature types differ between splits")
        # Publish a sample schema with neutral placeholders, independently of supplied values.
        sample_df = frame(prepared / sample)
        for c in spec.columns[1:]:
            sample_df[c] = 1 / len(spec.classes) if spec.probability else spec.classes[0]
        (inference / sample).parent.mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(inference / sample, index=False)
        shutil.copytree(inference, training, dirs_exist_ok=True)
        manifest = InputManifest(1, sample, splits)
        atomic_write_json(training / "input_manifest.json", manifest.payload())
        atomic_write_json(inference / "input_manifest.json", manifest.payload(inference=True))
        return manifest
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise InvalidInputError(f"task.yaml input binding: {exc}") from exc


def build_task_facts(spec, manifest, capabilities, run_id):
    return {
        "schema_version": 1, "run_id": run_id, "task_contract_hash": spec.signature,
        "split_id": spec.signature, "metric": spec.metric, "maximize": spec.maximize,
        "classes": spec.classes, "submission_columns": spec.columns, "probability": spec.probability,
        "input_manifest": manifest.payload(), "capabilities": capabilities,
    }
