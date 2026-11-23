from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f'Nonfinite JSON value: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=invalid)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ValueError('Missing or duplicate CSV columns')
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError('Malformed CSV row')
    return fields, rows


def write_csv(path, fields, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def inside_file(root, relative):
    root = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('File must be a relative path without parent traversal')
    path = root / rel
    if any(p.is_symlink() for p in (path, *path.parents) if p != root and p.is_relative_to(root)):
        raise ValueError('Symlinks are not accepted')
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError(f'Missing or out-of-bound file: {relative}')
    return path


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_predictions(path, sample_path, target, classes):
    columns, rows = read_csv(path)
    sample_columns, sample = read_csv(sample_path)
    if columns != ['id', target] or columns != sample_columns:
        raise ValueError('Prediction columns/order differ from prediction template')
    ids = [r['id'] for r in rows]
    expected = [r['id'] for r in sample]
    if any(not i for i in ids) or len(ids) != len(set(ids)) or len(expected) != len(set(expected)):
        raise ValueError('Empty or duplicate IDs')
    if len(ids) != len(expected) or set(ids) != set(expected):
        raise ValueError('Prediction IDs do not exactly cover final inputs')
    result = {}
    for row in rows:
        try:
            value = float(row[target])
        except (ValueError, TypeError):
            raise ValueError('Predicted class is not numeric') from None
        if not math.isfinite(value) or not value.is_integer() or int(value) not in classes:
            raise ValueError('Predictions must be finite declared class labels, not probabilities')
        result[row['id']] = int(value)
    return result
