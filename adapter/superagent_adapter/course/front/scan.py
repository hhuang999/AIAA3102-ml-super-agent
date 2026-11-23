"""Read a raw course folder into a small profile for writing its task spec.

Only training and validation files contribute sample rows and label values. The test split
contributes its header, row count and array shape, so held-out labels never reach the profile.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

SPLIT_WORDS = {
    'train': ('train',),
    'val': ('val', 'valid', 'validation', 'dev'),
    'test': ('test',),
}
DOC_SUFFIXES = ('.md', '.txt', '.rst')
MAX_DOC_CHARS = 6000
MAX_JSON_CHARS = 2000
SAMPLE_ROWS = 3
MAX_LABEL_VALUES = 50


def split_of(name):
    """Guess the split from a file name; ``None`` when it names none or several."""
    stem = Path(name).stem.lower()
    tokens = {t for t in stem.replace('-', '_').replace('.', '_').split('_') if t}
    found = [split for split, words in SPLIT_WORDS.items()
             if any(w in tokens or stem.startswith(w) or stem.endswith(w) for w in words)]
    return found[0] if len(found) == 1 else None


def _csv_profile(path, split):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        try:
            columns = next(reader)
        except StopIteration:
            return {'columns': [], 'rows': 0}
        rows, samples = 0, []
        values = {c: {} for c in columns} if split in ('train', 'val') else None
        for row in reader:
            rows += 1
            if split == 'train' and len(samples) < SAMPLE_ROWS:
                samples.append(dict(zip(columns, row)))
            if values is not None:
                for column, value in zip(columns, row):
                    seen = values[column]
                    if seen is not None:
                        seen[value] = seen.get(value, 0) + 1
                        if len(seen) > MAX_LABEL_VALUES:
                            values[column] = None
    info = {'columns': columns, 'rows': rows}
    if samples:
        info['sample_rows'] = [{k: v[:120] for k, v in r.items()} for r in samples]
    if values is not None:
        info['low_cardinality_values'] = {c: v for c, v in values.items() if v is not None}
    return info


def _npy_profile(path, split, trusted_pickle):
    import numpy as np
    try:
        array = np.load(path, allow_pickle=trusted_pickle, mmap_mode=None)
    except ValueError:
        return {'needs_pickle': True}
    info = {'shape': list(array.shape), 'dtype': str(array.dtype), 'needs_pickle': array.dtype == object}
    if split == 'train' and array.ndim == 2:
        info['sample_rows'] = [[str(v)[:60] for v in row] for row in array[:SAMPLE_ROWS]]
    if split in ('train', 'val') and array.ndim == 1:
        unique, counts = np.unique(array, return_counts=True)
        if len(unique) <= MAX_LABEL_VALUES:
            info['values'] = {str(u): int(c) for u, c in zip(unique, counts)}
    return info


def scan_folder(folder, *, trusted_pickle=False):
    """Profile every regular file directly inside ``folder``."""
    root = Path(folder).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f'Not a folder: {root}')
    files, docs = [], {}
    for path in sorted(root.iterdir()):
        if path.is_symlink() or not path.is_file() or path.name.startswith('.'):
            continue
        name, suffix = path.name, path.suffix.lower()
        entry = {'name': name, 'bytes': path.stat().st_size, 'split': split_of(name)}
        if suffix == '.csv':
            entry['kind'] = 'csv'
            entry.update(_csv_profile(path, entry['split']))
        elif suffix == '.npy':
            entry['kind'] = 'npy'
            entry.update(_npy_profile(path, entry['split'], trusted_pickle))
        elif suffix in DOC_SUFFIXES:
            entry['kind'] = 'doc'
            docs[name] = path.read_text(encoding='utf-8', errors='replace')[:MAX_DOC_CHARS]
        elif suffix == '.json':
            entry['kind'] = 'json'
            text = path.read_text(encoding='utf-8', errors='replace')
            try:
                json.loads(text)
                docs[name] = text[:MAX_JSON_CHARS]
            except ValueError:
                pass
        else:
            entry['kind'] = 'other'
        files.append(entry)
    return {'folder': str(root), 'name': root.name, 'files': files, 'docs': docs}


def data_files(profile, kind=None):
    return [f for f in profile['files'] if f['kind'] in ('csv', 'npy') and (kind is None or f['kind'] == kind)]
