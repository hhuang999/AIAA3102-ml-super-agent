"""Format adaptation only: do not choose algorithms or use final labels to select them."""
from __future__ import annotations

import math
import re
from pathlib import Path

from ..common import digest, inside_file, read_csv, read_json, write_csv, write_json


VALIDATION_POLICY = {
    'name': 'core_default',
    'status': 'provisional_pending_team_discussion',
    'adapter_resplits': False,
    'labeled_validation_visible_to_core': True,
    'note': '课程 train/val 均作为开发数据交给 Core，Adapter 保留原划分且不另切验证集；'
            'Core 默认自行安排内部验证，也可能将 val 并入开发训练数据。此为暂定默认，交付后须讨论。',
}
BOUNDARY = {
    'mode': 'separate_paths_on_assignment1_server',
    'status': 'provisional_pending_deployment_discussion',
    'os_enforced': False,
    'note': 'Core 只接收 task_dir；测试标签保存在其外的 Adapter 私有目录。当前同账号路径分离不是操作系统访问隔离。',
}


def _class(value, allowed):
    if isinstance(value, bool):
        raise ValueError('Boolean labels are ambiguous; use declared integer labels')
    try:
        numeric = float(value)
    except (ValueError, TypeError):
        raise ValueError('Labels must be declared integer classes') from None
    if not math.isfinite(numeric) or not numeric.is_integer() or int(numeric) not in allowed:
        raise ValueError('Label outside declared classes')
    return str(int(numeric))


def _csv_split(root, config, features, target, allowed, ignored=()):
    path = inside_file(root, config['features'])
    columns, rows = read_csv(path)
    if sorted(columns) != sorted([*features, target, *ignored]):
        raise ValueError('CSV input columns must be exactly the declared features, target and ignored columns')
    return [[r[k] for k in features] for r in rows], [_class(r[target], allowed) for r in rows], [path]


def _npy_split(root, config, features, target, allowed, trusted_pickle):
    import numpy as np
    xp = inside_file(root, config['features'])
    yp = inside_file(root, config['labels'])
    # NPY object arrays use pickle; explicit trusted-source acknowledgement is mandatory.
    x = np.load(xp, allow_pickle=trusted_pickle)
    y = np.load(yp, allow_pickle=False)
    if x.ndim != 2 or x.shape[1] != len(features) or y.ndim != 1 or len(x) != len(y):
        raise ValueError('Feature/label shapes do not match the declared contract')
    def cell(v):
        if v is None:
            return ''
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            return ''
        if not isinstance(v, (str, int, float, bool, np.generic)):
            raise ValueError('Only scalar features are supported')
        return str(v)
    return [[cell(v) for v in row] for row in x], [_class(v, allowed) for v in y], [xp, yp]


def prepare_task(spec_path, source_dir, task_dir, private_dir):
    spec = read_json(spec_path)
    version = spec.get('interface_version', '1.1')
    if version not in ('0.1', '1.1') or spec.get('metric') != 'accuracy':
        raise ValueError('Supported Core interfaces: 0.1 (legacy), 1.1; course metric: accuracy')
    sample_name = 'sample_predictions.csv' if version == '1.1' else 'sample_submission.csv'
    if spec.get('validation_policy', 'core_default') != 'core_default':
        raise ValueError('Alternative validation policy requires a separately agreed contract')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', spec['task_id']):
        raise ValueError('Unsafe task_id')
    features, target = spec['feature_columns'], spec['target']
    if not features or len(features) != len(set(features)) or target in features or 'id' in [target, *features]:
        raise ValueError('Feature/target/ID names must be distinct')
    ignored = spec.get('ignore_columns', [])
    if not isinstance(ignored, list) or not all(isinstance(c, str) for c in ignored) \
            or len(ignored) != len(set(ignored)) or set(ignored) & {target, *features}:
        raise ValueError('ignore_columns must be distinct column names outside the features and target')
    allowed = spec['classes']
    if len(allowed) < 2 or len(allowed) != len(set(allowed)) or any(type(v) is not int for v in allowed):
        raise ValueError('Declare at least two unique integer classes')
    if type(spec.get('trusted_pickle', False)) is not bool:
        raise ValueError('trusted_pickle must be an explicit boolean, not a string or number')
    choice = spec.get('final_choice', 'final' if version == '1.1' else 'best')
    if choice not in (('final',) if version == '1.1' else ('best', 'ensemble')):
        raise ValueError('Core 1.1 requires final; legacy 0.1 allows best or ensemble')
    source = Path(source_dir).resolve(strict=True)
    task, private = Path(task_dir).resolve(), Path(private_dir).resolve()
    if task.exists() or private.exists():
        raise ValueError('Task and private directories must be new')
    if task.is_relative_to(private) or private.is_relative_to(task):
        raise ValueError('Private labels cannot be inside the Core task directory')
    if task.is_relative_to(source) or private.is_relative_to(source) or source.is_relative_to(task):
        raise ValueError('Source data and generated outputs must be separate')
    split_data, input_hashes = {}, {}
    for split in ('train', 'val', 'test'):
        config = spec['splits'][split]
        if spec['input_format'] == 'csv':
            x, y, paths = _csv_split(source, config, features, target, allowed, ignored)
        elif spec['input_format'] == 'npy':
            x, y, paths = _npy_split(source, config, features, target, allowed, spec.get('trusted_pickle', False))
        else:
            raise ValueError('Supported input formats: csv, npy')
        if not x:
            raise ValueError('Empty splits are not supported')
        split_data[split] = x, y
        input_hashes.update({p.relative_to(source).as_posix(): digest(p) for p in paths})
    task.mkdir(parents=True)
    private.mkdir(parents=True, mode=0o700)
    data = task / 'data'
    data.mkdir()
    for split, (x, y) in split_data.items():
        records = []
        for index, row in enumerate(x):
            rec = dict(zip(features, row))
            rec = {'id': f'{split}_{index:08d}', **rec}
            if split != 'test':
                rec[target] = y[index]
            records.append(rec)
        write_csv(data / f'{split}.csv', ['id', *features] + ([target] if split != 'test' else []), records)
    labels = [{'id': f'test_{i:08d}', target: v} for i, v in enumerate(split_data['test'][1])]
    write_csv(private / 'test_labels.csv', ['id', target], labels)
    (private / 'test_labels.csv').chmod(0o600)
    sample = [{'id': row['id'], target: str(allowed[0])} for row in labels]
    write_csv(data / sample_name, ['id', target], sample)
    desc = (
        f"# Classification task\n\n{spec['objective']}\n\n"
        f"Classes: {allowed}. Target: `{target}`.\n"
        f"Features: {', '.join(features)}. IDs identify rows only; do not use them as predictive features.\n\n"
        f"The task data is available as `{'./data' if version == '1.1' else './input'}` during candidate execution.\n"
        "- train.csv: labeled development data.\n"
        "- val.csv: labeled development validation data, in the original provided split.\n"
        "- test.csv: final evaluation features only; never train or select solutions on this split.\n"
        f"- {sample_name}: all final IDs and exact required prediction columns. Values are placeholders.\n\n"
        "Validation policy (provisional core_default): use the Core's default internal validation behavior on "
        "allowed development data. The adapter does not resplit or merge train/val. The Core may combine "
        "development rows under its default behavior; record the actual validation strategy in the solution.\n\n"
        f"Metric: accuracy = correctly classified rows / evaluated rows, higher is better. "
        f"Prediction columns exactly `id,{target}`; `{target}` must be a class label from {allowed}, not a probability. "
        f"Include every test ID exactly once. Save {'./out/predictions.csv' if version == '1.1' else 'submission/submission.csv'} with no dataframe index.\n"
    )
    (task / 'description.md').write_text(desc, encoding='utf-8', newline='\n')
    public_hashes = {p.relative_to(task).as_posix(): digest(p) for p in sorted(task.rglob('*'), key=lambda p: p.relative_to(task).as_posix()) if p.is_file()}
    manifest = {
        'schema_version': '0.1',
        'core_interface_version': version, 'prediction_template': 'data/' + sample_name,
        'task': {'task_id': spec['task_id'], 'modality': spec['modality'], 'target': target,
                 'classes': allowed, 'feature_columns': features, 'metric': {'name': 'accuracy', 'higher_is_better': True},
                 'rows': {s: len(x) for s, (x, y) in split_data.items()}},
        'policy': {'validation': VALIDATION_POLICY, 'test_label_boundary': BOUNDARY,
                   'final_choice': choice, 'delivery_mode': 'full_script', 'cost_control_owner': 'external_team_module'},
        'source_hashes': input_hashes, 'spec_sha256': digest(spec_path), 'public_hashes': public_hashes,
        'private_labels_sha256': digest(private / 'test_labels.csv'),
        'row_id_rule': 'split name plus original zero-based row index, width 8',
        'formal_evaluation_eligible': False,
    }
    write_json(private / 'manifest.json', manifest)
    return manifest


def verify_inputs(task_dir, private_dir):
    task, private = Path(task_dir).resolve(strict=True), Path(private_dir).resolve(strict=True)
    if private.is_relative_to(task) or task.is_relative_to(private):
        raise ValueError('Private and public directories overlap')
    m = read_json(private / 'manifest.json')
    paths = [p for p in task.rglob('*') if p.is_file() or p.is_symlink()]
    if any(p.is_symlink() for p in task.rglob('*')):
        raise ValueError('Public task contains a symlink')
    if set(m['public_hashes']) != {p.relative_to(task).as_posix() for p in paths}:
        raise ValueError('Public task files changed')
    if any(digest(inside_file(task, p)) != sha for p, sha in m['public_hashes'].items()):
        raise ValueError('Public task hash mismatch')
    if digest(inside_file(private, 'test_labels.csv')) != m['private_labels_sha256']:
        raise ValueError('Private label hash mismatch')
    return m
