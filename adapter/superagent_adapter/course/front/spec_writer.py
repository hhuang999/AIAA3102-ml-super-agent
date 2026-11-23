"""Write a course task spec from a scanned folder, by rules or with the Core's model.

The model only proposes a draft. Classes, fixed contract fields and every file reference are
filled in or checked here against the folder, so a wrong proposal is reported, not trusted.
"""
from __future__ import annotations

import json
import re

from .scan import SPLIT_WORDS, data_files

FIXED = {'interface_version': '1.1', 'metric': 'accuracy', 'validation_policy': 'core_default',
         'final_choice': 'final'}
TARGET_NAMES = ('label', 'labels', 'target', 'y', 'class', 'outcome')
LABEL_PREFIXES = ('y_', 'label_', 'labels_', 'target_')
FEATURE_PREFIXES = ('x_', 'c_', 'features_', 'feature_', 'data_')

SYSTEM = (
    'You read the file listing and documentation of a classification dataset and describe how to load it. '
    'Use only file and column names that appear in the profile. Never guess test labels. '
    'Answer in JSON.'
)


def _safe_id(text):
    value = re.sub(r'[^A-Za-z0-9_-]+', '_', str(text)).strip('_')
    return value or 'course_task'


def _by_split(files):
    out = {}
    for f in files:
        if f['split']:
            out.setdefault(f['split'], []).append(f)
    return out


def _npy_pair(group):
    labels = [f for f in group if f.get('shape') and len(f['shape']) == 1
              or f['name'].lower().startswith(LABEL_PREFIXES)]
    features = [f for f in group if f not in labels]
    return (features[0]['name'] if len(features) == 1 else None,
            labels[0]['name'] if len(labels) == 1 else None)


NAME_LIST = re.compile(r'`[A-Za-z_]\w*`(?:\s*,\s*(?:and\s+)?`[A-Za-z_]\w*`)+')


def _doc_feature_names(profile, count):
    """A comma-separated list of exactly ``count`` distinct back-quoted names in the documentation."""
    for text in profile['docs'].values():
        for run in NAME_LIST.finditer(text):
            names = re.findall(r'`([A-Za-z_]\w*)`', run.group(0))
            split_words = {w for words in SPLIT_WORDS.values() for w in words}
            if len(names) == count and len(set(names)) == count and not {n.lower() for n in names} <= split_words:
                return names
    return None


def _objective(profile):
    for name, text in profile['docs'].items():
        if name.lower().endswith(('.md', '.txt', '.rst')):
            para = ' '.join(text.strip().split('\n\n')[0].split())
            if para:
                return para[:600]
    return f'Classify rows of the {profile["name"]} dataset.'


def heuristic_draft(profile):
    """A draft from file names, column names and documentation alone."""
    csvs, npys = data_files(profile, 'csv'), data_files(profile, 'npy')
    draft = {'task_id': _safe_id(profile['name']), 'objective': _objective(profile), 'ignore_columns': []}
    if csvs and not npys:
        split = {k: v[0]['name'] for k, v in _by_split(csvs).items() if len(v) == 1}
        draft.update(input_format='csv', train_file=split.get('train'), val_file=split.get('val'),
                     test_file=split.get('test'))
        train = next((f for f in csvs if f['name'] == split.get('train')), None)
        columns = train['columns'] if train else []
        target = next((c for c in columns if c.lower() in TARGET_NAMES), None)
        if target is None and train:
            values = train.get('low_cardinality_values', {})
            target = next((c for c in reversed(columns) if c in values and len(values[c]) <= 20), None)
        ignored = [c for c in columns if c.lower() == 'id' and c != target]
        draft.update(target=target, ignore_columns=ignored,
                     feature_columns=[c for c in columns if c != target and c not in ignored])
        text = any(any(len(v) > 40 for v in r.values()) for r in (train or {}).get('sample_rows', []))
        draft['modality'] = 'text' if text else 'tabular'
    elif npys and not csvs:
        groups = _by_split(npys)
        pairs = {k: _npy_pair(v) for k, v in groups.items()}
        draft.update(input_format='npy', modality='tabular', target='label')
        for split in ('train', 'val', 'test'):
            feats, labels = pairs.get(split, (None, None))
            draft[split + '_file'], draft[split + '_labels'] = feats, labels
        train = next((f for f in npys if f['name'] == draft.get('train_file')), None)
        count = train['shape'][1] if train and train.get('shape') and len(train['shape']) == 2 else 0
        draft['feature_columns'] = (_doc_feature_names(profile, count) or [f'f{i}' for i in range(count)]) if count else []
    else:
        draft['input_format'] = None
    return draft


DRAFT_FIELDS = ('task_id', 'modality', 'input_format', 'train_file', 'val_file', 'test_file',
                'train_labels', 'val_labels', 'test_labels', 'target', 'feature_columns',
                'ignore_columns', 'objective')


def llm_schema():
    from superagent.llm.schema import Field
    return {
        'task_id': Field('string', 'short identifier: letters, digits, _ or -'),
        'modality': Field('enum', 'main kind of input', values=('tabular', 'text')),
        'input_format': Field('enum', 'file format holding the data', values=('csv', 'npy')),
        'train_file': Field('string', 'training file name (features file for npy)'),
        'val_file': Field('string', 'validation file name (features file for npy)'),
        'test_file': Field('string', 'test file name (features file for npy)'),
        'train_labels': Field('string', 'npy only: training label file name', nullable=True),
        'val_labels': Field('string', 'npy only: validation label file name', nullable=True),
        'test_labels': Field('string', 'npy only: test label file name', nullable=True),
        'target': Field('string', 'csv: the label column; npy: a short name for the label, e.g. label'),
        'feature_columns': Field('string_list', 'input columns in file order; for npy, one name per array column'),
        'ignore_columns': Field('string_list', 'csv only: columns that are neither features nor the target, such as a row id'),
        'objective': Field('string', 'one or two English sentences: what is predicted and what each class means'),
    }


def llm_prompt(profile, draft):
    return (
        'Dataset profile (test files show only their header or shape):\n'
        + json.dumps(profile, ensure_ascii=False, indent=1)[:20000]
        + '\n\nA rule-based draft, which may be wrong or incomplete:\n'
        + json.dumps(draft, ensure_ascii=False, indent=1)
        + '\n\nDescribe how to load this dataset for training, validation and final testing.'
    )


def llm_draft(profile, ask):
    """``ask(system, user, schema)`` returns a dict. The rule draft fills anything left out."""
    draft = heuristic_draft(profile)
    answer = ask(SYSTEM, llm_prompt(profile, draft), llm_schema())
    merged = dict(draft)
    for key in DRAFT_FIELDS:
        value = answer.get(key)
        if value not in (None, '', []) or key in ('ignore_columns', 'train_labels', 'val_labels', 'test_labels'):
            merged[key] = value if value is not None else ([] if key == 'ignore_columns' else None)
    merged['task_id'] = _safe_id(merged.get('task_id') or profile['name'])
    return merged


def spec_to_draft(spec):
    """Turn a full spec (for example one edited by hand) back into a draft for ``build_spec``."""
    draft = {k: spec.get(k) for k in ('task_id', 'modality', 'input_format', 'target', 'objective')}
    draft['feature_columns'] = list(spec.get('feature_columns') or [])
    draft['ignore_columns'] = list(spec.get('ignore_columns') or [])
    for split, entry in (spec.get('splits') or {}).items():
        draft[split + '_file'] = (entry or {}).get('features')
        draft[split + '_labels'] = (entry or {}).get('labels')
    return draft


def _int_classes(values):
    out = set()
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not number.is_integer():
            return None
        out.add(int(number))
    return sorted(out)


def build_spec(draft, profile):
    """Return ``(spec, problems)``. An empty problem list means the spec matches the folder."""
    problems = []
    names = {f['name']: f for f in profile['files']}
    fmt = draft.get('input_format')
    if fmt not in ('csv', 'npy'):
        return None, ['需要 train、val、test 三份 CSV，或三组特征与标签 NPY；其他格式还不支持。']
    spec = {**FIXED, 'task_id': _safe_id(draft.get('task_id')), 'modality': draft.get('modality') or 'tabular',
            'input_format': fmt, 'feature_columns': list(draft.get('feature_columns') or []),
            'target': draft.get('target'), 'objective': (draft.get('objective') or '').strip()}
    splits, label_values = {}, []
    for split in ('train', 'val', 'test'):
        file = draft.get(split + '_file')
        if not file or file not in names or names[file]['kind'] != fmt:
            problems.append(f'{split} 文件没有找到：{file!r}')
            continue
        entry = {'features': file}
        if fmt == 'npy':
            labels = draft.get(split + '_labels')
            if not labels or labels not in names or names[labels]['kind'] != 'npy':
                problems.append(f'{split} 标签文件没有找到：{labels!r}')
            else:
                entry['labels'] = labels
                if split != 'test':
                    label_values += list((names[labels].get('values') or {}).keys())
        splits[split] = entry
    spec['splits'] = splits
    if not spec['target']:
        problems.append('没有确定标签列。')
    if not spec['feature_columns']:
        problems.append('没有确定特征列。')
    if not spec['objective']:
        problems.append('任务说明为空。')
    if fmt == 'csv':
        ignored = list(draft.get('ignore_columns') or [])
        if ignored:
            spec['ignore_columns'] = ignored
        declared = sorted([*spec['feature_columns'], spec['target'] or '', *ignored])
        for split, entry in splits.items():
            info = names[entry['features']]
            if sorted(info['columns']) != declared:
                problems.append(f'{entry["features"]} 的列与声明不一致：文件 {info["columns"]}')
            elif split != 'test':
                label_values += list(info.get('low_cardinality_values', {}).get(spec['target'], {}).keys())
        if 'id' in [spec['target'], *spec['feature_columns']]:
            problems.append('列名 id 会与生成的行号冲突，请把它放进 ignore_columns。')
    else:
        spec['trusted_pickle'] = any(names[e['features']].get('needs_pickle') for e in splits.values()
                                     if e['features'] in names)
        train = names.get(splits.get('train', {}).get('features'))
        shape = (train or {}).get('shape')
        if shape and (len(shape) != 2 or shape[1] != len(spec['feature_columns'])):
            problems.append(f'特征名数量 {len(spec["feature_columns"])} 与数组列数 {shape} 不一致。')
        if train and train.get('needs_pickle') and 'shape' not in train:
            problems.append('NPY 是对象数组，需要确认信任该文件后才能读取形状。')
    classes = _int_classes(set(label_values)) if label_values else None
    if classes is None:
        problems.append('训练或验证标签不是整数类别，当前只支持整数类别的 accuracy 任务。')
    elif len(classes) < 2:
        problems.append(f'训练和验证标签只有 {classes}，至少需要两个类别。')
    spec['classes'] = classes or []
    return spec, problems
