"""Prepare the two released course tasks and check row-by-row conversion, without scoring."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from superagent_adapter.course.common import digest, read_csv, read_json, write_json
from superagent_adapter.course.prepare import prepare_task, verify_inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', required=True, help='Directory containing the two original task subdirectories')
    p.add_argument('--output-root', required=True, help='New directory, outside the raw source')
    args = p.parse_args()
    source, out = Path(args.source_root).resolve(strict=True), Path(args.output_root).resolve()
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('Output must be a new directory separate from the source')
    results = []
    for config in ('headlines.json', 'coupon.json'):
        spec_path = ROOT / 'configs/course' / config
        spec = read_json(spec_path)
        name = spec['task_id']
        task, private = out / 'tasks' / name, out / 'private' / name
        manifest = prepare_task(spec_path, source / name, task, private)
        verify_inputs(task, private)
        features, target = spec['feature_columns'], spec['target']
        # Independent check against original rows; do not calculate test statistics or model scores.
        for split in ('train', 'val', 'test'):
            columns, rows = read_csv(task / 'data' / (split + '.csv'))
            assert columns == ['id', *features] + ([target] if split != 'test' else [])
            original = spec['splits'][split]
            if spec['input_format'] == 'csv':
                _, originals = read_csv(source / name / original['features'])
                x = [[r[k] for k in features] for r in originals]
                y = [int(r[target]) for r in originals]
            else:
                import numpy as np
                x = np.load(source / name / original['features'], allow_pickle=True)
                y = np.load(source / name / original['labels'], allow_pickle=False)
            assert len(rows) == len(x) == len(y)
            if split == 'test':
                _, label_rows = read_csv(private / 'test_labels.csv')
            else:
                label_rows = rows
            for i, row in enumerate(rows):
                expected_id = f'{split}_{i:08d}'
                assert row['id'] == label_rows[i]['id'] == expected_id
                assert int(label_rows[i][target]) == int(y[i])
                for j, column in enumerate(features):
                    value = x[i][j]
                    missing = value is None or (spec['input_format'] == 'npy' and isinstance(value, (float, np.floating)) and np.isnan(value))
                    assert row[column] == ('' if missing else str(value))
        assert set(p.relative_to(task).as_posix() for p in task.rglob('*') if p.is_file()) == {
            'description.md', 'data/train.csv', 'data/val.csv', 'data/test.csv', 'data/sample_predictions.csv'}
        results.append({'task_id': name, 'rows': manifest['task']['rows'], 'task_dir': str(task),
                        'private_dir': str(private), 'manifest_sha256': digest(private / 'manifest.json'),
                        'source_hashes': manifest['source_hashes'], 'public_hashes': manifest['public_hashes'],
                        'row_order_features_and_labels_verified': True,
                        'test_target_absent_from_public_test': True,
                        'validation_policy': manifest['policy']['validation'],
                        'test_label_boundary': manifest['policy']['test_label_boundary']})
    receipt = {'status': 'passed', 'created_at': datetime.now(timezone.utc).isoformat(), 'tasks': results,
               'official_test_scoring_performed': False, 'model_training_performed': False}
    write_json(out / 'preparation-receipt.json', receipt)
    print(json.dumps({'status': 'passed', 'receipt': str(out / 'preparation-receipt.json'),
                      'tasks': [{'task_id': r['task_id'], 'rows': r['rows']} for r in results]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
