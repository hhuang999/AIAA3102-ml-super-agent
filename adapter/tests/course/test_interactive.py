"""Interactive page: folder scan, spec writing, and one offline run from folder to file list."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from superagent_adapter.course import interactive
from superagent_adapter.course.common import read_json, write_csv
from superagent_adapter.course.front.prepare import prepare_task
from superagent_adapter.course.front.scan import scan_folder, split_of
from superagent_adapter.course.front.spec_writer import build_spec, heuristic_draft, llm_draft
from superagent_adapter.course.runner import launch_core
from .test_core11 import FIXTURE11, offline_environment


def csv_folder(root, *, with_id=False):
    root.mkdir(parents=True)
    for split in ('train', 'val', 'test'):
        rows = [{'label': i % 2, 'score': (i // 2) % 2, 'text': f'{split} row {i}'} for i in range(12)]
        fields = ['label', 'score', 'text']
        if with_id:
            fields = ['id', *fields]
            rows = [{'id': f'r{i}', **r} for i, r in enumerate(rows)]
        write_csv(root / f'{split}.csv', fields, rows)
    (root / 'README.md').write_text('Predict the label of each row (1 or 0).\n', encoding='utf-8')
    return root


def npy_folder(root):
    root.mkdir(parents=True)
    for split in ('train', 'val', 'test'):
        x = np.array([['a', 1], ['b', 2], ['c', 3], ['d', 4]], dtype=object)
        np.save(root / f'C_{split}.npy', x)
        np.save(root / f'y_{split}.npy', np.array([0, 1, 0, 1]))
    (root / 'Readme.md').write_text('Use `train`, `val` and `test`. Columns are `colour`, `size`.\n', encoding='utf-8')
    return root


def test_split_names():
    assert split_of('C_train.npy') == 'train'
    assert split_of('y_val.npy') == 'val'
    assert split_of('validation.csv') == 'val'
    assert split_of('test.csv') == 'test'
    assert split_of('train_val.csv') is None
    assert split_of('README.md') is None


def test_scan_reads_no_test_labels(tmp_path):
    csv_profile = scan_folder(csv_folder(tmp_path / 'csv'))
    test_csv = next(f for f in csv_profile['files'] if f['name'] == 'test.csv')
    assert 'sample_rows' not in test_csv and 'low_cardinality_values' not in test_csv
    assert test_csv['rows'] == 12
    npy_profile = scan_folder(npy_folder(tmp_path / 'npy'), trusted_pickle=True)
    test_y = next(f for f in npy_profile['files'] if f['name'] == 'y_test.npy')
    assert 'values' not in test_y
    assert next(f for f in npy_profile['files'] if f['name'] == 'y_train.npy')['values'] == {'0': 2, '1': 2}


def test_rule_spec_for_csv_with_id_column_prepares(tmp_path):
    folder = csv_folder(tmp_path / 'raw', with_id=True)
    profile = scan_folder(folder)
    spec, problems = build_spec(heuristic_draft(profile), profile)
    assert problems == []
    assert spec['target'] == 'label'
    assert spec['ignore_columns'] == ['id']
    assert spec['feature_columns'] == ['score', 'text']
    assert spec['classes'] == [0, 1]
    (tmp_path / 'spec.json').write_text(json.dumps(spec), encoding='utf-8')
    manifest = prepare_task(tmp_path / 'spec.json', folder, tmp_path / 'task', tmp_path / 'private')
    assert manifest['task']['rows'] == {'train': 12, 'val': 12, 'test': 12}
    header = (tmp_path / 'task/data/test.csv').read_text(encoding='utf-8').splitlines()[0]
    assert header == 'id,score,text'


def test_rule_spec_for_npy_uses_documented_names(tmp_path):
    profile = scan_folder(npy_folder(tmp_path / 'npy'), trusted_pickle=True)
    spec, problems = build_spec(heuristic_draft(profile), profile)
    assert problems == []
    assert spec['feature_columns'] == ['colour', 'size']
    assert spec['splits']['train'] == {'features': 'C_train.npy', 'labels': 'y_train.npy'}
    assert spec['trusted_pickle'] is True


def test_model_answer_is_checked_against_the_folder(tmp_path):
    profile = scan_folder(csv_folder(tmp_path / 'raw'))
    seen = {}

    def ask(system, user, schema):
        seen['user'] = user
        return {'task_id': 'my task!', 'modality': 'text', 'input_format': 'csv', 'train_file': 'train.csv',
                'val_file': 'val.csv', 'test_file': 'missing.csv', 'target': 'label',
                'feature_columns': ['score', 'text'], 'ignore_columns': [], 'objective': 'Predict label.'}

    draft = llm_draft(profile, ask)
    assert draft['task_id'] == 'my_task'
    assert 'test row' not in seen['user']
    spec, problems = build_spec(draft, profile)
    assert any('missing.csv' in p for p in problems)


@pytest.mark.parametrize('typed, expected', [
    ("'/tmp/my data/'", '/tmp/my data'),
    ('/tmp/my\\ data', '/tmp/my data'),
    ('file:///tmp/x', '/tmp/x'),
    ('  /tmp/x  ', '/tmp/x'),
])
def test_dragged_paths(typed, expected):
    assert interactive.clean_path(typed) == Path(expected)


def test_offline_run_from_folder_to_file_list(tmp_path):
    folder = csv_folder(tmp_path / 'raw')
    said = []

    def fixture_launcher(task, private, run, *, config=None, seed=42, extra_args=()):
        return launch_core(task, private, run, seed=seed, extra_args=extra_args,
                           command_prefix=[sys.executable, str(FIXTURE11), '--mode', 'single'],
                           execution_kind='fixture')

    with offline_environment():
        code = interactive.main(['--folder', str(folder), '--workspace', str(tmp_path / 'ws'), '--yes', '--no-llm'],
                                ask=lambda prompt: pytest.fail(f'unexpected prompt: {prompt}'),
                                say=said.append, launcher=fixture_launcher)
    text = '\n'.join(said)
    assert code == 0, text
    run = next((tmp_path / 'ws' / 'raw').iterdir()) / 'run'
    outcome = read_json(run / 'adapter/outcome.json')
    assert outcome['evaluation']['status'] == 'evaluated'
    assert '测试集 accuracy' in text
    assert outcome['files']['predictions_csv'] in text
    assert str(run / 'adapter/core.stdout.log') in text


def test_resolve_resources_defaults_and_clamping():
    parser = interactive.build_parser()
    defaults = interactive.core_defaults(None)

    extra, chosen = interactive.resolve_resources(parser.parse_args([]), defaults, detected_gpus=8, detected_cpus=112)
    assert chosen['num_gpus'] == 4 and chosen['parallel'] == 4 and chosen['num_cpus'] == 16
    assert extra == []

    extra2, chosen2 = interactive.resolve_resources(parser.parse_args([]), defaults, detected_gpus=2, detected_cpus=16)
    assert chosen2['num_gpus'] == 2 and chosen2['parallel'] == 2 and chosen2['num_cpus'] == 16
    assert extra2 == ['--num-gpus', '2', '--parallel', '2']

    extra3, chosen3 = interactive.resolve_resources(
        parser.parse_args(['--num-gpus', '2']), defaults, detected_gpus=8, detected_cpus=112,
    )
    assert chosen3['num_gpus'] == 2 and chosen3['parallel'] == 2 and chosen3['num_cpus'] == 16
    assert extra3 == ['--num-gpus', '2', '--parallel', '2']

    extra4, chosen4 = interactive.resolve_resources(
        parser.parse_args(['--num-gpus', '0']), defaults, detected_gpus=8, detected_cpus=112,
    )
    assert chosen4['num_gpus'] == 0 and chosen4['parallel'] == 4 and chosen4['num_cpus'] == 16


def test_interactive_default_flow_skips_resource_prompts(tmp_path):
    folder = csv_folder(tmp_path / 'raw')
    prompts, said = [], []

    def ask_enter(prompt):
        prompts.append(prompt)
        return ''

    def fixture_launcher(task, private, run, *, config=None, seed=42, extra_args=()):
        return launch_core(task, private, run, seed=seed, extra_args=extra_args,
                           command_prefix=[sys.executable, str(FIXTURE11), '--mode', 'single'],
                           execution_kind='fixture')

    with offline_environment():
        code = interactive.main(['--folder', str(folder), '--workspace', str(tmp_path / 'ws'), '--no-llm'],
                                ask=ask_enter, say=said.append, launcher=fixture_launcher)
    assert code == 0, '\n'.join(said)
    joined_prompts = '\n'.join(prompts)
    for unwanted in ('GPU', '并行', 'CPU', '总时间', '最多方案节点'):
        assert unwanted not in joined_prompts
    assert any('确认开始运行' in p for p in prompts)
    text = '\n'.join(said)
    assert '预算策略' in text and '18' in text


def test_interactive_customize_resources_on_c(tmp_path):
    folder = csv_folder(tmp_path / 'raw')
    answers = iter(['', 'c', '1000', '10', '2', '8', '2', '', ''])
    prompts, seen_extra = [], []

    def ask(prompt):
        prompts.append(prompt)
        return next(answers)

    def fixture_launcher(task, private, run, *, config=None, seed=42, extra_args=()):
        seen_extra.extend(extra_args)
        # Fixture script does not parse Core resource flags; run fixture without forwarding them.
        return launch_core(task, private, run, seed=seed,
                           command_prefix=[sys.executable, str(FIXTURE11), '--mode', 'single'],
                           execution_kind='fixture')

    with offline_environment():
        code = interactive.main(['--folder', str(folder), '--workspace', str(tmp_path / 'ws'), '--no-llm'],
                                ask=ask, say=lambda _m: None, launcher=fixture_launcher)
    assert code == 0
    assert seen_extra == ['--time-limit-secs', '1000', '--max-steps', '10', '--num-gpus', '2', '--num-cpus', '8', '--parallel', '2']
