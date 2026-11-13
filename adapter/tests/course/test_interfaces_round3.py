"""Offline regressions for course handoff and final-result transaction boundaries."""
import contextlib
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

from superagent_adapter.course import cli
from superagent_adapter.course.collect import collect_result, evaluate_final
from superagent_adapter.course.common import digest, read_csv, read_json, write_csv, write_json
from superagent_adapter.course.outcome import write_outcome
from superagent_adapter.course.prepare import prepare_task
from superagent_adapter.course.runner import launch_core
from .test_core11 import prepare_synthetic, launch_synthetic


@pytest.fixture
def prepared(tmp_path):
    task, private = prepare_synthetic(tmp_path)
    return task, private, tmp_path / 'run'


def invoke(prepared, *extra):
    task, private, run = prepared
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = cli.main(['run', '--task-dir', str(task), '--private-dir', str(private),
                         '--run-dir', str(run), *extra])
    return code, json.loads(output.getvalue())


def fake_launch(*args, **kwargs):
    return launch_synthetic(*args, mode='single')


def test_default_run_does_not_generate_report_and_indexes_delivery(prepared):
    with patch.object(cli, 'launch_core', side_effect=fake_launch), \
            patch.object(cli, 'report', side_effect=AssertionError('Report work is paused')):
        code, result = invoke(prepared)
    assert code == 0
    run = prepared[2]
    assert not (run / 'report').exists()
    assert result['outcome']['report']['state'] == 'not_generated'
    files = result['outcome']['files']
    assert Path(files['predictions_csv']) == run / 'delivery/predictions.csv'
    assert files['final_evaluation_json'] is None
    assert len(files['solution_scripts']) == 1
    assert all(Path(p).is_file() for p in files['solution_scripts'])
    assert Path(files['selection_json']).is_file()
    assert Path(files['combine_script']).is_file()
    assert result['outcome']['evaluation']['status'] == 'not_requested'


def test_report_still_available_when_explicitly_requested(prepared):
    with patch.object(cli, 'launch_core', side_effect=fake_launch):
        code, result = invoke(prepared, '--generate-report')
    assert code == 0
    assert result['outcome']['report']['state'] == 'generated'


@pytest.mark.parametrize('mutation', ['script', 'predictions'])
def test_ensemble_best_member_must_match_best_artifact(prepared, mutation):
    task, private, run = prepared
    launch_synthetic(task, private, run, 'ensemble')
    source = read_json(run / 'core/final/source.json')
    status = read_json(run / 'core/status.json')
    member = next(m for m in source['members'] if m['node_id'] == status['best_node_id'])
    if mutation == 'script':
        (run / 'core/final/members' / member['dir'] / 'solution.py').write_text(
            '# Different implementation from the validated best node\n', encoding='utf-8')
    else:
        path = run / 'core/best/predictions.csv'
        columns, rows = read_csv(path)
        rows[0]['label'] = 1 - int(rows[0]['label'])
        write_csv(path, columns, rows)
    result = collect_result(task, private, run)
    assert not result['collection']['valid_solution']
    assert any('best' in item.lower() for item in result['collection']['issues'])


def test_failed_evidence_write_recovers_committed_evaluation_without_rescoring(prepared):
    task, private, run = prepared
    launch_synthetic(task, private, run, 'single')
    collect_result(task, private, run)
    def fail_evidence(path, data):
        if Path(path) == run / 'adapter/evidence.json':
            raise OSError('Synthetic interrupted evidence write')
        write_json(path, data)
    with patch('superagent_adapter.course.collect.write_json', side_effect=fail_evidence), \
            pytest.raises(OSError, match='Synthetic'):
        evaluate_final(task, private, run)
    marker = run / 'adapter/final-evaluation.json'
    committed_hash = digest(marker)
    assert read_json(run / 'adapter/evidence.json')['evaluation']['status'] == 'not_requested'
    with patch('superagent_adapter.course.collect.read_csv',
               side_effect=AssertionError('Committed evaluation must not re-read labels')):
        evidence = evaluate_final(task, private, run)
    assert evidence['evaluation'] == read_json(marker)
    assert digest(marker) == committed_hash
    assert not (run / 'adapter/final-evaluation.lock').exists()
    with pytest.raises(ValueError, match='already recorded'):
        evaluate_final(task, private, run)


def test_unbound_evaluation_marker_is_never_overwritten(prepared):
    task, private, run = prepared
    launch_synthetic(task, private, run, 'single')
    collect_result(task, private, run)
    marker = run / 'adapter/final-evaluation.json'
    write_json(marker, {'status': 'evaluated', 'test_accuracy': 1.0})
    before = digest(marker)
    with pytest.raises(ValueError, match='evaluation record'):
        evaluate_final(task, private, run)
    assert digest(marker) == before


def test_existing_evaluation_lock_blocks_another_evaluator(prepared):
    task, private, run = prepared
    launch_synthetic(task, private, run, 'single')
    collect_result(task, private, run)
    lock = run / 'adapter/final-evaluation.lock'
    lock.write_text('another evaluator', encoding='utf-8')
    with pytest.raises(ValueError, match='evaluation.*lock'):
        evaluate_final(task, private, run)
    assert lock.read_text(encoding='utf-8') == 'another evaluator'
    assert not (run / 'adapter/final-evaluation.json').exists()


def test_outcome_does_not_expose_invalid_solution_as_deliverable(prepared):
    task, private, run = prepared
    launch_synthetic(task, private, run, 'novalid')
    evidence = collect_result(task, private, run)
    files = write_outcome(run, evidence)['files']
    assert files['predictions_csv'] is None
    assert files['solution_scripts'] == []
    assert Path(files['evidence_json']).is_file()


def test_handoff_lists_materialized_model_artifacts(prepared):
    task, private, run = prepared
    launch_synthetic(task, private, run, 'single')
    member = next((run / 'core/final/members').iterdir())
    artifacts = member / 'artifacts'
    artifacts.mkdir()
    (artifacts / 'fixture.txt').write_text('Synthetic artifact, not a trained model', encoding='utf-8')
    evidence = collect_result(task, private, run)
    files = write_outcome(run, evidence)['files']
    assert files['model_artifact_dirs'] == [str(run / 'delivery/final/members' / member.name / 'artifacts')]
    assert (Path(files['model_artifact_dirs'][0]) / 'fixture.txt').is_file()
    assert Path(files['core_logs_dir']).is_dir()


def test_cpu_only_resource_override_reaches_child(prepared):
    task, private, run = prepared
    script = task.parent / 'arguments.py'
    script.write_text('import sys\nprint(sys.argv[1:])\n', encoding='utf-8')
    receipt = launch_core(task, private, run, command_prefix=[sys.executable, str(script)],
                          execution_kind='fixture', extra_args=['--num-gpus', '0'])
    assert receipt['returncode'] == 0
    assert receipt['command'][-2:] == ['--num-gpus', '0']


@pytest.mark.parametrize('override', [['--num-gpus', '-1'], ['--parallel', '0'],
                                       ['--num-cpus', '1.5'], ['--num-gpus', '0', '--num-gpus', '1']])
def test_invalid_resource_overrides_create_no_run(prepared, override):
    task, private, run = prepared
    with pytest.raises(ValueError):
        launch_core(task, private, run, extra_args=override)
    assert not run.exists()


@pytest.mark.parametrize('trust', ['false', 1])
def test_npy_trust_requires_a_boolean_before_loading_arrays(tmp_path, trust):
    import numpy as np
    source = tmp_path / 'source'
    source.mkdir()
    splits = {}
    for split in ('train', 'val', 'test'):
        np.save(source / (split + '-x.npy'), np.array([['a'], ['b']], dtype=object))
        np.save(source / (split + '-y.npy'), np.array([0, 1]))
        splits[split] = {'features': split + '-x.npy', 'labels': split + '-y.npy'}
    spec = {'interface_version': '1.1', 'task_id': 'synthetic', 'metric': 'accuracy',
            'feature_columns': ['text'], 'target': 'label', 'classes': [0, 1],
            'input_format': 'npy', 'trusted_pickle': trust, 'splits': splits,
            'objective': 'Synthetic input type check', 'modality': 'text'}
    write_json(tmp_path / 'spec.json', spec)
    with pytest.raises(ValueError, match='trusted_pickle'):
        prepare_task(tmp_path / 'spec.json', source, tmp_path / 'task', tmp_path / 'private')
    assert not (tmp_path / 'task').exists()
    assert not (tmp_path / 'private').exists()
