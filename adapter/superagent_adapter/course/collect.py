"""Validate and seal the preselected Core output before any final-label evaluation."""
from __future__ import annotations

import math
from pathlib import Path
import shutil

from .common import digest, finite_number, inside_file, read_csv, read_json, write_json
from .prepare import verify_inputs
from .common import validate_predictions
from .core11 import collect_core11
from .consistency import check_summary



def _copy_tree(source, destination):
    source = Path(source)
    if source.is_symlink() or not source.is_dir():
        raise ValueError('Artifact source must be a real directory')
    for p in source.rglob('*'):
        if p.is_symlink() or not (p.is_file() or p.is_dir()):
            raise ValueError('Artifact contains a symlink or special file')
    shutil.copytree(source, destination)


def _validate_ensemble(core, task, target, classes):
    """Local v0.1 fixture schema; validate against the peer's first real sample later."""
    members = read_json(inside_file(core, 'ensemble/members.json'))
    if isinstance(members, dict):
        members = members.get('members')
    if not isinstance(members, list) or len(members) < 2:
        raise ValueError('Ensemble members must be a list of at least two nodes')
    script = inside_file(core, 'ensemble/ensemble.py').read_text(encoding='utf-8')
    if not script.strip():
        raise ValueError('Ensemble reconstruction script is empty')
    indexed = {}
    top = core / 'top_solution'
    if top.is_symlink() or not top.is_dir():
        raise ValueError('Ensemble member solution directory is missing')
    for directory in top.iterdir():
        if directory.is_symlink():
            raise ValueError('Ensemble member directory is a link')
        if directory.is_dir():
            node = inside_file(directory, 'node_id.txt').read_text(encoding='utf-8').strip()
            if not node or node in indexed:
                raise ValueError('Duplicate or missing top-solution node ID')
            indexed[node] = directory
    seen, weight_total = set(), 0.
    for member in members:
        if not isinstance(member, dict) or not isinstance(member.get('node_id'), str):
            raise ValueError('Each ensemble member must declare node_id')
        node = member['node_id']
        weight, score = member.get('weight'), member.get('validation_score')
        if node in seen or node not in indexed or not finite_number(weight) or weight < 0:
            raise ValueError('Invalid/duplicate/unavailable ensemble member or weight')
        if not finite_number(score) or not 0 <= score <= 1:
            raise ValueError('Invalid ensemble member validation accuracy')
        directory = indexed[node]
        actual = float(inside_file(directory, 'metric.txt').read_text(encoding='utf-8').strip())
        if not math.isfinite(actual) or abs(actual-score) > 1e-12:
            raise ValueError('Ensemble member score does not match its solution')
        if not inside_file(directory, 'solution.py').read_text(encoding='utf-8').strip():
            raise ValueError('Missing ensemble member full solution')
        validate_predictions(inside_file(directory, 'submission.csv'), task/'data/sample_submission.csv', target, classes)
        seen.add(node)
        weight_total += weight
    if not math.isfinite(weight_total) or weight_total <= 0:
        raise ValueError('Ensemble total weight must be positive and finite')


def collect_result(task_dir, private_dir, run_dir):
    task, private, run = Path(task_dir).resolve(), Path(private_dir).resolve(), Path(run_dir).resolve()
    m = verify_inputs(task, private)
    launch = read_json(run / 'adapter/launch.json')
    if launch['task_manifest_sha256'] != digest(private / 'manifest.json') or launch['final_choice'] != m['policy']['final_choice']:
        raise ValueError('Launch contract changed after Core started')
    if launch['state'] == 'running':
        raise ValueError('Cannot finalize while the Core process is running')
    runtime = None
    if launch.get('runtime_sha256'):
        runtime_path = inside_file(run / 'adapter', 'runtime.json')
        if digest(runtime_path) != launch['runtime_sha256']:
            raise ValueError('Runtime snapshot changed after launch')
        runtime = read_json(runtime_path)
    destination = run / 'delivery'
    if destination.exists():
        raise ValueError('Collected outputs are immutable; use a new run')
    destination.mkdir()
    core = run / 'core'
    issues, status, summary = [], None, {}
    supported = False
    try:
        status = read_json(inside_file(core, 'status.json'))
        if not isinstance(status, dict) or status.get('interface_version') != m.get('core_interface_version', '0.1'):
            raise ValueError('Unsupported/missing Core interface version')
        if status.get('status') not in ('running', 'success', 'no_valid_solution', 'failed'):
            raise ValueError('Unknown Core status')
        if status.get('higher_is_better') is not True:
            raise ValueError('Core metric direction differs from accuracy contract')
        score_key = 'best_score' if status['interface_version'] == '1.1' else 'best_metric'
        if status.get(score_key) is not None and not finite_number(status[score_key]):
            raise ValueError('Invalid Core validation metric')
        supported = True
    except (ValueError, OSError, TypeError) as exc:
        issues.append(f'Core status unavailable or invalid: {exc}')
        status = None
    try:
        summary = read_json(inside_file(core, 'report/run_summary.json'))
        if not isinstance(summary, dict):
            raise ValueError('Core summary must be an object')
    except (ValueError, OSError, TypeError) as exc:
        summary = {}
        issues.append(f'Core report summary unavailable: {exc}')
    valid, artifacts, solution_mode = False, False, 'full_script'
    selected_result = None
    partial = launch['returncode'] != 0 or (status or {}).get('status') != 'success'
    choice = launch['final_choice']
    try:
        if not supported:
            raise ValueError('Do not accept artifacts under an invalid interface/status')
        if status['status'] == 'no_valid_solution':
            raise ValueError('Core declared no valid solution')
        if status['interface_version'] == '1.1':
            selected_result, artifacts, solution_mode = collect_core11(
                core, destination, status, partial, task / m['prediction_template'],
                m['task']['target'], m['task']['classes'])
            if selected_result['source'] == 'best_partial':
                issues.append('Finalization incomplete; preserved the validated best single solution as a partial result')
            if not artifacts:
                issues.append('No saved model artifacts; full member scripts are required, inference-only is unverified')
        else:
            if not status.get('best_node_id') or not finite_number(status.get('best_metric')):
                raise ValueError('No identified, validated best node')
            if not 0 <= status['best_metric'] <= 1:
                raise ValueError('Accuracy must lie in [0, 1]')
            node = inside_file(core, 'best_solution/node_id.txt').read_text(encoding='utf-8').strip()
            score = float(inside_file(core, 'best_solution/metric.txt').read_text(encoding='utf-8').strip())
            if node != str(status['best_node_id']) or not math.isfinite(score) or abs(score-status['best_metric']) > 1e-12:
                raise ValueError('Best node/metric files disagree with status')
            solution = inside_file(core, 'best_solution/solution.py')
            if not solution.read_text(encoding='utf-8').strip():
                raise ValueError('Empty full solution script')
            prediction = 'best_submission/submission.csv' if choice == 'best' else 'ensemble/submission.csv'
            pp = inside_file(core, prediction)
            validate_predictions(pp, task / 'data/sample_submission.csv', m['task']['target'], m['task']['classes'])
            if choice == 'ensemble':
                if status.get('ensemble_available') is not True:
                    raise ValueError('Preselected ensemble unavailable; do not switch based on test results')
                _validate_ensemble(core, task, m['task']['target'], m['task']['classes'])
                _copy_tree(core / 'ensemble', destination / 'ensemble')
                _copy_tree(core / 'top_solution', destination / 'top_solution')
                solution_mode = 'full_member_scripts_plus_ensemble'
            _copy_tree(core / 'best_solution', destination / 'best_solution')
            shutil.copyfile(pp, destination / 'predictions.csv')
            model_dir = destination / 'best_solution/artifacts'
            artifacts = model_dir.is_dir() and any(p.is_file() for p in model_dir.rglob('*'))
            if bool(status.get('best_has_artifacts')) != artifacts:
                issues.append('Core artifact-presence flag disagrees with the collected files')
            if not artifacts:
                issues.append('No saved model artifacts; delivery uses the full training/prediction script per interface default')
        valid = True
    except (ValueError, OSError, TypeError) as exc:
        issues.append(f'Selected solution unavailable: {exc}')
    partial = launch['returncode'] != 0 or (status or {}).get('status') != 'success'
    if partial:
        issues.append('Core did not finish with normal success; any valid output is a partial result')
    # These fixed schema sections are the only Core prose sent to the report builder.
    summary_keys = ('task','run','best','score_timeline','branches','actions','key_events','top_solutions','ensemble','final','control','diagnostics','failures','resources')
    summary = {k:summary[k] for k in summary_keys if k in summary}
    consistency = check_summary(status, summary, selected_result)
    if consistency['status'] == 'conflict':
        issues.append('Core summary conflicts with verified best node/metric; exclude it from report conclusions')
    for name in ('logs', 'report'):
        if (core / name).exists():
            try:
                _copy_tree(core / name, destination / ('core_' + name))
            except (ValueError, OSError) as exc:
                issues.append(f'Could not preserve Core {name}: {exc}')
    sealed = {p.relative_to(destination).as_posix():digest(p) for p in destination.rglob('*') if p.is_file()}
    collection = {'valid_solution': valid, 'partial': partial, 'issues': issues,
                  'final_choice':choice, 'selected_result':selected_result, 'saved_model_artifacts_present':artifacts,
                  'solution_mode':solution_mode, 'inference_only_verified':False,
                  'full_script_replay_verified':False, 'combination_replay_verified':False, 'core_returncode':launch['returncode'],
                  'execution_kind':launch['execution_kind'], 'files_sha256':sealed}
    evidence = {'schema_version':'0.1', 'task':m['task'], 'policy':m['policy'], 'core_status':status,
                'core_summary':summary, 'collection':collection, 'consistency':consistency,
                'evaluation':{'status':'not_requested' if valid else 'unavailable','test_accuracy':None},
                'provenance':{'task_manifest_sha256':digest(private/'manifest.json'),
                              'launch_sha256':digest(run/'adapter/launch.json'),
                              'runtime_sha256':launch.get('runtime_sha256'), 'runtime_snapshot':runtime,
                              'execution_kind':launch['execution_kind'], 'formal_evaluation_eligible':False}}
    write_json(run / 'adapter/evidence.json', evidence)
    write_json(run / 'adapter/collection.json', collection)
    return evidence


def evaluate_final(task_dir, private_dir, run_dir):
    task, private, run = Path(task_dir).resolve(), Path(private_dir).resolve(), Path(run_dir).resolve()
    m = verify_inputs(task, private)
    evidence = read_json(run / 'adapter/evidence.json')
    if evidence['evaluation']['status'] != 'not_requested':
        raise ValueError('Final score is already recorded; do not silently reevaluate')
    if not evidence['collection']['valid_solution']:
        raise ValueError('No verified selected prediction to evaluate')
    if evidence['provenance']['task_manifest_sha256'] != digest(private / 'manifest.json'):
        raise ValueError('Task manifest changed after collection')
    for rel, sha in evidence['collection']['files_sha256'].items():
        if digest(inside_file(run / 'delivery', rel)) != sha:
            raise ValueError('Sealed delivery artifact changed')
    predictions = validate_predictions(run/'delivery/predictions.csv', task/m.get('prediction_template', 'data/sample_submission.csv'), m['task']['target'], m['task']['classes'])
    fields, labels = read_csv(private / 'test_labels.csv')
    target = m['task']['target']
    if fields != ['id', target] or len(labels) != len(predictions) or len({r['id'] for r in labels}) != len(labels):
        raise ValueError('Private labels are not uniquely aligned')
    if set(predictions) != {r['id'] for r in labels}:
        raise ValueError('Final label IDs differ from predictions')
    correct = sum(predictions[r['id']] == int(r[target]) for r in labels)
    evaluation = {'status':'evaluated', 'test_accuracy':correct/len(labels), 'correct':correct, 'rows':len(labels),
                  'selected_before_scoring':True, 'final_choice':evidence['collection']['final_choice'],
                  'predictions_sha256':digest(run/'delivery/predictions.csv'),
                  'execution_kind':evidence['provenance']['execution_kind']}
    evidence['evaluation'] = evaluation
    write_json(run / 'adapter/final-evaluation.json', evaluation)
    write_json(run / 'adapter/evidence.json', evidence)
    return evidence
