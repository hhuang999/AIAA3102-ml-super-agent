"""Persistent synthetic contract smoke test. No real model, LLM, or course scoring."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from superagent_adapter.course.common import digest, read_json, write_csv, write_json
from superagent_adapter.course.prepare import prepare_task
from superagent_adapter.course.runner import launch_core
from superagent_adapter.course.collect import collect_result, evaluate_final, validate_predictions
from superagent_adapter.course.reporting import generate_report


def mock_writer(payload):
    """Exercise the writer protocol without accessing any service."""
    assert payload['evidence']['provenance']['execution_kind'] == 'fixture'
    if payload['evidence'].get('consistency', {}).get('status') == 'conflict':
        assert payload['evidence']['core_summary'] == {}
        return {'sections':[{'title':'摘要冲突','text':'摘要未通过一致性检查，无法据此说明方法。',
                             'evidence_refs':['/core_status/best_metric']}],
                'usage':{'prompt_tokens':0,'completion_tokens':0},'model':'synthetic_mock_no_llm'}
    return {'sections': [
        {'title': '方法与运行过程', 'text': '本样例使用人工构造的单特征模拟 Core，未训练模型，不能代表课程任务性能。',
         'evidence_refs': ['/core_summary/best', '/provenance/execution_kind']},
        {'title': '默认策略与局限', 'text': 'Adapter 保留原有 train/val 划分；Core 的默认验证行为和测试标签路径分离均为暂定安排，之后需要团队讨论。',
         'evidence_refs': ['/policy/validation', '/policy/test_label_boundary']},
    ], 'usage': {'prompt_tokens': 0, 'completion_tokens': 0}, 'model': 'synthetic_mock_no_llm'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = out / 'synthetic_source'
    source.mkdir()
    for split in ('train', 'val', 'test'):
        write_csv(source / (split + '.csv'), ['feature', 'label'],
                  [{'feature': 0, 'label': 0}, {'feature': 1, 'label': 1}, {'feature': 0, 'label': 0}])
    spec = {'interface_version': '0.1', 'task_id': 'synthetic_contract_only', 'modality': 'tabular',
            'input_format': 'csv', 'feature_columns': ['feature'], 'target': 'label',
            'classes': [0, 1], 'metric': 'accuracy', 'objective': 'Synthetic contract fixture only.',
            'validation_policy': 'core_default', 'final_choice': 'best',
            'splits': {s: {'features': s + '.csv'} for s in ('train', 'val', 'test')}}
    write_json(out / 'spec.json', spec)
    task, private = out / 'task', out / 'private'
    manifest = prepare_task(out / 'spec.json', source, task, private)
    scenarios = []
    for mode, expected_exit in (('success', 0), ('partial', 1), ('no_valid', 2), ('summary_conflict', 0)):
        run = out / ('run_' + mode)
        launch = launch_core(task, private, run, command_prefix=[sys.executable, str(ROOT / 'tests/course/fixture_core.py'), '--mode', mode], execution_kind='fixture')
        assert launch['returncode'] == expected_exit
        assert launch['runtime_sha256'] == digest(run / 'adapter/runtime.json')
        assert read_json(run / 'adapter/preflight.json')['ready']
        if mode == 'success':
            # Simulate longer real-world summary shapes before collection seals the source files.
            summary_path = run / 'core/report/run_summary.json'
            summary = read_json(summary_path)
            summary['best']['code'] = 'SYNTHETIC_CODE_NOT_FOR_WRITER'
            summary['score_timeline'] = [{'explanation': 'synthetic long explanation ' * 100,
                                           'node_id': f'fixture-step-{i}', 'score': .51 + i / 100} for i in range(25)]
            summary['key_events'] = [{'event': f'fixture-event-{i}'} for i in range(25)]
            write_json(summary_path, summary)
        elif mode == 'summary_conflict':
            summary_path = run / 'core/report/run_summary.json'
            summary = read_json(summary_path)
            summary['best'].update(node_id='wrong-node', metric=.99, method_summary='INCONSISTENT_METHOD')
            write_json(summary_path, summary)
        evidence = collect_result(task, private, run)
        assert evidence['evaluation']['test_accuracy'] is None
        assert evidence['collection']['valid_solution'] == (mode != 'no_valid')
        assert evidence['collection']['partial'] == (mode in ('partial', 'no_valid'))
        assert evidence['provenance']['runtime_snapshot']['core']['source'] == 'fixture'
        assert evidence['consistency']['status'] == ('conflict' if mode == 'summary_conflict' else 'not_verifiable' if mode == 'no_valid' else 'consistent')
        before = generate_report(evidence, run / 'report_before_scoring')
        replay = False
        if mode == 'success':
            # Demonstrate a full fixture script in a fresh working directory, not inference-only.
            replay_dir = run / 'fixture_replay'
            replay_dir.mkdir()
            (replay_dir / 'working').mkdir()
            (replay_dir / 'submission').mkdir()
            shutil.copytree(task / 'data', replay_dir / 'input')
            shutil.copyfile(run / 'delivery/best_solution/solution.py', replay_dir / 'solution.py')
            replay_process = subprocess.run([sys.executable, str(replay_dir / 'solution.py')], cwd=replay_dir, check=True, timeout=30, capture_output=True, text=True)
            assert 'Final Validation Score: 0.75' in replay_process.stdout
            actual = validate_predictions(replay_dir / 'submission/submission.csv', task / 'data/sample_submission.csv', 'label', [0, 1])
            sealed = validate_predictions(run / 'delivery/predictions.csv', task / 'data/sample_submission.csv', 'label', [0, 1])
            assert actual == sealed
            replay = True
            evidence = evaluate_final(task, private, run)
            assert evidence['evaluation']['test_accuracy'] == 1.0
            assert not evidence['provenance']['formal_evaluation_eligible']
        report = generate_report(evidence, run / 'report_with_mock_writer', writer=mock_writer)
        assert report['mode'] == 'writer_human_review_required'
        if mode == 'success':
            context = read_json(run / 'report_with_mock_writer/writer_context.json')['evidence']
            assert context['core_summary']['score_timeline']['24']['node_id'] == 'fixture-step-24'
            assert context['core_summary']['key_events']['24']['event'] == 'fixture-event-24'
            assert 'code' not in context['core_summary']['best']
            assert 'code' in read_json(run / 'adapter/evidence.json')['core_summary']['best']
        fallback = generate_report(evidence, run / 'report_invalid_writer', writer=lambda _: {'sections': 'invalid'})
        assert fallback['mode'] == 'deterministic_after_rejection'
        # Exercise the public CLI against the same collected evidence.
        cli = subprocess.run([sys.executable, '-m', 'superagent_adapter.course', 'report', '--run-dir', str(run), '--output-dir', str(run / 'report_cli')], capture_output=True, text=True, encoding='utf-8', timeout=30)
        if cli.returncode:
            raise RuntimeError('Report CLI failed: ' + cli.stderr[-2000:])
        cli_result = json.loads(cli.stdout)
        assert cli_result['mode'] == 'deterministic_no_llm'
        assert cli_result['outcome']['report']['state'] == 'generated'
        broken_config = run / 'broken-writer-config.json'
        broken_config.write_text('{broken', encoding='utf-8')
        config_cli = subprocess.run([sys.executable, '-m', 'superagent_adapter.course', 'report', '--run-dir', str(run),
                                     '--output-dir', str(run / 'report_bad_config'), '--writer-config', str(broken_config)],
                                    capture_output=True, text=True, encoding='utf-8', timeout=30)
        assert config_cli.returncode == 0, config_cli.stderr
        assert json.loads(config_cli.stdout)['mode'] == 'deterministic_configuration_error'
        assert json.loads(config_cli.stdout)['outcome']['report']['state'] == 'fallback'
        assert (run / 'report_bad_config/report.md').is_file()
        assert not (run / 'report_bad_config/llm_calls.jsonl').exists()
        scenarios.append({'scenario': mode, 'core_returncode': launch['returncode'], 'valid_solution': evidence['collection']['valid_solution'],
                          'partial': evidence['collection']['partial'], 'evaluation_status': evidence['evaluation']['status'],
                          'fixture_script_replayed': replay, 'mock_writer': report['mode'], 'invalid_writer': fallback['mode'],
                          'deterministic_report': before['mode'], 'cli_report': cli_result['mode'],
                          'bad_config_cli_fallback': True, 'long_summary_checked': mode == 'success',
                          'summary_consistency':evidence['consistency']['status'],
                          'runtime_snapshot_bound':True,
                          'evidence_sha256': digest(run / 'adapter/evidence.json')})
    receipt = {'status': 'passed', 'created_at': datetime.now(timezone.utc).isoformat(), 'execution_kind': 'fixture',
               'real_model_calls': 0, 'model_training_performed': False, 'official_test_scoring_performed': False,
               'rows': manifest['task']['rows'], 'scenarios': scenarios}
    write_json(out / 'receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
