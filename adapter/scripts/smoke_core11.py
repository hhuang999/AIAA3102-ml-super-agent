"""Persistent offline Core 1.1 integration smoke; synthetic data and real writers only."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
for directory in ('adapter', 'core'):
    sys.path.insert(0, str(REPO / directory))
module_spec = importlib.util.spec_from_file_location(
    'core11_smoke_helpers', REPO / 'adapter/tests/course/test_core11.py')
helpers = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(helpers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--generate-report', action='store_true',
                        help='Explicitly include the optional legacy mock report writer')
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if root.exists():
        raise ValueError('Choose a new output directory; smoke evidence is immutable')
    root.mkdir(parents=True)
    from superagent_adapter.course.collect import collect_result
    from superagent_adapter.course.common import read_json, write_json
    from superagent_adapter.course.outcome import write_outcome
    from superagent_adapter.course.reporting import generate_report
    records = []
    for mode in ('single', 'ensemble', 'partial', 'novalid', 'summaryconflict'):
        directory = root / mode
        task, private = helpers.prepare_synthetic(directory)
        run = directory / 'run'
        receipt = helpers.launch_synthetic(task, private, run, mode)
        with helpers.offline_environment():
            evidence = collect_result(task, private, run)
            def writer(payload):
                return {'sections': [{'title': 'Offline fixture',
                                      'text': 'Synthetic integration only; zero real model calls, training runs, or formal scoring.',
                                      'evidence_refs': ['/collection/execution_kind']}]}
            report = generate_report(evidence, run / 'report', writer) if args.generate_report else None
            outcome = write_outcome(run, evidence, report_result=report)
            if not args.generate_report:
                assert outcome['report']['state'] == 'not_generated'
                assert not (run / 'report').exists()
        valid = evidence['collection']['valid_solution']
        assert valid == (mode != 'novalid'), evidence['collection']
        if mode == 'ensemble':
            selected = evidence['collection']['selected_result']
            assert selected['kind'] == 'ensemble' and selected['validation_score'] == 1.0
        if mode == 'partial':
            assert evidence['collection']['selected_result']['source'] == 'best_partial'
        if mode == 'summaryconflict':
            assert evidence['consistency']['status'] == 'conflict'
        audit = read_json(run / 'core/fixture_audit.json')
        assert audit['real_llm_calls'] == audit['model_training_runs'] == audit['formal_scoring_runs'] == 0
        records.append({'mode': mode, 'returncode': receipt['returncode'],
                        'valid_solution': valid, 'partial': evidence['collection']['partial'],
                        'selected_result': evidence['collection'].get('selected_result'),
                        'consistency': evidence['consistency']['status'],
                        'report_mode': report['mode'] if report else None, 'outcome': outcome,
                        'audit': audit})
    summary = {'execution_kind': 'fixture', 'real_llm_calls': 0, 'model_training_runs': 0,
               'formal_scoring_runs': 0, 'network_guard': True, 'modes': records,
               'note': 'Actual pinned Core writers on synthetic arrays; this is not real Core search or a course evaluation.'}
    write_json(root / 'smoke_summary.json', summary)
    print(json.dumps({'output_dir': str(root), 'modes': len(records), 'real_llm_calls': 0,
                      'model_training_runs': 0, 'formal_scoring_runs': 0}, indent=2))


if __name__ == '__main__':
    main()
