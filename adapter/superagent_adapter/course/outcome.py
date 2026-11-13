"""A readable latest-run view; immutable reports remain bound to their input evidence."""
from pathlib import Path

from .common import digest, read_json, write_json


def blocked_outcome(check):
    return {'core': {'state': 'not_started'}, 'evaluation': {'status': 'not_requested'},
            'report': {'state': 'not_generated'},
            'summary': '预检查未通过，Core 尚未启动。',
            'issues': [c['message'] for c in check['checks'] if not c['ok']]}


def _handoff_files(run, evidence):
    """Locate existing evidence and the selected scripts, without running a writer."""
    collection = evidence['collection']
    sealed = collection['files_sha256'] if collection['valid_solution'] else {}
    def record(relative):
        path = run / relative
        return str(path) if path.is_file() else None
    def delivery(relative):
        return record('delivery/' + relative) if relative in sealed else None
    selected = collection.get('selected_result') or {}
    if selected.get('source') == 'final':
        scripts = [p for p in sealed if p.startswith('final/members/') and p.endswith('/solution.py')]
        combine = 'final/combine.py'
    elif selected.get('source') == 'best_partial':
        scripts, combine = ['best/solution.py'], None
    elif collection['final_choice'] == 'ensemble':
        scripts = [p for p in sealed if p.startswith('top_solution/') and p.endswith('/solution.py')]
        combine = 'ensemble/ensemble.py'
    else:
        scripts, combine = ['best_solution/solution.py'], None
    artifact_dirs = sorted({str(Path(p).parent / 'artifacts') for p in scripts
                            if any(rel.startswith(str(Path(p).parent.as_posix()) + '/artifacts/')
                                   for rel in sealed)})
    return {'run_dir': str(run), 'evidence_json': record('adapter/evidence.json'),
            'collection_json': record('adapter/collection.json'),
            'launch_json': record('adapter/launch.json'), 'runtime_json': record('adapter/runtime.json'),
            'stdout_log': record('adapter/core.stdout.log'), 'stderr_log': record('adapter/core.stderr.log'),
            'predictions_csv': delivery('predictions.csv'),
            'selection_json': delivery('final/source.json'),
            'solution_scripts': [path for rel in sorted(scripts) if (path := delivery(rel))],
            'model_artifact_dirs': [str(run / 'delivery' / rel) for rel in artifact_dirs],
            'core_logs_dir': str(run / 'delivery/core_logs')
                if (run / 'delivery/core_logs').is_dir() else None,
            'combine_script': delivery(combine) if combine else None,
            'final_evaluation_json': record('adapter/final-evaluation.json')
                if evidence['evaluation']['status'] == 'evaluated' else None}


def write_outcome(run_dir, evidence, *, report_result=None, report_error=None):
    run = Path(run_dir).resolve()
    evidence_hash = digest(run / 'adapter/evidence.json')
    receipt_path = run / 'adapter/report_receipt.json'
    if report_result is not None or report_error is not None:
        previous = read_json(receipt_path) if receipt_path.is_file() else {}
        receipt = (previous if report_error and previous.get('result') else
                   {'source_evidence_sha256': evidence_hash, 'result': report_result, 'error': report_error})
        receipt['last_attempt'] = {'state':'failed' if report_error else 'generated', 'error':report_error,
                                   'source_evidence_sha256':evidence_hash}
        if report_result:
            receipt['reason'] = read_json(report_result['review_json']).get('rejection')
        write_json(receipt_path, receipt)
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        result = receipt.get('result') or {}
        mode = result.get('mode')
        state = ('failed' if receipt.get('error') else
                 'fallback' if mode in ('deterministic_after_rejection', 'deterministic_configuration_error') else 'generated')
        report = {'state': state, 'mode': mode, 'reason': receipt.get('error') or receipt.get('reason'),
                  'report_md': result.get('report_md'), 'report_json': result.get('report_json'),
                  'last_attempt':receipt.get('last_attempt'),
                  'source_evidence_sha256': receipt['source_evidence_sha256']}
        if result and receipt['source_evidence_sha256'] != evidence_hash:
            report.update(state='stale', reason='证据已更新；现有报告仍对应旧证据，需显式重新生成。')
    else:
        report = {'state': 'not_generated', 'mode': None, 'reason': None, 'report_md': None, 'report_json': None}
    collection = evidence['collection']
    state = ('partial_valid' if collection['partial'] else 'success') if collection['valid_solution'] else 'no_valid_result'
    core_text = {'success':'方案有效，Core 正常结束', 'partial_valid':'Core 未正常完成，已保留有效部分方案',
                 'no_valid_result':'没有通过检查的有效方案'}[state]
    evaluation = dict(evidence['evaluation'])
    eval_text = {'not_requested':'未进行最终评分', 'unavailable':'没有可评价的方案', 'evaluated':'最终评分已完成'}.get(evaluation['status'], '评价状态未知')
    report_text = {'not_generated':'未生成可选报告', 'generated':'报告已生成', 'fallback':'报告已回退到确定性草稿',
                   'failed':'报告生成失败', 'stale':'报告需要按新证据重新生成'}[report['state']]
    issues = list(collection['issues'])
    if report.get('reason'):
        issues.append(report['reason'])
    if report.get('last_attempt') and report['last_attempt']['state'] == 'failed':
        issues.append(report['last_attempt']['error'])
        if report['state'] != 'failed':
            report_text += '，最近一次重新生成失败，保留此前报告'
    outcome = {'schema_version':'0.1', 'execution_kind':evidence['provenance']['execution_kind'],
               'core':{'state':state, 'status':(evidence['core_status'] or {}).get('status'),
                       'returncode':collection['core_returncode'], 'valid_solution':collection['valid_solution']},
               'evaluation':evaluation, 'report':report,
               'files':_handoff_files(run, evidence),
               'consistency':evidence.get('consistency', {'status':'not_verifiable'}),
               'source_evidence_sha256':evidence_hash,
               'summary':f'{core_text}；{eval_text}；{report_text}。', 'issues':issues}
    write_json(run / 'adapter/outcome.json', outcome)
    return outcome
