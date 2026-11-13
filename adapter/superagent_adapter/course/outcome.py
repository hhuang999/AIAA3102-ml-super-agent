"""A readable latest-run view; immutable reports remain bound to their input evidence."""
from pathlib import Path

from .common import digest, read_json, write_json


def blocked_outcome(check):
    return {'core': {'state': 'not_started'}, 'evaluation': {'status': 'not_requested'},
            'report': {'state': 'not_generated'},
            'summary': '预检查未通过，Core 尚未启动。',
            'issues': [c['message'] for c in check['checks'] if not c['ok']]}


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
    report_text = {'not_generated':'报告尚未生成', 'generated':'报告已生成', 'fallback':'报告已回退到确定性草稿',
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
               'consistency':evidence.get('consistency', {'status':'not_verifiable'}),
               'source_evidence_sha256':evidence_hash,
               'summary':f'{core_text}；{eval_text}；{report_text}。', 'issues':issues}
    write_json(run / 'adapter/outcome.json', outcome)
    return outcome
