"""Regressions for real-summary variations and long autonomous runs. No model calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from superagent_adapter.course.cli import report as cli_report
from superagent_adapter.course.reporting import generate_report


def evidence():
    return {'schema_version': '0.1', 'task': {'task_id': 'synthetic', 'rows': {'train': 3, 'val': 3, 'test': 3}},
            'policy': {'validation': {'status': 'provisional_pending_team_discussion'}},
            'core_status': {'status': 'success', 'best_node_id': 'last-node', 'best_metric': .9},
            'core_summary': {'best': {'method_summary': 'Synthetic summary only.'}},
            'collection': {'valid_solution': True, 'partial': False},
            'evaluation': {'status': 'not_requested', 'test_accuracy': None},
            'provenance': {'execution_kind': 'fixture'}}


class ReportRound2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / 'report'

    def test_late_timeline_and_event_survive_with_original_references(self):
        data = evidence()
        timeline = [{'node_id': f'node-{i}', 'score': .5 + i / 100} for i in range(25)]
        timeline[-1] = {'explanation': 'long description ' * 1000, 'node_id': 'last-node', 'score': .9}
        data['core_summary'].update(score_timeline=timeline,
                                    key_events=[{'event': f'event-{i}'} for i in range(25)])
        seen = {}
        def writer(payload):
            seen.update(payload['evidence'])
            return {'sections': [{'title': '最后一次改善', 'text': '末次记录对应最终节点，原因需结合实验核对。',
                                  'evidence_refs': ['/core_summary/score_timeline/24/score',
                                                    '/core_summary/key_events/24/event']}]}
        result = generate_report(data, self.out, writer)
        self.assertEqual(result['mode'], 'writer_human_review_required')
        selected = seen['core_summary']['score_timeline']
        self.assertLessEqual(len(selected), 8)
        self.assertEqual(selected['24']['node_id'], 'last-node')
        self.assertEqual(selected['24']['score'], .9)
        self.assertEqual(selected['0'], timeline[0])
        self.assertNotIn('1', selected)  # the projection does not renumber retained records
        context = json.loads((self.out / 'writer_context.json').read_text(encoding='utf-8'))
        self.assertEqual(context['evidence'], seen)
        rendered = (self.out / 'report.md').read_text(encoding='utf-8')
        self.assertIn('event-24', rendered)
        self.assertIn('原索引 24', rendered)
        self.assertEqual(len(json.loads((self.out / 'evidence.json').read_text(encoding='utf-8'))['core_summary']['score_timeline']), 25)

    def test_large_early_section_does_not_consume_later_sections(self):
        data = evidence()
        data['core_summary'] = {
            'task': {f'preview_{i}': 'description ' * 1000 for i in range(25)},
            'run': {'num_nodes': 100}, 'best': {'method_summary': 'final method'},
            'score_timeline': [{'score': .5 + i / 100} for i in range(25)],
            'failures': {'count': 4}, 'resources': {'prompt_tokens': 12345}}
        seen = {}
        def writer(payload):
            seen.update(payload['evidence']['core_summary'])
            return {'sections': [{'title': '运行概况', 'text': '检查全部运行记录。',
                                  'evidence_refs': ['/core_summary/best/method_summary',
                                                    '/core_summary/score_timeline/24/score',
                                                    '/core_summary/resources/prompt_tokens']}]}
        result = generate_report(data, self.out, writer)
        self.assertEqual(result['mode'], 'writer_human_review_required')
        self.assertEqual(seen['resources']['prompt_tokens'], 12345)
        self.assertLess(len(json.dumps(seen, ensure_ascii=False)), 7500)

    def test_extra_core_data_is_omitted_without_losing_report_or_input(self):
        data = evidence()
        data['core_summary']['best'].update(code='DO_NOT_FORWARD_CODE', rows=[{'label': 'DO_NOT_FORWARD_ROW'}])
        data['core_status']['debug'] = {'api_key': 'DO_NOT_FORWARD_KEY'}
        original = copy.deepcopy(data)
        seen = {}
        def writer(payload):
            seen.update(payload)
            return {'sections': [{'title': '方案', 'text': '只根据方法摘要说明方案。',
                                  'evidence_refs': ['/core_summary/best/method_summary']}]}
        result = generate_report(data, self.out, writer)
        self.assertEqual(result['mode'], 'writer_human_review_required')
        self.assertEqual(data, original)
        self.assertNotIn('DO_NOT_FORWARD', json.dumps(seen))
        self.assertNotIn('DO_NOT_FORWARD', (self.out / 'evidence.json').read_text(encoding='utf-8'))
        review = json.loads((self.out / 'review.json').read_text(encoding='utf-8'))
        self.assertIn('/core_summary/best/code', review['omitted_core_fields'])
        self.assertIn('/core_summary/best/rows', review['omitted_core_fields'])
        self.assertIn('/core_status/debug/api_key', review['omitted_core_fields'])
        self.assertIsNone(json.loads((self.out / 'report.json').read_text(encoding='utf-8'))['authoritative_evidence']['evaluation']['test_accuracy'])

    def test_empty_nested_list_does_not_hide_following_metadata(self):
        data = evidence()
        data['core_summary']['best'] = {'changes': [], 'method_summary': 'still present'}
        writer = lambda _: {'sections': [{'title': '方法', 'text': '查看方法说明。',
                                          'evidence_refs': ['/core_summary/best/method_summary']}]}
        result = generate_report(data, self.out, writer)
        self.assertEqual(result['mode'], 'writer_human_review_required')

    def test_bad_writer_configuration_still_produces_report_without_a_call(self):
        for index, config in enumerate(('{broken', '{}',
                                       '{"base_url":"invalid","model":"fixture","api_key_env":"SECRET_NAME"}',
                                       '{"base_url":123,"model":"fixture","api_key_env":"KEY"}',
                                       '{"base_url":"https://example.invalid","model":123,"api_key_env":"KEY"}',
                                       '{"base_url":"https://example.invalid","model":"fixture","api_key_env":["KEY"]}')):
            with self.subTest(config=index):
                path = self.root / f'config-{index}.json'
                path.write_text(config, encoding='utf-8')
                out = self.root / f'report-{index}'
                with patch('urllib.request.build_opener') as network:
                    result = cli_report(evidence(), out, path)
                    network.assert_not_called()
                self.assertEqual(result['mode'], 'deterministic_configuration_error')
                self.assertTrue((out / 'report.md').is_file())
                self.assertFalse((out / 'llm_calls.jsonl').exists())
                review = json.loads((out / 'review.json').read_text(encoding='utf-8'))
                self.assertIn('配置', review['rejection'])
                self.assertNotIn('SECRET_NAME', review['rejection'])


if __name__ == '__main__':
    unittest.main()
