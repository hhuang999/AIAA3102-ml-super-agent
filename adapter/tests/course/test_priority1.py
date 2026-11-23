"""Integration regressions for preflight, provenance, outcomes and contradictory summaries."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from superagent_adapter.course import cli
from superagent_adapter.course.common import digest, read_json, write_csv, write_json
from superagent_adapter.course.prepare import prepare_task
from superagent_adapter.course.runner import launch_core
from superagent_adapter.course.collect import collect_result, evaluate_final
from superagent_adapter.course.reporting import generate_report

FIXTURE = Path(__file__).with_name('fixture_core.py')


class Priority1Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / 'source'; source.mkdir()
        for split in ('train', 'val', 'test'):
            write_csv(source/(split+'.csv'), ['feature', 'label'], [{'feature': 0, 'label': 0}, {'feature': 1, 'label': 1}])
        spec = {'interface_version':'0.1','task_id':'synthetic','modality':'tabular','input_format':'csv',
                'feature_columns':['feature'],'target':'label','classes':[0,1],'metric':'accuracy',
                'objective':'Synthetic fixture','validation_policy':'core_default','final_choice':'best',
                'splits':{s:{'features':s+'.csv'} for s in ('train','val','test')}}
        write_json(self.root/'spec.json', spec)
        self.task, self.private, self.run = self.root/'task', self.root/'private', self.root/'run'
        prepare_task(self.root/'spec.json', source, self.task, self.private)

    def launch(self, *args, **kwargs):
        return launch_core(self.task, self.private, self.run,
                           command_prefix=[sys.executable, str(FIXTURE)], execution_kind='fixture')

    def command(self, action, *extra):
        args = [action, '--run-dir', str(self.run)]
        if action != 'report':
            args += ['--task-dir', str(self.task), '--private-dir', str(self.private)]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = cli.main([*args, *extra])
        return code, json.loads(output.getvalue())

    def test_run_exposes_report_fallback_and_runtime_identity(self):
        config = self.root/'broken.json'; config.write_text('{broken', encoding='utf-8')
        with patch('superagent_adapter.course.cli.launch_core', side_effect=self.launch):
            code, result = self.command('run', '--report-writer-config', str(config))
        self.assertEqual(code, 0)
        outcome = result['outcome']
        self.assertEqual(outcome['core']['state'], 'success')
        self.assertEqual(outcome['evaluation']['status'], 'not_requested')
        self.assertEqual(outcome['report']['state'], 'fallback')
        self.assertIn('配置', outcome['report']['reason'])
        self.assertEqual(read_json(self.run/'adapter/outcome.json'), outcome)
        launch = read_json(self.run/'adapter/launch.json')
        self.assertEqual(launch['runtime_sha256'], digest(self.run/'adapter/runtime.json'))
        snapshot = read_json(self.run/'adapter/runtime.json')
        self.assertEqual(snapshot['adapter']['version'], '0.3.1')
        self.assertEqual(result['provenance']['runtime_snapshot'], snapshot)
        self.assertIsNone(snapshot['model'])

    def test_runtime_file_mutation_is_detected_before_collection(self):
        self.launch()
        runtime = read_json(self.run/'adapter/runtime.json')
        runtime['adapter']['version'] = 'different'
        write_json(self.run/'adapter/runtime.json', runtime)
        with self.assertRaisesRegex(ValueError, 'runtime|Runtime'):
            collect_result(self.task, self.private, self.run)

    def test_conflicting_summary_keeps_valid_solution_but_is_not_used_for_writing(self):
        self.launch()
        summary_path = self.run/'core/report/run_summary.json'
        summary = read_json(summary_path)
        summary['best'] = {'node_id':'wrong-node','metric':.99,'method_summary':'WRONG_METHOD_MUST_NOT_BE_USED'}
        write_json(summary_path, summary)
        evidence = collect_result(self.task, self.private, self.run)
        self.assertTrue(evidence['collection']['valid_solution'])
        self.assertEqual(evidence['consistency']['status'], 'conflict')
        seen = {}
        def writer(payload):
            seen.update(payload)
            return {'sections':[{'title':'运行说明','text':'摘要冲突，方法未知。','evidence_refs':['/core_status/best_metric']}]}
        result = generate_report(evidence, self.run/'report', writer)
        self.assertEqual(result['mode'], 'writer_human_review_required')
        self.assertEqual(seen['evidence']['core_summary'], {})
        self.assertNotIn('WRONG_METHOD_MUST_NOT_BE_USED', json.dumps(seen))
        self.assertIn('WRONG_METHOD_MUST_NOT_BE_USED', (self.run/'adapter/evidence.json').read_text(encoding='utf-8'))
        review = read_json(self.run/'report/review.json')
        self.assertEqual(review['summary_consistency']['status'], 'conflict')
        self.assertIn('冲突', (self.run/'report/report.md').read_text(encoding='utf-8'))
        self.assertEqual(read_json(self.run/'report/evidence.json')['core_status']['best_metric'], .75)

    def test_report_is_marked_stale_after_evaluation_and_refreshed_explicitly(self):
        self.launch(); collect_result(self.task, self.private, self.run)
        code, result = self.command('report', '--output-dir', str(self.run/'report-before'))
        self.assertEqual(code, 0)
        self.assertEqual(result['outcome']['report']['state'], 'generated')
        code, result = self.command('evaluate-final')
        self.assertEqual(code, 0)
        self.assertEqual(result['outcome']['report']['state'], 'stale')
        self.assertEqual(result['outcome']['evaluation']['test_accuracy'], 1.)
        code, result = self.command('report', '--output-dir', str(self.run/'report-after'))
        self.assertEqual(result['outcome']['report']['state'], 'generated')
        self.assertEqual(read_json(self.run/'report-before/evidence.json')['evaluation']['status'], 'not_requested')

    def test_missing_core_check_and_run_do_not_launch_or_create_run(self):
        missing = {'source':'real_core','status':'missing','origin':None,'run_origin':None,'message':'Core unavailable'}
        with patch('superagent_adapter.course.runtime._discover_core', return_value=missing), patch('subprocess.Popen', side_effect=AssertionError('must not launch')):
            code, result = self.command('check')
            self.assertEqual(code, 3)
            self.assertFalse(result['ready'])
            code, result = self.command('run')
            self.assertEqual(code, 3)
            self.assertEqual(result['outcome']['core']['state'], 'not_started')
        self.assertFalse(self.run.exists())

    def test_failed_report_retry_preserves_previous_readable_report(self):
        self.launch(); collect_result(self.task, self.private, self.run)
        directory = self.run/'report'
        code, result = self.command('report', '--output-dir', str(directory))
        before = read_json(self.run/'adapter/report_receipt.json')
        self.assertEqual(code, 0)
        code, retried = self.command('report', '--output-dir', str(directory))
        self.assertEqual(code, 4)
        self.assertEqual(retried['outcome']['report']['state'], 'generated')
        self.assertEqual(retried['outcome']['report']['report_md'], result['report_md'])
        self.assertEqual(retried['outcome']['report']['last_attempt']['state'], 'failed')
        self.assertEqual(read_json(self.run/'adapter/report_receipt.json')['result'], before['result'])
        self.command('evaluate-final')
        code, retried = self.command('report', '--output-dir', str(directory))
        self.assertEqual(code, 4)
        self.assertEqual(retried['outcome']['report']['state'], 'stale')
        self.assertEqual(retried['outcome']['report']['report_md'], result['report_md'])

    def test_postcheck_oserror_still_leaves_collectable_evidence(self):
        with patch('subprocess.Popen', side_effect=OSError('synthetic launch failure')):
            receipt = self.launch()
        self.assertEqual(receipt['state'], 'launch_failed')
        evidence = collect_result(self.task, self.private, self.run)
        self.assertFalse(evidence['collection']['valid_solution'])
        self.assertTrue(evidence['provenance']['runtime_snapshot']['adapter']['source_sha256'])

    def test_never_created_report_is_failed_not_stale_after_evaluation(self):
        self.launch(); collect_result(self.task, self.private, self.run)
        directory = self.run/'occupied'; directory.mkdir()
        code, result = self.command('report', '--output-dir', str(directory))
        self.assertEqual(code, 4)
        self.assertEqual(result['outcome']['report']['state'], 'failed')
        code, result = self.command('evaluate-final')
        self.assertEqual(result['outcome']['report']['state'], 'failed')
        self.assertIsNone(result['outcome']['report']['report_md'])


if __name__ == '__main__':
    unittest.main()
