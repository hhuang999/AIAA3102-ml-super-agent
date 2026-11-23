"""Offline 1.1 integration checks: real Core writers, synthetic data, zero LLM calls."""
from __future__ import annotations

import contextlib
import csv
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from superagent_adapter.course.collect import collect_result, evaluate_final
from superagent_adapter.course.common import digest, read_json, write_csv, write_json
from superagent_adapter.course.outcome import write_outcome
from superagent_adapter.course.prepare import prepare_task
from superagent_adapter.course.reporting import generate_report
from superagent_adapter.course.runner import launch_core

REPO = Path(__file__).resolve().parents[3]
FIXTURE11 = Path(__file__).with_name('fixture_core11.py').resolve()
PYTHONPATH11 = os.pathsep.join(str(REPO / name) for name in ('core', 'adapter'))


@contextlib.contextmanager
def offline_environment():
    def denied(*args, **kwargs):
        raise AssertionError('Offline integration test attempted network access')
    with patch.dict(os.environ, {'PYTHONPATH': PYTHONPATH11}), \
         patch.object(socket.socket, 'connect', denied), \
         patch.object(socket, 'create_connection', denied):
        yield


def prepare_synthetic(root):
    root = Path(root)
    source = root / 'source'
    source.mkdir(parents=True)
    for split in ('train', 'val', 'test'):
        write_csv(source / (split + '.csv'), ['feature', 'label'],
                  [{'feature': i % 2, 'label': i % 2} for i in range(12)])
    spec = {'interface_version': '1.1', 'task_id': 'synthetic_core11',
            'modality': 'tabular', 'input_format': 'csv', 'feature_columns': ['feature'],
            'target': 'label', 'classes': [0, 1], 'metric': 'accuracy',
            'objective': 'Synthetic offline fixture; no training or real benchmark scoring.',
            'validation_policy': 'core_default', 'final_choice': 'final',
            'splits': {name: {'features': name + '.csv'} for name in ('train', 'val', 'test')}}
    write_json(root / 'spec.json', spec)
    task, private = root / 'task', root / 'private'
    prepare_task(root / 'spec.json', source, task, private)
    return task, private


def launch_synthetic(task, private, run, mode):
    with offline_environment():
        receipt = launch_core(task, private, run,
                              command_prefix=[sys.executable, str(FIXTURE11), '--mode', mode],
                              execution_kind='fixture')
    audit_path = Path(run) / 'core/fixture_audit.json'
    if not audit_path.exists():
        stderr = (Path(run) / 'adapter/core.stderr.log').read_text(encoding='utf-8', errors='replace')
        raise AssertionError(f'Real-Core offline fixture did not complete: {stderr}')
    return receipt


class Core11OfflineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task, self.private = prepare_synthetic(self.root)
        self.run = self.root / 'run'

    def launch(self, mode='single'):
        return launch_synthetic(self.task, self.private, self.run, mode)

    def collect(self):
        with offline_environment():
            return collect_result(self.task, self.private, self.run)

    def test_prepare_uses_core11_data_and_output_contract(self):
        self.assertTrue((self.task / 'data/sample_predictions.csv').is_file())
        text = (self.task / 'description.md').read_text(encoding='utf-8')
        self.assertIn('./data', text)
        self.assertIn('./out', text)
        self.assertIn('predictions.csv', text)
        self.assertNotIn('submission/submission.csv', text)
        self.assertNotIn('input/', text)
        self.assertEqual(read_json(self.private / 'manifest.json')['policy']['final_choice'], 'final')

    def test_single_preserves_real_status_summary_and_selection(self):
        self.assertEqual(self.launch()['returncode'], 0)
        evidence = self.collect()
        self.assertTrue(evidence['collection']['valid_solution'], evidence['collection']['issues'])
        selected = evidence['collection']['selected_result']
        self.assertEqual((selected['source'], selected['kind'], selected['method']), ('final', 'single', 'single'))
        self.assertIsNone(selected['metric'])
        self.assertIsNone(selected['validation_score'])
        self.assertEqual(selected['node_ids'], ['fixture11-node-1'])
        self.assertEqual(evidence['core_status']['best_score'], .75)
        self.assertNotIn('best_metric', evidence['core_status'])
        self.assertEqual(evidence['core_summary']['best']['score'], .75)
        self.assertEqual(evidence['collection']['final_choice'], 'final')
        self.assertFalse(evidence['provenance']['formal_evaluation_eligible'])
        audit = read_json(self.run / 'core/fixture_audit.json')
        self.assertEqual(audit['real_llm_calls'], 0)
        self.assertEqual(audit['model_training_runs'], 0)
        self.assertEqual(audit['formal_scoring_runs'], 0)
        self.assertIn('final.build_final', audit['writers'])
        self.assertEqual(evidence['core_summary']['resources']['llm']['calls'], 0)

    def test_real_ensemble_selection_and_independent_replay(self):
        self.launch('ensemble')
        source = read_json(self.run / 'core/final/source.json')
        self.assertEqual(source['kind'], 'ensemble', source)
        self.assertEqual(source['validation']['score'], 1.0)
        self.assertEqual(source['validation']['single_full_score'], .75)
        self.assertEqual(len(source['members']), 3)
        evidence = self.collect()
        self.assertTrue(evidence['collection']['valid_solution'], evidence['collection']['issues'])
        selected = evidence['collection']['selected_result']
        self.assertEqual(selected['source'], 'final')
        self.assertEqual(selected['kind'], 'ensemble')
        self.assertEqual(selected['metric'], 'accuracy')
        self.assertEqual(selected['validation_score'], 1.0)
        self.assertEqual(len(selected['node_ids']), 3)
        delivery = self.run / 'delivery'
        replay = self.root / 'replay'
        shutil.copytree(delivery / 'final', replay / 'final')
        original_hash = digest(replay / 'final/predictions.csv')
        for member in source['members']:
            directory = replay / 'final/members' / member['dir']
            expected_hash = digest(directory / 'predictions.csv')
            workspace = replay / ('workspace_' + member['node_id'])
            shutil.copytree(self.task / 'data', workspace / 'data')
            shutil.copyfile(directory / 'solution.py', workspace / 'solution.py')
            result = subprocess.run([sys.executable, str(workspace / 'solution.py')], cwd=workspace,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(digest(workspace / 'out/predictions.csv'), expected_hash)
            shutil.copyfile(workspace / 'out/predictions.csv', directory / 'predictions.csv')
        (replay / 'final/predictions.csv').unlink()
        result = subprocess.run([sys.executable, str(replay / 'final/combine.py')], cwd=replay,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(digest(replay / 'final/predictions.csv'), original_hash)

    def test_partial_is_explicit_best_fallback(self):
        self.assertEqual(self.launch('partial')['returncode'], 1)
        evidence = self.collect()
        self.assertTrue(evidence['collection']['valid_solution'], evidence['collection']['issues'])
        self.assertTrue(evidence['collection']['partial'])
        selected = evidence['collection']['selected_result']
        self.assertEqual(selected['source'], 'best_partial')
        self.assertTrue(selected['reason'])
        self.assertEqual(selected['node_ids'], ['fixture11-node-1'])

    def test_synthetic_final_evaluation_updates_report_lifecycle(self):
        # Synthetic fixture labels only; no course dataset or model is used.
        from superagent_adapter.course.outcome import write_outcome
        self.launch('ensemble')
        evidence = self.collect()
        report = generate_report(evidence, self.run / 'before_evaluation')
        write_outcome(self.run, evidence, report_result=report)
        with offline_environment():
            evidence = evaluate_final(self.task, self.private, self.run)
        self.assertEqual(evidence['evaluation']['test_accuracy'], 1.0)
        self.assertEqual(evidence['evaluation']['execution_kind'], 'fixture')
        self.assertEqual(evidence['evaluation']['final_choice'], 'final')
        self.assertEqual(write_outcome(self.run, evidence)['report']['state'], 'stale')
        with self.assertRaises(ValueError):
            evaluate_final(self.task, self.private, self.run)

    def test_integrated_cli_report_entry(self):
        self.launch()
        self.collect()
        result = subprocess.run([sys.executable, '-m', 'superagent_adapter.cli', 'course', 'report',
                                 '--run-dir', str(self.run), '--output-dir', str(self.run / 'cli_report')],
                                env={**os.environ, 'PYTHONPATH': PYTHONPATH11, 'PYTHONIOENCODING': 'utf-8'},
                                capture_output=True, text=True, encoding='utf-8', timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload['mode'], 'deterministic_no_llm')
        self.assertEqual(payload['outcome']['report']['state'], 'generated')

    def test_changed_ensemble_vote_is_rejected_without_reselection(self):
        self.launch('ensemble')
        path = self.run / 'core/final/predictions.csv'
        with path.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        rows[0]['label'] = 1 - int(rows[0]['label'])
        write_csv(path, fields, rows)
        evidence = self.collect()
        self.assertFalse(evidence['collection']['valid_solution'])
        self.assertTrue(any('recorded member vote' in issue for issue in evidence['collection']['issues']))

    def test_malformed_vote_normalization_cannot_be_delivered(self):
        for value in ('missing', True, 0):
            with self.subTest(normalize=value):
                self.run = self.root / ('normalize_' + str(value))
                self.launch('ensemble')
                path = self.run / 'core/final/source.json'
                source = read_json(path)
                if value == 'missing':
                    del source['format']['normalize']
                else:
                    source['format']['normalize'] = value
                write_json(path, source)
                self.assertFalse(self.collect()['collection']['valid_solution'])

    def test_vote_preserves_cores_raw_string_tie_semantics(self):
        self.launch('ensemble')
        source = read_json(self.run / 'core/final/source.json')
        path = self.run / 'core/final/members' / source['members'][2]['dir'] / 'predictions.csv'
        with path.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        for row in rows:
            row['label'] = row['label'] + '.0'
        write_csv(path, fields, rows)
        result = subprocess.run([sys.executable, str(self.run / 'core/final/combine.py')],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        evidence = self.collect()
        self.assertTrue(evidence['collection']['valid_solution'], evidence['collection']['issues'])

    def test_no_valid_output_stays_unavailable(self):
        self.assertEqual(self.launch('novalid')['returncode'], 2)
        evidence = self.collect()
        self.assertFalse(evidence['collection']['valid_solution'])
        self.assertEqual(evidence['evaluation']['status'], 'unavailable')

    def test_missing_final_on_success_does_not_fallback(self):
        self.launch()
        shutil.rmtree(self.run / 'core/final')
        evidence = self.collect()
        self.assertFalse(evidence['collection']['valid_solution'])

    def test_duplicate_predictions_are_rejected(self):
        self.launch()
        path = self.run / 'core/final/predictions.csv'
        with path.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        rows[-1]['id'] = rows[0]['id']
        write_csv(path, fields, rows)
        self.assertFalse(self.collect()['collection']['valid_solution'])

    def test_wrong_class_predictions_are_rejected(self):
        self.launch()
        path = self.run / 'core/final/predictions.csv'
        with path.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        rows[0]['label'] = '0.4'
        write_csv(path, fields, rows)
        self.assertFalse(self.collect()['collection']['valid_solution'])

    def test_missing_member_script_is_rejected(self):
        self.launch('ensemble')
        next((self.run / 'core/final/members').glob('*/solution.py')).unlink()
        self.assertFalse(self.collect()['collection']['valid_solution'])

    def test_source_member_metadata_mismatch_is_rejected(self):
        self.launch()
        path = self.run / 'core/final/source.json'
        source = read_json(path)
        source['members'][0]['score'] = .9
        write_json(path, source)
        self.assertFalse(self.collect()['collection']['valid_solution'])

    def test_corrupt_final_never_falls_back_even_on_partial_exit(self):
        self.launch()
        status_path = self.run / 'core/status.json'
        status = read_json(status_path)
        status.update(status='failed', final_available=False)
        write_json(status_path, status)
        launch_path = self.run / 'adapter/launch.json'
        launch = read_json(launch_path)
        launch['returncode'] = 1
        write_json(launch_path, launch)
        (self.run / 'core/final/source.json').write_text('{malformed', encoding='utf-8')
        self.assertFalse(self.collect()['collection']['valid_solution'])

    def test_summary_conflict_keeps_selection_excludes_writer_prose(self):
        self.launch('summaryconflict')
        evidence = self.collect()
        self.assertTrue(evidence['collection']['valid_solution'], evidence['collection']['issues'])
        self.assertEqual(evidence['consistency']['status'], 'conflict')
        seen = {}
        def writer(payload):
            seen.update(payload)
            return {'sections': [{'title': '离线夹具', 'text': '已验证选择来自真实 Core 的 final 输出；未训练模型。',
                                  'evidence_refs': ['/collection/selected_result/source']}]}
        with offline_environment():
            report = generate_report(evidence, self.run / 'report', writer)
            outcome = write_outcome(self.run, evidence, report_result=report)
        self.assertEqual(report['mode'], 'writer_human_review_required')
        self.assertEqual(seen['evidence']['core_summary'], {})
        self.assertNotIn('UNTRUSTED_CONFLICTING_FIXTURE_METHOD', json.dumps(seen))
        self.assertTrue((self.run / 'report/report.md').is_file())
        self.assertEqual(outcome['core']['state'], 'success')
        self.assertEqual(outcome['evaluation']['status'], 'not_requested')
        self.assertEqual(outcome['report']['state'], 'generated')
        self.assertEqual(read_json(self.run / 'adapter/outcome.json'), outcome)

    @unittest.skipIf(os.name == 'nt', 'Core artifact symlink behavior verified on Linux; Windows requires privileges')
    def test_allowed_core_model_link_is_materialized(self):
        self.launch()
        model = self.run / 'core/runs/fixture11-node-1/out/model'
        model.mkdir()
        (model / 'fixture.txt').write_text('Synthetic placeholder, not a trained model.', encoding='utf-8')
        member = next((self.run / 'core/final/members').iterdir())
        (member / 'artifacts').symlink_to(os.path.relpath(model, member), target_is_directory=True)
        evidence = self.collect()
        self.assertTrue(evidence['collection']['valid_solution'], evidence['collection']['issues'])
        copied = self.run / 'delivery/final/members' / member.name / 'artifacts/fixture.txt'
        self.assertTrue(copied.is_file())
        self.assertFalse(copied.parent.is_symlink())
        shutil.rmtree(model)
        self.assertIn('placeholder', copied.read_text(encoding='utf-8'))

    @unittest.skipIf(os.name == 'nt', 'Requires unprivileged Linux symlink support')
    def test_outside_artifact_link_is_rejected(self):
        self.launch()
        outside = self.root / 'outside'
        outside.mkdir()
        (outside / 'fixture.txt').write_text('outside', encoding='utf-8')
        member = next((self.run / 'core/final/members').iterdir())
        (member / 'artifacts').symlink_to(outside, target_is_directory=True)
        self.assertFalse(self.collect()['collection']['valid_solution'])


if __name__ == '__main__':
    unittest.main()


def test_collect_core11_accepts_verified_prob_argmax_ensemble(tmp_path):
    import json
    from superagent_adapter.course.back.core11 import collect_core11

    core = tmp_path / "core"
    sample = tmp_path / "sample.csv"
    sample.write_text("id,target\nt0,0\nt1,0\n", encoding="utf-8")
    (core / "best/artifacts").mkdir(parents=True)
    (core / "best/artifacts/m.bin").write_text("w", encoding="utf-8")
    (core / "best/solution.py").write_text("print(1)\n", encoding="utf-8")
    (core / "best/predictions.csv").write_text("id,target\nt0,0\nt1,1\n", encoding="utf-8")
    (core / "best/score.json").write_text(json.dumps({"node_id": "n1", "score": 0.85, "higher_is_better": True}), encoding="utf-8")

    for rank, nid, p_t0, p_t1, pred_t0, sc in [
        (1, "n1", ("0.55", "0.45"), ("0.20", "0.80"), 0, 0.85),
        (2, "n2", ("0.20", "0.80"), ("0.10", "0.90"), 1, 0.84),
    ]:
        run_m = core / f"runs/{nid}/out/model"
        run_m.mkdir(parents=True)
        (run_m / "w.bin").write_text("w", encoding="utf-8")
        mdir = core / f"final/members/rank_{rank:02d}_{nid}"
        mdir.mkdir(parents=True)
        (mdir / "artifacts").symlink_to(f"../../../runs/{nid}/out/model")
        (mdir / "solution.py").write_text("print(1)\n", encoding="utf-8")
        (mdir / "score.json").write_text(json.dumps({"node_id": nid, "rank": rank, "score": sc}), encoding="utf-8")
        (mdir / "predictions.csv").write_text(f"id,target\nt0,{pred_t0}\nt1,1\n", encoding="utf-8")
        (mdir / "test_probabilities.csv").write_text(
            f"id,0,1\nt0,{p_t0[0]},{p_t0[1]}\nt1,{p_t1[0]},{p_t1[1]}\n", encoding="utf-8"
        )
        (mdir / "val_probabilities.csv").write_text("id,0,1\nv0,0.8,0.2\n", encoding="utf-8")

    (core / "final/combine.py").write_text("print(1)\n", encoding="utf-8")
    (core / "final/predictions.csv").write_text("id,target\nt0,1\nt1,1\n", encoding="utf-8")
    source = {
        "available": True,
        "kind": "ensemble",
        "method": "prob_top2",
        "metric": "accuracy",
        "validation": {"score": 0.88},
        "members": [
            {"dir": "rank_01_n1", "rank": 1, "node_id": "n1", "score": 0.85, "weight": 0.5},
            {"dir": "rank_02_n2", "rank": 2, "node_id": "n2", "score": 0.84, "weight": 0.5},
        ],
        "format": {
            "id_col": "id",
            "pred_cols": ["target"],
            "prob_cols": ["0", "1"],
            "copy_cols": [],
            "strategy": "prob_argmax",
            "normalize": False,
            "columns": ["id", "target"],
        },
    }
    (core / "final/source.json").write_text(json.dumps(source), encoding="utf-8")
    status = {
        "best_node_id": "n1",
        "best_score": 0.85,
        "final_available": True,
        "final_kind": "ensemble",
        "final_method": "prob_top2",
        "final_metric": "accuracy",
        "final_validation_score": 0.88,
    }
    dest = tmp_path / "delivery"
    dest.mkdir()
    selected, has_models, mode = collect_core11(core, dest, status, False, sample, "target", [0, 1])
    assert selected["kind"] == "ensemble" and selected["method"] == "prob_top2"
    assert has_models is True
    assert (dest / "final/members/rank_01_n1/test_probabilities.csv").is_file()
