import copy
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from superagent_adapter.course.common import read_csv, read_json, write_csv, write_json
from superagent_adapter.course.prepare import prepare_task, verify_inputs
from superagent_adapter.course.runner import launch_core, PreflightError
from superagent_adapter.course.collect import collect_result, evaluate_final


FIXTURE = Path(__file__).with_name('fixture_core.py')


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.source=self.root/'source';self.source.mkdir()
        self.task=self.root/'task';self.private=self.root/'private';self.run=self.root/'run'
        self.spec={'interface_version':'0.1','task_id':'synthetic','modality':'tabular','input_format':'csv',
                   'feature_columns':['feature'],'target':'label','classes':[0,1],'metric':'accuracy',
                   'objective':'Synthetic classifier fixture only.','validation_policy':'core_default',
                   'final_choice':'best','splits':{s:{'features':s+'.csv'} for s in ('train','val','test')}}
        for split in self.spec['splits']:
            write_csv(self.source/(split+'.csv'),['feature','label'],[{'feature':0,'label':0},{'feature':1,'label':1}])
        self.spec_path=self.root/'spec.json'

    def prepare(self):
        write_json(self.spec_path,self.spec)
        return prepare_task(self.spec_path,self.source,self.task,self.private)

    def launch(self,mode='success'):
        return launch_core(self.task,self.private,self.run,command_prefix=[sys.executable,str(FIXTURE),'--mode',mode],execution_kind='fixture')

    def collect(self):
        return collect_result(self.task,self.private,self.run)

    def test_preparation_hides_test_labels_and_marks_default(self):
        manifest=self.prepare()
        self.assertEqual(read_csv(self.task/'data/test.csv')[0],['id','feature'])
        self.assertIn('label',read_csv(self.task/'data/val.csv')[0])
        self.assertFalse(manifest['policy']['validation']['adapter_resplits'])
        self.assertEqual(manifest['policy']['validation']['status'],'provisional_pending_team_discussion')
        self.assertEqual(read_csv(self.private/'test_labels.csv')[1][1],{'id':'test_00000001','label':'1'})
        self.assertEqual(set(p.name for p in (self.task/'data').iterdir()),{'train.csv','val.csv','test.csv','sample_submission.csv'})
        self.assertFalse(manifest['policy']['test_label_boundary']['os_enforced'])
        self.assertNotIn(b'\r\n', (self.task/'description.md').read_bytes())
        self.assertNotIn(b'\r\n', (self.private/'manifest.json').read_bytes())
        self.assertEqual(list(manifest['public_hashes']), sorted(manifest['public_hashes']))

    def test_private_inside_task_rejected(self):
        write_json(self.spec_path,self.spec)
        with self.assertRaises(ValueError):prepare_task(self.spec_path,self.source,self.task,self.task/'hidden')

    def test_unknown_metric_or_policy_not_silently_changed(self):
        for key,value in [('metric','auc'),('validation_policy','merge_without_record')]:
            other=copy.deepcopy(self.spec);other[key]=value;write_json(self.spec_path,other)
            with self.assertRaises(ValueError):prepare_task(self.spec_path,self.source,self.task,self.private)

    def test_extra_public_file_and_mutated_inputs_rejected(self):
        self.prepare()
        (self.task/'data/extra.csv').write_text('label\n1',encoding='utf-8')
        with self.assertRaises(ValueError):verify_inputs(self.task,self.private)
        (self.task/'data/extra.csv').unlink()
        with (self.task/'data/train.csv').open('a') as f:f.write('changed')
        with self.assertRaises(ValueError):verify_inputs(self.task,self.private)

    def test_private_labels_hash_checked(self):
        self.prepare()
        (self.private/'test_labels.csv').write_text('id,label\ntest_00000000,1',encoding='utf-8')
        with self.assertRaises(ValueError):verify_inputs(self.task,self.private)

    def test_shuffled_predictions_are_scored_by_id(self):
        self.prepare();self.launch();e=self.collect()
        self.assertTrue(e['collection']['valid_solution'])
        self.assertFalse(e['collection']['saved_model_artifacts_present'])
        self.assertEqual(e['evaluation']['status'],'not_requested')
        e=evaluate_final(self.task,self.private,self.run)
        self.assertEqual(e['evaluation']['test_accuracy'],1.)
        self.assertEqual(e['evaluation']['execution_kind'],'fixture')
        self.assertFalse(e['provenance']['formal_evaluation_eligible'])
        with self.assertRaises(ValueError):evaluate_final(self.task,self.private,self.run)

    def test_partial_nonzero_exit_preserves_valid_result(self):
        self.prepare();r=self.launch('partial');e=self.collect()
        self.assertEqual(r['returncode'],1)
        self.assertTrue(e['collection']['valid_solution'])
        self.assertTrue(e['collection']['partial'])

    def test_stale_running_status_is_explicit_partial(self):
        self.prepare();self.launch('stale_running');e=self.collect()
        self.assertTrue(e['collection']['valid_solution'])
        self.assertTrue(e['collection']['partial'])

    def test_no_valid_solution_never_scored(self):
        self.prepare();self.launch('no_valid');e=self.collect()
        self.assertFalse(e['collection']['valid_solution'])
        with self.assertRaises(ValueError):evaluate_final(self.task,self.private,self.run)

    def test_bad_predictions_and_binding_are_rejected(self):
        self.prepare()
        for mode in ('duplicate_id','probability','bad_binding','wrong_direction','unknown_version'):
            with self.subTest(mode=mode):
                self.run=self.root/mode;self.launch(mode);e=self.collect()
                self.assertFalse(e['collection']['valid_solution'])

    def test_choice_frozen_before_launch(self):
        self.prepare();self.launch()
        path=self.private/'manifest.json';m=read_json(path);m['policy']['final_choice']='ensemble';write_json(path,m)
        with self.assertRaises(ValueError):self.collect()

    def test_preselected_ensemble_unavailable_does_not_fallback(self):
        self.spec['final_choice']='ensemble';self.prepare();self.launch();e=self.collect()
        self.assertFalse(e['collection']['valid_solution'])
        self.assertEqual(e['collection']['final_choice'],'ensemble')

    def test_nonobject_status_retains_failure_evidence(self):
        self.prepare()
        for index, value in enumerate(([1], 'invalid', 1, None)):
            self.run=self.root/('bad_status_'+str(index));self.launch()
            write_json(self.run/'core/status.json',value)
            e=self.collect()
            self.assertFalse(e['collection']['valid_solution'])
            self.assertTrue(e['collection']['partial'])
            self.assertTrue((self.run/'adapter/evidence.json').is_file())

    def make_ensemble(self):
        core=self.run/'core'
        status=read_json(core/'status.json');status['ensemble_available']=True;write_json(core/'status.json',status)
        (core/'ensemble').mkdir()
        shutil.copyfile(core/'best_submission/submission.csv',core/'ensemble/submission.csv')
        (core/'ensemble/ensemble.py').write_text('# Synthetic ensemble reconstruction fixture\nprint("fixture")',encoding='utf-8')
        members=[]
        for index in range(2):
            path=core/'top_solution'/('top'+str(index+1));path.mkdir(parents=True)
            for name in ('solution.py','metric.txt'):shutil.copyfile(core/'best_solution'/name,path/name)
            node='member'+str(index)
            (path/'node_id.txt').write_text(node,encoding='utf-8')
            shutil.copyfile(core/'best_submission/submission.csv',path/'submission.csv')
            members.append({'node_id':node,'validation_score':0.75,'weight':0.5})
        write_json(core/'ensemble/members.json',{'members':members,'method':'weighted_vote'})

    def test_complete_ensemble_contract(self):
        self.spec['final_choice']='ensemble';self.prepare();self.launch();self.make_ensemble()
        e=self.collect();self.assertTrue(e['collection']['valid_solution'])
        self.assertEqual(e['collection']['solution_mode'],'full_member_scripts_plus_ensemble')
        self.assertEqual(evaluate_final(self.task,self.private,self.run)['evaluation']['test_accuracy'],1.)

    def test_broken_ensemble_rejected(self):
        self.spec['final_choice']='ensemble';self.prepare()
        for issue in ('bad_json','empty_script','missing_member','wrong_score','bad_weight'):
            self.run=self.root/issue;self.launch();self.make_ensemble();core=self.run/'core'
            if issue=='bad_json':(core/'ensemble/members.json').write_text('not json',encoding='utf-8')
            elif issue=='empty_script':(core/'ensemble/ensemble.py').write_text('',encoding='utf-8')
            elif issue=='missing_member':shutil.rmtree(core/'top_solution/top2')
            elif issue=='wrong_score':(core/'top_solution/top2/metric.txt').write_text('0.5',encoding='utf-8')
            else:
                m=read_json(core/'ensemble/members.json');m['members'][0]['weight']=-1;write_json(core/'ensemble/members.json',m)
            e=self.collect();self.assertFalse(e['collection']['valid_solution'],issue)

    def test_prediction_tamper_after_sealing_rejected(self):
        self.prepare();self.launch();self.collect()
        (self.run/'delivery/predictions.csv').write_text('id,label\nanything,1',encoding='utf-8')
        with self.assertRaises(ValueError):evaluate_final(self.task,self.private,self.run)

    def test_no_reuse_of_output_directories(self):
        self.prepare();self.launch();self.collect()
        with self.assertRaises(ValueError):self.launch()
        with self.assertRaises(ValueError):self.collect()
        with self.assertRaises(ValueError):self.prepare()

    def test_missing_executable_is_stopped_before_launch(self):
        self.prepare()
        with self.assertRaises(PreflightError) as context:
            launch_core(self.task,self.private,self.run,command_prefix=[str(self.root/'absent-python')],execution_kind='fixture')
        self.assertFalse(context.exception.result['ready'])
        self.assertFalse(self.run.exists())

    def test_custom_command_cannot_be_labeled_real(self):
        self.prepare()
        with self.assertRaises(ValueError):launch_core(self.task,self.private,self.run,command_prefix=[sys.executable])

    def test_npy_object_arrays_keep_order_and_missing_values(self):
        import numpy as np
        self.spec.update(input_format='npy',trusted_pickle=True,feature_columns=['first','second'])
        for split in ('train','val','test'):
            np.save(self.source/f'C_{split}.npy',np.array([['one',None],['two','x']],dtype=object))
            np.save(self.source/f'y_{split}.npy',np.array([0,1]))
            self.spec['splits'][split]={'features':f'C_{split}.npy','labels':f'y_{split}.npy'}
        self.prepare()
        header,rows=read_csv(self.task/'data/test.csv')
        self.assertEqual(header,['id','first','second'])
        self.assertEqual(rows[0]['second'],'');self.assertEqual(rows[1]['first'],'two')

    def test_npy_requires_explicit_trust_and_matching_shape(self):
        import numpy as np
        self.spec.update(input_format='npy',trusted_pickle=False)
        for split in ('train','val','test'):
            np.save(self.source/f'C_{split}.npy',np.array([['one'],['two']],dtype=object))
            np.save(self.source/f'y_{split}.npy',np.array([0,1]))
            self.spec['splits'][split]={'features':f'C_{split}.npy','labels':f'y_{split}.npy'}
        with self.assertRaises(ValueError):self.prepare()
        self.spec['trusted_pickle']=True
        np.save(self.source/'y_val.npy',np.array([0]))
        with self.assertRaises(ValueError):self.prepare()

    @unittest.skipIf(os.name=='nt','Symlink creation may require local privilege on Windows')
    def test_core_artifact_symlink_rejected(self):
        self.prepare();self.launch()
        (self.run/'core/best_solution/link').symlink_to(self.private/'test_labels.csv')
        e=self.collect();self.assertFalse(e['collection']['valid_solution'])


if __name__=='__main__':unittest.main()
