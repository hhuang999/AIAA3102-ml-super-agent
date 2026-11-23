"""Contract simulator only. No LLM calls and no model training."""
import argparse
import csv
import json
from pathlib import Path


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--higher-is-better',required=True)
    parser.add_argument('--seed')
    parser.add_argument('--mode',default='success')
    parser.add_argument('--config')
    args=parser.parse_args()
    assert args.higher_is_better=='true'
    task,out=Path(args.task_dir),Path(args.output_dir)
    out.mkdir()
    with (task/'data/sample_submission.csv').open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f); columns=reader.fieldnames; sample=list(reader)
    target=columns[1]
    with (task/'data/test.csv').open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f); feature=reader.fieldnames[1]; tests=list(reader)
    assert target not in tests[0]
    assert not any('test_labels' in p.name or p.suffix=='.zip' for p in task.rglob('*'))
    status={'interface_version':'0.1','status':'success','stop_reason':'max_steps','higher_is_better':True,
            'metric_direction_source':'cli','best_metric':0.75,'best_node_id':'fixture-node',
            'best_has_artifacts':False,'ensemble_available':False,'num_nodes':1,'num_valid_nodes':1,'num_branches':1}
    if args.mode=='no_valid':
        status.update(status='no_valid_solution',best_metric=None,best_node_id=None,num_valid_nodes=0)
    else:
        best=out/'best_solution';best.mkdir()
        (best/'node_id.txt').write_text('fixture-node',encoding='utf-8')
        (best/'metric.txt').write_text('0.75',encoding='utf-8')
        script="""# Synthetic contract fixture; no trained model.
import csv
from pathlib import Path
with open('input/test.csv',newline='',encoding='utf-8') as f:
    reader=csv.DictReader(f); feature=reader.fieldnames[1]; rows=list(reader)
with open('input/sample_submission.csv',newline='',encoding='utf-8') as f:
    fields=next(csv.reader(f))
Path('submission').mkdir(exist_ok=True)
with open('submission/submission.csv','w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    writer.writerows({'id':r['id'],fields[1]:int(float(r[feature]))} for r in reversed(rows))
print('Final Validation Score: 0.75')
"""
        (best/'solution.py').write_text(script,encoding='utf-8')
        predictions=[{'id':r['id'],target:int(float(r[feature]))} for r in reversed(tests)]
        if args.mode=='duplicate_id':predictions[-1]['id']=predictions[0]['id']
        if args.mode=='probability':predictions[0][target]=0.4
        if args.mode=='bad_binding':(best/'node_id.txt').write_text('other-node',encoding='utf-8')
        (out/'best_submission').mkdir()
        with (out/'best_submission/submission.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(predictions)
    if args.mode=='partial':status.update(status='failed',stop_reason='error')
    if args.mode=='stale_running':status.update(status='running',stop_reason='interrupted')
    if args.mode=='unknown_version':status['interface_version']='99'
    if args.mode=='wrong_direction':status['higher_is_better']=False
    write(out/'status.json',status)
    write(out/'report/run_summary.json',{'task':{'metric':'accuracy'},'run':{'num_nodes':1},
        'best':{'node_id':'fixture-node','metric':0.75,'method_summary':'Synthetic fixture based on one feature; no training performed.'},
        'key_events':[],'failures':[],'resources':{'llm_calls':0,'prompt_tokens':0,'completion_tokens':0}})
    write(out/'logs/journal.json',[{'id':'fixture-node','stage':'draft','metric':0.75}])
    (out/'logs/decisions.jsonl').write_text('{}\n',encoding='utf-8')
    (out/'logs/llm_calls.jsonl').write_text('',encoding='utf-8')
    return 2 if args.mode=='no_valid' else 1 if args.mode in ('partial','stale_running') else 0


if __name__=='__main__':raise SystemExit(main())
