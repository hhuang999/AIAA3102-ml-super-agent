import argparse
import json
from pathlib import Path

from .common import read_json, write_json
from .prepare import prepare_task
from .runner import launch_core, PreflightError
from .collect import collect_result, evaluate_final
from .reporting import generate_report
from .report_llm import make_chat_writer
from .runtime import preflight
from .outcome import blocked_outcome, write_outcome


def report(evidence, directory, writer_config=None):
    writer, setup_error = None, None
    if writer_config:
        try:
            config = read_json(writer_config)
            # Real report calls must go through the team's configured provider/gateway.
            writer = make_chat_writer(base_url=config['base_url'], model=config['model'],
                                      api_key_env=config['api_key_env'], timeout=config.get('timeout',60))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            setup_error = f'报告写作器配置不可用 ({type(exc).__name__})'
    return generate_report(evidence, directory, writer=writer, writer_setup_error=setup_error)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Course Adapter: prepare -> Core -> collect -> final evaluation/report')
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--spec', required=True)
    prep.add_argument('--source-dir', required=True)
    prep.add_argument('--task-dir', required=True)
    prep.add_argument('--private-dir', required=True)
    for name in ('check','run','collect','evaluate-final'):
        p = sub.add_parser(name)
        p.add_argument('--task-dir', required=True)
        p.add_argument('--private-dir', required=True)
        p.add_argument('--run-dir', required=True)
        if name in ('check', 'run'):
            p.add_argument('--core-config')
        if name == 'run':
            p.add_argument('--seed', type=int, default=42)
            p.add_argument('--evaluate-final', action='store_true')
            p.add_argument('--report-writer-config')
    p = sub.add_parser('report')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--writer-config')
    args = parser.parse_args(argv)
    if args.action == 'check':
        result = preflight(args.task_dir,args.private_dir,args.run_dir,config=args.core_config)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0 if result['ready'] else 3
    if args.action == 'prepare':
        result = prepare_task(args.spec,args.source_dir,args.task_dir,args.private_dir)
    elif args.action == 'run':
        try:
            launch_core(args.task_dir,args.private_dir,args.run_dir,config=args.core_config,seed=args.seed)
        except PreflightError as exc:
            print(json.dumps({'preflight':exc.result,'outcome':blocked_outcome(exc.result)},ensure_ascii=False,indent=2))
            return 3
        result = collect_result(args.task_dir,args.private_dir,args.run_dir)
        if args.evaluate_final and result['collection']['valid_solution']:
            result = evaluate_final(args.task_dir,args.private_dir,args.run_dir)
        try:
            report_result = report(result,Path(args.run_dir)/'report',args.report_writer_config)
            result['outcome'] = write_outcome(args.run_dir,result,report_result=report_result)
        except (OSError, ValueError, TypeError) as exc:
            result['outcome'] = write_outcome(args.run_dir,result,report_error=f'报告生成失败 ({type(exc).__name__})')
    elif args.action == 'collect':
        result = collect_result(args.task_dir,args.private_dir,args.run_dir)
        result['outcome'] = write_outcome(args.run_dir,result)
    elif args.action == 'evaluate-final':
        result = evaluate_final(args.task_dir,args.private_dir,args.run_dir)
        result['outcome'] = write_outcome(args.run_dir,result)
    else:
        evidence = read_json(Path(args.run_dir)/'adapter/evidence.json')
        try:
            result = report(evidence,args.output_dir,args.writer_config)
        except (OSError, ValueError, TypeError) as exc:
            outcome = write_outcome(args.run_dir,evidence,report_error=f'报告生成失败 ({type(exc).__name__})')
            print(json.dumps({'outcome':outcome},ensure_ascii=False,indent=2))
            return 4
        result['outcome'] = write_outcome(args.run_dir,evidence,report_result=result.copy())
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if args.action in ('run','collect'):
        if not result['collection']['valid_solution']:
            return 2
        if result['collection']['partial']:
            return 1
        if result['outcome']['report']['state'] == 'failed':
            return 4
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
