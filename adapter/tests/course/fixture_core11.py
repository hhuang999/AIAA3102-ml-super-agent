"""Offline integration fixture built by the pinned Core's real output writers.

This never executes search, trains models, or instantiates an LLM gateway. Validation
scores concern hand-constructed synthetic arrays only, not a course benchmark.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import socket
import time
from types import SimpleNamespace


def deny_network(*args, **kwargs):
    raise AssertionError('Offline Core 1.1 fixture attempted network access')


def install_network_guard():
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.create_connection = deny_network


def csv_write(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def member_script(member):
    # Each member is a complete, deterministic synthetic prediction program.
    return '''"""Synthetic fixture only: no model training or external services."""
import csv
from pathlib import Path
with Path('data/test.csv').open(encoding='utf-8', newline='') as stream:
    reader = csv.DictReader(stream)
    feature = reader.fieldnames[1]
    rows = list(reader)
with Path('data/sample_predictions.csv').open(encoding='utf-8', newline='') as stream:
    columns = next(csv.reader(stream))
Path('out').mkdir(exist_ok=True)
with Path('out/predictions.csv').open('w', encoding='utf-8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows({'id': row['id'], columns[1]: int(float(row[feature])) ^ (index % 4 == MEMBER)}
                     for index, row in enumerate(rows))
'''.replace('MEMBER', str(member))


def build(task, output, mode='single', seed=42):
    install_network_guard()
    # Intentionally import only real bookkeeping/final/report paths, never run/search.
    from superagent.bookkeeping import write_best, write_search_tree, write_status
    from superagent.config import load_defaults
    from superagent.context import LockedRandom, OutputPaths, RunContext
    from superagent.final import Candidate, build_final, half_of
    from superagent.report import build_summary
    from superagent.tree import Direction, Node, SearchTree
    from superagent.utils import atomic_write_json, atomic_write_text, now_iso

    task, output = Path(task).resolve(), Path(output).resolve()
    paths = OutputPaths(output)
    paths.make_dirs()
    cfg = load_defaults()
    cfg['run'].update(seed=seed, max_steps=3, time_limit_seconds=60)
    cfg['pretrained']['mode'] = 'off'
    cfg['resources'].update(parallel=1, num_gpus=0)
    direction = Direction(True)
    tree = SearchTree(direction)
    ctx = RunContext(cfg=cfg, paths=paths, task_dir=task, direction=direction,
                     direction_source='cli', tree=tree, gateway=None,
                     executor=SimpleNamespace(run_dir=lambda node_id: paths.runs / node_id),
                     experience=None, task_desc=(task / 'description.md').read_text(encoding='utf-8'),
                     data_preview='Synthetic fixture; no real dataset preview.', catalog_text='',
                     modalities=['tabular'], sample_path=task / 'data/sample_predictions.csv',
                     rng=LockedRandom(seed), process_start=time.monotonic(), started_at=now_iso())
    ctx.stop_reason = 'offline_fixture_complete'
    with ctx.sample_path.open(encoding='utf-8', newline='') as stream:
        columns = next(csv.reader(stream))
    with (task / 'data/test.csv').open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        feature, tests = reader.fieldnames[1], list(reader)
    assert columns[1] not in tests[0], 'Final labels must remain absent from Core inputs'
    halves = [[], []]
    for i in range(1000):
        sample = f'synthetic_val_{i:04d}'
        half = half_of(sample)
        if len(halves[half]) < 12:
            halves[half].append(sample)
        if all(len(part) == 12 for part in halves):
            break
    candidates = []
    count = 0 if mode in ('novalid', 'no_valid') else 3 if mode in ('ensemble', 'summaryconflict') else 1
    for index in range(count):
        node_id = f'fixture11-node-{index + 1}'
        out = paths.runs / node_id / 'out'
        out.mkdir(parents=True)
        predictions = [{'id': row['id'], columns[1]: int(float(row[feature])) ^ (j % 4 == index)}
                       for j, row in enumerate(tests)]
        csv_write(out / 'predictions.csv', columns, predictions)
        labels, val_predictions = [], []
        for part in halves:
            for j, sample in enumerate(part):
                label = j % 2
                labels.append({'id': sample, columns[1]: label})
                val_predictions.append({'id': sample, columns[1]: label ^ (j % 4 == index)})
        csv_write(out / 'val_labels.csv', columns, labels)
        csv_write(out / 'val_predictions.csv', columns, val_predictions)
        node = Node(id=node_id, kind='draft', branch_id=index + 1, score=.75,
                    failed=False, predictions_ok=True, val_files_ok=True, exec_seconds=0.0,
                    code=member_script(index), plan='Hand-constructed synthetic fixture; no training.',
                    method_summary='Synthetic parity prediction and deterministic label flips; no training.',
                    score_source='synthetic_fixture', finished_at=now_iso())
        tree.attach(node, tree.root)
        tree.append(node)
        tree.register_ok(node)
        candidates.append(Candidate(index + 1, node_id, .75, out, node.code))
    if candidates:
        tree.best = tree.nodes[1]
        tree.top_list = tree.ok_nodes()
        write_best(ctx, tree.best)
    final = None
    if mode != 'partial' and candidates:
        final_cfg = dict(cfg['final'], min_validation_rows=12, small_validation_rows=12)
        final = build_final(output, candidates, True, final_cfg)
    state = 'no_valid_solution' if not candidates else 'failed' if mode == 'partial' else 'success'
    write_status(ctx, state, final=final)
    write_search_tree(ctx)
    summary = build_summary(ctx, final=final)
    if mode == 'summaryconflict':
        summary['best'].update(node_id='WRONG-NODE', score=.99,
                               method_summary='UNTRUSTED_CONFLICTING_FIXTURE_METHOD')
    atomic_write_json(paths.report / 'run_summary.json', summary)
    atomic_write_text(paths.llm_calls, '')
    atomic_write_text(paths.decisions, '')
    atomic_write_json(output / 'fixture_audit.json', {
        'execution_kind': 'fixture', 'mode': mode, 'real_llm_calls': 0, 'model_training_runs': 0,
        'formal_scoring_runs': 0, 'network_guard': True,
        'writers': ['bookkeeping.write_best', 'bookkeeping.write_status',
                    'bookkeeping.write_search_tree', 'final.build_final', 'report.build_summary'],
    })
    return 2 if not candidates else 1 if mode == 'partial' else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--higher-is-better', required=True, choices=['true'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config')  # Accepted for the launcher contract; no runtime starts.
    parser.add_argument('--mode', default='single',
                        choices=['single', 'ensemble', 'partial', 'novalid', 'no_valid', 'summaryconflict'])
    args = parser.parse_args()
    return build(args.task_dir, args.output_dir, args.mode, args.seed)


if __name__ == '__main__':
    raise SystemExit(main())
