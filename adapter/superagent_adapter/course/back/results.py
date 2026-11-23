"""List the files a finished run hands back, for display after the Core stops."""
from __future__ import annotations

from pathlib import Path

CORE_FILES = (
    ('Core 运行日志', 'logs/superagent.log'),
    ('Core 模型调用记录', 'logs/llm_calls.jsonl'),
    ('Core 决策记录', 'logs/decisions.jsonl'),
    ('Core 用量账单', 'logs/api_usage.html'),
    ('Core 运行摘要', 'report/run_summary.md'),
)


def result_files(run_dir, outcome):
    """Return ``[(label, path)]`` for files that exist, the selected predictions first."""
    run = Path(run_dir).resolve()
    files = outcome.get('files') or {}
    rows = [('最终预测 CSV', files.get('predictions_csv')),
            ('最终选择说明', files.get('selection_json')),
            ('组合脚本', files.get('combine_script'))]
    rows += [('方案脚本', p) for p in files.get('solution_scripts') or []]
    rows += [('模型文件目录', p) for p in files.get('model_artifact_dirs') or []]
    rows += [('测试集评分', files.get('final_evaluation_json')),
             ('交接记录 outcome.json', str(run / 'adapter/outcome.json')),
             ('Core 标准输出', files.get('stdout_log')),
             ('Core 错误输出', files.get('stderr_log'))]
    for base in (run / 'delivery', run / 'core'):
        prefix = 'core_' if base.name == 'delivery' else ''
        for label, rel in CORE_FILES:
            path = base / (prefix + rel)
            if path.is_file() and label not in {r[0] for r in rows if r[1]}:
                rows.append((label, str(path)))
    return [(label, path) for label, path in rows if path and Path(path).exists()]


def headline(outcome):
    """The short facts to print above the file list."""
    evaluation = outcome.get('evaluation') or {}
    accuracy = evaluation.get('test_accuracy')
    return {
        'summary': outcome.get('summary'),
        'core_state': (outcome.get('core') or {}).get('state'),
        'test_accuracy': accuracy,
        'correct': evaluation.get('correct'),
        'rows': evaluation.get('rows'),
        'issues': outcome.get('issues') or [],
    }
