"""Interactive course page: drop in a data folder, get back the selected predictions and logs.

    python -m superagent_adapter.cli course ui [--folder DIR] [--yes] [--no-llm]

Steps: scan the folder, write the task spec (with the Core's model), prepare and check,
run the Core, collect and score, then list the files. Only the spec writer and the Core run
call a model; the Core run is the expensive step and is confirmed separately.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import threading

from .back.collect import collect_result, evaluate_final
from .back.outcome import write_outcome
from .back.results import headline, result_files
from .common import read_json, write_json
from .front.llm import ADAPTER_DIR, CoreModelWriter, ensure_core_importable, load_env
from .front.prepare import prepare_task
from .front.scan import data_files, scan_folder
from .front.spec_writer import build_spec, heuristic_draft, llm_draft, spec_to_draft
from .runner import PreflightError, launch_core
from .runtime import preflight

DEFAULT_WORKSPACE = ADAPTER_DIR / 'outputs' / 'course'
STATUS_POLL_SECONDS = 15
RESOURCE_OPTIONS = (
    ('time_limit_secs', '--time-limit-secs', '总时间（秒）', ('run', 'time_limit_seconds')),
    ('max_steps', '--max-steps', '最多方案节点', ('run', 'max_steps')),
    ('num_gpus', '--num-gpus', 'GPU 数', ('resources', 'num_gpus')),
    ('num_cpus', '--num-cpus', 'CPU 核数', ('resources', 'num_cpus')),
    ('parallel', '--parallel', '并行方案数', ('resources', 'parallel')),
)


class Page:
    """Console input and output; tests replace ``ask`` and ``say``."""

    def __init__(self, ask=input, say=print, assume_yes=False):
        self.ask, self.say, self.assume_yes = ask, say, assume_yes

    def step(self, number, title):
        self.say(f'\n[{number}/6] {title}')
        self.say('-' * 48)

    def prompt(self, text, default=''):
        if self.assume_yes:
            return default
        answer = self.ask(f'{text}' + (f' [{default}]' if default != '' else '') + ': ').strip()
        return answer or default

    def confirm(self, text, default=False):
        if self.assume_yes:
            return default
        hint = 'Y/n' if default else 'y/N'
        answer = self.ask(f'{text} [{hint}]: ').strip().lower()
        return default if not answer else answer in ('y', 'yes', '是')


def clean_path(text):
    """Accept a path typed, pasted, or dragged into the terminal."""
    text = text.strip()
    if text.startswith('file://'):
        text = text[len('file://'):]
    try:
        parts = shlex.split(text)
    except ValueError:
        parts = [text.strip('\'"')]
    return Path(' '.join(parts) if len(parts) > 1 else (parts[0] if parts else '')).expanduser()


def core_defaults(config_path):
    ensure_core_importable()
    from superagent.config import build_config, get_path
    cfg = build_config(config_path)
    return {name: get_path(cfg, path) for name, _, _, path in RESOURCE_OPTIONS}


def core_cost_policy(config_path):
    ensure_core_importable()
    from superagent.config import build_config
    cap = build_config(config_path)['control']['cost_cap']
    limit = float(cap['limit'])
    warn_fraction = float(cap['warn_fraction'])
    return {
        'mode': str(cap['mode']),
        'limit': limit,
        'warn_fraction': warn_fraction,
        'warn_limit': round(limit * warn_fraction, 4),
        'stop_on_warning': bool(cap.get('stop_on_warning', False)),
        'currency': str(cap.get('currency', 'CNY')),
    }


def resolve_resources(args, defaults, *, detected_gpus=None, detected_cpus=None):
    """Choose 4 GPUs + maximum dedicated-GPU parallelism by default, honoring explicit CLI overrides."""
    ensure_core_importable()
    from superagent.control.hardware import available_cpus, detect_gpus

    if detected_gpus is None:
        detected_gpus = detect_gpus().count
    if detected_cpus is None:
        detected_cpus = available_cpus()
    detected_gpus = max(0, int(detected_gpus))
    detected_cpus = max(1, int(detected_cpus))

    chosen = {}
    for name in ('time_limit_secs', 'max_steps'):
        given = getattr(args, name, None)
        chosen[name] = int(given) if given is not None else int(defaults[name])

    given_gpus = getattr(args, 'num_gpus', None)
    if given_gpus is not None:
        chosen['num_gpus'] = max(0, int(given_gpus))
    else:
        default_gpus = int(defaults['num_gpus'])
        chosen['num_gpus'] = min(default_gpus, detected_gpus) if detected_gpus > 0 else default_gpus

    given_parallel = getattr(args, 'parallel', None)
    if given_parallel is not None:
        chosen['parallel'] = max(1, int(given_parallel))
    elif chosen['num_gpus'] > 0:
        chosen['parallel'] = max(1, min(chosen['num_gpus'], max(1, detected_cpus // 2)))
    else:
        chosen['parallel'] = max(1, min(int(defaults['parallel']), max(1, detected_cpus // 2)))

    given_cpus = getattr(args, 'num_cpus', None)
    if given_cpus is not None:
        chosen['num_cpus'] = max(chosen['parallel'], int(given_cpus))
    else:
        chosen['num_cpus'] = max(
            chosen['parallel'],
            min(detected_cpus, max(int(defaults['num_cpus']), chosen['parallel'] * 4)),
        )

    extra = []
    for name, flag, _, _ in RESOURCE_OPTIONS:
        if chosen[name] != defaults[name]:
            extra += [flag, str(chosen[name])]
    return extra, chosen


def show_profile(page, profile):
    page.say(f'文件夹：{profile["folder"]}')
    for f in profile['files']:
        detail = ''
        if f['kind'] == 'csv':
            detail = f'{f.get("rows", 0)} 行，列 {f.get("columns")}'
        elif f['kind'] == 'npy':
            detail = f'形状 {f.get("shape", "?")}，{f.get("dtype", "对象数组")}'
        page.say(f'  {f["name"]:<28} {f["kind"]:<5} {f["split"] or "-":<5} {detail}')
    if not data_files(profile):
        page.say('  没有找到 CSV 或 NPY 数据文件。')


def write_spec_draft(page, profile, args, job):
    if args.no_llm:
        page.say('按文件名和列名规则生成规格（不调用模型）。')
        return heuristic_draft(profile), None
    try:
        writer = CoreModelWriter(job / 'front' / 'logs' / 'llm_calls.jsonl', args.core_config)
    except Exception as exc:  # noqa: BLE001 - any setup failure falls back to the rule draft
        page.say(f'模型不可用，改用规则草稿：{exc}')
        return heuristic_draft(profile), None
    page.say(f'调用 {writer.model}（{writer.base_url}）生成规格……')
    try:
        draft = llm_draft(profile, writer)
    except Exception as exc:  # noqa: BLE001
        page.say(f'模型生成失败，改用规则草稿：{type(exc).__name__}: {exc}')
        draft = heuristic_draft(profile)
    totals = writer.totals
    page.say(f'模型调用 {int(totals["calls"])} 次，费用约 {totals["cost"]:.4f} 元。')
    return draft, writer


def edit_file(page, path):
    editor = os.environ.get('EDITOR') or next((e for e in ('nano', 'vim', 'vi') if shutil.which(e)), None)
    if editor and not page.assume_yes:
        subprocess.run([*shlex.split(editor), str(path)], check=False)
    else:
        page.prompt(f'请编辑 {path}，保存后按回车继续')


def settle_spec(page, profile, args, job):
    """Loop until the user accepts a spec without problems. Returns the spec, or None to quit."""
    spec_path = job / 'spec.json'
    draft, _ = write_spec_draft(page, profile, args, job)
    while True:
        spec, problems = build_spec(draft, profile)
        if spec is not None:
            write_json(spec_path, spec)
            page.say(json.dumps(spec, ensure_ascii=False, indent=2))
            page.say(f'规格已写入：{spec_path}')
        for p in problems:
            page.say(f'  问题：{p}')
        if not problems:
            if page.assume_yes:
                return spec
            choice = page.prompt('回车接受；e 编辑；r 重新生成；h 改用规则草稿；q 退出', '').lower()
        else:
            if page.assume_yes:
                return None
            choice = page.prompt('规格有问题。e 编辑；r 重新生成；h 改用规则草稿；q 退出', 'e').lower()
        if choice == '' and not problems:
            return spec
        if choice == 'q':
            return None
        if choice == 'e':
            if spec is None:
                write_json(spec_path, draft)
            edit_file(page, spec_path)
            try:
                draft = spec_to_draft(read_json(spec_path))
            except (OSError, ValueError) as exc:
                page.say(f'无法读取编辑后的 JSON：{exc}')
        elif choice == 'r':
            draft, _ = write_spec_draft(page, profile, args, job)
        elif choice == 'h':
            draft = heuristic_draft(profile)


def ask_resources(page, args, defaults, *, base_defaults=None):
    baseline = defaults if base_defaults is None else base_defaults
    extra, chosen = [], {}
    for name, flag, label, _ in RESOURCE_OPTIONS:
        given = getattr(args, name, None)
        value = given if given is not None else page.prompt(f'{label}', str(defaults[name]))
        value = str(value).strip()
        minimum = 0 if name == 'num_gpus' else 1
        if not value.isdigit() or int(value) < minimum:
            page.say(f'{label} 需要是不小于 {minimum} 的整数，改用默认值 {defaults[name]}。')
            value = str(defaults[name])
        chosen[name] = int(value)
        if int(value) != baseline[name]:
            extra += [flag, value]
    return extra, chosen


def _live_cost(calls_path):
    try:
        lines = calls_path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return None
    total = 0.0
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        total += float(row.get('cost') or 0.0)
    return total


def _new_cost_notices(control_path, seen):
    try:
        lines = control_path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return []
    notices = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        event = row.get('event')
        if event in ('cost_warning', 'cost_limit') and event not in seen:
            seen.add(event)
            cost = float(row.get('cost') or 0.0)
            warn_limit = float(row.get('warn_limit') or 0.0)
            limit = float(row.get('limit') or 0.0)
            if row.get('finished'):
                notices.append(
                    f'  [费用预警] 已达 ¥{cost:.2f}（预警线 ¥{warn_limit:g} / 上限 ¥{limit:g}），'
                    '已停止开启新分支，等待运行中方案完成后收尾……'
                )
            else:
                notices.append(
                    f'  [费用提示] 当前已达 ¥{cost:.2f}（预警线 ¥{warn_limit:g} / 上限 ¥{limit:g}）。'
                )
    return notices


def watch_status(page, status_path, stop):
    last = None
    seen_cost_events = set()
    logs_dir = status_path.parent / 'logs'
    calls_path = logs_dir / 'llm_calls.jsonl'
    control_path = logs_dir / 'control.jsonl'
    while not stop.wait(STATUS_POLL_SECONDS):
        for notice in _new_cost_notices(control_path, seen_cost_events):
            page.say(notice)
        try:
            status = read_json(status_path)
        except (OSError, ValueError):
            continue
        view = (status.get('status'), status.get('num_nodes'), status.get('num_ok_nodes'), status.get('best_score'))
        if view != last:
            last = view
            cost = _live_cost(calls_path)
            cost_note = f'，累计费用 ¥{cost:.2f}' if cost else ''
            page.say(f'  已运行 {status.get("elapsed_secs", 0):.0f} 秒：方案 {view[1] or 0} 个，'
                     f'成功 {view[2] or 0} 个，最佳验证分 {view[3]}{cost_note}')


def run_core(page, task, private, run, args, extra, launcher):
    stop = threading.Event()
    watcher = threading.Thread(target=watch_status, args=(page, run / 'core' / 'status.json', stop), daemon=True)
    watcher.start()
    try:
        return launcher(task, private, run, config=args.core_config, seed=args.seed, extra_args=extra)
    finally:
        stop.set()
        watcher.join(timeout=1)


def show_results(page, run, outcome):
    head = headline(outcome)
    page.say(head['summary'] or '')
    if head['test_accuracy'] is not None:
        page.say(f'测试集 accuracy：{head["test_accuracy"]:.4f}（{head["correct"]}/{head["rows"]}）')
    for issue in head['issues']:
        page.say(f'  提示：{issue}')
    page.say('产出文件：')
    for label, path in result_files(run, outcome):
        page.say(f'  {label:<16} {path}')


def build_parser():
    p = argparse.ArgumentParser(prog='superagent_adapter course ui', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--folder', help='数据文件夹；不提供时在页面里拖入或输入')
    p.add_argument('--workspace', default=str(DEFAULT_WORKSPACE), help='保存规格、任务目录和运行结果的位置')
    p.add_argument('--core-config', help='Core 配置文件；前端模型与 Core 运行共用')
    p.add_argument('--no-llm', action='store_true', help='只用规则生成规格')
    p.add_argument('--yes', action='store_true', help='全部采用默认值，不再询问')
    p.add_argument('--no-evaluate', action='store_true', help='跑完后不计算测试集 accuracy')
    p.add_argument('--trust-pickle', action='store_true', help='允许读取对象类型的 NPY')
    p.add_argument('--seed', type=int, default=42)
    for name, flag, label, _ in RESOURCE_OPTIONS:
        p.add_argument(flag, dest=name, type=int, help=label)
    return p


def main(argv=None, *, ask=input, say=print, launcher=launch_core):
    args = build_parser().parse_args(argv)
    page = Page(ask, say, args.yes)
    load_env()
    if args.core_config:
        args.core_config = str(Path(args.core_config).expanduser().resolve(strict=True))

    page.step(1, '选择数据文件夹')
    folder = clean_path(args.folder) if args.folder else None
    while folder is None or not folder.is_dir():
        if folder is not None:
            page.say(f'不是可用的文件夹：{folder}')
        if args.yes:
            return 3
        folder = clean_path(page.ask('把数据文件夹拖进来，或输入路径，然后回车: '))
    folder = folder.resolve()

    page.step(2, '扫描文件')
    profile = scan_folder(folder, trusted_pickle=args.trust_pickle)
    if any(f.get('needs_pickle') and 'shape' not in f for f in data_files(profile, 'npy')):
        if args.trust_pickle or page.confirm('NPY 是对象数组，读取需要 allow_pickle。确认这是你信任的数据？', True):
            profile = scan_folder(folder, trusted_pickle=True)
    show_profile(page, profile)

    workspace = Path(args.workspace).expanduser().resolve()
    job = workspace / profile['name'] / datetime.now().strftime('%Y%m%d-%H%M%S')
    if job.is_relative_to(folder) or folder.is_relative_to(job):
        page.say('输出位置不能放在数据文件夹里面。请用 --workspace 换一个位置。')
        return 3
    job.mkdir(parents=True)
    page.say(f'本次工作目录：{job}')

    page.step(3, '生成任务规格 JSON')
    spec = settle_spec(page, profile, args, job)
    if spec is None:
        page.say('已退出，没有启动 Core。')
        return 3
    task, private, run = job / 'task', job / 'private', job / 'run'

    page.step(4, '准备任务目录并预检')
    try:
        manifest = prepare_task(job / 'spec.json', folder, task, private)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        page.say(f'准备失败：{exc}')
        return 3
    page.say(f'任务目录：{task}（{manifest["task"]["rows"]}）')
    page.say(f'测试标签已封存：{private / "test_labels.csv"}')
    check = preflight(task, private, run, config=args.core_config)
    for c in check['checks']:
        page.say(f'  {"通过" if c["ok"] else "未通过"}  {c["name"]}: {c["message"]}')
    if not check['ready']:
        page.say('预检没有通过，Core 没有启动。')
        return 3

    page.step(5, '运行 AutoML（会调用模型并产生费用）')
    defaults = core_defaults(args.core_config)
    extra, chosen = resolve_resources(args, defaults)
    cost_policy = core_cost_policy(args.core_config)
    page.say('默认资源：' + '，'.join(f'{label} {chosen[name]}' for name, _, label, _ in RESOURCE_OPTIONS))
    if cost_policy['mode'] == 'enforce' and cost_policy['limit'] > 0:
        if cost_policy['stop_on_warning']:
            page.say(
                f'预算策略：达到 ¥{cost_policy["warn_limit"]:g} 预警时停止开启新分支并收尾'
                f'（硬上限 ¥{cost_policy["limit"]:g} {cost_policy["currency"]}）。'
            )
        else:
            page.say(
                f'预算策略：预警线 ¥{cost_policy["warn_limit"]:g}，达到上限 ¥{cost_policy["limit"]:g} {cost_policy["currency"]} 停止。'
            )
    if not page.assume_yes:
        choice = page.prompt('确认开始运行？（回车/y 直接启动，c 手动修改资源，n 取消）', 'y').lower()
        if choice == 'c':
            extra, chosen = ask_resources(page, args, chosen, base_defaults=defaults)
            page.say('已更新设置：' + '，'.join(f'{label} {chosen[name]}' for name, _, label, _ in RESOURCE_OPTIONS))
            if not page.confirm('确认开始运行？', True):
                page.say(f'没有启动。之后可以运行：python -m superagent_adapter.cli course run '
                         f'--task-dir {task} --private-dir {private} --run-dir {run}')
                return 0
        elif choice not in ('y', 'yes', '是'):
            page.say(f'没有启动。之后可以运行：python -m superagent_adapter.cli course run '
                     f'--task-dir {task} --private-dir {private} --run-dir {run}')
            return 0
    try:
        receipt = run_core(page, task, private, run, args, extra, launcher)
    except PreflightError as exc:
        page.say(f'启动前检查失败：{exc}')
        return 3
    page.say(f'Core 已结束，返回码 {receipt.get("returncode")}，用时 {receipt.get("elapsed_secs", 0):.0f} 秒。')

    page.step(6, '收集结果')
    evidence = collect_result(task, private, run)
    if evidence['collection']['valid_solution'] and not args.no_evaluate:
        if page.confirm('用封存的测试标签计算 accuracy？', True):
            evidence = evaluate_final(task, private, run)
    outcome = write_outcome(run, evidence)
    show_results(page, run, outcome)
    if not evidence['collection']['valid_solution']:
        return 2
    return 1 if evidence['collection']['partial'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
