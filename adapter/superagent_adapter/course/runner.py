"""Invoke the agreed Core process without coupling to its implementation internals."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .common import digest, write_json
from .front.prepare import verify_inputs
from .runtime import preflight, runtime_snapshot


class PreflightError(ValueError):
    def __init__(self, result):
        super().__init__('Preflight failed; no Core process was started')
        self.result = result


def launch_core(task_dir, private_dir, run_dir, *, config=None, seed=42, command_prefix=None,
                execution_kind='real_core', extra_args=()):
    task, private, run = Path(task_dir).resolve(), Path(private_dir).resolve(), Path(run_dir).resolve()
    if execution_kind not in ('real_core', 'fixture'):
        raise ValueError('Unknown execution kind')
    if command_prefix is not None and execution_kind != 'fixture':
        raise ValueError('Custom commands are for explicitly labeled contract fixtures only')
    permitted = {'--time-limit-secs', '--max-steps', '--num-gpus', '--num-cpus', '--parallel', '--keep-artifacts-top-k'}
    extra_args = list(extra_args)
    if len(extra_args) % 2 or any(extra_args[i] not in permitted for i in range(0, len(extra_args), 2)):
        raise ValueError('Unsupported Core argument override')
    if len(set(extra_args[::2])) != len(extra_args[::2]):
        raise ValueError('Duplicate Core argument override')
    for name, value in zip(extra_args[::2], extra_args[1::2]):
        minimum = 0 if name == '--num-gpus' else 1
        if not str(value).isascii() or not str(value).isdigit() or int(value) < minimum:
            raise ValueError('Resource overrides must be positive integers; num-gpus may be zero')
    check = preflight(task, private, run, config=config, execution_kind=execution_kind, command_prefix=command_prefix)
    if not check['ready']:
        raise PreflightError(check)
    manifest = verify_inputs(task, private)
    command = list(command_prefix) if command_prefix is not None else [sys.executable, '-m', 'superagent.run']
    command += ['--task-dir', str(task), '--output-dir', str(run / 'core'), '--higher-is-better', 'true', '--seed', str(seed)]
    config_hash = None
    if config:
        config = Path(config).resolve(strict=True)
        config_hash = digest(config)
        command += ['--config', str(config)]
    command += [str(a) for a in extra_args]
    snapshot = runtime_snapshot(config=config, execution_kind=execution_kind,
                                command_prefix=command_prefix, core_info=check['core_info'])
    if snapshot['config_sha256'] != config_hash:
        raise ValueError('Core configuration changed during launch preparation')
    (run / 'adapter').mkdir(parents=True)
    snapshot['captured_at'] = datetime.now(timezone.utc).isoformat()
    snapshot['phase'] = 'before_core_launch'
    write_json(run / 'adapter/runtime.json', snapshot)
    write_json(run / 'adapter/preflight.json', check)
    receipt = {'schema_version': '0.1', 'state': 'running', 'execution_kind': execution_kind,
               'command': command, 'task_manifest_sha256': digest(private / 'manifest.json'),
               'config_sha256': config_hash, 'runtime_sha256': digest(run / 'adapter/runtime.json'),
               'final_choice': manifest['policy']['final_choice'],
               'core_interface_version': manifest.get('core_interface_version', '0.1'),
               'started_at': datetime.now(timezone.utc).isoformat(), 'higher_is_better': True,
               'returncode': None, 'interrupted': False, 'formal_evaluation_eligible': False}
    receipt_path = run / 'adapter/launch.json'
    write_json(receipt_path, receipt)
    started, process = time.monotonic(), None
    try:
        with (run / 'adapter/core.stdout.log').open('wb') as out, (run / 'adapter/core.stderr.log').open('wb') as err:
            options = {'start_new_session': True} if os.name != 'nt' else {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
            process = subprocess.Popen(command, cwd=run, stdout=out, stderr=err, stdin=subprocess.DEVNULL, **options)
            receipt['returncode'] = process.wait()
        receipt['state'] = 'exited'
    except KeyboardInterrupt:
        receipt['interrupted'] = True
        receipt['state'] = 'interrupted'
        if process is not None:
            if os.name == 'nt':
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == 'nt':
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            receipt['returncode'] = process.returncode
    except OSError as exc:
        receipt['state'] = 'launch_failed'
        receipt['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        receipt['elapsed_secs'] = time.monotonic() - started
        receipt['finished_at'] = datetime.now(timezone.utc).isoformat()
        write_json(receipt_path, receipt)
    return receipt
