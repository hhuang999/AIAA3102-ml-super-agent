"""Use the Core's model settings for the front end's own calls.

Endpoint, model, key variable and retries come from the Core configuration (defaults plus the
optional ``--core-config`` file). Keys come from the environment or the repository ``.env`` files.
Calls use the Core's ``judgement`` group and are logged to their own ``llm_calls.jsonl``.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[4]
CORE_DIR = REPO / 'core'
ADAPTER_DIR = REPO / 'adapter'
ENV_FILES = (REPO / '.env', CORE_DIR / '.env')
ROLE = 'course_spec_writer'


def ensure_core_importable():
    """Make ``superagent`` importable here and discoverable by the Core child process."""
    for directory in (CORE_DIR, ADAPTER_DIR):
        if str(directory) not in sys.path:
            sys.path.append(str(directory))
    entries = [e for e in os.environ.get('PYTHONPATH', '').split(os.pathsep) if e]
    for directory in (str(CORE_DIR), str(ADAPTER_DIR)):
        if directory not in entries:
            entries.append(directory)
    os.environ['PYTHONPATH'] = os.pathsep.join(entries)


def load_env():
    """Read the repository ``.env`` files; existing variables keep their value. Returns names set."""
    ensure_core_importable()
    from superagent.utils import load_env_file
    names = []
    for path in ENV_FILES:
        names += load_env_file(path)
    return names


def core_llm_config(config_path=None):
    ensure_core_importable()
    from superagent.config import build_config
    return build_config(config_path)['llm']


class CoreModelWriter:
    """Callable ``(system, user, schema) -> dict`` backed by the Core gateway."""

    def __init__(self, calls_log_path, config_path=None):
        load_env()
        from superagent.llm.gateway import LLMGateway, OpenAITransport

        llm = core_llm_config(config_path)
        key = os.environ.get(str(llm['api_key_env']), '')
        if not key:
            raise RuntimeError(f'环境变量 {llm["api_key_env"]} 没有设置（也可写在 .env 里）。')

        class _JudgementGateway(LLMGateway):
            def _group(self, role):
                return self.cfg['judgement']

        transport = OpenAITransport(llm['base_url'], key, float(llm.get('request_timeout_seconds', 1200)))
        self.gateway = _JudgementGateway(llm, calls_log_path, transport)
        self.model = llm['judgement']['model']
        self.base_url = llm['base_url']
        self.calls_log_path = Path(calls_log_path)

    def __call__(self, system, user, schema):
        return self.gateway.structured(ROLE, system, user, schema)

    @property
    def totals(self):
        return dict(self.gateway.totals)
