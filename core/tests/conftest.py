import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from superagent.utils import load_env_file  # noqa: E402

load_env_file(ROOT / ".env")
KEY_ENV = "DEEPSEEK_API_KEY"
EMBED_KEY_ENV = "AIHUBMIX_API_KEY"


def pytest_configure(config):
    config.addinivalue_line("markers", f"llm: test calls the real LLM endpoint (needs {KEY_ENV}, costs a little)")
    config.addinivalue_line("markers", f"embed_api: test calls the real embeddings endpoint (needs {EMBED_KEY_ENV})")
    config.addinivalue_line("markers", "slow: test takes more than a few seconds")


def pytest_collection_modifyitems(config, items):
    for marker, env in (("llm", KEY_ENV), ("embed_api", EMBED_KEY_ENV)):
        if os.environ.get(env):
            continue
        skip = pytest.mark.skip(reason=f"{env} is not set")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
