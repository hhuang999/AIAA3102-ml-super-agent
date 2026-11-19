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


def pytest_configure(config):
    config.addinivalue_line("markers", f"llm: test calls the real LLM endpoint (needs {KEY_ENV}, costs a little)")
    config.addinivalue_line("markers", "slow: test takes more than a few seconds")


def pytest_collection_modifyitems(config, items):
    if not os.environ.get(KEY_ENV):
        skip = pytest.mark.skip(reason=f"{KEY_ENV} is not set")
        for item in items:
            if "llm" in item.keywords:
                item.add_marker(skip)
