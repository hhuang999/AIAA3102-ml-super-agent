"""Every catalog example must at least parse; pretrained examples must run offline when downloaded."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from superagent.catalog import load_catalog, load_manifest, offline_env

LOCAL = Path(os.environ.get("SUPERAGENT_PRETRAIN_DIR", "pretrained_cache")).resolve()


@pytest.mark.parametrize("entry", load_catalog(), ids=lambda e: e["id"])
def test_example_parses(entry):
    ast.parse(entry["example"])


@pytest.mark.slow
@pytest.mark.parametrize("entry", [e for e in load_catalog() if e["pretrained"]], ids=lambda e: e["id"])
def test_pretrained_example_runs_offline(entry, tmp_path):
    manifest = load_manifest(LOCAL) or {}
    if not manifest.get(entry["id"], {}).get("ok"):
        pytest.skip(f"{entry['id']} is not downloaded in {LOCAL}")
    env = {k: v for k, v in os.environ.items() if not k.startswith("HF_")}
    env.update(offline_env("catalog", LOCAL))
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "")
    script = tmp_path / "example.py"
    script.write_text(entry["example"])
    r = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
