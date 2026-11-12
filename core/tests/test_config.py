from pathlib import Path

import pytest
import yaml

from superagent.config import build_config, dump_config, load_defaults
from superagent.errors import ConfigError


def write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_defaults_load():
    cfg = load_defaults()
    assert cfg["run"]["max_steps"] == 500
    assert cfg["llm"]["generation"]["model"] == "deepseek-flash"


def test_partial_override_merges(tmp_path):
    cfg = build_config(write(tmp_path, {"search": {"max_drafts": 7}}), environ={})
    assert cfg["search"]["max_drafts"] == 7
    assert cfg["search"]["initial_drafts"] == 3


def test_unknown_key(tmp_path):
    with pytest.raises(ConfigError, match="search.no_such_key"):
        build_config(write(tmp_path, {"search": {"no_such_key": 1}}), environ={})


def test_wrong_type(tmp_path):
    with pytest.raises(ConfigError, match="run.max_steps"):
        build_config(write(tmp_path, {"run": {"max_steps": "many"}}), environ={})
    with pytest.raises(ConfigError, match="plateau.enabled"):
        build_config(write(tmp_path, {"plateau": {"enabled": "yes"}}), environ={})


def test_int_accepted_for_float(tmp_path):
    cfg = build_config(write(tmp_path, {"search": {"min_gain": 0}}), environ={})
    assert cfg["search"]["min_gain"] == 0.0


def test_enum(tmp_path):
    with pytest.raises(ConfigError, match="pretrained.mode"):
        build_config(write(tmp_path, {"pretrained": {"mode": "maybe"}}), environ={})


def test_precedence_cli_env(tmp_path):
    cfg = build_config(
        write(tmp_path, {"run": {"max_steps": 10}, "llm": {"base_url": "http://file"}}),
        cli_overrides={"max_steps": 20, "pretrained_mode": "off"},
        environ={"SUPERAGENT_LLM_BASE_URL": "http://env", "SUPERAGENT_PRETRAIN_DIR": "/tmp/pt"},
    )
    assert cfg["run"]["max_steps"] == 20
    assert cfg["pretrained"]["mode"] == "off"
    assert cfg["llm"]["base_url"] == "http://env"
    assert cfg["pretrained"]["local_dir"] == "/tmp/pt"


def test_relative_path_resolved(tmp_path):
    cfg = build_config(None, environ={}, cwd=tmp_path)
    assert cfg["pretrained"]["local_dir"] == str((tmp_path / "pretrained_cache").resolve())


def test_dump_has_no_key(tmp_path):
    cfg = build_config(None, environ={"DEEPSEEK_API_KEY": "sk-secret-value"})
    text = dump_config(cfg)
    assert "sk-secret-value" not in text
    assert cfg["llm"]["api_key_env"] == "DEEPSEEK_API_KEY"  # only the variable name is stored


def test_embedding_backend_enum(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("experience:\n  vector:\n    backend: remote\n")
    with pytest.raises(ConfigError):
        build_config(f, environ={})
    f.write_text("experience:\n  vector:\n    backend: api\n")
    assert build_config(f, environ={})["experience"]["vector"]["backend"] == "api"
