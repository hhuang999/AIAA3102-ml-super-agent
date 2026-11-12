"""Configuration loading, merging and validation.

The built-in defaults live in ``resources/default_config.yaml``. A user file only
lists the keys it overrides. Command-line arguments and environment variables
take precedence over both.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ConfigError

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
DEFAULT_CONFIG_PATH = RESOURCE_DIR / "default_config.yaml"

# Keys whose values must come from a fixed set.
ENUM_KEYS: dict[tuple[str, ...], tuple[str, ...]] = {
    ("llm", "generation", "thinking"): ("enabled", "disabled"),
    ("llm", "judgement", "thinking"): ("enabled", "disabled"),
    ("llm", "generation", "reasoning_effort"): ("low", "high", "max"),
    ("llm", "judgement", "reasoning_effort"): ("low", "high", "max"),
    ("experience", "vector", "device"): ("cuda", "cpu"),
    ("experience", "vector", "backend"): ("local", "api"),
    ("pretrained", "mode"): ("catalog", "open", "off"),
    ("control", "time_allocation", "mode"): ("off", "observe", "enforce"),
    ("search", "reward_mode"): ("per_step", "chain_end"),
    ("control", "cost_cap", "mode"): ("off", "observe", "enforce"),
    ("control", "resources", "mode"): ("off", "observe", "enforce"),
}

# Keys holding filesystem paths, resolved against the current directory.
PATH_KEYS: tuple[tuple[str, ...], ...] = (("pretrained", "local_dir"),)

# Command-line option name -> configuration key path.
CLI_TO_CONFIG: dict[str, tuple[str, ...]] = {
    "time_limit_secs": ("run", "time_limit_seconds"),
    "max_steps": ("run", "max_steps"),
    "seed": ("run", "seed"),
    "num_gpus": ("resources", "num_gpus"),
    "num_cpus": ("resources", "num_cpus"),
    "parallel": ("resources", "parallel"),
    "keep_artifacts_top_k": ("execution", "keep_artifacts_top_k"),
    "pretrained_mode": ("pretrained", "mode"),
}

ENV_TO_CONFIG: dict[str, tuple[str, ...]] = {
    "SUPERAGENT_LLM_BASE_URL": ("llm", "base_url"),
    "SUPERAGENT_PRETRAIN_DIR": ("pretrained", "local_dir"),
}

POSITIVE_INT_KEYS: tuple[tuple[str, ...], ...] = (
    ("resources", "parallel"),
    ("resources", "num_cpus"),
    ("run", "max_steps"),
)


def load_defaults() -> dict[str, Any]:
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _type_name(value: Any) -> str:
    return type(value).__name__


def _check_value(path: tuple[str, ...], value: Any, default: Any) -> None:
    key = ".".join(path)
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"config key '{key}' must be a boolean, got {_type_name(value)}")
    elif isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"config key '{key}' must be an integer, got {_type_name(value)}")
    elif isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"config key '{key}' must be a number, got {_type_name(value)}")
    elif isinstance(default, str):
        if not isinstance(value, str):
            raise ConfigError(f"config key '{key}' must be a string, got {_type_name(value)}")
    elif isinstance(default, list):
        if not isinstance(value, list):
            raise ConfigError(f"config key '{key}' must be a list, got {_type_name(value)}")
        if default:
            proto = default[0]
            for i, item in enumerate(value):
                _check_value(path + (f"[{i}]",), item, proto)
        if len(default) == 2 and all(isinstance(d, (int, float)) for d in default) and len(value) != 2:
            raise ConfigError(f"config key '{key}' must be a list of two numbers")
    elif isinstance(default, dict):
        if not isinstance(value, dict):
            raise ConfigError(f"config key '{key}' must be a mapping, got {_type_name(value)}")
    enum = ENUM_KEYS.get(path)
    if enum is not None and value not in enum:
        raise ConfigError(f"config key '{key}' must be one of {list(enum)}, got {value!r}")


def _merge(base: dict[str, Any], override: Mapping[str, Any], prefix: tuple[str, ...]) -> None:
    for k, v in override.items():
        path = prefix + (str(k),)
        if k not in base:
            raise ConfigError(f"unknown config key '{'.'.join(path)}'")
        default = base[k]
        _check_value(path, v, default)
        if isinstance(default, dict):
            _merge(base[k], v, path)
        elif isinstance(default, float) and not isinstance(v, bool):
            base[k] = float(v)
        else:
            base[k] = copy.deepcopy(v)


def get_path(cfg: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = cfg
    for p in path:
        cur = cur[p]
    return cur


def set_path(cfg: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cur = cfg
    for p in path[:-1]:
        cur = cur[p]
    default = cur[path[-1]]
    _check_value(path, value, default)
    if isinstance(default, float) and not isinstance(value, bool):
        value = float(value)
    cur[path[-1]] = value


def build_config(
    user_config_path: str | os.PathLike[str] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return the effective configuration. Raises :class:`ConfigError` on any problem."""
    cfg = load_defaults()
    if user_config_path is not None:
        p = Path(user_config_path)
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        try:
            with open(p, encoding="utf-8") as fh:
                user = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"config file is not valid YAML: {exc}") from exc
        if not isinstance(user, dict):
            raise ConfigError("config file must contain a mapping at the top level")
        _merge(cfg, user, ())
    env = os.environ if environ is None else environ
    for var, path in ENV_TO_CONFIG.items():
        val = env.get(var)
        if val:
            set_path(cfg, path, val)
    for name, value in (cli_overrides or {}).items():
        if value is None:
            continue
        path = CLI_TO_CONFIG.get(name)
        if path is None:
            raise ConfigError(f"unknown option '{name}'")
        set_path(cfg, path, value)
    for path in POSITIVE_INT_KEYS:
        if get_path(cfg, path) < 1:
            raise ConfigError(f"config key '{'.'.join(path)}' must be at least 1")
    if get_path(cfg, ("resources", "num_gpus")) < 0:
        raise ConfigError("config key 'resources.num_gpus' must not be negative")
    base = Path(cwd) if cwd is not None else Path.cwd()
    for path in PATH_KEYS:
        val = get_path(cfg, path)
        if val:
            pv = Path(os.path.expanduser(val))
            if not pv.is_absolute():
                pv = (base / pv).resolve()
            cur = cfg
            for p in path[:-1]:
                cur = cur[p]
            cur[path[-1]] = str(pv)
    return cfg


def dump_config(cfg: Mapping[str, Any]) -> str:
    """YAML text of the effective configuration. It never contains the API key."""
    return yaml.safe_dump(dict(cfg), sort_keys=False, allow_unicode=True)
