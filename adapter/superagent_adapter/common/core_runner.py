"""Start the Core through its command line and read its status file."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

# Core command-line options the adapter passes on (name -> flag).
CORE_OPTIONS = {
    "time_limit_secs": "--time-limit-secs",
    "max_steps": "--max-steps",
    "higher_is_better": "--higher-is-better",
    "num_gpus": "--num-gpus",
    "num_cpus": "--num-cpus",
    "parallel": "--parallel",
    "keep_artifacts_top_k": "--keep-artifacts-top-k",
    "pretrained_mode": "--pretrained-mode",
    "seed": "--seed",
}


def core_command(python: str, task_dir: Path, output_dir: Path, config: Path | None = None,
                 options: Mapping[str, Any] | None = None) -> list[str]:
    """The Core command line. ``options`` uses the names of :data:`CORE_OPTIONS`; None values are left out."""
    cmd = [python, "-m", "superagent.run", "--task-dir", str(task_dir), "--output-dir", str(output_dir)]
    if config is not None:
        cmd += ["--config", str(config)]
    for name, value in (options or {}).items():
        if value is None:
            continue
        if name not in CORE_OPTIONS:
            raise ValueError(f"unknown Core option {name!r}")
        if isinstance(value, bool):
            value = "true" if value else "false"
        cmd += [CORE_OPTIONS[name], str(value)]
    return cmd


def run_core(cmd: list[str], cwd: Path, log_path: Path) -> tuple[int, dict[str, Any] | None]:
    """Run the Core as a child process (``cwd`` holds the .env file). Returns (exit code, status.json or None)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
    out = Path(cmd[cmd.index("--output-dir") + 1])
    if not out.is_absolute():
        out = Path(cwd) / out
    status_file = out / "status.json"
    status = json.loads(status_file.read_text()) if status_file.exists() else None
    return proc.returncode, status
