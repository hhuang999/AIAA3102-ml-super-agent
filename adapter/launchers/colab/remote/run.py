"""Run the Core through the adapter with the settings of launch.json: one task ("task_dir", "output_dir",
"config", "core_options") or a batch plan ("batch_plan", optional "batch_results"). Appends to the driver log;
the last line is '== run end <time> exit <code>'."""

import datetime
import json
import os
import subprocess


def settings():
    """Settings from launch.json, with defaults derived from its workdir."""
    path = os.environ.get("LAUNCH_JSON", "/content/launch/launch.json")
    with open(path) as fh:
        cfg = json.load(fh)
    work = cfg.get("workdir", "/content")
    cfg.setdefault("launch_dir", os.path.join(work, "launch"))
    cfg.setdefault("python", os.path.join(work, "venv", "bin", "python"))
    cfg.setdefault("driver_log", os.path.join(work, "run_driver.log"))
    cfg.setdefault("env_dir", cfg["launch_dir"])
    cfg.setdefault("pretrain_dir", os.path.join(work, "pretrained_cache"))
    return cfg


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


cfg = settings()
env_file = os.path.join(cfg["env_dir"], ".env")
if os.path.exists(env_file):
    os.chmod(env_file, 0o600)
if cfg.get("batch_plan"):
    cmd = [cfg["python"], "-m", "superagent_adapter.cli", "batch", "--plan", cfg["batch_plan"],
           "--results", cfg.get("batch_results", os.path.join(cfg.get("workdir", "/content"), "batch_results.json"))]
else:
    cmd = [cfg["python"], "-m", "superagent_adapter.cli", "run", "--task-dir", cfg["task_dir"], "--output-dir", cfg["output_dir"],
           "--core-python", cfg["python"], "--cwd", cfg["env_dir"]]
    if cfg.get("config"):
        cmd += ["--config", cfg["config"]]
    for key, value in (cfg.get("core_options") or {}).items():
        cmd += ["--" + key.replace("_", "-"), str(value).lower() if isinstance(value, bool) else str(value)]
env = dict(os.environ, SUPERAGENT_PRETRAIN_DIR=cfg["pretrain_dir"])
with open(cfg["driver_log"], "a") as fh:
    fh.write(f"== run start {now()}\n")
    fh.flush()
    code = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    fh.write(f"== run end {now()} exit {code}\n")
raise SystemExit(code)
