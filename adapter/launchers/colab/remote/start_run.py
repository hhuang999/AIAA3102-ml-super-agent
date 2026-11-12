"""Start run.py (in the launch directory) in the background with the environment's Python."""

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


cfg = settings()
subprocess.Popen([cfg["python"], os.path.join(cfg["launch_dir"], "run.py")], cwd=cfg["launch_dir"], start_new_session=True,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("run started")
