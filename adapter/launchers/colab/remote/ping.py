"""Print one 'ALIVE' line with the time and the last line of the driver log."""

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


cfg = settings()
last = subprocess.run(["tail", "-1", cfg["driver_log"]], capture_output=True, text=True).stdout.strip()
print("ALIVE", datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S"), last)
