"""Pack the output directory (without model weights and caches) and the logs into <workdir>/snapshot.tar.gz,
then print the run status."""

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
work = cfg.get("workdir", "/content")
out = cfg["output_dir"].rstrip("/")
name, parent = os.path.basename(out), os.path.dirname(out)
excludes = [f"{name}/prepared_data", f"{name}/runs/*/out/model", f"{name}/runs/*/work", f"{name}/best/artifacts"]
logs = [p for p in (cfg["driver_log"], os.path.join(parent, f"{name}_core_stdout.log")) if os.path.exists(p)]
args = ["tar", "czf", os.path.join(work, "snapshot.tar.gz")] + [f"--exclude={e}" for e in excludes] + ["-C", parent, name]
for p in logs:
    args += ["-C", os.path.dirname(p), os.path.basename(p)]
subprocess.run(args, stderr=subprocess.DEVNULL)
try:
    with open(os.path.join(out, "status.json")) as fh:
        s = json.load(fh)
    print("STATUS", s["status"], s["stop_reason"], s["num_nodes"], s["num_ok_nodes"], s["best_score"], s["elapsed_secs"])
except Exception as exc:  # noqa: BLE001
    print("STATUS ?", exc)
print(subprocess.run(["tail", "-2", cfg["driver_log"]], capture_output=True, text=True).stdout)
