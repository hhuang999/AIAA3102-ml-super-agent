"""Run the Core on several tasks or seeds one after another, grade each run, and summarise.

Plan file (YAML):

    core_python: /path/to/python          # interpreter with superagent installed
    cwd: /path/to/dir/with/.env           # where the Core reads its .env file
    config: /path/to/config.yaml          # optional, used by every run
    options: {time_limit_secs: 28800, parallel: 2, num_gpus: 1, num_cpus: 8}
    grader:                                # optional: MLE-bench grading
      mlebench_bin: mlebench
      data_dir: /path/to/mlebench/data
    runs:
      - name: spooky_seed1
        task: spooky                       # optional label used to group runs in the summary
        task_dir: tasks/spooky
        output_dir: runs/spooky_seed1
        competition: spooky-author-identification
        options: {seed: 1, higher_is_better: false}

A run whose output directory already holds a finished ``status.json`` is not started again.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Mapping

import yaml

from .common.core_runner import core_command, run_core
from .mlebench.grader import grade_all

FINISHED = ("success", "no_valid_solution")


def load_plan(path: Path) -> dict[str, Any]:
    plan = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for key in ("core_python", "runs"):
        if key not in plan:
            raise ValueError(f"{path}: missing '{key}'")
    names = [r["name"] for r in plan["runs"]]
    if len(names) != len(set(names)):
        raise ValueError(f"{path}: run names must be unique")
    return plan


def _status(out: Path) -> dict[str, Any] | None:
    f = out / "status.json"
    return json.loads(f.read_text()) if f.exists() else None


def run_one(plan: Mapping[str, Any], run: Mapping[str, Any], base: Path) -> dict[str, Any]:
    task = (base / run["task_dir"]).resolve()
    out = (base / run["output_dir"]).resolve()
    status = _status(out)
    if status is None or status.get("status") not in FINISHED:
        options = {**(plan.get("options") or {}), **(run.get("options") or {})}
        config = plan.get("config")
        cmd = core_command(plan["core_python"], task, out, (base / config).resolve() if config else None, options)
        _code, status = run_core(cmd, Path(plan.get("cwd") or base), out.parent / f"{out.name}_core_stdout.log")
    label = run.get("task") or run.get("competition") or str(run["task_dir"])
    row: dict[str, Any] = {"name": run["name"], "task": label, "output_dir": str(out),
                           "status": (status or {}).get("status"), "final_kind": (status or {}).get("final_kind"),
                           "validation_score": ((status or {}).get("final_validation_score") if (status or {}).get("final_validation_score") is not None else (status or {}).get("best_score")),
                           "test_score": None, "medal": None}
    g = plan.get("grader")
    if g and run.get("competition") and row["status"] == "success":
        sample = (base / run["sample"]).resolve() if run.get("sample") else task / "data" / "sample_predictions.csv"
        res = grade_all(out, sample, g.get("mlebench_bin", "mlebench"), run["competition"], Path(g["data_dir"]),
                        out.parent / f"{out.name}_grades")
        rep = res["grades"][res["primary"]]
        row["test_score"] = rep.get("score")
        row["medal"] = next((m for m in ("gold", "silver", "bronze") if rep.get(f"{m}_medal")), "none")
    return row


def summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Per task: number of runs and mean, standard deviation, min and max of the test and validation scores."""
    out: dict[str, Any] = {}
    for task in sorted({r["task"] for r in rows}):
        rs = [r for r in rows if r["task"] == task]
        entry: dict[str, Any] = {"runs": len(rs), "finished": sum(1 for r in rs if r["status"] == "success")}
        for key in ("test_score", "validation_score"):
            vals = [float(r[key]) for r in rs if r.get(key) is not None]
            entry[key] = None if not vals else {
                "mean": round(statistics.mean(vals), 6), "std": round(statistics.stdev(vals), 6) if len(vals) > 1 else 0.0,
                "min": round(min(vals), 6), "max": round(max(vals), 6)}
        entry["medals"] = {m: sum(1 for r in rs if r.get("medal") == m) for m in ("gold", "silver", "bronze", "none")}
        out[task] = entry
    return out


def run_plan(plan_path: Path, results_path: Path) -> dict[str, Any]:
    plan = load_plan(plan_path)
    base = Path(plan_path).resolve().parent
    rows = [run_one(plan, r, base) for r in plan["runs"]]
    result = {"runs": rows, "summary": summarize(rows)}
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
