"""Command-line entry of the adapter.

    python -m superagent_adapter.cli prepare --public P --private Q --task-dir T
    python -m superagent_adapter.cli run     --task-dir T --output-dir O [--config C] [Core options]
    python -m superagent_adapter.cli grade   --core-output O --sample S --data-dir D --results R --competition C
    python -m superagent_adapter.cli batch   --plan plan.yaml --results batch_results.json
    python -m superagent_adapter.cli cost    --core-output O --prices prices.yaml
    python -m superagent_adapter.cli analyze --core-output O [--grades R/grades.json] [--catalog C] --out A
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis import analyze, plot_timeline
from .batch import run_plan
from .common.core_runner import core_command, run_core
from .common.costs import load_prices, read_usage
from .common.task_dir import build_task_dir, check_isolation
from .mlebench.grader import grade_all


def _bool(v: str) -> bool:
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="superagent_adapter")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="build a Core task directory from a public split and check isolation")
    p.add_argument("--public", required=True, type=Path)
    p.add_argument("--private", required=True, type=Path)
    p.add_argument("--task-dir", required=True, type=Path)
    p.add_argument("--contract", type=Path, help="task-owned classification YAML; paths refer to public data")
    r = sub.add_parser("run", help="start the Core on a task directory")
    r.add_argument("--task-dir", required=True, type=Path)
    r.add_argument("--output-dir", required=True, type=Path)
    r.add_argument("--config", type=Path)
    r.add_argument("--core-python", default=sys.executable)
    r.add_argument("--cwd", type=Path, default=Path.cwd(), help="directory holding the .env file")
    r.add_argument("--time-limit-secs", type=int)
    r.add_argument("--max-steps", type=int)
    r.add_argument("--higher-is-better", type=_bool)
    r.add_argument("--num-gpus", type=int)
    r.add_argument("--num-cpus", type=int)
    r.add_argument("--parallel", type=int)
    r.add_argument("--keep-artifacts-top-k", type=int)
    r.add_argument("--pretrained-mode", choices=["catalog", "open", "off"])
    r.add_argument("--seed", type=int)
    g = sub.add_parser("grade", help="grade the Core's predictions with the official MLE-bench grader")
    g.add_argument("--core-output", required=True, type=Path)
    g.add_argument("--sample", required=True, type=Path)
    g.add_argument("--data-dir", required=True, type=Path)
    g.add_argument("--results", required=True, type=Path)
    g.add_argument("--competition", required=True)
    g.add_argument("--mlebench-bin", default="mlebench")
    g.add_argument("--include-top", action="store_true", help="also grade top/rank_* (comparison only)")
    b = sub.add_parser("batch", help="run, grade and summarise several runs from a plan file")
    b.add_argument("--plan", required=True, type=Path)
    b.add_argument("--results", required=True, type=Path)
    c = sub.add_parser("cost", help="recompute the model cost of a run with a price table")
    c.add_argument("--core-output", required=True, type=Path)
    c.add_argument("--prices", required=True, type=Path)
    c.add_argument("--after-cursor")
    c.add_argument("--process-exited", action="store_true")
    z = sub.add_parser("analyze", help="summarise a run (operators, selection modes, failures, usage, grades)")
    z.add_argument("--core-output", required=True, type=Path)
    z.add_argument("--grades", type=Path)
    z.add_argument("--catalog", type=Path)
    z.add_argument("--out", required=True, type=Path)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "prepare":
        build_task_dir(a.public, a.task_dir, contract=a.contract)
        rep = check_isolation(a.task_dir, a.private)
        print(json.dumps({"isolation_ok": rep.ok, "files_checked": rep.files_checked, "problems": rep.problems}, indent=2))
        return 0 if rep.ok else 1
    if a.cmd == "run":
        options = {k: getattr(a, k) for k in ("time_limit_secs", "max_steps", "higher_is_better", "num_gpus", "num_cpus",
                                              "parallel", "keep_artifacts_top_k", "pretrained_mode", "seed")}
        cmd = core_command(a.core_python, a.task_dir.resolve(), a.output_dir.resolve(),
                           a.config.resolve() if a.config else None, options)
        print("running:", " ".join(cmd), flush=True)
        code, status = run_core(cmd, a.cwd, a.output_dir.resolve().parent / f"{a.output_dir.name}_core_stdout.log")
        print(json.dumps({"exit_code": code, "status": status}, indent=2))
        return code if code in (0, 2) else 1
    if a.cmd == "grade":
        res = grade_all(a.core_output, a.sample, a.mlebench_bin, a.competition, a.data_dir, a.results, a.include_top)
        print(json.dumps({"primary": res["primary"],
                          "grades": {k: {"score": v["score"], "any_medal": v["any_medal"]} for k, v in res["grades"].items()}},
                         indent=2))
        return 0
    if a.cmd == "batch":
        res = run_plan(a.plan, a.results)
        print(json.dumps(res["summary"], indent=2))
        return 0
    if a.cmd == "cost":
        try:
            result = read_usage(a.core_output, load_prices(a.prices), a.after_cursor, a.process_exited)
        except ValueError as exc:
            print(f"superagent_adapter: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2))
        return 0
    if a.cmd == "analyze":
        res = analyze(a.core_output, a.grades, a.catalog)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "analysis.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            plot_timeline(a.core_output, a.out / "score_timeline.png")
        except ImportError:
            print("matplotlib is not installed: no plot written", file=sys.stderr)
        print(json.dumps({k: res[k] for k in ("operators", "selection_modes", "failures") if k in res}, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
