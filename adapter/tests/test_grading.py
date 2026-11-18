import json

import pandas as pd
import pytest

from superagent_adapter.batch import load_plan, run_one, summarize
from superagent_adapter.cli import build_parser
from superagent_adapter.common.core_runner import core_command
from superagent_adapter.common.costs import recompute
from superagent_adapter.common.submission import candidate_files, make_submission, primary_name
from superagent_adapter.mlebench.grader import parse_report


def test_make_submission_reorders_columns(tmp_path):
    pd.DataFrame({"id": ["a"], "B": [0.2], "A": [0.8]}).to_csv(tmp_path / "p.csv", index=False)
    pd.DataFrame({"id": ["a"], "A": [0.5], "B": [0.5]}).to_csv(tmp_path / "s.csv", index=False)
    out = make_submission(tmp_path / "p.csv", tmp_path / "s.csv", tmp_path / "sub.csv")
    assert list(pd.read_csv(out).columns) == ["id", "A", "B"]


def test_submission_preserves_opaque_ids_and_labels(tmp_path):
    pred=tmp_path/"pred.csv"
    pred.write_text("id,label\n0001,NA\n0002,01\n")
    out=make_submission(pred,pred,tmp_path/"submit.csv")
    assert out.read_text()==pred.read_text()


@pytest.mark.parametrize("rows, expected", [
    ("0002,NA\n0001,01\n", "id,label\n0001,01\n0002,NA\n"),
    ("0001,NA\n0001,01\n", None),
    ("0001,NA\n0003,01\n", None),
    ("0001,NA\n", None),
])
def test_submission_matches_sample_by_exact_id(tmp_path, rows, expected):
    pred, sample, out = (tmp_path / f"{name}.csv" for name in ("pred", "sample", "submission"))
    pred.write_text("id,label\n" + rows)
    sample.write_text("id,label\n0001,placeholder\n0002,placeholder\n")
    if expected is None:
        with pytest.raises(ValueError, match="ID"):
            make_submission(pred, sample, out)
        assert not out.exists()
    else:
        assert make_submission(pred, sample, out).read_text() == expected


def test_run_exit_reflects_no_deliverable(monkeypatch, tmp_path):
    from superagent_adapter import cli

    monkeypatch.setattr(cli, "run_core", lambda *_: (2, {"status": "no_valid_solution"}))
    assert cli.main(["run", "--task-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")]) == 2


def test_adapter_relays_sigterm_to_core(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time

    if os.name != "posix":
        pytest.skip("POSIX signal forwarding")
    child = tmp_path / "child.py"
    child.write_text("""import os, signal, time
from pathlib import Path
def finish(*_):
    Path("submission-kept").touch()
    raise SystemExit(0)
signal.signal(signal.SIGTERM, finish)
Path("child-id").write_text(str(os.getpid()))
time.sleep(6)
""")
    wrapper = (
        "import sys; from pathlib import Path; "
        "from superagent_adapter.common.core_runner import run_core; "
        "raise SystemExit(run_core([sys.executable, 'child.py', '--output-dir', 'out'], "
        "Path.cwd(), Path('child.log'))[0])"
    )
    proc = subprocess.Popen([sys.executable, "-c", wrapper], cwd=tmp_path)
    marker = tmp_path / "child-id"
    try:
        end = time.monotonic() + 4
        while not marker.is_file() and time.monotonic() < end:
            time.sleep(.02)
        assert marker.is_file()
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=4) == 0
        assert (tmp_path / "submission-kept").is_file()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if marker.is_file() and not (tmp_path / "submission-kept").is_file():
            try:
                os.kill(int(marker.read_text()), signal.SIGTERM)
            except ProcessLookupError:
                pass


def _core_out(tmp_path, final=True):
    for d in ("best", "ensemble", "top/rank_01") + (("final",) if final else ()):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    (tmp_path / "best" / "predictions.csv").write_text("id,A\n")
    (tmp_path / "top" / "rank_01" / "predictions.csv").write_text("id,A\n")
    for n in ("predictions.csv", "predictions_top2.csv"):
        (tmp_path / "ensemble" / n).write_text("id,A\n")
    if final:
        (tmp_path / "final" / "predictions.csv").write_text("id,A\n")
    return tmp_path


def test_final_is_the_primary_result(tmp_path):
    out = _core_out(tmp_path)
    files = candidate_files(out)
    assert list(files)[0] == "final" and primary_name(out) == "final"
    assert "best" in files and "ensemble_top2" in files and "top_rank_01" not in files
    assert "top_rank_01" in candidate_files(out, include_top=True)


def test_best_is_primary_for_interface_1_0_outputs(tmp_path):
    out = _core_out(tmp_path, final=False)
    assert primary_name(out) == "best" and list(candidate_files(out))[0] == "best"


def test_core_command_options(tmp_path):
    cmd = core_command("py", tmp_path / "t", tmp_path / "o", tmp_path / "c.yaml",
                       {"time_limit_secs": 100, "higher_is_better": False, "parallel": 2, "seed": None})
    assert cmd[:3] == ["py", "-m", "superagent.run"]
    assert cmd[cmd.index("--higher-is-better") + 1] == "false"
    assert cmd[cmd.index("--config") + 1].endswith("c.yaml")
    assert "--seed" not in cmd
    with pytest.raises(ValueError):
        core_command("py", tmp_path, tmp_path, None, {"bogus": 1})


def test_report_regex_handles_log_prefix():
    text = ('warnings.warn(\n[2026-11-12 04:21:12,571] [cli.py:202] Competition report:\n'
            '[2026-11-12 04:21:12,571] [cli.py:203] {\n    "score": 0.24125,\n    "any_medal": true\n}\n')
    assert parse_report(text)["score"] == 0.24125
    assert parse_report("no report here") is None


def test_cost_recompute(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    rows = [{"agent": "patch_coder", "model": "m1", "prompt_tokens": 1_000_000, "completion_tokens": 500_000, "cost": 0.1},
            {"agent": "reviewer", "model": "m1", "prompt_tokens": None, "completion_tokens": None, "cost": None},
            {"agent": "reviewer", "model": "m2", "prompt_tokens": 10, "completion_tokens": 10, "cost": 0.0}]
    (logs / "llm_calls.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (logs / "embedding_calls.jsonl").write_text(json.dumps({"model": "e1", "tokens": 2_000_000, "cost": 0.01}) + "\n")
    res = recompute(tmp_path, {"currency": "CNY", "models": {"m1": {"input_per_million": 2.0, "output_per_million": 8.0}},
                               "embeddings": {"e1": {"per_million": 0.5}}})
    assert res["llm"]["cost"] == pytest.approx(2.0 + 4.0)
    assert res["embedding"]["cost"] == pytest.approx(1.0)
    assert res["total_cost"] == pytest.approx(7.0)
    assert res["models_without_price"] == ["m2"]
    assert res["recorded_by_core"]["total"] == pytest.approx(0.11)


def test_batch_summary():
    rows = [{"task": "t1", "status": "success", "test_score": 0.2, "validation_score": 0.19, "medal": "silver"},
            {"task": "t1", "status": "success", "test_score": 0.3, "validation_score": 0.28, "medal": "bronze"},
            {"task": "t2", "status": "failed", "test_score": None, "validation_score": None, "medal": None}]
    s = summarize(rows)
    assert s["t1"]["runs"] == 2 and s["t1"]["test_score"]["mean"] == pytest.approx(0.25)
    assert s["t1"]["medals"]["silver"] == 1 and s["t2"]["test_score"] is None and s["t2"]["finished"] == 0


def test_batch_skips_finished_runs(tmp_path):
    out = tmp_path / "runs" / "r1"
    out.mkdir(parents=True)
    (out / "status.json").write_text(json.dumps({"status": "success", "final_kind": "ensemble", "final_validation_score": 0.3}))
    (tmp_path / "task").mkdir()
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text("core_python: /nonexistent/python\nruns:\n  - name: r1\n    task_dir: task\n    output_dir: runs/r1\n")
    plan = load_plan(plan_file)
    row = run_one(plan, plan["runs"][0], tmp_path)
    assert row["status"] == "success" and row["final_kind"] == "ensemble" and row["validation_score"] == 0.3


def test_plan_needs_unique_names(tmp_path):
    f = tmp_path / "plan.yaml"
    f.write_text("core_python: py\nruns:\n  - {name: a, task_dir: t, output_dir: o}\n  - {name: a, task_dir: t, output_dir: p}\n")
    with pytest.raises(ValueError):
        load_plan(f)


def test_cli_parser():
    a = build_parser().parse_args(["run", "--task-dir", "t", "--output-dir", "o", "--higher-is-better", "false"])
    assert a.higher_is_better is False and a.time_limit_secs is None
    a = build_parser().parse_args(["grade", "--core-output", "o", "--sample", "s", "--data-dir", "d", "--results", "r",
                                   "--competition", "c", "--include-top"])
    assert a.include_top


def test_batch_groups_by_task_label(tmp_path):
    for name in ("r1", "r2"):
        out = tmp_path / "runs" / name
        out.mkdir(parents=True)
        (out / "status.json").write_text(json.dumps({"status": "success", "best_score": 0.5}))
    (tmp_path / "a" / "task").mkdir(parents=True)
    (tmp_path / "b" / "task").mkdir(parents=True)
    plan = {"core_python": "py", "runs": [{"name": "r1", "task_dir": "a/task", "output_dir": "runs/r1"},
                                          {"name": "r2", "task_dir": "b/task", "output_dir": "runs/r2", "task": "B"}]}
    rows = [run_one(plan, r, tmp_path) for r in plan["runs"]]
    assert [r["task"] for r in rows] == ["a/task", "B"]
