"""Run real Core/adapter subprocesses against an offline SSE fixture and grade exports."""

import argparse
import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml


SOLUTION = Path(__file__).with_name("fixture_solution.py").read_text()
COLAB = Path(__file__).resolve().parents[3] / "adapter/launchers/colab/remote"


def write_csv(path, header, rows):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def fixture(root):
    public, private = root / "public", root / "private"
    public.mkdir()
    private.mkdir()
    (public / "description.md").write_text(
        "Classify x into classes 01 and NA. Use the frozen validation split. Score accuracy."
    )
    write_csv(public / "train.csv", ["id", "x", "label"],
              [(f"x{i}", i - 4.5, "01" if i < 5 else "NA") for i in range(10)])
    write_csv(public / "val.csv", ["id", "x"], [("v0", -3), ("v1", 3), ("v2", -1), ("v3", 1)])
    write_csv(public / "val_truth.csv", ["id", "label"], [("v0", "01"), ("v1", "NA"), ("v2", "01"), ("v3", "NA")])
    write_csv(public / "test.csv", ["id", "x"], [("0002", -2), ("0001", 2)])
    write_csv(public / "sample_submission.csv", ["id", "label"], [("0001", "placeholder"), ("0002", "placeholder")])
    write_csv(private / "test.csv", ["id", "label"], [("0001", "NA"), ("0002", "01")])
    contract = root / "contract.yaml"
    contract.write_text(yaml.safe_dump({
        "version": 1, "metric": "accuracy", "classes": ["01", "NA"],
        "sample": "sample_submission.csv",
        "inference_inputs": ["val.csv", "test.csv", "sample_submission.csv"],
        "validation": {"format": "csv", "path": "val_truth.csv", "id": "id", "label": "label"},
    }))
    return public, private, contract


def main(root):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    public, private, contract = fixture(root)
    task = root / "task"
    adapter = [sys.executable, "-m", "superagent_adapter.cli"]
    core = [sys.executable, "-m", "superagent.run"]
    subprocess.run([*adapter, "prepare", "--public", str(public), "--private", str(private),
                    "--task-dir", str(task), "--contract", str(contract)], check=True)
    observed = []
    stage = ""
    blocked = threading.Event()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            system = req["messages"][0]["content"]
            structured = bool(req.get("response_format"))
            role = "review" if "code reviewer" in system else "plan" if structured else "script"
            observed.append(dict(case=stage, role=role, model=req["model"], thinking=req["thinking"]["type"],
                                 effort=req.get("reasoning_effort"), cap=req["max_tokens"], stream=req["stream"],
                                 json_mode=structured, chars=sum(len(x["content"]) for x in req["messages"])))
            if role == "plan" and stage in ("interrupt", "deadline", "generation_cutoff"):
                blocked.set()
                if stage == "interrupt":
                    release.wait(4)
                elif stage == "generation_cutoff":
                    threading.Event().wait(24)
                self.send_response(524)
                self.send_header("Retry-After", "60")
                self.end_headers()
                return
            if role == "review":
                body = '{"verdict":"pass","findings":"fixture review","patch":null}'
            else:
                assert role == "script", role
                body = "Fit a running two-class centroid model.\n\n```python\n" + SOLUTION + "\n```"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            frames = [
                {"id": "fixture", "model": req["model"], "choices": [{"delta": {"content": body}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens":
                 (50_000 if role == "script" else 15_000) if stage == "budget_tail" else 30}},
            ]
            for frame in frames:
                self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *_):
            pass

    def check_delivery(out):
        status = json.loads((out / "status.json").read_text())
        source = json.loads((out / "final/source.json").read_text())
        tree = json.loads((out / "logs/search_tree.json").read_text())["nodes"]
        published = out / "final/predictions.csv"
        matches = [n for n in tree if n["id"] == source["node_id"]]
        assert len(matches) == 1 and matches[0]["score"] == 1 and not matches[0]["failed"]
        node = out / "runs" / source["node_id"] / "out"
        assert published.read_bytes() == (node / "predictions.csv").read_bytes()

        def accuracy(expected, predicted):
            with expected.open() as f:
                gold = {x["id"]: x["label"] for x in csv.DictReader(f)}
            with predicted.open() as f:
                actual = list(csv.DictReader(f))
            assert len(gold) == len(actual) and set(gold) == {x["id"] for x in actual}
            return sum(gold[x["id"]] == x["label"] for x in actual) / len(gold)

        val = accuracy(task / "data/val_truth.csv", node / "val_predictions.csv")
        test = accuracy(private / "test.csv", published)
        assert val == status["best_score"] == test == 1
        return dict(status=status, source=source, nodes=[{
            k: n.get(k) for k in ("id", "kind", "parent_id", "score", "rung", "continuation_verification", "artifact_verification")
        } for n in tree if n["kind"] != "root"],
                    decisions=[json.loads(line) for line in (out / "logs/decisions.jsonl").read_text().splitlines()],
                    calls=[json.loads(line) for line in (out / "logs/llm_calls.jsonl").read_text().splitlines()],
                    accounting=json.loads((out / "logs/accounting.json").read_text()) if status["status"] != "running" else None,
                    actual_test_accuracy=test, independent_val_accuracy=val,
                    submission_sha256=hashlib.sha256(published.read_bytes()).hexdigest())

    scenarios = {}
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        api = f"http://127.0.0.1:{server.server_port}"
        # Use a file-backed endpoint to exercise the real CLI's configuration path.
        (root / ".env").write_text(f"DEEPSEEK_API_KEY=fixture\nSUPERAGENT_LLM_BASE_URL={api}\n")
        for stage in ("normal_one", "normal", "budget", "budget_tail", "interrupt", "deadline", "generation_cutoff"):
            out = root / stage
            config = root / (stage + ".yaml")
            budget = stage == "budget"
            normal = stage.startswith("normal")
            config.write_text(yaml.safe_dump({
                "run": {"finalize_reserve_seconds": 4, "idle_backoff_seconds": 1},
                "search": {"initial_drafts": 1, "max_drafts": 1},
                "codegen": {"staged_drafting": False},
                "experience": {"enabled": False}, "pretrained": {"mode": "off"},
                "control": {"cost_cap": {"mode": "enforce", "limit": .01 if budget else .565 if stage == "budget_tail" else 1.0}},
                "llm": {"request_timeout_seconds": 30 if stage == "generation_cutoff" else 6},
                "final": {"allow_ensemble": False},
            }))
            args = ["--task-dir", str(task), "--output-dir", str(out), "--config", str(config),
                    "--time-limit-secs", "24" if stage in ("deadline", "generation_cutoff") else "90",
                    "--max-steps", "2" if normal else "6",
                    "--num-cpus", "2", "--num-gpus", "0", "--parallel", "2" if stage == "normal" else "1"]
            command = [*core, *args] if stage in ("deadline", "generation_cutoff") else [*adapter, "run", *args, "--cwd", str(root)]
            env = os.environ.copy()
            env.pop("SUPERAGENT_LLM_BASE_URL", None)
            env["DEEPSEEK_API_KEY"] = "fixture"
            env["SUPERAGENT_TEST_NO_CONTINUE"] = str(int(not normal))
            if budget:
                launch = root / "budget-launch.json"
                launch.write_text(json.dumps({
                    "workdir": str(root), "task_dir": str(task), "output_dir": str(out),
                    "config": str(config), "env_dir": str(root), "python": sys.executable,
                    "driver_log": str(root / "budget-driver.log"),
                    "core_options": {"time_limit_secs": 90, "max_steps": 6,
                                     "num_cpus": 2, "num_gpus": 0, "parallel": 1},
                }))
                env["LAUNCH_JSON"] = str(launch)
                command = [sys.executable, str(COLAB / "run.py")]
            with (root / (stage + ".log")).open("w") as log:
                proc = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, env=env)
                if stage == "interrupt":
                    try:
                        assert blocked.wait(20), "expected the second operation to enter the gateway"
                        before = check_delivery(out)
                        assert before["status"]["status"] == "running" and before["status"]["final_available"]
                        proc.send_signal(signal.SIGTERM)
                    finally:
                        release.set()
                code = proc.wait(timeout=110)
            if budget:
                assert code == 2 and not (out / "final").exists()
                scenarios[stage] = {"exit": code, "status": json.loads((out / "status.json").read_text())}
            else:
                assert code == 0
                scenarios[stage] = check_delivery(out)
                scenarios[stage]["exit"] = code
                if stage == "interrupt":
                    assert scenarios[stage]["status"]["stop_reason"] == "interrupted"
                    assert scenarios[stage]["submission_sha256"] == before["submission_sha256"]
                if stage == "deadline":
                    assert scenarios[stage]["status"]["stop_reason"] == "time_limit"
                if stage == "generation_cutoff":
                    assert scenarios[stage]["status"]["stop_reason"] == "no_deliverable_action"
                    discarded = [r for r in scenarios[stage]["decisions"] if r["details"].get("generation_cutoff")]
                    assert len(discarded) == 1 and not discarded[0]["new_node_id"]
                    assert scenarios[stage]["submission_sha256"] == scenarios["deadline"]["submission_sha256"]
                    assert scenarios[stage]["accounting"]["unknown_or_inflight"] == 1
                if stage == "budget_tail":
                    assert scenarios[stage]["status"]["stop_reason"] == "no_deliverable_action"
                    assert len(scenarios[stage]["calls"]) == 2
                if normal:
                    assert len(scenarios[stage]["nodes"]) == 2 and scenarios[stage]["nodes"][-1]["kind"] == "promote"
                    assert scenarios[stage]["nodes"][-1]["continuation_verification"]["passed"]
    grader = root / "fixture-grader"
    grader.write_text(f"#!{sys.executable}\n" + """import csv, json, sys
from pathlib import Path
def table(path):
    with Path(path).open() as f:
        return {r["id"]: r["label"] for r in csv.DictReader(f)}
assert sys.argv[1] == "grade-sample" and sys.argv[4] == "--data-dir"
actual = table(sys.argv[2])
truth = table(Path(sys.argv[5]) / "test.csv")
assert set(actual) == set(truth)
score = sum(actual[i] == truth[i] for i in truth) / len(truth)
print("Competition report:")
print(json.dumps({"score": score, "any_medal": False}, indent=2))
""")
    grader.chmod(0o700)
    subprocess.run([*adapter, "grade", "--core-output", str(root / "normal"), "--sample",
                    str(task / "data/sample_predictions.csv"), "--data-dir", str(private), "--results",
                    str(root / "grades"), "--competition", "fixture", "--mlebench-bin", str(grader)], check=True)
    grades = json.loads((root / "grades/grades.json").read_text())
    assert grades["primary"] == "final" and grades["grades"]["final"]["score"] == 1
    with (root / "grades/submission.csv").open() as f:
        assert list(csv.reader(f)) == [["id", "label"], ["0001", "NA"], ["0002", "01"]]
    launch = root / "snapshot-launch.json"
    settings = dict(workdir=str(root), output_dir=str(root / "normal"), driver_log=str(root / "normal.log"))
    launch.write_text(json.dumps(settings))
    snapshot = [sys.executable, str(COLAB / "snapshot.py")]
    snapshot_env = dict(os.environ, LAUNCH_JSON=str(launch))
    subprocess.run(snapshot, env=snapshot_env, check=True, stdout=subprocess.DEVNULL)
    path = root / "snapshot.tar.gz"
    with tarfile.open(path) as tar:
        names = tar.getnames()
        assert not any("/checkpoints" in name or "/feature_cache" in name for name in names)
        pointer = tar.getmember("normal/final")
        prediction = tar.extractfile("normal/" + pointer.linkname + "/predictions.csv")
        assert prediction and hashlib.sha256(prediction.read()).hexdigest() == scenarios["normal"]["submission_sha256"]
    previous = hashlib.sha256(path.read_bytes()).hexdigest()
    launch.write_text(json.dumps({**settings, "output_dir": str(root / "does-not-exist")}))
    failure = subprocess.run(snapshot, env=snapshot_env, capture_output=True)
    assert failure.returncode != 0 and hashlib.sha256(path.read_bytes()).hexdigest() == previous
    assert [r["role"] for r in observed if r["case"] == "normal"] == ["script", "review"]
    (root / "evidence.json").write_text(json.dumps({
        "runs": scenarios, "wire": observed, "grades": grades, "snapshot": {
            "file_entries": len(names), "sha256": previous, "failed_export_kept_archive": True,
        }
    }, indent=2))
    print({k: (v["exit"], v["status"]["stop_reason"], v["status"]["elapsed_secs"]) for k, v in scenarios.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    main(p.parse_args().root)
