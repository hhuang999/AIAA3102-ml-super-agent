import os
import time
from pathlib import Path

import pytest

from superagent.catalog import offline_env
from superagent.executor import Executor, parse_exception


def make_exec(tmp_path, parallel=2, num_cpus=None, mode="catalog"):
    prep = tmp_path / "prepared"
    prep.mkdir(exist_ok=True)
    cpus = num_cpus or len(os.sched_getaffinity(0))
    return Executor(tmp_path / "runs", prep, parallel, cpus, 0, offline_env(mode, tmp_path / "pt"))


def test_parallel_runs_isolated(tmp_path):
    import threading

    ex = make_exec(tmp_path)
    code = 'import sys, time\ntime.sleep(0.5)\nopen("./out/predictions.csv","w").write("{v}")\n'
    res = {}

    def go(nid, v):
        res[nid] = ex.run(nid, code.replace("{v}", v), timeout=30)

    ts = [threading.Thread(target=go, args=("a1", "AAA")), threading.Thread(target=go, args=("b2", "BBB"))]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert (tmp_path / "runs/a1/out/predictions.csv").read_text() == "AAA"
    assert (tmp_path / "runs/b2/out/predictions.csv").read_text() == "BBB"
    assert (tmp_path / "runs/a1/solution.py").read_bytes() == code.replace("{v}", "AAA").encode()


def test_timeout_kills_group(tmp_path):
    ex = make_exec(tmp_path)
    code = ("import subprocess, sys, time\n"
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "open('child.pid','w').write(str(p.pid))\n"
            "time.sleep(60)\n")
    t0 = time.monotonic()
    r = ex.run("t1", code, timeout=3)
    assert time.monotonic() - t0 < 6
    assert r.error_type == "TimeoutError"
    pid = int((tmp_path / "runs/t1/child.pid").read_text())
    time.sleep(0.3)
    alive = Path(f"/proc/{pid}").exists() and "Z" not in Path(f"/proc/{pid}/stat").read_text().split()[2]
    assert not alive


def test_exception_parsed(tmp_path):
    ex = make_exec(tmp_path)
    r = ex.run("e1", 'raise ValueError("bad")\n', timeout=30)
    assert r.error_type == "ValueError"
    assert r.error_message == "bad"


def test_normal_end_removes_work(tmp_path):
    ex = make_exec(tmp_path)
    r = ex.run("n1", 'open("./work/tmp.txt","w").write("x")\nprint("VALIDATION_SCORE: 0.5")\n', timeout=30)
    assert r.error_type is None
    assert not (tmp_path / "runs/n1/work").exists()
    assert "VALIDATION_SCORE: 0.5" in r.output
    assert (tmp_path / "runs/n1/stdout.log").read_text().startswith("VALIDATION_SCORE: 0.5")


def test_offline_env_visible(tmp_path):
    ex = make_exec(tmp_path, mode="catalog")
    r = ex.run("o1", 'import os\nprint("OFF", os.environ.get("HF_HUB_OFFLINE"), os.environ.get("DEEPSEEK_API_KEY"))\n', timeout=30)
    assert "OFF 1 None" in r.output


def test_cpu_affinity(tmp_path):
    n_avail = len(os.sched_getaffinity(0))
    if n_avail < 2:
        pytest.skip("needs at least 2 cores")
    ex = make_exec(tmp_path, parallel=2, num_cpus=n_avail)
    n = n_avail // 2
    assert ex.slot_cores[0] and ex.slot_cores[1]
    assert len(ex.slot_cores[0]) == n
    assert not set(ex.slot_cores[0]) & set(ex.slot_cores[1])
    r = ex.run("c1", 'import os\nprint("CORES", len(os.sched_getaffinity(0)))\n', timeout=30)
    assert f"CORES {n}" in r.output


def test_parse_exception_chained():
    out = ("Traceback (most recent call last):\n  File x\nKeyError: 'a'\n\nDuring handling of the above exception, another exception occurred:\n\n"
           "Traceback (most recent call last):\n  File y\n    foo()\nRuntimeError: second\n")
    assert parse_exception(out) == ("RuntimeError", "second")


def test_nonzero_without_traceback(tmp_path):
    ex = make_exec(tmp_path)
    r = ex.run("x1", "import sys\nsys.exit(3)\n", timeout=30)
    assert r.error_type == "RuntimeError" and "3" in r.error_message
