"""Runs solution scripts in per-node run directories, inside execution slots."""

from __future__ import annotations

import logging
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

log = logging.getLogger("superagent")

OUTPUT_KEEP_EACH_SIDE = 200_000
OUTPUT_MAX_CHARS = 2 * OUTPUT_KEEP_EACH_SIDE
TERMINATE_GRACE_SECONDS = 2.0
POLL_SECONDS = 0.2
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS", "TORCH_NUM_THREADS")
SECRET_ENV_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)
_TRACEBACK_HEADER = "Traceback (most recent call last):"
SMOKE_ENV = "SUPERAGENT_SMOKE"
SMOKE_FAILED_NOTE = (
    "[superagent] SMOKE RUN FAILED: the script was first run with the environment variable SUPERAGENT_SMOKE=1 "
    "on a tiny subset of the data and failed, so the full run was skipped. The error may lie in the pipeline itself "
    "or in the smoke-mode subset handling.\n"
)
_EXC_LINE_RE = re.compile(r"^([A-Za-z_][\w.]*)(?::\s?(.*))?$")


@dataclass
class ExecResult:
    output: str
    exec_seconds: float
    error_type: str | None
    error_message: str | None
    timed_out: bool
    exit_code: int | None
    timeout_seconds: float
    smoke: str | None = None           # passed | failed | timed_out | None (no smoke run)
    smoke_seconds: float = 0.0
    artifact_verification: dict | None = None
    continuation_verification: dict | None = None


def parse_exception(output: str) -> tuple[str, str] | None:
    """Return (exception class, message) from the last traceback in ``output``."""
    idx = output.rfind(_TRACEBACK_HEADER)
    if idx < 0:
        return None
    lines = output[idx:].splitlines()[1:]
    for line in lines:
        if not line.strip():
            continue
        if line.startswith((" ", "\t")):
            continue
        m = _EXC_LINE_RE.match(line.strip())
        if m:
            return m.group(1), (m.group(2) or "").strip()
        # Some exceptions print a non-identifier line (e.g. "During handling ..."); keep scanning.
    return None


def classify_result(output: str, exit_code: int | None, timed_out: bool, timeout_seconds: float) -> tuple[str | None, str | None]:
    if timed_out:
        return "TimeoutError", f"script exceeded the time limit of {timeout_seconds:.0f} seconds"
    if exit_code == 0:
        return None, None
    parsed = parse_exception(output)
    if parsed is not None:
        return parsed
    if exit_code is not None and exit_code < 0:
        try:
            name = signal.Signals(-exit_code).name
        except ValueError:
            name = str(-exit_code)
        return "RuntimeError", f"process was killed by signal {name}"
    return "RuntimeError", f"process exited with code {exit_code}"


class _OutputCollector:
    """Keeps the head and tail of a stream in memory and writes everything to a log file."""

    def __init__(self, log_path: Path) -> None:
        self._fh = open(log_path, "w", encoding="utf-8", errors="replace")
        self._head: list[str] = []
        self._head_len = 0
        self._tail: list[str] = []
        self._tail_len = 0
        self._dropped = 0

    def add(self, text: str) -> None:
        self._fh.write(text)
        self._fh.flush()
        if self._head_len < OUTPUT_KEEP_EACH_SIDE:
            room = OUTPUT_KEEP_EACH_SIDE - self._head_len
            self._head.append(text[:room])
            self._head_len += len(text[:room])
            text = text[room:]
            if not text:
                return
        self._tail.append(text)
        self._tail_len += len(text)
        while self._tail_len > OUTPUT_KEEP_EACH_SIDE and len(self._tail) > 1:
            first = self._tail.pop(0)
            self._tail_len -= len(first)
            self._dropped += len(first)
        if self._tail_len > OUTPUT_KEEP_EACH_SIDE:
            extra = self._tail_len - OUTPUT_KEEP_EACH_SIDE
            self._tail[0] = self._tail[0][extra:]
            self._tail_len -= extra
            self._dropped += extra

    def text(self) -> str:
        head = "".join(self._head)
        tail = "".join(self._tail)
        if self._dropped:
            return head + f"\n... [{self._dropped} characters omitted; full output in stdout.log] ...\n" + tail
        return head + tail

    def close(self) -> None:
        try:
            self._fh.close()
        except OSError:
            pass


def _kill_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


class Executor:
    def __init__(
        self,
        runs_dir: Path,
        prepared_dir: Path,
        parallel: int,
        num_cpus: int,
        num_gpus: int,
        policy_env: Mapping[str, str],
    ) -> None:
        self.runs_dir = Path(runs_dir)
        self.prepared_dir = Path(prepared_dir)
        self.parallel = int(parallel)
        from .control.hardware import cuda_mask

        mask = cuda_mask()
        self.gpu_devices = mask[:num_gpus] if mask is not None else tuple(map(str, range(num_gpus)))
        self.num_gpus = len(self.gpu_devices)
        self.policy_env = dict(policy_env)
        try:
            available = sorted(os.sched_getaffinity(0))
        except AttributeError:  # pragma: no cover - non-Linux
            available = list(range(os.cpu_count() or 1))
        effective = int(num_cpus)
        if effective > len(available):
            log.warning("num_cpus=%d but only %d cores are available to this process; using %d",
                        num_cpus, len(available), len(available))
            effective = len(available)
        self.cores_per_slot = max(1, effective // self.parallel)
        self.slot_cores: list[list[int]] = []
        for i in range(self.parallel):
            cores = [available[(i * self.cores_per_slot + j) % len(available)] for j in range(self.cores_per_slot)]
            self.slot_cores.append(cores)
        self._free: queue.Queue[int] = queue.Queue()
        for i in range(self.parallel):
            self._free.put(i)
        self._procs: dict[int, subprocess.Popen[bytes]] = {}
        self._lock = threading.Lock()
        self._kill_all = threading.Event()
        self._intervals = []
        self._running_slots = {}

    # ------------------------------------------------------------------ run dirs
    def run_dir(self, node_id: str) -> Path:
        return self.runs_dir / node_id

    def prepare_run_dir(self, node_id: str, code: str | None) -> Path:
        d = self.run_dir(node_id)
        d.mkdir(parents=True, exist_ok=True)
        if code is not None:
            with open(d / "solution.py", "w", encoding="utf-8", newline="") as fh:
                fh.write(code)
        link = d / "data"
        if not link.exists() and not link.is_symlink():
            os.symlink(self.prepared_dir.resolve(), link)
        (d / "work").mkdir(exist_ok=True)
        (d / "out").mkdir(exist_ok=True)
        return d

    # ------------------------------------------------------------------ environment
    def _env(self, slot: int) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not SECRET_ENV_RE.search(k)}
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        n = str(self.cores_per_slot)
        for var in THREAD_ENV_VARS:
            env[var] = n
        env["CUDA_VISIBLE_DEVICES"] = self.gpu_devices[slot % self.num_gpus] if self.num_gpus else ""
        env.update(self.policy_env)
        return env

    # ------------------------------------------------------------------ execution
    def acquire_slot(self, abort: threading.Event | None = None, deadline: float | None = None) -> int | None:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                return None
            try:
                return self._free.get(timeout=POLL_SECONDS)
            except queue.Empty:
                if self._kill_all.is_set() or (abort is not None and abort.is_set()):
                    return None

    def utilization(self, start, end):
        """Time a GPU had at least one assigned script slot; not device compute utilization."""
        with self._lock:
            spans = self._intervals + [(s, t, end) for s, t in self._running_slots.values()]
        used = 0.0
        for gpu in range(self.num_gpus):
            last = start
            for a, b in sorted((max(start,a), min(end,b)) for s,a,b in spans if s % self.num_gpus == gpu):
                used += max(0, b-max(last,a))
                last = max(last,b)
        return {"assigned_gpu_seconds":round(used,3), "unassigned_gpu_seconds":round(max(0,(end-start)*self.num_gpus-used),3)}

    def release_slot(self, slot: int) -> None:
        self._free.put(slot)

    def run(self, node_id: str, code: str, timeout: float, abort: threading.Event | None = None,
            smoke_timeout: float | None = None, trial_env: Mapping[str,str] | None = None,
            inference_inputs: tuple[str, ...] | None = None, probability: bool = False) -> ExecResult:
        """Run ``code`` as ``solution.py`` in the node's run directory (acquires a slot).

        With ``smoke_timeout``, the script first runs with ``SUPERAGENT_SMOKE=1``; when that run
        fails, the full run is skipped and the smoke run's result is returned.
        """
        started=time.monotonic()
        deadline=started+timeout
        run_dir = self.prepare_run_dir(node_id, code)
        trial_env=dict(trial_env or {})
        trial_env["PYTHONPATH"]=os.pathsep.join(filter(None,[str(Path(__file__).resolve().parents[1]),os.environ.get("PYTHONPATH")]))
        trial_env["SUPERAGENT_TRAIN_UNTIL"]=str(deadline-float(trial_env.get("SUPERAGENT_DELIVERY_SECONDS",0)))
        slot = self.acquire_slot(abort,deadline)
        if slot is None:
            return ExecResult("[superagent] Run cancelled before start.\n", 0.0, "RuntimeError",
                              "run cancelled before start", False, None, timeout)
        with self._lock:
            self._running_slots[node_id]=(slot,time.monotonic())
        try:
            prefix = ""
            smoke: str | None = None
            smoke_secs = 0.0
            if smoke_timeout:
                sres = self._run_in_slot(run_dir, slot, min(smoke_timeout,max(0,deadline-time.monotonic())), abort, {**trial_env,SMOKE_ENV: "1"}, "smoke_stdout.log")
                self._reset_outputs(run_dir)
                smoke_secs = sres.exec_seconds
                if sres.error_message and "cancelled" in sres.error_message:
                    return sres
                if sres.timed_out:
                    smoke = "timed_out"
                    prefix = (f"[superagent] Smoke run (SUPERAGENT_SMOKE=1) did not finish within {smoke_timeout:.0f} seconds; "
                              "the full run follows.\n")
                elif sres.error_type is not None:
                    return ExecResult(SMOKE_FAILED_NOTE + sres.output, sres.exec_seconds, sres.error_type, sres.error_message,
                                      False, sres.exit_code, smoke_timeout, "failed", smoke_secs)
                else:
                    smoke = "passed"
                    prefix = f"[superagent] Smoke run (SUPERAGENT_SMOKE=1) passed in {smoke_secs:.1f} seconds.\n"
            left=deadline-time.monotonic()
            if left <= 0:
                return ExecResult("No time after smoke/queue",0,"TimeoutError","execution budget exhausted",True,None,timeout)
            res = self._run_in_slot(run_dir, slot, left, abort, trial_env)
            res.output = prefix + res.output
            res.smoke, res.smoke_seconds = smoke, smoke_secs
            if inference_inputs is not None and not res.error_type and res.exit_code == 0:
                from .artifacts import verify
                from .utils import atomic_write_json
                res.artifact_verification = verify(self, run_dir, slot, deadline, abort, inference_inputs, probability)
                atomic_write_json(run_dir/"artifact_verification.json",res.artifact_verification)
                if not res.artifact_verification["passed"]:
                    res.error_type, res.error_message = "ArtifactError", res.artifact_verification["error"]
                    res.output += "\n[artifact verifier] " + res.error_message
                res.exec_seconds = time.monotonic()-started
            return res
        finally:
            with self._lock:
                s, t = self._running_slots.pop(node_id)
                self._intervals.append((s,t,time.monotonic()))
            self.release_slot(slot)
            shutil.rmtree(run_dir / "work", ignore_errors=True)

    @staticmethod
    def _reset_outputs(run_dir: Path) -> None:
        for name in ("work", "out"):
            shutil.rmtree(run_dir / name, ignore_errors=True)
            (run_dir / name).mkdir(exist_ok=True)

    def _run_in_slot(self, run_dir: Path, slot: int, timeout: float, abort: threading.Event | None,
                     extra_env: Mapping[str, str] | None = None, log_name: str = "stdout.log", *, argv=None) -> ExecResult:
        collector = _OutputCollector(run_dir / log_name)
        t0 = time.monotonic()
        env = self._env(slot)
        if extra_env:
            env.update(extra_env)
        proc = subprocess.Popen(
            [sys.executable, *(argv if argv is not None else ["solution.py"])],
            cwd=str(run_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        try:
            os.sched_setaffinity(proc.pid, self.slot_cores[slot])
        except (AttributeError, OSError):
            pass
        with self._lock:
            self._procs[proc.pid] = proc
        pgid = proc.pid

        def reader() -> None:
            assert proc.stdout is not None
            while True:
                chunk = proc.stdout.read1(65536) if hasattr(proc.stdout, "read1") else proc.stdout.read(65536)
                if not chunk:
                    break
                collector.add(chunk.decode("utf-8", errors="replace"))

        rt = threading.Thread(target=reader, daemon=True)
        rt.start()
        timed_out = False
        cancelled = False
        while proc.poll() is None:
            elapsed = time.monotonic() - t0
            if elapsed >= timeout:
                timed_out = True
                break
            if self._kill_all.is_set() or (abort is not None and abort.is_set()):
                cancelled = True
                break
            time.sleep(POLL_SECONDS)
        if timed_out or cancelled:
            _kill_group(pgid, signal.SIGINT)
            deadline = time.monotonic() + TERMINATE_GRACE_SECONDS
            while proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            _kill_group(pgid, signal.SIGKILL)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                pass
        # Clean up any children the script left behind.
        _kill_group(pgid, signal.SIGKILL)
        rt.join(timeout=5)
        exec_seconds = time.monotonic() - t0
        with self._lock:
            self._procs.pop(proc.pid, None)
        collector.close()
        output = collector.text()
        exit_code = proc.returncode
        if cancelled:
            error_type, error_message = "RuntimeError", "run cancelled by the agent (shutdown)"
            output += f"\n[superagent] Run cancelled after {exec_seconds:.1f} seconds.\n"
        else:
            error_type, error_message = classify_result(output, exit_code, timed_out, timeout)
            if timed_out:
                output += f"\n[superagent] TIMEOUT: the script exceeded the time limit of {timeout:.0f} seconds and was stopped.\n"
            else:
                output += f"\n[superagent] Script finished in {exec_seconds:.1f} seconds (time limit {timeout:.0f} seconds).\n"
        return ExecResult(output, exec_seconds, error_type, error_message, timed_out, exit_code, timeout)

    def kill_all(self) -> None:
        """Stop every running script and refuse new ones."""
        self._kill_all.set()
        with self._lock:
            procs = list(self._procs.values())
        for p in procs:
            _kill_group(p.pid, signal.SIGINT)
        time.sleep(0.5)
        for p in procs:
            _kill_group(p.pid, signal.SIGKILL)
