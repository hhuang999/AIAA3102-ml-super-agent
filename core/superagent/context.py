"""Run-wide state shared by every component: configuration, paths, clocks and services."""

from __future__ import annotations

import logging
import random
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .control.base import MIN_SCRIPT_SECONDS, required_start_seconds
from .control.state import GpuInfo, StepTiming
from .prompts.blocks import Resources
from .tree import Direction, SearchTree

if TYPE_CHECKING:  # pragma: no cover
    from .control.base import Controller, StepLimits
    from .executor import Executor
    from .experience import ExperienceStore
    from .llm.gateway import LLMGateway

__all__ = ["MIN_SCRIPT_SECONDS", "required_start_seconds", "OutputPaths", "LockedRandom", "RunContext"]

log = logging.getLogger("superagent")


class OutputPaths:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.status = self.root / "status.json"
        self.report = self.root / "report"
        self.best = self.root / "best"
        self.top = self.root / "top"
        self.ensemble = self.root / "ensemble"
        self.logs = self.root / "logs"
        self.experience = self.root / "experience"
        self.prepared = self.root / "prepared_data"
        self.runs = self.root / "runs"
        self.search_tree = self.logs / "search_tree.json"
        self.decisions = self.logs / "decisions.jsonl"
        self.llm_calls = self.logs / "llm_calls.jsonl"
        self.embedding_calls = self.logs / "embedding_calls.jsonl"
        self.run_log = self.logs / "superagent.log"
        self.config_used = self.logs / "config_used.yaml"
        self.control_log = self.logs / "control.jsonl"
        self.final = self.root / "final"
        self.records = self.experience / "records.json"

    def make_dirs(self) -> None:
        for d in (self.report, self.logs, self.experience, self.prepared, self.runs):
            d.mkdir(parents=True, exist_ok=True)


class LockedRandom:
    """random.Random guarded by a lock, so worker threads can share it."""

    def __init__(self, seed: int) -> None:
        self._r = random.Random(seed)
        self._lock = threading.Lock()

    def random(self) -> float:
        with self._lock:
            return self._r.random()

    def choice(self, seq: Any) -> Any:
        with self._lock:
            return self._r.choice(seq)

    def shuffle(self, seq: list[Any]) -> None:
        with self._lock:
            self._r.shuffle(seq)


@dataclass
class RunContext:
    cfg: dict[str, Any]
    paths: OutputPaths
    task_dir: Path
    direction: Direction
    direction_source: str
    tree: SearchTree
    gateway: "LLMGateway"
    executor: "Executor"
    experience: "ExperienceStore | None"
    task_desc: str
    data_preview: str
    catalog_text: str
    modalities: list[str]
    sample_path: Path | None
    rng: LockedRandom
    process_start: float
    started_at: str
    stop_event: threading.Event = field(default_factory=threading.Event)    # no new tasks
    hard_event: threading.Event = field(default_factory=threading.Event)    # abandon generation now
    stop_reason: str | None = None
    llm_failure_streak: int = 0
    state_lock: threading.RLock = field(default_factory=threading.RLock)
    io_lock: threading.RLock = field(default_factory=threading.RLock)
    in_flight: int = 0
    key_events: list[dict[str, Any]] = field(default_factory=list)
    flags: dict[str, Any] = field(default_factory=dict)
    control: "Controller | None" = None
    gpu_info: GpuInfo | None = None
    task_spec: Any = None
    active_tasks: int = 0
    step_timings: list[StepTiming] = field(default_factory=list)
    _local: threading.local = field(default_factory=threading.local)

    # ---------------------------------------------------------------- clocks
    @property
    def time_limit(self) -> float:
        return float(self.cfg["run"]["time_limit_seconds"])

    @property
    def deadline(self) -> float:
        return self.process_start + self.time_limit

    @property
    def reserve(self) -> float:
        if self.control is not None and self.control.name=="adaptive":
            return self.control.reserve()
        return float(self.cfg["run"]["finalize_reserve_seconds"])

    def time_left(self) -> float:
        return self.deadline - time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.process_start

    def progress(self) -> float | None:
        start = self.tree.search_start
        if start is None:
            return None
        return (time.monotonic() - start) / self.time_limit

    def run_limit(self) -> float:
        return min(float(self.cfg["execution"]["script_timeout_seconds"]), self.time_left() - self.reserve)

    def typical_script_seconds(self) -> float | None:
        """Median run time of the successful scripts so far (None before the first success)."""
        with self.tree.lock:
            times = [n.exec_seconds for n in self.tree.nodes if n.ok and n.exec_seconds is not None]
        return statistics.median(times) if times else None

    def can_start_task(self) -> bool:
        """Whether a new step is still worth starting.

        A script needs at least MIN_SCRIPT_SECONDS. Once successful scripts exist, a step started
        when less than ``late_start_guard`` times their median run time is left would most likely
        be cut off by the deadline, so it is not started.
        """
        return self.time_left() - self.reserve >= required_start_seconds(
            float(self.cfg["run"]["late_start_guard"]), self.typical_script_seconds())

    def should_abort_generation(self) -> bool:
        return self.hard_event.is_set() or self.time_left() <= self.reserve

    def generation_window_expired(self) -> bool:
        """Protect the execution estimate of the current worker's admitted action."""
        execution = getattr(self._local, "generation_execution_reserve", 0.0)
        return execution > 0 and self.time_left() <= self.reserve + execution

    def steps_left(self) -> int:
        return max(0, int(self.cfg["run"]["max_steps"]) - len(self.tree.solution_nodes()))

    @property
    def mode(self) -> str:
        return str(self.cfg["pretrained"]["mode"])

    # ---------------------------------------------------------------- control
    def begin_step(self) -> "StepLimits | None":
        """Ask the controller for this step's limits and keep them for the current thread."""
        if self.control is None:
            self._local.limits = None
            return None
        from .control.state import build_state

        try:
            limits = self.control.step_limits(build_state(self))
        except Exception as exc:  # noqa: BLE001 - a broken controller must not stop the run
            log.warning("controller.step_limits failed (%s); using the fixed script time limit", exc)
            limits = None
        self._local.limits = limits
        return limits

    def current_limits(self) -> "StepLimits | None":
        return getattr(self._local, "limits", None)

    def current_run_limit(self) -> float:
        """Script time limit for the current thread's step; never beyond the time left."""
        lim = self.current_limits()
        hard = self.time_left() - self.reserve
        if lim is None:
            return min(float(self.cfg["execution"]["script_timeout_seconds"]), hard)
        return min(lim.run_limit_secs, hard)

    def search_cfg(self) -> dict[str, Any]:
        """Configuration used by selection, dispatch and chain control (from the controller)."""
        if self.control is None:
            return self.cfg
        from .control.state import build_state

        try:
            return self.control.search_config(build_state(self))  # type: ignore[return-value]
        except Exception as exc:  # noqa: BLE001 - a broken controller must not stop the run
            log.warning("controller.search_config failed (%s); using the configuration", exc)
            return self.cfg

    def smoke_seconds(self) -> float:
        ex = self.cfg["execution"]
        return float(ex["smoke_timeout_seconds"]) if ex["smoke_run"] else 0.0

    def record_step(self, timing: StepTiming) -> None:
        with self.state_lock:
            self.step_timings.append(timing)

    def resources(self) -> Resources:
        lim = self.current_limits()
        if lim is None:
            return Resources(
                time_left_secs=self.time_left(),
                steps_left=self.steps_left(),
                run_limit_secs=max(0.0, self.run_limit()),
                cpus_per_run=self.executor.cores_per_slot,
                num_gpus=int(self.cfg["resources"]["num_gpus"]),
                mode=self.mode,
                smoke_seconds=self.smoke_seconds(),
            )
        return Resources(
            time_left_secs=self.time_left(),
            steps_left=lim.steps_left,
            run_limit_secs=max(0.0, lim.run_limit_secs),
            cpus_per_run=self.executor.cores_per_slot,
            num_gpus=int(self.cfg["resources"]["num_gpus"]),
            mode=self.mode,
            steps_left_estimated=lim.steps_left_estimated,
            budget_text=lim.budget_text,
            gpu_memory_gb=lim.gpu_memory_gb,
            gpu_sharing=lim.gpu_sharing,
            smoke_seconds=self.smoke_seconds(),
        )

    def add_event(self, type_: str, node_id: str | None, detail: str, step: int | None = None) -> None:
        with self.state_lock:
            self.key_events.append({
                "step": step if step is not None else len(self.tree.nodes),
                "elapsed_secs": round(self.elapsed(), 1),
                "type": type_,
                "node_id": node_id,
                "detail": detail,
            })
