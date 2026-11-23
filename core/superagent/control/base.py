"""The controller interface and its static implementation.

The search asks the controller every resource question: how long a script may run, whether a new
step may start, whether the search should finish early, and which search parameters to use. The
static controller answers with fixed rules taken directly from the configuration.
"""

from __future__ import annotations

import threading
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from ..utils import append_jsonl, now_iso
from .state import RunState, StepTiming

if TYPE_CHECKING:  # pragma: no cover
    from ..context import RunContext

log = logging.getLogger("superagent")

MIN_SCRIPT_SECONDS = 60.0


def required_start_seconds(guard: float, typical_script_seconds: float | None) -> float:
    """Usable time that must be left before a new step may start (static rule)."""
    if guard <= 0 or typical_script_seconds is None:
        return MIN_SCRIPT_SECONDS
    return max(MIN_SCRIPT_SECONDS, guard * typical_script_seconds)


@dataclass
class StepLimits:
    """Resource figures for one step. Shown to the model and used to run the script."""

    run_limit_secs: float
    steps_left: int
    steps_left_estimated: bool = False
    gpu_memory_gb: float | None = None
    gpu_sharing: int = 1
    budget_text: bool = False          # True: describe the budget as a planned allocation
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    allow: bool
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class Controller:
    """Static controller. Subclasses override any of the hooks below.

    All hooks are called from worker threads and from the main loop; implementations must be
    thread-safe and fast (no LLM calls, no blocking I/O except small log writes).
    """

    name = "static"

    def __init__(self, ctx: "RunContext") -> None:
        self.ctx = ctx

    # ------------------------------------------------------------------ lifecycle
    def start(self, state: RunState) -> None:
        """Called once before the first step."""

    def on_step_finished(self, state: RunState, timing: StepTiming) -> None:
        """Called after every evaluated node (``state`` already contains it)."""

    def summary(self) -> dict[str, Any]:
        """Block written to ``report/run_summary.json`` under ``control``."""
        return {"controller": self.name}

    # ------------------------------------------------------------------ decisions
    def step_limits(self, state: RunState) -> StepLimits:
        cfg = self.ctx.cfg
        limit = min(float(cfg["execution"]["script_timeout_seconds"]), state.usable_secs)
        return StepLimits(run_limit_secs=max(0.0, limit), steps_left=max(0, state.max_steps - state.nodes_total))

    def may_start_step(self, state: RunState) -> Decision:
        guard = float(self.ctx.cfg["run"]["late_start_guard"])
        need = required_start_seconds(guard, state.median_ok_script_seconds())
        if state.usable_secs >= need:
            return Decision(True, "enough_time", {"need_secs": round(need, 1)})
        return Decision(False, "not_enough_time", {"need_secs": round(need, 1), "usable_secs": round(state.usable_secs, 1)})

    def should_finish(self, state: RunState) -> Decision:
        return Decision(False)

    def search_config(self, state: RunState | None = None) -> Mapping[str, Any]:
        """Configuration used by selection, dispatch and chain control."""
        return self.ctx.cfg


class ControlLog:
    """Appends rows to ``logs/control.jsonl``; skips rows identical to the previous one of the same key."""

    def __init__(self, ctx: "RunContext") -> None:
        self.ctx = ctx
        self._last: dict[str, Any] = {}
        self._lock = threading.Lock()

    def write(self, event: str, payload: Mapping[str, Any], key: str | None = None, state: RunState | None = None) -> None:
        if key is not None:
            with self._lock:
                if self._last.get(key) == payload:
                    return
                self._last[key] = dict(payload)
        row: dict[str, Any] = {"ts": now_iso(), "elapsed_secs": round(self.ctx.elapsed(), 1), "event": event}
        if state is not None:
            row["nodes"] = state.nodes_total
            row["cost"] = round(state.cost, 6)
        row.update(payload)
        try:
            append_jsonl(self.ctx.paths.control_log, row)
        except OSError as exc:  # pragma: no cover - logging must never break the run
            log.warning("could not write the control log: %s", exc)
