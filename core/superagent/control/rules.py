"""Control rules. Each rule proposes a change to the static decision; its mode says what happens to it.

Modes:
- ``off``: the rule is not evaluated.
- ``observe``: the proposal is computed and written to the control log, but not applied.
- ``enforce``: the proposal is applied.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping

from .base import Decision, StepLimits
from .state import RunState

MODES = ("off", "observe", "enforce")


class Rule:
    name = "rule"

    def __init__(self, block: Mapping[str, Any]) -> None:
        self.mode = str(block["mode"])
        if self.mode not in MODES:
            raise ValueError(f"control rule {self.name}: unknown mode {self.mode!r}")

    @property
    def active(self) -> bool:
        return self.mode != "off"

    @property
    def enforced(self) -> bool:
        return self.mode == "enforce"


class TimeAllocationRule(Rule):
    """Splits the remaining time into steps of a planned size.

    The time one execution slot has left is divided among the steps still planned for it. A step
    consists of generation (mostly LLM calls) and the script run; the script receives what is left
    of the step's share after the measured generation time. The result is kept between a floor
    and the configured script timeout, and never exceeds the time left.
    """

    name = "time_allocation"

    def __init__(self, block: Mapping[str, Any], script_timeout_seconds: float) -> None:
        super().__init__(block)
        self.target_nodes = int(block["target_nodes"])
        self.min_script = float(block["min_script_seconds"])
        self.tail_min_script = float(block["tail_min_script_seconds"])
        self.other_prior = float(block["llm_seconds_prior"])
        self.recent = int(block["recent_steps"])
        self.max_script = float(script_timeout_seconds)

    def other_seconds(self, state: RunState) -> float:
        measured = state.median_other_seconds(self.recent)
        return measured if measured is not None else self.other_prior

    def plan(self, state: RunState) -> dict[str, Any]:
        other = self.other_seconds(state)
        usable = state.usable_secs
        in_flight_others = max(0, state.active_tasks - 1)
        remaining = max(1, self.target_nodes - state.nodes_total - in_flight_others)
        slot_budget = usable * max(1, state.parallel) / remaining
        script = min(max(slot_budget - other, self.min_script), self.max_script)
        script = max(0.0, min(script, usable - other))
        measured_step = state.median_step_seconds(self.recent)
        step_est = measured_step if measured_step is not None else script + other
        steps_left = int(usable * max(1, state.parallel) // max(step_est, 1.0))
        steps_left = min(steps_left, max(0, state.max_steps - state.nodes_total))
        return {
            "script_secs": round(script, 1),
            "other_secs": round(other, 1),
            "slot_budget_secs": round(slot_budget, 1),
            "remaining_target_steps": remaining,
            "steps_left": steps_left,
            "step_estimate_measured": measured_step is not None,
        }

    def limits(self, state: RunState, base: StepLimits) -> StepLimits:
        p = self.plan(state)
        details = dict(base.details)
        details[self.name] = p
        return replace(base, run_limit_secs=p["script_secs"], steps_left=p["steps_left"], steps_left_estimated=True,
                       budget_text=True, details=details)

    def start(self, state: RunState) -> Decision:
        other = self.other_seconds(state)
        room = state.usable_secs - other
        if room >= self.tail_min_script:
            return Decision(True, "enough_time_for_a_short_step", {"script_room_secs": round(room, 1)})
        return Decision(False, "not_enough_time", {"script_room_secs": round(room, 1), "need_secs": self.tail_min_script})


class CostCapRule(Rule):
    """Finishes the search before the recorded model cost would pass a limit."""

    name = "cost_cap"

    def __init__(self, block: Mapping[str, Any]) -> None:
        super().__init__(block)
        self.limit = float(block["limit"])
        self.currency = str(block["currency"])
        self.rate = float(block["rate_to_cap_currency"])
        self.warn_fraction = float(block["warn_fraction"])
        self.stop_on_warning = bool(block.get("stop_on_warning", False))

    @property
    def warn_limit(self) -> float:
        return self.warn_fraction * self.limit

    def capped_cost(self, state: RunState) -> float:
        return state.cost * self.rate

    def evaluate(self, state: RunState, *, extra_known_cost: float = 0.0) -> Decision:
        cost = max(self.capped_cost(state), float(extra_known_cost) * self.rate)
        warn_limit = self.warn_limit
        per_step = cost / max(1, state.nodes_total)
        projected = cost + per_step * max(1, state.active_tasks)
        details = {
            "cost": round(cost, 4),
            "projected": round(projected, 4),
            "warn_limit": round(warn_limit, 4),
            "limit": self.limit,
            "currency": self.currency,
            "stop_on_warning": self.stop_on_warning,
        }
        if self.limit > 0 and cost >= self.limit:
            return Decision(True, "cost_limit", details)
        if self.limit > 0 and cost >= warn_limit and self.stop_on_warning:
            return Decision(True, "cost_warning", details)
        if self.limit > 0 and projected >= self.limit:
            return Decision(True, "cost_limit", details)
        if self.limit > 0 and cost >= warn_limit:
            return Decision(False, "cost_warning", details)
        return Decision(False, "within_limit", details)


class ResourceRule(Rule):
    """Tells each script how much GPU memory it can count on, from the detected hardware."""

    name = "resources"

    def __init__(self, block: Mapping[str, Any]) -> None:
        super().__init__(block)
        self.min_cpus_per_script = int(block["min_cpus_per_script"])

    def gpu_share(self, state: RunState) -> tuple[float | None, int]:
        gpu = state.gpu
        if state.configured_gpus <= 0 or gpu is None or gpu.count <= 0 or not gpu.memory_mb:
            return None, 1
        n = min(state.configured_gpus, gpu.count)
        sharing = max(1, math.ceil(max(1, state.parallel) / n))
        mem_gb = min(gpu.memory_mb[:n]) / 1024.0 / sharing
        return round(mem_gb, 1), sharing

    def limits(self, state: RunState, base: StepLimits) -> StepLimits:
        mem, sharing = self.gpu_share(state)
        if mem is None:
            return base
        details = dict(base.details)
        details[self.name] = {"gpu_memory_gb": mem, "gpu_sharing": sharing}
        return replace(base, gpu_memory_gb=mem, gpu_sharing=sharing, details=details)

    def report(self, state: RunState, cpus_available: int, cores_per_slot: int) -> dict[str, Any]:
        mem, sharing = self.gpu_share(state)
        suggested = max(1, min(state.parallel, cpus_available // max(1, self.min_cpus_per_script)))
        gpu = state.gpu
        return {
            "cpus_available": cpus_available,
            "cores_per_script": cores_per_slot,
            "gpus_configured": state.configured_gpus,
            "gpus_detected": gpu.count if gpu is not None else None,
            "gpu_names": list(gpu.names) if gpu is not None else [],
            "gpu_memory_mb": list(gpu.memory_mb) if gpu is not None else [],
            "gpu_memory_per_script_gb": mem,
            "scripts_per_gpu": sharing,
            "parallel": state.parallel,
            "suggested_parallel": suggested,
        }
