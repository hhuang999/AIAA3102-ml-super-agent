"""The rule-based controller: the static decisions, adjusted by the configured rules."""

from __future__ import annotations

import logging
import statistics
import threading
from typing import TYPE_CHECKING, Any

from .base import ControlLog, Controller, Decision, StepLimits
from .hardware import available_cpus
from .rules import CostCapRule, ResourceRule, TimeAllocationRule
from .state import RunState, StepTiming

if TYPE_CHECKING:  # pragma: no cover
    from ..context import RunContext

log = logging.getLogger("superagent")


def _limits_view(lim: StepLimits) -> dict[str, Any]:
    return {"run_limit_secs": round(lim.run_limit_secs, 1), "steps_left": lim.steps_left,
            "gpu_memory_gb": lim.gpu_memory_gb, "gpu_sharing": lim.gpu_sharing}


class RuleController(Controller):
    name = "rules"

    def __init__(self, ctx: "RunContext") -> None:
        super().__init__(ctx)
        c = ctx.cfg["control"]
        self.time_rule = TimeAllocationRule(c["time_allocation"], float(ctx.cfg["execution"]["script_timeout_seconds"]))
        self.cost_rule = CostCapRule(c["cost_cap"])
        self.resource_rule = ResourceRule(c["resources"])
        self.log = ControlLog(ctx)
        self._lock = threading.Lock()
        self._once: set[str] = set()
        self._applied_limits: list[float] = []
        self._cost_curve: list[dict[str, Any]] = []
        self._cost_events: dict[str, dict[str, Any]] = {}
        self._resources: dict[str, Any] = {}
        self._plan: dict[str, Any] = {}

    # ------------------------------------------------------------------ helpers
    def _rules(self) -> dict[str, str]:
        return {r.name: r.mode for r in (self.time_rule, self.cost_rule, self.resource_rule)}

    def _first(self, key: str) -> bool:
        with self._lock:
            if key in self._once:
                return False
            self._once.add(key)
            return True

    # ------------------------------------------------------------------ lifecycle
    def start(self, state: RunState) -> None:
        cores = int(getattr(self.ctx.executor, "cores_per_slot", 1))
        self._resources = self.resource_rule.report(state, available_cpus(), cores)
        if self._resources["gpus_detected"] is not None and state.configured_gpus > (self._resources["gpus_detected"] or 0):
            log.warning("resources.num_gpus=%d but %s GPU(s) were detected", state.configured_gpus, self._resources["gpus_detected"])
        if self._resources["suggested_parallel"] < state.parallel:
            log.warning("resources.parallel=%d with %d CPU cores leaves fewer than %d cores per script",
                        state.parallel, self._resources["cpus_available"], self.resource_rule.min_cpus_per_script)
        self._plan = self.time_rule.plan(state) if self.time_rule.active else {}
        self.log.write("plan", {"controller": self.name, "rules": self._rules(), "resources": self._resources,
                                "time_allocation": self._plan,
                                "cost_cap": {"limit": self.cost_rule.limit, "currency": self.cost_rule.currency,
                                             "rate_to_cap_currency": self.cost_rule.rate}}, state=state)

    def on_step_finished(self, state: RunState, timing: StepTiming) -> None:
        capped = self.cost_rule.capped_cost(state)
        point = {"nodes": state.nodes_total, "elapsed_secs": round(state.elapsed_secs, 1), "cost": round(capped, 6),
                 "best_score": state.best_score}
        with self._lock:
            self._cost_curve.append(point)
        self.log.write("step", {"node_id": timing.node_id, "kind": timing.kind, "ok": timing.ok,
                                "step_secs": round(timing.step_seconds, 1),
                                "exec_secs": round(timing.exec_seconds, 1) if timing.exec_seconds is not None else None,
                                "capped_cost": round(capped, 6), "best_score": state.best_score}, state=state)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            limits = list(self._applied_limits)
            curve = list(self._cost_curve)
            events = dict(self._cost_events)
        lim_stats = None
        if limits:
            lim_stats = {"count": len(limits), "min": round(min(limits), 1), "median": round(statistics.median(limits), 1),
                         "max": round(max(limits), 1)}
        return {
            "controller": self.name,
            "rules": self._rules(),
            "resources": self._resources,
            "time_allocation": {"plan_at_start": self._plan, "run_limits": lim_stats},
            "cost_cap": {"limit": self.cost_rule.limit, "currency": self.cost_rule.currency,
                         "rate_to_cap_currency": self.cost_rule.rate, "events": events, "cost_curve": curve},
        }

    # ------------------------------------------------------------------ decisions
    def step_limits(self, state: RunState) -> StepLimits:
        base = super().step_limits(state)
        out = base
        proposals: dict[str, Any] = {}
        for rule in (self.time_rule, self.resource_rule):
            if not rule.active:
                continue
            prop = rule.limits(state, out)
            proposals[rule.name] = {"mode": rule.mode, "proposed": _limits_view(prop)}
            if rule.enforced:
                out = prop
        out.steps_left = min(out.steps_left, max(0, state.max_steps - state.nodes_total))
        with self._lock:
            self._applied_limits.append(out.run_limit_secs)
        if proposals:
            self.log.write("limits", {"static": _limits_view(base), "rules": proposals, "applied": _limits_view(out)}, state=state)
        return out

    def may_start_step(self, state: RunState) -> Decision:
        base = super().may_start_step(state)
        if not self.time_rule.active:
            return base
        prop = self.time_rule.start(state)
        if prop.allow != base.allow:
            self.log.write("start_decision", {"rule": self.time_rule.name, "mode": self.time_rule.mode,
                                              "static": {"allow": base.allow, "reason": base.reason, **base.details},
                                              "proposed": {"allow": prop.allow, "reason": prop.reason, **prop.details}},
                           key="start_decision", state=state)
        return prop if self.time_rule.enforced else base

    def should_finish(self, state: RunState) -> Decision:
        rule = self.cost_rule
        if not rule.active:
            return Decision(False)
        d = rule.evaluate(state)
        if d.reason in ("cost_warning", "cost_limit") and self._first(d.reason):
            outcome_key = "finished" if rule.enforced else "would_finish"
            event = {**d.details, "elapsed_secs": round(state.elapsed_secs, 1), "nodes": state.nodes_total,
                     "best_score": state.best_score, "mode": rule.mode, outcome_key: d.allow}
            with self._lock:
                self._cost_events[d.reason] = event
            self.log.write("cost_" + ("limit" if d.allow else "warning"), event, state=state)
        if d.allow and rule.enforced:
            return d
        return Decision(False, d.reason, d.details)
