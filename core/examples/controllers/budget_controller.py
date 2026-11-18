"""Example controller: the rule-based controller plus two extra rules.

- Finish the search once the model cost reaches a fixed share of the cap, without waiting for the cap.
- Give scripts at most a fixed share of the remaining usable time.

Use it with a config file containing:

    control:
      controller: "budget_controller:BudgetController"

and run the agent from a directory where ``budget_controller.py`` is importable
(for example with ``PYTHONPATH=examples/controllers``).
"""

from __future__ import annotations

from dataclasses import replace

from superagent.control import Decision, RuleController, RunState, StepLimits

COST_SHARE = 0.9
MAX_SHARE_OF_REMAINING = 0.25


class BudgetController(RuleController):
    name = "budget"

    def should_finish(self, state: RunState) -> Decision:
        base = super().should_finish(state)
        if base.allow:
            return base
        rule = self.cost_rule
        cost = rule.capped_cost(state)
        if rule.limit > 0 and cost >= COST_SHARE * rule.limit:
            self.log.write("budget_finish", {"cost": round(cost, 4), "limit": rule.limit}, key="budget_finish", state=state)
            return Decision(True, "cost_share_reached", {"cost": round(cost, 4), "limit": rule.limit})
        return base

    def step_limits(self, state: RunState) -> StepLimits:
        lim = super().step_limits(state)
        cap = MAX_SHARE_OF_REMAINING * state.usable_secs
        if lim.run_limit_secs > cap:
            lim = replace(lim, run_limit_secs=round(cap, 1))
        return lim
