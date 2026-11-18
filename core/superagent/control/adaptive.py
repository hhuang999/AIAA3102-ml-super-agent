"""Deadline-aware admission of explicit competing trial actions."""

import ast
import hashlib
from functools import lru_cache
import math
import time
from dataclasses import dataclass

from .base import Controller, ControlLog, Decision
from ..training import progress_available
from ..scheduling import branch_plateau, synthesis_conditions


def quantile(values, q):
    return sorted(values)[math.ceil(q * len(values)) - 1]


@lru_cache(maxsize=256)
def code_signature(code):
    try:
        canonical = ast.dump(ast.parse(code), include_attributes=False)
    except SyntaxError:
        canonical = code
    return hashlib.sha256(canonical.encode()).hexdigest()


@lru_cache(maxsize=256)
def method_shape(code):
    """A hint only: changing a literal or hyperparameter can still be informative."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code_signature(code)
    for expr in ast.walk(tree):
        if isinstance(expr, ast.Constant):
            expr.value = type(expr.value).__name__
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


@dataclass
class Offer:
    parent: object
    action: str
    reason: str
    seconds: float
    execution: float
    value: float
    fee: float = 0.0

    def record(self):
        return dict(
            parent=self.parent.id,
            action=self.action,
            reason=self.reason,
            parent_score=self.parent.score,
            parent_gain=self.parent.parent_gain,
            complementarity=self.parent.complementarity,
            predicted_seconds=round(self.seconds, 2),
            execution_seconds=round(self.execution, 2),
            estimated_fee=round(self.fee, 6),
            priority=round(self.value, 6),
        )


class AdaptiveController(Controller):
    name = "adaptive"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.policy = ctx.cfg["control"]["adaptive"]
        self.log = ControlLog(ctx)
        self.leases = set()
        self.scripts = set()
        self.active_shapes = {}
        self.exhausted = set()
        self.turn = 0
        self.censored = []
        self.publish_seconds = []
        self.denials = 0

    def start(self, state):
        self.log.write(
            "adaptive_start",
            {"policy": self.policy, "limit_seconds": state.time_limit_secs},
            state=state,
        )

    def reserve(self):
        floor = float(self.ctx.cfg["run"]["finalize_reserve_seconds"])
        return max(
            min(floor, self.ctx.time_limit * 0.1),
            2 * max(self.publish_seconds, default=0.0),
        )

    def estimate(self, kind, state, parent=None):
        cfg = self.policy
        samples = [
            s
            for s in state.step_timings
            if s.kind == kind and s.exec_seconds is not None
        ]
        # A cheap supplied draft is not an observation of planning and patching.
        q = (
            0.9
            if state.time_left_secs < 0.1 * state.time_limit_secs
            else float(cfg["admission_quantile"])
        )
        if kind == "promote":
            cp = parent.checkpoint
            run = cp["next_seconds"] + cp["delivery_seconds"]
            return 1.25 * run + 2, 1.25 * run
        cold = min(
            float(cfg["initial_script_seconds"]), max(5, state.time_limit_secs * 0.15)
        )
        overhead = min(
            float(cfg["initial_other_seconds"]), max(2, state.time_limit_secs * 0.04)
        )
        whole = [s.step_seconds for s in samples[-24:] if s.ok]
        lower = [t for action, t in self.censored[-16:] if action == kind] + [
            s.step_seconds for s in samples[-24:] if not s.ok
        ]
        script = [s.exec_seconds for s in samples[-24:] if s.exec_seconds]
        run = max(2, quantile(script, q) if script else cold)
        # Censored durations are lower bounds; never treat them as faster completions.
        need = max(
            quantile(whole, q) if whole else run + overhead,
            quantile(lower, q) if lower else 0,
            run,
        )
        if len(whole) < 3:
            need = max(need, run + overhead)
        return need, run

    def may_start_step(self, state):
        # The action-specific admission follows under the tree lock. Cheap promotions
        # can still fit when a full code change cannot.
        return Decision(state.usable_secs >= 2, "before_delivery_reserve")

    def step_limits(self, state):
        base = super().step_limits(state)
        _, run = self.estimate("draft", state)
        base.run_limit_secs = min(base.run_limit_secs, max(2, run * 1.4))
        return base

    def summary(self):
        return {
            "controller": self.name,
            "admission_denials": self.denials,
            "censored_operations": len(self.censored),
        }

    def choose(self, state, *, reserve=True):
        """Caller owns the tree lock until the selected child slot is reserved."""
        ctx = self.ctx
        tree = ctx.tree
        cfg = ctx.cfg
        offers = []
        infeasible = []
        has_pending = bool(self.leases)
        remaining = state.usable_secs
        g = ctx.gateway
        cooling = getattr(g, "cooldown_until", 0) > time.monotonic()
        ledger = getattr(g, "ledger", None)
        account = ledger.summary() if ledger and ledger.limit is not None else None
        recent = state.step_timings[-50:]
        charges = ledger.charges(t.node_id for t in recent) if account else {}
        balance = (
            ledger.limit - account["known_cost"] - account["risk_reserve"]
            if account
            else None
        )
        money_blocked = bool(
            account and account["known_cost"] + account["risk_reserve"] >= ledger.limit
        )

        def add(n, kind, reason, score):
            if (n.id, kind) in self.leases or (n.id, kind) in self.exhausted:
                return
            if kind != "promote" and (
                cooling or money_blocked or getattr(g, "endpoint_failure", None)
            ):
                return
            need, run = self.estimate(kind, state, n)
            sample = [s for s in state.step_timings if s.kind == kind][-20:]
            prob = (1 + sum(s.ok for s in sample)) / (2 + len(sample))
            # Estimate whole operations for selection. Per-send SQLite reservations
            # remain authoritative if two workers compete for the same balance.
            fees = [
                charges[t.node_id]
                for t in recent
                if t.kind == kind and t.node_id in charges
            ]
            fee = (
                0.0
                if kind == "promote"
                else (
                    quantile(fees, 0.8)
                    if fees
                    else self.policy["initial_operation_cost"]
                )
            )
            pressure = 1 + fee / max(balance, 1e-9) if balance is not None else 1
            item = Offer(
                n,
                kind,
                reason,
                need,
                run,
                score * prob / (max(need, 1) * pressure),
                fee,
            )
            (
                offers
                if need <= remaining
                and (balance is None or fee <= balance or kind == "promote")
                else infeasible
            ).append(item)

        root = tree.root
        drafts = sum(n.kind == "draft" for n in root.children) + root.pending_children
        if drafts < cfg["search"]["max_drafts"]:
            add(root, "draft", "test a different root approach", 0.7 / (1 + drafts))
        if synthesis_conditions(tree, cfg, ctx.progress()):
            add(
                root,
                "synthesis",
                "combine distinct successful branches on a global plateau",
                1.3,
            )
        rank = {n.id: i for i, n in enumerate(tree.sorted_ok())}
        for n in tree.nodes[1:]:
            if n.dead_end:
                continue
            children = n.children
            if n.is_failed:
                if not children and not n.pending_children:
                    add(
                        n,
                        "repair",
                        "recover an untested branch with a diagnosed failure",
                        0.45,
                    )
                continue
            if not n.ok:
                continue
            strength = 1 / (1 + rank.get(n.id, 0) * 0.12)
            novelty = 1 / (1 + len(children))
            merit = 0.2 + strength + novelty * 0.4 + min(0.5, n.complementarity or 0)
            cp = n.checkpoint
            if (
                cp
                and cp["can_continue"]
                and progress_available(cp)
                and n.rung < cfg["control"]["adaptive"]["max_rung"]
                and not any(c.kind == "promote" for c in children)
            ):
                if n.rung == 0 or (n.parent_gain is not None and n.parent_gain > 0):
                    add(
                        n,
                        "promote",
                        "continue a verified checkpoint without regenerating code",
                        merit,
                    )
            if (
                len(children) + n.pending_children
                >= cfg["search"]["max_refinements_per_node"]
            ):
                continue
            plateau = branch_plateau(
                tree,
                n.branch_id,
                cfg["plateau"]["branch_window"],
                cfg["plateau"]["enabled"],
                True,
            )
            if plateau and cfg["operators"]["retrospect"]["enabled"]:
                add(
                    n,
                    "retrospect",
                    "test a revised hypothesis using the branch trajectory",
                    merit * 0.85,
                )
                if (
                    cfg["operators"]["borrow"]["enabled"]
                    and len(rank) > 1
                    and (ctx.progress() or 0)
                    >= cfg["operators"]["borrow"]["start_fraction"]
                ):
                    add(n, "borrow", "try a method from another branch", merit * 0.7)
            else:
                add(
                    n,
                    "refine",
                    "change the parent's method; compare with unchanged-code continuation",
                    merit * novelty,
                )
        self.denials += len(infeasible)
        if not offers:
            pending = has_pending or any(
                not n.in_tree for n in tree.by_id.values() if not n.is_root
            )
            reason = (
                "wait_for_pending" if pending or cooling else "no_deliverable_action"
            )
            return None, dict(
                reason=reason,
                available=remaining,
                rejected=[i.record() for i in infeasible],
            )
        self.turn += 1
        exploration = [
            i
            for i in offers
            if i.action in ("draft", "retrospect", "borrow", "synthesis")
        ]
        pool = (
            exploration
            if exploration and self.turn % self.policy["explore_every"] == 0
            else offers
        )
        choice = max(pool, key=lambda o: o.value)
        if reserve:
            self.leases.add((choice.parent.id, choice.action))
            if choice.action == "synthesis":
                tree.synthesis_count += 1
            else:
                choice.parent.pending_children += 1
        log = dict(
            available_seconds=remaining,
            selected=choice.record(),
            alternatives=[
                o.record()
                for o in sorted(offers, key=lambda o: -o.value)
                if o is not choice
            ][:4],
            rejected=[o.record() for o in infeasible],
        )
        return choice, log
