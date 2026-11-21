"""Deadline-aware admission of explicit competing trial actions."""

import ast
import hashlib
from functools import lru_cache
import math
import statistics
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


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    parent_id: str
    action: str
    hypothesis: str
    source: str
    local: bool
    as_of_seq: int
    evidence_ids: tuple[str, ...] = ()
    target: str | None = None
    expected_seconds: float = 0.0
    expected_sends: int = 0
    method_id: str | None = None


@dataclass(frozen=True)
class Snapshot:
    as_of_seq: int
    facts: tuple[dict, ...]
    tried: frozenset[str]
    repairable: str | None
    incumbent: float | None
    api_available: bool
    dispatched: int
    resumable: frozenset[str]


def rank_proposals(snapshot, drafts, config, maximize):
    """Make deterministic decisions from committed observations and current leases."""
    from dataclasses import replace

    from ..llm.ledger import fingerprint

    search = config["search"]
    policy = config["control"]["adaptive"]
    threshold = search["min_gain"]
    direction = 1 if maximize else -1
    facts = snapshot.facts
    rejected = []
    scored = []
    children = {}
    for row in facts:
        children.setdefault(row["parent_id"], []).append(row)
    parents = sorted(
        (row for row in facts if row.get("score_verified")),
        key=lambda row: (-direction * row["score"], row["operation_id"]),
    )
    strength = {row["node_id"]: 1 / (1 + index * .12) for index, row in enumerate(parents)}

    def add(prop, reason, tier=2):
        record = {"proposal_id": prop.proposal_id, "parent_id": prop.parent_id, "action": prop.action, "reason": reason}
        if prop.proposal_id in snapshot.tried:
            rejected.append({**record, "reason": "duplicate_or_already_tried"})
            return
        if not prop.local and not snapshot.api_available:
            rejected.append({**record, "reason": "api_route_unavailable"})
            return
        samples = [r for r in facts if r["action"] == prop.action and r["method"] == (prop.method_id or prop.source)
                   and r.get("status") != "cancelled"][-20:]
        complete = [r["seconds"] for r in samples if r.get("deliverable")]
        timeouts = [r["seconds"] for r in samples if (r.get("reason") or "").startswith("timeout")]
        seconds = prop.expected_seconds or float(policy["initial_script_seconds"]) + float(policy["initial_other_seconds"])
        if complete:
            seconds = statistics.median(complete)
        if timeouts:
            seconds = max(seconds, max(timeouts))
        probability = (1 + len(complete)) / (2 + len(samples))
        merit = .2 + strength.get(prop.parent_id, .5) + .4 / (1 + len(children.get(prop.parent_id, ())))
        priority = merit * probability / max(seconds, 1)
        prop = replace(prop, expected_seconds=seconds)
        scored.append((tier, -priority, prop.expected_sends, prop.proposal_id, prop, {
            **record, "tier": tier, "merit": merit, "completion_probability": probability,
            "estimated_seconds": seconds, "estimated_sends": prop.expected_sends, "priority": priority,
        }))

    for draft in drafts:
        add(draft, "supplied_baseline" if draft.source == "supplied" else "distinct_root_hypothesis",
            -1 if draft.source == "supplied" else 2)

    for row in facts:
        parent = row.get("node_id")
        if not parent or not row.get("source_sha256"):
            continue
        branch_children = children.get(parent, ())
        source = "generated"

        def propose(action, why, tier=2, *, target=None, extra=(), seconds=0, slot=0):
            key = fingerprint([parent, action, target, row.get("trial_id"), row.get("rung"), row["seed"], slot])[:24]
            add(Proposal(key, parent, action, why, source, action == "promote", snapshot.as_of_seq,
                         tuple(dict.fromkeys((row["operation_id"], *extra))), target, seconds,
                         {"promote": 0, "repair": 2}.get(action, 3)), why, tier)

        non_cancelled = [r for r in branch_children if r.get("status") != "cancelled"]
        if not row["deliverable"]:
            valuable = snapshot.incumbent is None or row.get("score") is not None and direction * (row["score"] - snapshot.incumbent) > 0
            repairs_all = sum(r["action"] == "repair" for r in branch_children)
            repairs = sum(r["action"] == "repair" for r in non_cancelled)
            if row["action"] == "repair" and not row["score_verified"]:
                continue
            if row["score_verified"] and parent == snapshot.repairable:
                if repairs < search["max_repairs_per_node"] and valuable:
                    propose("repair", "repair isolated inference using the retained verified-score model",
                            0, target="predict_only", slot=repairs_all)
            elif not row["score_verified"] and row.get("status") != "cancelled":
                if snapshot.incumbent is None and repairs < search["max_repairs_per_node"]:
                    propose("repair", "repair the observed training or scoring failure", 1, slot=repairs_all)
                elif (snapshot.incumbent is not None and row["action"] == "draft"
                      and row.get("source") == "generated" and repairs < min(1, search["max_repairs_per_node"])):
                    propose("repair", "repair the failed draft to unlock a distinct model family",
                            2, slot=repairs_all)
            continue
        if len(non_cancelled) >= search["max_refinements_per_node"]:
            continue
        trial = row.get("trial_id")
        comparable = sorted(
            (r for r in facts if trial and r.get("trial_id") == trial and r.get("score_verified")
             and all(r.get(k) == row.get(k) for k in ("split_id", "metric", "source_sha256", "seed", "trial_signature"))
             and progress_available(r.get("checkpoint"))),
            key=lambda r: (r["rung"], r["seq"]),
        )
        levels = {r["rung"]: r for r in comparable}
        past = [r for rung, r in sorted(levels.items()) if rung <= row["rung"]]
        gain = None
        if len(past) > 1 and past[-1]["checkpoint"]["progress"]["optimizer_steps"] > past[-2]["checkpoint"]["progress"]["optimizer_steps"]:
            gain = direction * (past[-1]["score"] - past[-2]["score"])
        cp = row.get("checkpoint") or {}
        trials = [r for r in facts if trial and r.get("trial_id") == trial and r["action"] == "promote"]
        newest = next(reversed(levels), row.get("rung")) == row.get("rung")
        if parent in snapshot.resumable and newest and row["rung"] < policy["max_rung"]:
            extra = tuple(r["operation_id"] for r in past[-2:])
            seconds = cp["next_seconds"] + cp["delivery_seconds"] + 2
            if gain is not None and gain > threshold:
                propose("promote", "verified checkpoint and positive comparable training gain", 1, extra=extra, seconds=seconds)
            elif gain is None and not trials:
                propose("promote", "one probe of the next training rung", 1, extra=extra, seconds=seconds)
            else:
                rejected.append({"parent_id": parent, "action": "promote", "reason": "no_positive_comparable_gain" if gain is not None else "probe_already_attempted"})
        diagnostic = row.get("diagnostic") or {}
        collapsed = diagnostic.get("total", 0) >= 20 and diagnostic.get("truth_classes", 0) > 1 and diagnostic.get("predicted") == 1
        refine_slot = sum(r["action"] == "refine" for r in branch_children)
        if (collapsed and gain is not None and gain <= threshold) or row.get("verified_feature_issue"):
            why = ("test one representation, optimizer or input hypothesis after sustained class collapse"
                   if collapsed else "test a representation that preserves the observed missing information")
            propose("refine", why, 1, target=None, extra=tuple(r["operation_id"] for r in past[-2:]), slot=refine_slot)
        else:
            propose("refine", "test a targeted single-model improvement", target=None, slot=refine_slot)
        peers = [r for r in facts if r.get("branch_id") == row.get("branch_id") and r.get("gain") is not None
                 and r.get("status") == "completed"][-3:]
        if len(peers) == 3 and all(r["gain"] <= threshold for r in peers):
            retro_slot = sum(r["action"] == "retrospect" for r in branch_children)
            if not any(r["action"] == "retrospect" for r in non_cancelled):
                propose("retrospect", "change direction after three comparable non-improvements",
                        extra=tuple(r["operation_id"] for r in peers), target=None, slot=retro_slot)
            donor = next((r for r in parents if r.get("branch_id") != row.get("branch_id")
                          and r["deliverable"] and direction * (r["score"] - row["score"]) > threshold), None)
            if donor and not any(r["action"] == "borrow" for r in non_cancelled):
                borrow_slot = sum(r["action"] == "borrow" for r in branch_children)
                propose("borrow", "adapt a useful method observed on a different branch", 1,
                        extra=(donor["operation_id"],), target=None, slot=borrow_slot)
    scored.sort(key=lambda item: item[:4])
    if scored and scored[0][0] > 0 and snapshot.dispatched > 0 and (snapshot.dispatched + 1) % policy["explore_every"] == 0:
        root = next((i for i, offer in enumerate(scored) if offer[4].action == "draft"), None)
        if root is not None:
            scored.insert(0, scored.pop(root))
    return [item[4] for item in scored], [item[5] for item in scored], rejected


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
        with self.ctx.state_lock:
            pub = list(self.publish_seconds)
        return max(
            min(floor, self.ctx.time_limit * 0.1),
            2 * max(pub, default=0.0),
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
