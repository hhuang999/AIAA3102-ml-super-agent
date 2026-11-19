"""Candidate claims, stage recovery and coordination for the isolated search."""

from __future__ import annotations

import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import asdict, dataclass

from . import single_model
from .control.adaptive import Proposal, Snapshot, rank_proposals
from .experience import Record, decision_context
from .errors import LLMUnavailableError, StructuredOutputError
from .llm.ledger import CancelledCall, DeferredCall, fingerprint
from .review import apply_policy, review
from .search import Discarded
from .utils import append_jsonl, new_id, now_iso

log = logging.getLogger("superagent")
DRAFT_ROLES = ("draft_writer.data_prep", "draft_writer.model_def", "draft_writer.train_eval", "draft_integrator")


@dataclass
class Dispatch:
    id: str
    proposal: Proposal
    signature: str
    decision_id: str
    node: object | None = None
    stage: str = "claimed"
    resume_at: float = 0.0
    started: float = 0.0
    evidence: tuple[dict, ...] = ()


class Operations:
    def __init__(self, controller):
        self.controller = controller
        self.ctx = controller.ctx
        self.ledger = self.ctx.gateway.ledger
        self.claimed = {}
        self.deferred = {}
        self.facts = []
        self.ctx.flags["active_parent_leases"] = set()
        self.parent_claims = {}
        self.ctx.flags["baseline_claimed"] = True  # The supplied path is picked explicitly by this scheduler.
        self.ctx.tree.search_start = time.monotonic()

    def _drafts(self, snapshot):
        ctx = self.ctx
        parent = ctx.tree.root.id
        sequence = snapshot.as_of_seq
        if (ctx.task_dir / "baseline.py").is_file():
            yield Proposal("supplied", parent, "draft", "Supplied baseline", "supplied", True, sequence)
        max_drafts = min(ctx.cfg["search"]["initial_drafts"], ctx.cfg["search"]["max_drafts"])
        position = 1 if (ctx.task_dir / "baseline.py").is_file() else 0
        methods = ctx.flags["draft_methods"]
        if not methods:
            if position < max_drafts:
                yield Proposal("draft-discover", parent, "draft", "Use the declared reader with a feasible single model",
                               "generated", False, sequence, expected_sends=len(self._roles_draft()))
            return
        explore = snapshot.dispatched > 0 and (snapshot.dispatched + 1) % ctx.cfg["control"]["adaptive"]["explore_every"] == 0
        limit = ctx.cfg["search"]["max_drafts"] if explore else max_drafts
        for method in methods[:max(0, limit - position)]:
            recipe = ctx.cfg["search"]["local_recipes"] and method.recipe_id == "tab_linear@1"
            hypothesis = f"{method.key} — {method.hypothesis}"
            yield Proposal(f"draft-{method.key}", parent, "draft", hypothesis, "recipe" if recipe else "generated",
                           recipe, sequence, expected_seconds=method.expected_seconds,
                           expected_sends=0 if recipe else len(self._roles_draft()), method_id=method.key)

    def snapshot(self):
        from .training import progress_available

        ctx = self.ctx
        with ctx.tree.lock:
            resumable = frozenset(n.id for n in ctx.tree.nodes if n.checkpoint and n.checkpoint.get("can_continue")
                                  and progress_available(n.checkpoint))
            best = ctx.tree.best
        return Snapshot(len(self.facts), tuple(self.facts), frozenset(self.claimed), ctx.flags.get("repairable_id"),
                        best.score if best else None, not ctx.gateway.endpoint_failure, len(self.claimed), resumable)

    def propose(self, snapshot=None):
        snapshot = self.snapshot() if snapshot is None else snapshot
        return rank_proposals(snapshot, self._drafts(snapshot), self.ctx.cfg, self.ctx.direction.higher_is_better)

    def _signature(self, proposal):
        ctx = self.ctx
        parent = ctx.tree.by_id[proposal.parent_id]
        return fingerprint([
            ctx.flags["trial_environment"], proposal.proposal_id, parent.id, parent.code,
            ctx.cfg["llm"], ctx.cfg["execution"], ctx.cfg["review"], ctx.cfg["run"]["seed"],
            parent.checkpoint.get("sha256") if parent.checkpoint else None,
        ])

    def claim(self, proposal, snapshot_seq):
        """Acquire one candidate name and slot before the next selection."""
        ctx = self.ctx
        if ctx.stop_event.is_set() or len(self.claimed) >= ctx.cfg["run"]["max_steps"]:
            return None
        with ctx.tree.lock:
            if proposal.proposal_id in self.claimed or snapshot_seq != len(self.facts):
                return None
            dispatch = Dispatch(new_id(), proposal, self._signature(proposal), new_id(), started=time.monotonic())
            self.claimed[proposal.proposal_id] = dispatch
        normal = self._roles(proposal)
        try:
            self.ledger.register_operation(dispatch.id, dispatch.signature, normal)
        except Exception:
            with ctx.tree.lock:
                del self.claimed[proposal.proposal_id]
            raise
        ctx.flags["claimed_steps"] = len(self.claimed)
        dispatch.evidence = tuple(decision_context(self.facts, proposal, ctx.experience))
        if not ctx.tree.by_id[proposal.parent_id].is_root:
            self.parent_claims[proposal.parent_id] = self.parent_claims.get(proposal.parent_id, 0) + 1
            ctx.flags["active_parent_leases"].add(proposal.parent_id)
            ctx.flags["artifact_leases"].add(proposal.parent_id)
        return dispatch

    def _roles(self, proposal):
        if proposal.local:
            return ()
        if proposal.action == "draft":
            return self._roles_draft()
        if proposal.action == "repair" and proposal.target == "predict_only" and "predict_only" in self.ctx.tree.by_id[proposal.parent_id].code:
            return ("repair_coder", "reviewer")
        return ("planner.structured", "patch_coder", "reviewer")

    def _roles_draft(self):
        return (*DRAFT_ROLES, "reviewer") if self.ctx.cfg["codegen"]["staged_drafting"] else ("script_coder", "reviewer")

    def _state(self, dispatch, state, delay=0):
        dispatch.stage = state
        self.ledger.state(dispatch.id, state, delay)

    def _candidate(self, dispatch):
        ctx = self.ctx
        proposal = dispatch.proposal
        parent = ctx.tree.by_id[proposal.parent_id]
        ctx._local.operation_id = dispatch.id
        ctx._local.selected_hypothesis = proposal.hypothesis
        try:
            if proposal.source == "supplied":
                path = ctx.task_dir / "baseline.py"
                return self.controller.ops._new_node(dispatch.id, "draft", parent, proposal.hypothesis,
                                                     path.read_text(encoding="utf8"))
            if proposal.source == "recipe":
                from pathlib import Path

                facts = ctx.flags["task_facts"]
                recipe = {key: facts[key] for key in ("input_manifest", "classes", "submission_columns", "probability")}
                recipe["seed"] = ctx.cfg["run"]["seed"]
                template = Path(__file__).with_name("tabular_recipe.py").read_text(encoding="utf8")
                code = template.replace("\nTASK = None\n", f"\nTASK = {recipe!r}\n", 1)
                return self.controller.ops._new_node(dispatch.id, "draft", parent, proposal.hypothesis, code)
            if proposal.action == "promote":
                node = self.controller.ops._new_node(dispatch.id, "promote", parent, parent.plan, parent.code)
                node.trial_id = parent.trial_id or parent.id
                node.rung = parent.rung + 1
                return node
            if proposal.action == "draft":
                return self.controller.ops.draft(parent).node

            def planned(plan):
                self.ledger.put(dispatch.id, "accepted_plan", dispatch.signature, plan)
                self._state(dispatch, "planned")

            return self.controller.ops.experiment_candidate(dispatch, parent, dispatch.evidence, planned)
        finally:
            ctx._local.operation_id = None
            ctx._local.selected_hypothesis = None

    def _work(self, dispatch):
        ctx = self.ctx
        if ctx.stop_event.is_set():
            raise CancelledCall("cancelled")
        if dispatch.signature != self._signature(dispatch.proposal):
            raise ValueError("candidate inputs changed during recovery")
        if dispatch.node is None:
            dispatch.node = self._candidate(dispatch)
        node = dispatch.node
        self._state(dispatch, "generated")
        if node.is_failed:
            return node, {"failure_stage": "generated", "reason": node.summary}
        if not dispatch.proposal.local:
            from .codegen import plan_coverage

            outcome = review(ctx, node)
            if not (outcome.verdict == "pass" or outcome.changed):
                node.failed = True
                node.summary = outcome.findings
                node.failure_reasons.append("review_unresolved")
                return node, {"failure_stage": "reviewed", "reason": outcome.findings}
            if node.structured_plan and plan_coverage(node.structured_plan, node.parent.code, node.code):
                node.failed = True
                node.summary = "plan_incomplete: review removed a required change"
                node.failure_reasons.append("plan_incomplete")
                return node, {"failure_stage": "reviewed", "reason": node.summary}
        self._state(dispatch, "reviewed")
        if apply_policy(ctx, node):
            return node, {"failure_stage": "reviewed", "reason": node.summary}
        if ctx.stop_event.is_set():
            raise CancelledCall("cancelled")

        def execute(candidate):
            result = self.controller._execute_or_restore(candidate)
            self._state(dispatch, "executed")
            return result

        details = single_model.evaluate(ctx, node, execute, on_assessed=lambda: self._state(dispatch, "assessed"))
        if details["deliverable"]:
            self._state(dispatch, "verified")
        if not details["deliverable"]:
            details.update(
                failure_stage="verified" if details["score_verified"] else "executed" if node.error_type else "assessed",
                reason=(node.artifact_verification or {}).get("error", node.summary),
            )
        return node, details

    def _record(self, dispatch, node, details, terminal):
        ctx = self.ctx
        parent = ctx.tree.by_id[dispatch.proposal.parent_id]
        parent_fact = next((row for row in self.facts if row.get("node_id") == parent.id), None)
        gain = (ctx.direction.gain(node.score, parent_fact["score"])
                if parent_fact and parent_fact["score_verified"] and details.get("score_verified") else None)
        if node:
            node.parent_gain = gain
        from .training import digest

        path = ctx.executor.run_dir(node.id) / "solution.py" if node else None
        source_hash = digest(path) if path and path.is_file() else fingerprint(node.code) if node and node.code else None
        data = {
            "schema_version": 1, "operation_id": dispatch.id, "proposal_id": dispatch.proposal.proposal_id,
            "node_id": node.id if node else None, "parent_id": parent.id, "action": dispatch.proposal.action,
            "method": dispatch.proposal.method_id or dispatch.proposal.source, "source": dispatch.proposal.source,
            "hypothesis": dispatch.proposal.hypothesis,
            "method_summary": node.method_summary if node else None,
            "source_sha256": source_hash, "target": dispatch.proposal.target, "branch_id": node.branch_id if node else parent.branch_id,
            "seed": ctx.cfg["run"]["seed"], "split_id": ctx.task_spec.signature, "metric": ctx.task_spec.metric,
            "status": terminal, "failed_stage": details.get("failure_stage"), "reason": details.get("reason"),
            "seconds": time.monotonic() - dispatch.started,
            "exec_seconds": node.exec_seconds if node else None,
            "score": node.score if node else None, "score_verified": details.get("score_verified", False),
            "deliverable": details.get("deliverable", False), "trial_id": node.trial_id if node else None,
            "rung": node.rung if node else None, "checkpoint": node.checkpoint if node else None,
            "trial_signature": (node.checkpoint or {}).get("signature") if node else None,
            "gain": gain, "evidence_ids": [item["operation_id"] for item in dispatch.evidence],
            "verification": node.artifact_verification if node else None,
            "diagnostic": details.get("validation_class_coverage"),
            "structured_plan": node.structured_plan if node else None,
            "publication": details.get("publication"),
        }
        seq = self.ledger.record_experiment(dispatch.id, data, terminal)
        self.facts.append(deepcopy({**data, "seq": seq}))
        if ctx.experience is not None:
            try:
                ctx.experience.add(Record(dispatch.id, dispatch.proposal.action, dispatch.proposal.hypothesis,
                    node.method_summary if node else "", (gain > 0) - (gain < 0) if gain is not None else 0,
                    data["score"], parent_fact["score"] if parent_fact else None, data["exec_seconds"], data["reason"], now_iso()))
            except OSError as exc:
                log.warning("could not update derived experience index: %s", exc)
        return data, seq

    def _complete(self, dispatch, future):
        ctx = self.ctx
        node = None
        try:
            node, details = future.result()
            terminal = "completed" if details.get("deliverable") else "cancelled" if ctx.stop_event.is_set() else "failed"
        except DeferredCall as exc:
            if exc.resume_at is None:
                raise
            if ctx.stop_event.is_set():
                self._record(dispatch, None, {"failure_stage": dispatch.stage, "reason": "cancelled"}, "cancelled")
                return
            dispatch.resume_at = exc.resume_at
            self._state(dispatch, "deferred", max(0, exc.resume_at - time.monotonic()))
            self.deferred[dispatch.id] = dispatch
            return
        except (CancelledCall, Discarded):
            details, terminal = {"failure_stage": dispatch.stage, "reason": "cancelled"}, "cancelled"
        except (LLMUnavailableError, StructuredOutputError, OSError, ValueError) as exc:
            node = dispatch.node
            details, terminal = {"failure_stage": dispatch.stage, "reason": str(exc)}, "failed"
            if node is not None:
                node.summary = str(exc)
                node.failed = True
                node.failure_reasons.append("candidate_failed")
        if node is None:
            if dispatch.node:
                ctx.tree.discard(dispatch.node)
            self._record(dispatch, None, details, terminal)
        else:
            single_model.finish(ctx, node, details, lambda: self._record(dispatch, node, details, terminal))
        parent_id = dispatch.proposal.parent_id
        if dispatch.proposal.action == "repair" and ctx.flags.get("repairable_id") == parent_id:
            ctx.flags.pop("repairable_id", None)
        if parent_id in self.parent_claims:
            self.parent_claims[parent_id] -= 1
            if not self.parent_claims[parent_id]:
                del self.parent_claims[parent_id]
                ctx.flags["active_parent_leases"].discard(parent_id)
                if ctx.flags.get("repairable_id") != parent_id:
                    ctx.flags["artifact_leases"].discard(parent_id)
        if dispatch.proposal.action == "repair":
            from .bookkeeping import apply_retention

            apply_retention(ctx, ctx.tree.top_list)
        append_jsonl(ctx.paths.decisions, {
            "schema_version": 1, "ts": now_iso(), "decision_id": dispatch.decision_id,
            "as_of_seq": dispatch.proposal.as_of_seq, "operation_id": dispatch.id,
            "selected_node_id": parent_id, "action": dispatch.proposal.action, "result": self.facts[-1],
        })

    def run(self):
        ctx = self.ctx
        parallel = ctx.cfg["resources"]["parallel"]
        api_slots = ctx.cfg["llm"]["max_in_flight"] or parallel
        pending = {}
        with ThreadPoolExecutor(max_workers=parallel + api_slots, thread_name_prefix="candidate") as pool:
            while not ctx.stop_event.is_set():
                done = [f for f in pending if f.done()]
                for future in done:
                    self._complete(pending.pop(future), future)
                local_jobs = sum(d.proposal.local for d in pending.values())
                api_jobs = len(pending) - local_jobs
                for dispatch in list(self.deferred.values()):
                    if (dispatch.resume_at <= time.monotonic() or ctx.gateway.endpoint_failure) and api_jobs < api_slots:
                        del self.deferred[dispatch.id]
                        pending[pool.submit(self._work, dispatch)] = dispatch
                        api_jobs += 1
                snapshot = self.snapshot()
                proposals, ranked, rejected = self.propose(snapshot) if len(self.claimed) < ctx.cfg["run"]["max_steps"] else ([], [], [])
                for proposal in proposals:
                    if proposal.local and local_jobs >= parallel or not proposal.local and api_jobs >= api_slots:
                        continue
                    if not proposal.local and time.monotonic() < ctx.gateway.cooldown_until:
                        continue
                    dispatch = self.claim(proposal, proposal.as_of_seq)
                    if dispatch is None:
                        continue
                    append_jsonl(ctx.paths.decisions, {
                        "schema_version": 1, "ts": now_iso(), "decision_id": dispatch.decision_id,
                        "as_of_seq": snapshot.as_of_seq, "operation_id": dispatch.id, "event": "selected",
                        "selected": asdict(proposal), "ranked": ranked, "rejected": rejected,
                        "evidence_sent": [item["operation_id"] for item in dispatch.evidence],
                    })
                    pending[pool.submit(self._work, dispatch)] = dispatch
                    local_jobs += int(proposal.local)
                    api_jobs += int(not proposal.local)
                ctx.active_tasks = len(pending) + len(self.deferred)
                if not pending and not self.deferred:
                    if proposals and time.monotonic() < ctx.gateway.cooldown_until:
                        ctx.stop_event.wait(max(0, ctx.gateway.cooldown_until - time.monotonic()))
                        continue
                    break
                if pending:
                    wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                else:
                    ctx.stop_event.wait(max(0, min(d.resume_at for d in self.deferred.values()) - time.monotonic()))
            for future, dispatch in pending.items():
                self._complete(dispatch, future)
            for dispatch in self.deferred.values():
                self._record(dispatch, None, {"failure_stage": dispatch.stage, "reason": "cancelled"}, "cancelled")
            ctx.active_tasks = 0
        ctx.stop_reason = ("interrupted" if ctx.stop_event.is_set() else
                           "max_steps" if len(self.claimed) >= ctx.cfg["run"]["max_steps"] else "no_candidate")
