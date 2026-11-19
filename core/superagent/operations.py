"""Candidate claims, stage recovery and coordination for the isolated search."""

from __future__ import annotations

import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass

from . import single_model
from .errors import LLMUnavailableError, StructuredOutputError
from .llm.ledger import CancelledCall, DeferredCall, fingerprint
from .review import apply_policy, review
from .search import Discarded
from .utils import append_jsonl, new_id, now_iso

log = logging.getLogger("superagent")
DRAFT_ROLES = ("draft_writer.data_prep", "draft_writer.model_def", "draft_writer.train_eval", "draft_integrator")


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


class Operations:
    def __init__(self, controller):
        self.controller = controller
        self.ctx = controller.ctx
        self.ledger = self.ctx.gateway.ledger
        self.claimed = {}
        self.deferred = {}
        self.facts = []
        self.ctx.flags["active_parent_leases"] = set()
        self.ctx.flags["baseline_claimed"] = True  # The supplied path is picked explicitly by this scheduler.
        self.ctx.tree.search_start = time.monotonic()

    def _drafts(self):
        ctx = self.ctx
        parent = ctx.tree.root.id
        sequence = len(self.facts)
        if (ctx.task_dir / "baseline.py").is_file():
            yield Proposal("supplied", parent, "draft", "Supplied baseline", "supplied", True, sequence)
        if ctx.gateway.endpoint_failure:
            return
        kinds = {
            "tabular": ("low-variance regularized baseline", "nonlinear tabular interactions"),
            "text": ("sparse word or character representation", "complementary text representation"),
            "image": ("lightweight spatial image features", "small classification backbone"),
        }
        primary = ctx.modalities[0] if ctx.modalities else None
        hypotheses = kinds.get(primary, ("simple task-specific baseline", "alternative representation"))
        max_drafts = min(ctx.cfg["search"]["initial_drafts"], ctx.cfg["search"]["max_drafts"])
        position = 1 if (ctx.task_dir / "baseline.py").is_file() else 0
        for hypothesis in hypotheses[:max(0, max_drafts - position)]:
            yield Proposal(f"draft-{position}", parent, "draft", hypothesis, "generated", False, sequence)
            position += 1

    def propose(self):
        return [proposal for proposal in self._drafts() if proposal.proposal_id not in self.claimed]

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
        if not ctx.tree.by_id[proposal.parent_id].is_root:
            ctx.flags["active_parent_leases"].add(proposal.parent_id)
            ctx.flags["artifact_leases"].add(proposal.parent_id)
        return dispatch

    def _roles(self, proposal):
        if proposal.local:
            return ()
        if proposal.action == "draft":
            return (*DRAFT_ROLES, "reviewer") if self.ctx.cfg["codegen"]["staged_drafting"] else ("script_coder", "reviewer")
        if proposal.action == "repair":
            return ("repair_coder", "reviewer")
        return ("planner.structured", "patch_coder", "reviewer")

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
            return self.controller.ops.draft(parent).node
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
            outcome = review(ctx, node)
            if not (outcome.verdict == "pass" or outcome.changed):
                node.failed = True
                node.summary = outcome.findings
                node.failure_reasons.append("review_unresolved")
                return node, {"failure_stage": "reviewed", "reason": outcome.findings}
        self._state(dispatch, "reviewed")
        if apply_policy(ctx, node):
            return node, {"failure_stage": "reviewed", "reason": node.summary}
        if ctx.stop_event.is_set():
            raise CancelledCall("cancelled")

        def execute(candidate):
            result = self.controller._execute_or_restore(candidate)
            self._state(dispatch, "executed")
            return result

        details = single_model.evaluate(ctx, node, execute)
        self._state(dispatch, "verified" if details["deliverable"] else "assessed")
        if not details["deliverable"]:
            details.update(
                failure_stage="verified" if details["score_verified"] else "executed" if node.error_type else "assessed",
                reason=(node.artifact_verification or {}).get("error", node.summary),
            )
        return node, details

    def _record(self, dispatch, node, details, terminal):
        ctx = self.ctx
        parent = ctx.tree.by_id[dispatch.proposal.parent_id]
        data = {
            "schema_version": 1, "operation_id": dispatch.id, "proposal_id": dispatch.proposal.proposal_id,
            "node_id": node.id if node else None, "parent_id": parent.id, "action": dispatch.proposal.action,
            "method": dispatch.proposal.source, "hypothesis": dispatch.proposal.hypothesis,
            "source_sha256": fingerprint(node.code) if node and node.code else None,
            "seed": ctx.cfg["run"]["seed"], "split_id": ctx.task_spec.signature, "metric": ctx.task_spec.metric,
            "status": terminal, "failed_stage": details.get("failure_stage"), "reason": details.get("reason"),
            "seconds": time.monotonic() - dispatch.started,
            "exec_seconds": node.exec_seconds if node else None,
            "score": node.score if node else None, "score_verified": details.get("score_verified", False),
            "deliverable": details.get("deliverable", False), "trial_id": node.trial_id if node else None,
            "rung": node.rung if node else None, "checkpoint": node.checkpoint if node else None,
            "verification": node.artifact_verification if node else None,
            "diagnostic": details.get("validation_class_coverage"),
            "structured_plan": node.structured_plan if node else None,
            "publication": details.get("publication"),
        }
        seq = self.ledger.record_experiment(dispatch.id, data, terminal)
        self.facts.append({**data, "seq": seq})
        return data, seq

    def _complete(self, dispatch, future):
        ctx = self.ctx
        node = None
        try:
            node, details = future.result()
            terminal = "completed" if details.get("deliverable") else "failed"
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
        ctx.flags["active_parent_leases"].discard(parent_id)
        if ctx.flags.get("repairable_id") != parent_id:
            ctx.flags["artifact_leases"].discard(parent_id)
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
                proposals = self.propose() if len(self.claimed) < ctx.cfg["run"]["max_steps"] else []
                for proposal in proposals:
                    if proposal.local and local_jobs >= parallel or not proposal.local and api_jobs >= api_slots:
                        continue
                    if not proposal.local and time.monotonic() < ctx.gateway.cooldown_until:
                        continue
                    dispatch = self.claim(proposal, proposal.as_of_seq)
                    if dispatch is None:
                        continue
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
