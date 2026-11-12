"""The search controller: worker threads, task queue, deadlines and per-step bookkeeping."""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from . import report
from .assessment import assess
from .bookkeeping import after_assessment_files, update_rankings, write_search_tree, write_status
from .chain import ChainOutcome, add_reward, backpropagate, chain_control, step_reward
from .context import MIN_SCRIPT_SECONDS, RunContext
from .control.base import StepLimits
from .control.state import StepTiming, build_state
from .errors import LLMUnavailableError
from .executor import ExecResult
from .operators import GenOutcome, Operators
from .review import apply_policy, review
from .scheduling import Dispatch, dispatch, synthesis_blocker, synthesis_conditions, synthesis_pending
from .selection import elite_pick, explore_weight, select
from .tree import Node
from .utils import append_jsonl, now_iso

log = logging.getLogger("superagent")

LLM_FAILURE_STREAK_LIMIT = 10
MAIN_POLL_SECONDS = 0.5
FINISH_CHECK_SECONDS = 5.0
CREATING_ACTIONS = ("draft", "refine", "repair", "retrospect", "borrow")


@dataclass
class Task:
    kind: str                     # "run_draft" | "search"
    node: Node | None = None      # draft to run, or node to continue the chain from
    selection_mode: str | None = None


@dataclass
class StepRecord:
    selected: Node | None = None
    mode: str | None = None
    weight: float | None = None
    action: str = "none"
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    new_node: Node | None = None
    outcome: dict[str, Any] = field(default_factory=dict)
    limits: StepLimits | None = None
    started: float = field(default_factory=time.monotonic)


class Discarded(Exception):
    """The step was abandoned because the run is shutting down."""


class SearchController:
    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self.ops = Operators(ctx)
        self.queue: queue.Queue[Task | None] = queue.Queue()
        self._lock = threading.Lock()
        self.active = 0
        self.done = threading.Event()
        self.seq = 0
        self.workers: list[threading.Thread] = []
        self.time_stop = False
        self._pending_records: dict[str, StepRecord] = {}

    def _add_active(self, delta: int) -> None:
        """Change the number of queued or running tasks. Caller holds ``self._lock``."""
        self.active += delta
        self.ctx.active_tasks = self.active

    def _may_start(self) -> bool:
        ctx = self.ctx
        if ctx.control is None:
            return ctx.can_start_task()
        try:
            return ctx.control.may_start_step(build_state(ctx)).allow
        except Exception as exc:  # noqa: BLE001 - a broken controller must not stop the run
            log.warning("controller.may_start_step failed (%s); using the fixed rule", exc)
            return ctx.can_start_task()

    def _limits_for(self, rec: StepRecord) -> None:
        """Fix the step's resource limits (once) and make them current for this thread."""
        ctx = self.ctx
        if rec.limits is None:
            rec.limits = ctx.begin_step()
            if rec.limits is not None:
                rec.details["limits"] = {"run_limit_secs": round(rec.limits.run_limit_secs, 1), "steps_left": rec.limits.steps_left,
                                         "gpu_memory_gb": rec.limits.gpu_memory_gb}
        else:
            ctx._local.limits = rec.limits

    # ------------------------------------------------------------------ public
    def run(self) -> None:
        ctx = self.ctx
        parallel = int(ctx.cfg["resources"]["parallel"])
        for i in range(parallel):
            t = threading.Thread(target=self._worker, name=f"worker-{i}", daemon=True)
            t.start()
            self.workers.append(t)
        self._startup()
        self._main_loop()

    # ------------------------------------------------------------------ startup
    def _startup(self) -> None:
        """Write the initial drafts and hand each one to the workers as soon as it is ready."""
        ctx = self.ctx
        s = ctx.cfg["search"]
        root = ctx.tree.root
        n_drafts = min(int(s["initial_drafts"]), int(ctx.cfg["run"]["max_steps"]))
        with ctx.tree.lock:
            ctx.tree.search_start = time.monotonic()
        # Reserve the draft slots under the root first (cheap, no LLM call).
        recs: list[StepRecord] = []
        for _ in range(n_drafts):
            if ctx.stop_event.is_set() or not self._may_start():
                break
            rec = StepRecord(selected=root, mode="initial")
            d = dispatch(ctx.tree, root, ctx.search_cfg(), ctx.progress(), ctx.rng)  # type: ignore[arg-type]
            rec.action, rec.reason = d.action, d.reason
            if d.action != "draft":
                break
            recs.append(rec)

        stagger = float(ctx.cfg["execution"]["draft_launch_stagger_seconds"])
        launched = 0
        last_launch: float | None = None

        def launch(node: Node, rec: StepRecord) -> None:
            nonlocal launched, last_launch
            if last_launch is not None:
                wait = stagger - (time.monotonic() - last_launch)
                if wait > 0:
                    ctx.stop_event.wait(wait)
            with self._lock:
                self._add_active(1)
            self._pending_records[node.id] = rec
            self.queue.put(Task("run_draft", node, "initial"))
            last_launch = time.monotonic()
            launched += 1

        if bool(s["concurrent_initial_drafts"]) and len(recs) > 1:
            with ThreadPoolExecutor(max_workers=len(recs), thread_name_prefix="draft") as pool:
                futures = [pool.submit(self._write_initial_draft, rec) for rec in recs]
                for fut in as_completed(futures):
                    node = fut.result()
                    if node is not None:
                        launch(node, recs[futures.index(fut)])
        else:
            for k, rec in enumerate(recs):
                node = self._write_initial_draft(rec)
                if node is not None:
                    launch(node, rec)
                elif ctx.should_abort_generation() or ctx.llm_failure_streak >= LLM_FAILURE_STREAK_LIMIT:
                    with ctx.tree.lock:  # give back the slots that will not be used
                        root.pending_children = max(0, root.pending_children - (len(recs) - k - 1))
                    break

        extra = int(ctx.cfg["resources"]["parallel"]) - launched
        for _ in range(max(0, extra)):
            if not self._can_continue_locked_free():
                break
            with self._lock:
                self._add_active(1)
            self.queue.put(Task("search"))
        with self._lock:
            if self.active == 0:
                self.done.set()

    def _write_initial_draft(self, rec: StepRecord) -> Node | None:
        """Generate and pre-check one initial draft. Returns None (and frees the slot) on failure."""
        ctx = self.ctx
        root = ctx.tree.root
        rec.started = time.monotonic()
        self._limits_for(rec)
        try:
            gen = self.ops.draft(root)
        except Exception as exc:  # noqa: BLE001
            with ctx.tree.lock:
                root.pending_children = max(0, root.pending_children - 1)
            if isinstance(exc, LLMUnavailableError):
                with self._lock:
                    ctx.llm_failure_streak += 1
            log.error("initial draft generation failed: %s", exc)
            return None
        node = gen.node
        assert node is not None
        rec.details.update(gen.details)
        rec.new_node = node
        if not node.is_failed:
            self._prerun(node, rec)
        log.info("initial draft %s generated (%s)", node.id, rec.details.get("codegen"))
        return node

    # ------------------------------------------------------------------ main loop
    def _main_loop(self) -> None:
        ctx = self.ctx
        reserve = ctx.reserve
        w_started = False
        next_finish_check = time.monotonic()
        while True:
            if self.done.is_set():
                break
            if ctx.control is not None and not ctx.stop_event.is_set() and time.monotonic() >= next_finish_check:
                next_finish_check = time.monotonic() + FINISH_CHECK_SECONDS
                try:
                    fin = ctx.control.should_finish(build_state(ctx))
                except Exception as exc:  # noqa: BLE001 - a broken controller must not stop the run
                    log.warning("controller.should_finish failed: %s", exc)
                    fin = None
                if fin is not None and fin.allow:
                    ctx.stop_reason = fin.reason or "controller"
                    ctx.add_event("controller_finish", None, f"{ctx.stop_reason}: {fin.details}")
                    log.info("the controller finished the search: %s %s", ctx.stop_reason, fin.details)
                    ctx.stop_event.set()
            if ctx.stop_event.is_set() and ctx.stop_reason == "interrupted":
                ctx.hard_event.set()
                break
            if ctx.time_left() <= reserve:
                ctx.hard_event.set()
                ctx.stop_event.set()
                self.time_stop = True
                break
            p = ctx.progress()
            if p is not None:
                w = explore_weight(p, ctx.cfg)
                if w < 1.0 and not w_started:
                    w_started = True
                    ctx.add_event("exploit_phase_started", None, f"explore weight {w:.2f}")
                if w <= float(ctx.cfg["search"]["explore"]["min_weight"]) and not ctx.flags.get("exploit_full"):
                    ctx.flags["exploit_full"] = True
                    ctx.add_event("exploit_phase_full", None, f"explore weight {w:.2f}")
            time.sleep(MAIN_POLL_SECONDS)
        if ctx.hard_event.is_set():
            # Let nodes already under assessment finish, but at most half of the reserve.
            limit = time.monotonic() + reserve / 2.0
            while time.monotonic() < limit and not self.done.is_set():
                time.sleep(0.2)
        ctx.flags["finalizing"] = True
        for _ in self.workers:
            self.queue.put(None)

    # ------------------------------------------------------------------ task flow
    def _can_continue_locked_free(self) -> bool:
        ctx = self.ctx
        if ctx.stop_event.is_set():
            return False
        if not self._may_start():
            self.time_stop = True
            return False
        n_nodes = len(ctx.tree.solution_nodes())
        return n_nodes + self.active < int(ctx.cfg["run"]["max_steps"])

    def _after_task(self, follow: Node | None) -> None:
        with self._lock:
            self._add_active(-1)
            if self._can_continue_locked_free() and not self.ctx.flags.get("finalizing"):
                self._add_active(1)
                self.queue.put(Task("search", follow, "chain" if follow is not None else None))
                return
            if self.active <= 0:
                self.done.set()
        if follow is not None:
            # The chain cannot continue: release the branch so other tasks may enter it again.
            with self.ctx.tree.lock:
                backpropagate(follow, 0.0, False)

    def _worker(self) -> None:
        while True:
            task = self.queue.get()
            if task is None:
                return
            follow: Node | None = None
            try:
                follow = self._execute(task)
            except Exception:  # noqa: BLE001 - _execute handles its own errors; this is a last resort
                log.error("unexpected worker failure:\n%s", traceback.format_exc())
            finally:
                self._after_task(follow)

    # ------------------------------------------------------------------ a single step
    def _execute(self, task: Task) -> Node | None:
        ctx = self.ctx
        if task.kind == "run_draft":
            assert task.node is not None
            rec = self._pending_records.pop(task.node.id, StepRecord(selected=ctx.tree.root, mode="initial", action="draft",
                                                                    reason="root_has_capacity"))
            return self._step(rec, task.node.parent or ctx.tree.root, pre_generated=task.node)
        rec = StepRecord()
        scfg = ctx.search_cfg()
        ctx.tree.synthesis_time_ok = self._synthesis_time_ok(scfg)
        handled, result = self._synthesis_trigger(rec, task, scfg)
        if handled:
            return result
        if task.node is not None:
            handled, result = self._leave_chain(rec, task.node, scfg)
            if handled:
                return result
            rec.selected, rec.mode = task.node, "chain"
            return self._step(rec, rec.selected)
        try:
            # Selection and capacity reservation happen atomically, so two workers cannot both
            # claim the last free slot of the same node.
            with ctx.tree.lock:
                p = ctx.progress()
                sel = select(ctx.tree, scfg, p, lambda: synthesis_conditions(ctx.tree, scfg, ctx.progress()), ctx.rng)  # type: ignore[arg-type]
                rec.selected, rec.mode, rec.weight = sel.node, sel.mode, sel.explore_weight
                d = dispatch(ctx.tree, sel.node, scfg, p, ctx.rng)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            log.error("selection failed: %s\n%s", exc, traceback.format_exc())
            rec.details["error"] = f"{type(exc).__name__}: {exc}"[:300]
            with ctx.tree.lock:
                if rec.selected is not None:
                    backpropagate(rec.selected, 0.0, False)
            self._write_decision(rec)
            return None
        return self._step(rec, rec.selected, dispatched=d)

    def _synthesis_time_ok(self, scfg: dict[str, Any]) -> bool:
        """Whether a synthesis step still fits in the usable time.

        A synthesis step writes a whole script before running it, so it needs the typical non-script
        time of a step (generation, review, assessment) plus a script run. Without measurements yet,
        synthesis is allowed (its time window keeps it away from the very end).
        """
        try:
            st = build_state(self.ctx)
        except Exception as exc:  # noqa: BLE001 - never block the search on a measurement problem
            log.warning("could not measure the time left for synthesis: %s", exc)
            return True
        other = st.median_other_seconds(10)
        script = st.median_ok_script_seconds()
        if other is None and script is None:
            return True
        factor = float(scfg["operators"]["synthesis"].get("time_factor", 1.0))
        need = factor * ((other or 0.0) + max(MIN_SCRIPT_SECONDS, script or 0.0))
        return st.usable_secs >= need

    def _synthesis_trigger(self, rec: StepRecord, task: Task, scfg: dict[str, Any]) -> tuple[bool, Node | None]:
        """Start a synthesis step as soon as its conditions hold (one at a time)."""
        ctx = self.ctx
        if not scfg["operators"]["synthesis"]["trigger_on_stagnation"]:
            return False, None
        tree = ctx.tree
        with tree.lock:
            p = ctx.progress()
            blocker = "synthesis_running" if synthesis_pending(tree) > 0 else synthesis_blocker(tree, scfg, p)
            counts = ctx.flags.setdefault("synthesis_blockers", {})
            counts[blocker or "triggered"] = counts.get(blocker or "triggered", 0) + 1
            if blocker is not None:
                return False, None
            tree.synthesis_count += 1
            if task.node is not None:
                backpropagate(task.node, 0.0, False)  # leave the chain; its branch is free again
        rec.selected, rec.mode = tree.root, "trigger"
        if task.node is not None:
            rec.details["left_chain_at"] = task.node.id
        return True, self._step(rec, tree.root, dispatched=Dispatch("synthesis", "synthesis_trigger"))

    def _leave_chain(self, rec: StepRecord, follow: Node, scfg: dict[str, Any]) -> tuple[bool, Node | None]:
        """Apply the tree_walk/elite_pick switch before a chain step: with the exploit chance, pick an elite node instead."""
        ctx = self.ctx
        if not scfg["search"]["chain_soft_switch"]:
            return False, None
        p = ctx.progress()
        if p is None:
            return False, None
        w = explore_weight(p, scfg)
        if w >= 1.0 or ctx.rng.random() < w:
            return False, None
        tree = ctx.tree
        try:
            with tree.lock:
                backpropagate(follow, 0.0, False)  # leave the chain; its branch is free again
                node, mode = elite_pick(tree, scfg, p, lambda: synthesis_conditions(tree, scfg, ctx.progress()), ctx.rng)  # type: ignore[arg-type]
                d = dispatch(tree, node, scfg, p, ctx.rng)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            log.error("selection after leaving a chain failed: %s\n%s", exc, traceback.format_exc())
            rec.details["error"] = f"{type(exc).__name__}: {exc}"[:300]
            self._write_decision(rec)
            return True, None
        rec.selected, rec.mode, rec.weight = node, mode, w
        rec.details["left_chain_at"] = follow.id
        return True, self._step(rec, node, dispatched=d)

    def _step(self, rec: StepRecord, parent: Node, pre_generated: Node | None = None, dispatched: Dispatch | None = None) -> Node | None:
        ctx = self.ctx
        node: Node | None = pre_generated
        reserved = False
        d: Dispatch | None = None
        self._limits_for(rec)
        try:
            if node is None:
                d = dispatched or dispatch(ctx.tree, parent, ctx.search_cfg(), ctx.progress(), ctx.rng)  # type: ignore[arg-type]
                rec.action, rec.reason = d.action, d.reason
                rec.details.update(d.details)
                reserved = d.action in CREATING_ACTIONS or d.action == "synthesis"
                if d.action == "none":
                    if d.reason == "dead_end":
                        with ctx.tree.lock:
                            backpropagate(parent, 0.0, True)
                    self._write_decision(rec)
                    ctx.stop_event.wait(float(ctx.cfg["run"]["idle_backoff_seconds"]))
                    return None
                gen: GenOutcome = self.ops.generate(d.action, parent)
                reserved = False
                node = gen.node
                rec.details.update(gen.details)
                if gen.fallback_reason:
                    rec.details["requested_action"] = d.action
                    rec.reason = gen.fallback_reason
                    rec.action = gen.kind
                if node is None:
                    self._write_decision(rec)
                    ctx.stop_event.wait(float(ctx.cfg["run"]["idle_backoff_seconds"]))
                    return None
                rec.new_node = node
                if rec.action in ("retrospect", "borrow"):
                    ctx.add_event("plateau_response", node.id, f"{rec.action} on node {parent.id}")
                if rec.details.get("escalated"):
                    ctx.add_event("escalated_prompt", node.id, f"escalated requirements for {rec.action}")
                if not node.is_failed:
                    self._prerun(node, rec)
            else:
                rec.new_node = node
            self._check_shutdown()
            if not node.is_failed:
                res = self._run(node)
                self._check_shutdown()
                info = assess(ctx, node, res)
                rec.details.update({k: v for k, v in info.items() if v})
            else:
                node.finished_at = now_iso()
                ctx.executor.prepare_run_dir(node.id, node.code if node.code else None)
            after_assessment_files(ctx, node)
            self._check_shutdown()
            outcome = self._finish_node(parent, node, rec)
            ctx.llm_failure_streak = 0
            return None if outcome.returned_to_root else node
        except Discarded:
            self._discard(parent, node, reserved, d, rec)
            return None
        except Exception as exc:  # noqa: BLE001
            if ctx.hard_event.is_set() or ctx.flags.get("finalizing"):
                self._discard(parent, node, reserved, d, rec)
                return None
            log.error("step failed with %s: %s\n%s", type(exc).__name__, exc, traceback.format_exc())
            if isinstance(exc, LLMUnavailableError):
                ctx.llm_failure_streak += 1
                if ctx.llm_failure_streak >= LLM_FAILURE_STREAK_LIMIT:
                    log.error("the LLM has been unavailable for %d tasks in a row; stopping the search", ctx.llm_failure_streak)
                    ctx.stop_reason = "error"
                    ctx.stop_event.set()
            self._handle_error(parent, node, reserved, d, rec, exc)
            return None

    def _check_shutdown(self) -> None:
        if self.ctx.flags.get("finalizing") or self.ctx.hard_event.is_set() and self.ctx.stop_reason == "interrupted":
            raise Discarded()

    def _prerun(self, node: Node, rec: StepRecord) -> None:
        ctx = self.ctx
        try:
            outcome = review(ctx, node)
            rec.details["review_changed_code"] = outcome.changed
        except Exception as exc:  # noqa: BLE001 - review must never block a run
            log.warning("review failed for %s: %s", node.id, exc)
            rec.details["review_changed_code"] = False
        if apply_policy(ctx, node):
            rec.details["policy_violation"] = True

    def _run(self, node: Node) -> ExecResult:
        ctx = self.ctx
        limit = ctx.current_run_limit()
        if limit < MIN_SCRIPT_SECONDS:
            raise Discarded()
        ex = ctx.cfg["execution"]
        smoke = float(ex["smoke_timeout_seconds"]) if ex["smoke_run"] else None
        res = ctx.executor.run(node.id, node.code, limit, abort=ctx.hard_event if ctx.stop_reason == "interrupted" else None,
                               smoke_timeout=smoke)
        node.smoke = res.smoke
        if ctx.flags.get("finalizing") or (ctx.stop_reason == "interrupted" and res.error_message and "cancelled" in res.error_message):
            raise Discarded()
        node.output = res.output
        node.exec_seconds = round(res.exec_seconds, 3)
        node.error_type = res.error_type
        node.error_message = res.error_message
        return res

    def _finish_node(self, parent: Node, node: Node, rec: StepRecord) -> ChainOutcome:
        ctx = self.ctx
        tree = ctx.tree
        scfg = ctx.search_cfg()
        with tree.lock:
            if ctx.flags.get("finalizing"):
                raise Discarded()
            if parent.is_failed and node.ok:
                if not parent.repair_succeeded:
                    ctx.add_event("repair_succeeded", node.id, f"repaired node {parent.id}")
                parent.repair_succeeded = True
            outcome = chain_control(tree, node, scfg, ctx.progress(), ctx.rng)  # type: ignore[arg-type]
            tree.append(node)
            if str(scfg["search"]["reward_mode"]) == "per_step":
                outcome.reward = step_reward(node, tree, float(scfg["search"]["min_gain"]))
                add_reward(node, outcome.reward)
        if outcome.chain_ended_no_gain:
            ctx.add_event("chain_ended", node.id, "chain ended after repeated attempts without progress", step=node.step)
        update_rankings(ctx, node)
        rec.outcome = {
            "failed": node.failed,
            "score": node.score,
            "forced_return": outcome.forced_return,
            "returned_to_root": outcome.returned_to_root,
            "reward": outcome.reward,
        }
        self._persist(node)
        self._write_decision(rec)
        self._record_timing(rec, node)
        best = tree.best.score if tree.best else None
        log.info("node %s step %s %s on %s: %s score=%s best=%s", node.id, node.step, node.kind,
                 parent.id if not parent.is_root else "root", "ok" if node.ok else f"FAILED {node.failure_reasons}",
                 node.score, best)
        return outcome

    def _record_timing(self, rec: StepRecord, node: Node) -> None:
        ctx = self.ctx
        timing = StepTiming(step=node.step, node_id=node.id, kind=node.kind, ok=node.ok,
                            step_seconds=time.monotonic() - rec.started, exec_seconds=node.exec_seconds)
        ctx.record_step(timing)
        if ctx.control is not None:
            try:
                ctx.control.on_step_finished(build_state(ctx), timing)
            except Exception as exc:  # noqa: BLE001 - a broken controller must not stop the run
                log.warning("controller.on_step_finished failed: %s", exc)

    def _persist(self, node: Node | None) -> None:
        ctx = self.ctx
        with ctx.io_lock:
            if ctx.flags.get("finalized"):
                return
            write_search_tree(ctx)
            write_status(ctx, "running")
            try:
                report.write_summary(ctx)
            except Exception as exc:  # noqa: BLE001
                log.warning("could not update the run summary: %s", exc)

    def _write_decision(self, rec: StepRecord) -> None:
        ctx = self.ctx
        with self._lock:
            self.seq += 1
            seq = self.seq
        p = ctx.progress()
        row = {
            "seq": seq,
            "ts": now_iso(),
            "progress": round(p, 6) if p is not None else None,
            "selected_node_id": rec.selected.id if rec.selected is not None else None,
            "selection_mode": rec.mode,
            "explore_weight": rec.weight,
            "action": rec.action,
            "reason": rec.reason,
            "details": rec.details,
            "new_node_id": rec.new_node.id if rec.new_node is not None and rec.new_node.in_tree else None,
            "outcome": rec.outcome or {"failed": None, "score": None, "forced_return": False, "returned_to_root": False, "reward": None},
        }
        append_jsonl(ctx.paths.decisions, row)

    def _discard(self, parent: Node, node: Node | None, reserved: bool, d: Dispatch | None, rec: StepRecord | None = None) -> None:
        tree = self.ctx.tree
        if rec is not None:
            rec.details["discarded"] = True
            try:
                self._write_decision(rec)
            except Exception:  # noqa: BLE001
                pass
        with tree.lock:
            if node is None:
                if reserved and d is not None:
                    if d.action == "synthesis":
                        tree.synthesis_count = max(0, tree.synthesis_count - 1)
                    else:
                        parent.pending_children = max(0, parent.pending_children - 1)
            else:
                tree.discard(node)
            backpropagate(parent, 0.0, False)

    def _handle_error(self, parent: Node, node: Node | None, reserved: bool, d: Dispatch | None, rec: StepRecord, exc: Exception) -> None:
        ctx = self.ctx
        tree = ctx.tree
        with tree.lock:
            if node is None:
                if reserved and d is not None:
                    if d.action == "synthesis":
                        tree.synthesis_count = max(0, tree.synthesis_count - 1)
                    else:
                        parent.pending_children = max(0, parent.pending_children - 1)
        if node is not None and not node.in_tree:
            with tree.lock:
                if node.branch_id is not None and node in tree.branch_ok.get(node.branch_id, []):
                    tree.branch_ok[node.branch_id].remove(node)
            if ctx.experience is not None:
                try:
                    ctx.experience.remove(node.id)
                except Exception:  # noqa: BLE001
                    pass
            shutil.rmtree(ctx.executor.run_dir(node.id) / "out", ignore_errors=True)
            node.has_artifacts = False
            node.failed = True
            node.score = None
            node.error_type = "InternalError"
            node.error_message = f"{type(exc).__name__}: {exc}"[:500]
            if "internal_error" not in node.failure_reasons:
                node.failure_reasons.append("internal_error")
            node.dead_end = True
            node.summary = node.summary or f"INTERNAL_ERROR: {type(exc).__name__}: {exc}"[:300]
            node.finished_at = now_iso()
            with tree.lock:
                tree.append(node)
            rec.new_node = node
            rec.outcome = {"failed": True, "score": None, "forced_return": False, "returned_to_root": True, "reward": None}
            try:
                update_rankings(ctx, node)
                self._persist(node)
            except Exception as exc2:  # noqa: BLE001
                log.warning("could not persist after an internal error: %s", exc2)
            self._record_timing(rec, node)
        rec.details["error"] = f"{type(exc).__name__}: {exc}"[:300]
        with tree.lock:
            backpropagate(parent, 0.0, False)
        try:
            self._write_decision(rec)
        except Exception:  # noqa: BLE001
            pass

    def stop_reason(self) -> str:
        ctx = self.ctx
        if ctx.stop_reason:
            return ctx.stop_reason
        if self.time_stop:
            return "time_limit"
        return "max_steps"
