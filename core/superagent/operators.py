"""The six operators: draft, refine, repair, retrospect, borrow, synthesis."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .codegen import CodeGenerator, Extracted
from .context import RunContext
from .errors import StructuredOutputError, LLMUnavailableError
from .llm.ledger import fingerprint, PermanentCall
from .planner import render_plan, structured_plan
from .prompts import blocks, builders
from .scheduling import synthesis_representatives
from .textfmt import attempts_summary, node_card, trajectory, trajectory_nodes, trim_with
from .tree import Node
from .utils import new_id

log = logging.getLogger("superagent")

CODEGEN_REPLY_CHARS = 2000
REPAIR_ERROR_HEAD_CHARS = 200


@dataclass
class GenOutcome:
    node: Node | None
    kind: str                                   # the kind actually produced
    details: dict[str, Any] = field(default_factory=dict)
    fallback_reason: str | None = None          # fallback_to_refine when retrospect/borrow fell back


class Operators:
    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        cg = ctx.cfg["codegen"]
        self.codegen = CodeGenerator(ctx.gateway, int(cg["extract_max_attempts"]), int(cg["patch_max_attempts"]))

    # ------------------------------------------------------------------ helpers
    def _include_torch(self) -> bool:
        if self.ctx.resources().num_gpus > 0:
            return True
        facts = getattr(self.ctx, "flags", {}).get("task_facts") or {}
        modalities = {str(m).lower() for m in (facts.get("modalities") or getattr(self.ctx, "modalities", ()) or ())}
        return bool(modalities & {"image", "text", "audio", "multimodal"}) or not modalities

    def _rules(self) -> str:
        v = self.ctx.cfg["validation_rule"]
        return blocks.implementation_rules(
            self.ctx.resources(),
            int(v["split_seed"]),
            float(v["validation_fraction"]),
            declared_split=self.ctx.task_spec is not None,
            include_torch=self._include_torch(),
        )

    def _env(self) -> str:
        if self.ctx.flags.get("formal"):
            caps = self.ctx.flags["task_facts"]["capabilities"]
            packages = ", ".join(sorted(caps["dependencies"]))
            return (
                f"Available packages: {packages}. Training and inference run offline. "
                "Use only the declared paths. Embed everything needed for standalone inference in out/model."
            )
        return blocks.environment_block(self.ctx.mode, self.ctx.rng)  # type: ignore[arg-type]

    def _new_node(self, node_id: str, kind: str, parent: Node, plan: str, code: str) -> Node:
        ctx = self.ctx
        tree = ctx.tree
        with tree.lock:
            n = Node(id=node_id, kind=kind, plan=plan, code=code)
            if kind in ("draft", "synthesis"):
                n.chain_best = tree.root
                n.branch_id = tree.new_branch_id()
                n.in_progress = True
            else:
                n.chain_best = parent.chain_best
                n.branch_id = parent.branch_id
            n.via_elite = bool(getattr(ctx._local, "via_elite", False) or parent.picked_by_elite)
            tree.attach(n, parent)
            if kind != "synthesis":
                parent.pending_children = max(0, parent.pending_children - 1)
        return n

    def _codegen_failure(self, node: Node, last_reply: str) -> None:
        node.code = ""
        node.failed = True
        node.error_type = "CodegenError"
        node.error_message = "no parseable code could be generated"
        node.output = ("Code generation failed: no usable Python script could be extracted from the model's replies.\n"
                       "Start of the last reply:\n" + (last_reply or "")[:CODEGEN_REPLY_CHARS])
        node.summary = "CODEGEN_ERROR: code generation failed; no runnable script was produced."
        node.failure_reasons = ["codegen_error"]

    def _branch_counts(self, branch: int) -> tuple[int, int, float | None]:
        tree = self.ctx.tree
        with tree.lock:
            ok = tree.branch_ok.get(branch, [])
            allb = tree.branch_all.get(branch, [])
            best = tree.branch_best(branch)
            if best is None:
                return 0, len(allb), None
            ok_since = len(ok) - 1 - ok.index(best)
            all_since = len(allb) - 1 - allb.index(best) if best in allb else len(allb)
            return ok_since, all_since, best.score

    def _escalation(self, parent: Node) -> tuple[bool, int, int, float | None]:
        e = self.ctx.cfg["operators"]["escalation"]
        ok_since, all_since, best = self._branch_counts(parent.branch_id or 0)
        esc = ok_since >= int(e["ok_since_best"]) or all_since >= int(e["all_since_best"])
        return esc, ok_since, all_since, best

    def _summary(self, n: Node) -> str:
        with self.ctx.tree.lock:
            return attempts_summary(n, self.ctx.direction)

    # ------------------------------------------------------------------ entry point
    def generate(self, action: str, parent: Node) -> GenOutcome:
        ledger = getattr(self.ctx.gateway, "ledger", None)
        if not ledger:
            return self._generate(action,parent)
        sig=fingerprint([action,parent.id,parent.code,self.ctx.task_desc,self.ctx.cfg["llm"]])
        op=ledger.claim(sig)
        self.ctx._local.operation_id=op
        try:
            ready=ledger.get(op,"candidate",sig)
            if ready:
                node=self._new_node(op,ready["kind"],parent,ready["plan"],ready["code"])
                node.structured_plan=ready["structured_plan"]
                outcome=GenOutcome(node,ready["kind"],ready["details"],ready["fallback"])
            else:
                outcome=self._generate(action,parent)
                if outcome.node and not outcome.node.is_failed:
                    node=outcome.node
                    ledger.put(op,"candidate",sig,dict(kind=outcome.kind,plan=node.plan,code=node.code,
                               structured_plan=node.structured_plan,details=outcome.details,fallback=outcome.fallback_reason))
        except PermanentCall:
            ledger.state(op,"abandoned")
            raise
        except LLMUnavailableError as exc:
            wait=max(0,self.ctx.gateway.cooldown_until-time.monotonic())
            ledger.state(op,"suspended" if "budget exhausted" not in str(exc) else "abandoned",wait)
            raise
        except Exception:
            ledger.state(op,"failed")
            raise
        else:
            ledger.state(op,"generated")
            return outcome
        finally:
            self.ctx._local.operation_id=None

    def _generate(self, action: str, parent: Node) -> GenOutcome:
        if action == "draft":
            return self.draft(parent)
        if action == "refine":
            return self.refine(parent)
        if action == "repair":
            return self.repair(parent)
        if action == "retrospect":
            return self.retrospect(parent)
        if action == "borrow":
            return self.borrow(parent)
        if action == "synthesis":
            return self.synthesis(parent)
        raise ValueError(f"unknown action {action}")

    def experiment_candidate(self, dispatch, parent, evidence, planned):
        """Generate a descendant of the selected parent and retain its approved plan."""
        from .codegen import plan_coverage, planned_patch
        from .planner import plan_experiment, python_symbols

        proposal = dispatch.proposal
        targeted = proposal.action == "repair" and proposal.target == "predict_only" and "predict_only" in python_symbols(parent.code)
        if targeted:
            plan = {
                "schema_version": 1, "action": "repair", "evidence_ids": list(proposal.evidence_ids),
                "hypothesis": "Correct the failing isolated inference path using the allowlisted inputs",
                "changes": [{"id": "inference", "component": "train_eval", "target": "predict_only",
                             "edit": "Fix the diagnosed inference failure while using the existing fitted model",
                             "check": "The same model reproduces both CSVs and rejects corrupt weights"}],
                "expected_observation": "The independent score is preserved and isolated reload passes",
                "preserve": ["training", "fitted model", "split", "submission schema"],
            }
        else:
            plan = plan_experiment(self.ctx, dispatch, parent, evidence)
        planned(plan)
        rules = self.ctx.task_desc + "\n" + self._rules()
        if targeted:
            rules += "\nFailure to repair: " + str((parent.artifact_verification or {}).get("error", ""))[-1500:]
        extract = None
        if plan["changes"][0]["target"] != "whole_script":
            extract = planned_patch(self.ctx.gateway, dispatch.id, dispatch.signature, plan, parent.code, rules,
                                    role="repair_coder" if targeted else "patch_coder")
        if extract is None:
            system = "Write a complete single-model classification script implementing the approved plan."
            user = (f"{rules}\nPlan: {plan}\nObservations: {evidence}\n"
                    f"Previous source:\n{parent.code}\nReturn a short method summary and one complete Python code block.")
            fallback = self.codegen.full_script("script_coder", system, user, dispatch.id)
            if fallback.extracted and not plan_coverage(plan, parent.code, fallback.extracted.code):
                extract = fallback.extracted.code
        node = self._new_node(dispatch.id, proposal.action, parent, plan["hypothesis"], extract or "")
        node.structured_plan = plan
        if extract is None:
            self._codegen_failure(node, "plan_incomplete: not every approved change is present in the candidate")
            node.summary = "plan_incomplete: not every approved change is present in the candidate"
        return node

    # ------------------------------------------------------------------ draft
    def draft(self, root: Node) -> GenOutcome:
        ctx = self.ctx
        node_id = getattr(self.ctx._local, "operation_id", None) or new_id()
        baseline = ctx.task_dir/"baseline.py"
        if ctx.task_spec and baseline.is_file():
            with ctx.tree.lock:
                use = not ctx.flags.get("baseline_claimed")
                ctx.flags["baseline_claimed"]=True
            if use:
                node=self._new_node(node_id,"draft",root,"Execute the supplied task baseline.",baseline.read_text(encoding="utf-8"))
                return GenOutcome(node,"draft",{"codegen":"supplied_baseline"})
        ledger = ctx.gateway.ledger if ctx.flags.get("formal") else None
        saved = ledger.get(node_id, "draft_inputs", "v1") if ledger else None
        if saved:
            summary, reqs = saved
        else:
            summary = self._summary(root)
            reqs = builders.draft_requirements(self._rules(), self._env(), ctx.catalog_text)
            hypothesis = getattr(ctx._local, "selected_hypothesis", None)
            if hypothesis:
                reqs += "\nThe assigned model hypothesis is: " + hypothesis
            if ledger:
                ledger.put(node_id, "draft_inputs", "v1", [summary, reqs])
        details: dict[str, Any] = {"planning": "none"}
        result: Extracted | None = None
        last = ""
        if ctx.cfg["codegen"]["staged_drafting"]:
            details["codegen"] = "staged"

            declared = ctx.task_spec is not None

            def build_stage(stage: str, prev: list[tuple[str, str, str]]) -> tuple[str, str]:
                return builders.draft_stage(ctx.task_desc, ctx.data_preview, summary, reqs, stage, prev, ctx.mode, declared_split=declared)

            def build_int(steps: list[tuple[str, str, str]]) -> tuple[str, str]:
                return builders.draft_integrator(ctx.task_desc, ctx.data_preview, summary, reqs, steps, declared_split=declared)

            result, _plan = self.codegen.staged(build_stage, build_int, node_id)
            if result is None:
                details["fallback"] = "staged_to_full_script"
        if result is None:
            details["codegen"] = "full_script" if "fallback" not in details else "staged"
            system, user = builders.draft_full(ctx.task_desc, ctx.data_preview, summary, reqs)
            r = self.codegen.full_script("script_coder", system, user, node_id)
            result, last = r.extracted, r.last_reply
        node = self._new_node(node_id, "draft", root, result.plan if result else "", result.code if result else "")
        if result is None:
            self._codegen_failure(node, last)
            if ledger:
                node.summary = r.failure_reason or node.summary
        return GenOutcome(node, "draft", details)

    # ------------------------------------------------------------------ planning + patching shared by refine-like ops
    def _plan_and_patch(self, op: builders.OpPrompt, parent: Node, node_id: str, two_stage: bool, details: dict[str, Any]) -> tuple[str, str, dict[str, Any] | None, str]:
        """Returns (plan text, code, structured plan, last reply). Empty code means codegen failure."""
        ctx = self.ctx
        trimmed = trim_with(ctx.cfg, parent.output)
        ledger=getattr(ctx.gateway,"ledger",None)
        sig=fingerprint([parent.code,op.kind])
        plan_obj=ledger.get(node_id,"validated_plan",sig) if ledger else None
        try:
            if plan_obj is not None:
                details["planning"]="resumed"
            elif two_stage and ctx.cfg["experience"]["planning"] == "two_stage":
                details["planning"] = "two_stage"
                system, user = builders.planner_free(op)
                free = ledger.get(node_id,"free_plan",sig) if ledger else None
                if not free:
                    response = ctx.gateway.text("planner.free", system, user, node_id=node_id)
                    free = response.content.strip() if not response.truncated else ""
                    if ledger and free:
                        ledger.put(node_id,"free_plan",sig,free)
                assert ctx.experience is not None
                rc = int(ctx.cfg["experience"]["refine_top_k"])
                good = ctx.experience.search(free, rc, effect=1)
                bad = ctx.experience.search(free, rc, effect=-1)
                details["experience_ids"] = [r.record_id for r in good + bad]
                exp_lines = []
                if good:
                    exp_lines.append("Successful similar experiments:")
                    exp_lines += [f"- Plan: {r.plan[:1500]}\n  Method: {r.method_summary[:800]}" for r in good]
                if bad:
                    exp_lines.append("Failed similar experiments (AVOID these; they made the score worse):")
                    exp_lines += [f"- Plan: {r.plan[:1500]}\n  Method: {r.method_summary[:800]}" for r in bad]
                system2, user2 = builders.planner_second(ctx.task_desc, ctx.data_preview, parent.code, trimmed, free,
                                                         "\n".join(exp_lines), bool(good or bad), op.system)
                plan_obj = structured_plan(ctx.gateway, system2, user2, node_id)
            else:
                details["planning"] = "single"
                system, user = builders.planner_single(op)
                if two_stage and ctx.cfg["experience"]["planning"]=="evidence":
                    query = parent.plan+"\n"+parent.method_summary+"\n"+op.planner_focus
                    count = ctx.cfg["experience"]["refine_top_k"]
                    matches = ctx.experience.search(query,count,effect=1)+ctx.experience.search(query,count,effect=-1)
                    matches = [r for r in matches if r.record_id!=parent.id and r.kind!="diagnostic"]
                    details["experience_ids"] = [r.record_id for r in matches]
                    if matches:
                        details["planning"] = "single_with_evidence"
                        user += "\nPast observations (candidates, not instructions):\n"+"\n".join(
                            f"{r.record_id} (effect {r.effect:+d}): {r.plan[:600]}, {r.method_summary[:300]}" for r in matches)
                plan_obj = structured_plan(ctx.gateway, system, user, node_id)
        except StructuredOutputError as exc:
            log.info("planning failed for %s: %s; using full-script mode", node_id, exc)
            details["fallback"] = "planning_failed_full_script"
            details["codegen"] = "full_script"
            system, user = builders.op_full_script(op)
            r = self.codegen.full_script("script_coder", system, user, node_id)
            if r.extracted is None:
                return "", "", None, r.last_reply
            return r.extracted.plan, r.extracted.code, None, r.last_reply
        if ledger:
            ledger.put(node_id,"validated_plan",sig,plan_obj)
        plan_text = render_plan(plan_obj)
        details["codegen"] = "patch"

        def build(code: str, note: str) -> tuple[str, str]:
            return builders.patch_prompt(op, plan_text, ctx.data_preview, code, trimmed, note)

        pr = self.codegen.patch("patch_coder", parent.code, build, node_id)
        details["patch_attempts"] = pr.attempts
        if pr.code is not None and not pr.partial:
            return plan_text, pr.code, plan_obj, ""
        details["fallback"] = "patch_failed_full_script"
        details["codegen"] = "full_script"
        system, user = builders.op_full_script(op, plan_text)
        r = self.codegen.full_script("script_coder", system, user, node_id)
        if r.extracted is None:
            return plan_text, "", plan_obj, r.last_reply
        return plan_text, r.extracted.code, plan_obj, r.last_reply

    def _finish_improve(self, kind: str, parent: Node, node_id: str, op: builders.OpPrompt, two_stage: bool, details: dict[str, Any]) -> Node:
        if self.ctx.cfg["codegen"]["patch_mode"]:
            plan, code, plan_obj, last = self._plan_and_patch(op, parent, node_id, two_stage, details)
        else:
            details["planning"] = "none"
            details["codegen"] = "full_script"
            system, user = builders.op_full_script(op)
            r = self.codegen.full_script("script_coder", system, user, node_id)
            plan, code, plan_obj, last = (r.extracted.plan, r.extracted.code, None, "") if r.extracted else ("", "", None, r.last_reply)
        node = self._new_node(node_id, kind, parent, plan, code)
        node.structured_plan = plan_obj
        if not code:
            self._codegen_failure(node, last)
        return node

    # ------------------------------------------------------------------ refine
    def refine(self, parent: Node, details: dict[str, Any] | None = None) -> GenOutcome:
        ctx = self.ctx
        details = details if details is not None else {}
        node_id = getattr(self.ctx._local, "operation_id", None) or new_id()
        esc, ok_since, all_since, best = self._escalation(parent)
        details.update({"ok_since_best": ok_since, "all_since_best": all_since, "escalated": esc})
        op = builders.refine_op(ctx.task_desc, ctx.data_preview, self._summary(parent), parent.code,
                                trim_with(ctx.cfg, parent.output), self._rules(), esc, best, ok_since, all_since)
        two_stage = bool(esc and ctx.experience is not None and ctx.cfg["experience"]["enabled"] and len(ctx.experience) > 0)
        node = self._finish_improve("refine", parent, node_id, op, two_stage, details)
        return GenOutcome(node, "refine", details)

    # ------------------------------------------------------------------ retrospect
    def retrospect(self, parent: Node) -> GenOutcome:
        ctx = self.ctx
        rcfg = ctx.cfg["operators"]["retrospect"]
        with ctx.tree.lock:
            steps = trajectory_nodes(parent, int(rcfg["max_steps"]))
            traj = trajectory(parent, ctx.direction, int(rcfg["max_steps"]))
        if len(steps) < int(rcfg["min_steps"]):
            out = self.refine(parent, {"requested": "retrospect"})
            out.fallback_reason = "fallback_to_refine"
            return out
        node_id = getattr(self.ctx._local, "operation_id", None) or new_id()
        esc, ok_since, all_since, best = self._escalation(parent)
        details: dict[str, Any] = {"ok_since_best": ok_since, "all_since_best": all_since, "escalated": esc, "trajectory_steps": len(steps)}
        op = builders.retrospect_op(ctx.task_desc, ctx.data_preview, self._summary(parent), parent.code,
                                    trim_with(ctx.cfg, parent.output), self._rules(), traj, esc, best, ok_since, all_since)
        node = self._finish_improve("retrospect", parent, node_id, op, False, details)
        return GenOutcome(node, "retrospect", details)

    # ------------------------------------------------------------------ borrow
    def borrow_references(self, parent: Node) -> list[Node]:
        bcfg = self.ctx.cfg["operators"]["borrow"]
        tree = self.ctx.tree
        with tree.lock:
            cands: list[Node] = []
            for b in tree.branches():
                if b == parent.branch_id:
                    continue
                ok = tree.sorted_ok([n for n in tree.branch_ok.get(b, []) if n.score is not None])
                cands.extend(ok[: int(bcfg["per_branch"])])
            refs = tree.sorted_ok(cands)[: int(bcfg["max_refs"])]
            if not refs:
                own = tree.sorted_ok([n for n in tree.branch_ok.get(parent.branch_id or 0, []) if n.score is not None])[:2]
                refs = [n for n in own if n is not parent]
            return refs

    def borrow(self, parent: Node) -> GenOutcome:
        ctx = self.ctx
        refs = self.borrow_references(parent)
        if not refs:
            out = self.refine(parent, {"requested": "borrow"})
            out.fallback_reason = "fallback_to_refine"
            return out
        node_id = getattr(self.ctx._local, "operation_id", None) or new_id()
        with ctx.tree.lock:
            if len(refs) == 1:
                ref_text = node_card(refs[0], ctx.direction, with_code=True)
            else:
                ref_text = "\n\n".join(f"## Reference {i} (branch {r.branch_id}, score {r.score:.4f})\n{node_card(r, ctx.direction)}"
                                       for i, r in enumerate(refs, 1))
        details: dict[str, Any] = {"reference_ids": [r.id for r in refs]}
        op = builders.borrow_op(ctx.task_desc, ctx.data_preview, parent.plan, parent.code, trim_with(ctx.cfg, parent.output),
                                self._rules(), ref_text)
        node = self._finish_improve("borrow", parent, node_id, op, False, details)
        return GenOutcome(node, "borrow", details)

    # ------------------------------------------------------------------ repair
    def repair(self, parent: Node) -> GenOutcome:
        ctx = self.ctx
        node_id = getattr(self.ctx._local, "operation_id", None) or new_id()
        details: dict[str, Any] = {"planning": "none"}
        rules = self._rules()
        if parent.error_type == "CodegenError" or not parent.code.strip():
            details["codegen"] = "full_script"
            details["fallback"] = "no_parent_code"
            system, user = builders.repair_from_plan(ctx.task_desc, ctx.data_preview, parent.plan, rules)
            r = self.codegen.full_script("repair_coder", system, user, node_id)
            node = self._new_node(node_id, "repair", parent, r.extracted.plan if r.extracted else parent.plan, r.extracted.code if r.extracted else "")
            if r.extracted is None:
                self._codegen_failure(node, r.last_reply)
            return GenOutcome(node, "repair", details)
        trimmed = trim_with(ctx.cfg, parent.output)
        exp_text = ""
        if ctx.experience is not None and len(ctx.experience) > 0:
            recs = ctx.experience.search(trimmed, int(ctx.cfg["experience"]["repair_top_k"]), effect=1, kind="repair")
            details["experience_ids"] = [r.record_id for r in recs]
            exp_text = "\n\n".join(
                f"- Error (start): {(r.error_text or '')[:REPAIR_ERROR_HEAD_CHARS]}\n  Fix plan: {r.plan[:1200]}\n  Method: {r.method_summary[:600]}"
                for r in recs)
        op = builders.repair_op(ctx.task_desc, ctx.data_preview, parent.code, trimmed, rules, exp_text)
        plan = ""
        code = ""
        last = ""
        if ctx.cfg["codegen"]["patch_mode"]:
            details["codegen"] = "patch"

            def build(code_now: str, note: str) -> tuple[str, str]:
                o = builders.repair_op(ctx.task_desc, ctx.data_preview, code_now, trimmed, rules, exp_text)
                return builders.repair_patch_prompt(o, ctx.data_preview, code_now, trimmed, note)

            pr = self.codegen.patch("repair_coder", parent.code, build, node_id)
            details["patch_attempts"] = pr.attempts
            if pr.code is not None and not pr.partial:
                code = pr.code
                plan = pr.first_explanation or f"Fix the {parent.error_type or 'failure'}: {parent.summary}".strip()
                if pr.partial:
                    details["patch_partial"] = True
            else:
                details["fallback"] = "patch_failed_full_script"
        if not code:
            details["codegen"] = "full_script"
            system, user = builders.op_full_script(op, emphasise_complete=True)
            r = self.codegen.full_script("repair_coder", system, user, node_id)
            if r.extracted is not None:
                plan, code = r.extracted.plan, r.extracted.code
            else:
                last = r.last_reply
        node = self._new_node(node_id, "repair", parent, plan, code)
        if not code:
            self._codegen_failure(node, last)
        return GenOutcome(node, "repair", details)

    # ------------------------------------------------------------------ synthesis
    def synthesis(self, root: Node) -> GenOutcome:
        ctx = self.ctx
        reps = synthesis_representatives(ctx.tree)
        if len(reps) < 2:
            with ctx.tree.lock:
                ctx.tree.synthesis_count = max(0, ctx.tree.synthesis_count - 1)
            return GenOutcome(None, "synthesis", {"reason": "fewer than two representatives"})
        node_id = getattr(self.ctx._local, "operation_id", None) or new_id()
        with ctx.tree.lock:
            text = "\n\n".join(f"## Branch {r.branch_id} (score {r.score:.4f})\n{node_card(r, ctx.direction)}" for r in reps)
        system, user = builders.synthesis_prompt(ctx.task_desc, ctx.data_preview, text, self._rules())
        r = self.codegen.full_script("script_coder", system, user, node_id)
        node = self._new_node(node_id, "synthesis", root, r.extracted.plan if r.extracted else "", r.extracted.code if r.extracted else "")
        if r.extracted is None:
            self._codegen_failure(node, r.last_reply)
        ctx.add_event("synthesis", node.id, f"synthesis from branches {[x.branch_id for x in reps]}")
        return GenOutcome(node, "synthesis", {"codegen": "full_script", "planning": "none", "reference_ids": [x.id for x in reps]})
