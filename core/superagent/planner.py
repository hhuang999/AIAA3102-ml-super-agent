"""Structured planner: which pipeline components to change and how."""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

from .llm import Field
from .llm.gateway import LLMGateway
from .textfmt import trim_with

log = logging.getLogger("superagent")

COMPONENT_NAMES = ("data_prep", "model_def", "train_eval")
DEFAULT_RATIONALE = "Improve the components most likely to raise the test performance."
GENERIC_GUIDANCE = ("Make targeted improvements to the components above, focusing on test performance while "
                    "keeping the existing functionality and interfaces.")

PLAN_SCHEMA = {
    "rationale": Field("string", "1-4 sentences of root-cause analysis", required=False, nullable=True),
    "components": Field("any", "array of component names", required=False, nullable=True),
    "changes": Field("any", "object mapping component name to plan text", required=False, nullable=True),
}


def normalize_and_check(obj: dict[str, Any], is_last: bool) -> tuple[dict[str, Any], list[str]]:
    """Normalise a planner answer, then apply the semantic checks. Returns (plan, errors)."""
    rationale = obj.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        rationale = DEFAULT_RATIONALE
    comps = obj.get("components")
    if comps is None:
        comps = []
    elif isinstance(comps, str):
        comps = [comps] if comps.strip() else []
    elif not isinstance(comps, list):
        comps = [str(comps)]
    comps = [str(c).strip() for c in comps if str(c).strip()]
    changes = obj.get("changes")
    if isinstance(changes, str):
        text = changes.strip()
        if text:
            rationale = rationale + "\n\n" + text
            changes = {c: text for c in comps}
        else:
            changes = {}
    elif not isinstance(changes, dict):
        changes = {}
    invalid_changes = [k for k, v in changes.items() if not isinstance(v, str) or not v.strip()]
    if invalid_changes:
        return obj, [f"give a nonempty text description for each change: {invalid_changes}"]
    changes = {k: v.strip() for k, v in changes.items()}
    unknown = sorted((set(comps) | set(changes)) - set(COMPONENT_NAMES))
    if unknown:
        log.warning("planner returned unknown component names: %s", unknown)
    plan = {"rationale": rationale, "components": comps, "changes": changes}
    if unknown:
        return plan, [f"unknown components: {unknown}; use only {COMPONENT_NAMES}"]
    if comps and changes:
        missing = [c for c in comps if c not in changes]
        if missing or set(changes) - set(comps):
            return plan, ["changes must contain exactly one target for each selected component"]
        return plan, []
    if not comps and changes:
        plan["components"] = list(changes.keys())
        return plan, []
    if comps and not changes:
        return plan, ["'changes' is empty: give a concrete plan for every component listed in 'components'"]
    return plan, ["both 'components' and 'changes' are empty: choose 1-3 components and describe the changes"]


def render_plan(plan: dict[str, Any]) -> str:
    lines = ["Analysis: " + str(plan.get("rationale", "")).strip(), "",
             "Target components: " + ", ".join(plan.get("components", []))]
    changes = plan.get("changes") or {}
    if changes:
        for comp in plan.get("components", []) or list(changes):
            if comp in changes:
                lines.append("")
                lines.append(f"[{comp}] {changes[comp]}")
        for comp, text in changes.items():
            if comp not in (plan.get("components") or []):
                lines.append("")
                lines.append(f"[{comp}] {text}")
    else:
        lines.append("")
        lines.append(GENERIC_GUIDANCE)
    return "\n".join(lines)


def structured_plan(gw: LLMGateway, system: str, user: str, node_id: str) -> dict[str, Any]:
    """One planner.structured call. Raises StructuredOutputError when every attempt fails."""
    return gw.structured("planner.structured", system, user, PLAN_SCHEMA, node_id=node_id, extra_check=normalize_and_check)


EXPERIMENT_PLAN = {
    "schema_version": Field("any", "the number 1"),
    "action": Field("string", "the dispatched action"),
    "evidence_ids": Field("string_list", "IDs from the supplied observations"),
    "hypothesis": Field("string", "one falsifiable hypothesis for this task"),
    "changes": Field("any", "list of {id, component, target, edit, check}"),
    "expected_observation": Field("string", "the observation that tests the hypothesis"),
    "preserve": Field("string_list", "unchanged interfaces or behavior"),
}


def python_symbols(code):
    found = {}

    def collect(nodes, prefix=""):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + node.name
                found.setdefault(name, []).append(node)
                if isinstance(node, ast.ClassDef):
                    collect(node.body, name + ".")

    collect(ast.parse(code).body)
    return {name: nodes[0] for name, nodes in found.items() if len(nodes) == 1}


def validate_experiment_plan(plan, action, visible_ids, code):
    errors = []
    if type(plan.get("schema_version")) is not int or plan["schema_version"] != 1 or plan.get("action") != action:
        errors.append("schema_version must be 1 and action must match the dispatch")
    ids = plan.get("evidence_ids", [])
    if not isinstance(ids, list) or not ids or any(not isinstance(k, str) or k not in visible_ids for k in ids):
        errors.append("evidence_ids must reference the provided snapshot observations")
    for key in ("hypothesis", "expected_observation"):
        if not isinstance(plan.get(key), str) or not plan[key].strip():
            errors.append(f"{key} must describe this experiment")
    preserve = plan.get("preserve")
    if not isinstance(preserve, list) or not preserve or any(not isinstance(k, str) or not k.strip() for k in preserve):
        errors.append("preserve must identify at least one unchanged interface")
    changes = plan.get("changes")
    if not isinstance(changes, list) or not 1 <= len(changes) <= 6:
        return errors + ["changes must contain 1 to 6 target edits"]
    symbols = python_symbols(code)
    used_ids, targets = set(), set()
    for change in changes:
        fields = ("id", "component", "target", "edit", "check")
        if not isinstance(change, dict) or any(not isinstance(change.get(k), str) or not change[k].strip() for k in fields):
            errors.append("each change requires id, component, target, edit and check")
            continue
        ident, target = change["id"], change["target"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,24}", ident) or ident in used_ids:
            errors.append(f"invalid or duplicate change ID {ident}")
        if change["component"] not in COMPONENT_NAMES:
            errors.append(f"unknown component {change['component']}")
        if target not in symbols and target not in ("whole_script", "__imports__"):
            errors.append(f"target {target} does not identify a unique Python symbol")
        if any(target == p or target.startswith(p + ".") or p.startswith(target + ".") for p in targets):
            errors.append(f"overlapping edit target {target}")
        targets.add(target)
        used_ids.add(ident)
        if change["edit"].strip().lower().rstrip(".") in {"improve model", "optimize model", "improve accuracy", "optimize performance"}:
            errors.append(f"change {ident} needs a specific edit")
    if "whole_script" in targets and len(changes) != 1:
        errors.append("whole_script is a single broad change")
    return errors


def _smart_source_excerpt(code: str, max_chars: int = 24000) -> str:
    """Keep header/imports, symbol outline and entry points (predict_only, fit, main, __main__) within max_chars."""
    if len(code) <= max_chars:
        return code
    lines = code.splitlines()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        half = max(200, max_chars // 2 - 30)
        return code[:half] + "\n# ... [middle omitted] ...\n" + code[-half:]
    outline = []
    preserved_spans: list[tuple[int, int]] = []
    imp_end = 0
    for node in tree.body:
        end_ln = getattr(node, "end_lineno", node.lineno)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imp_end = max(imp_end, end_ln)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            outline.append(f"# L{node.lineno}-{end_ln}: {node.name}")
            if node.name in ("predict_only", "fit", "main"):
                preserved_spans.append((node.lineno, end_ln))
        elif isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            preserved_spans.append((node.lineno, end_ln))
    if imp_end == 0:
        imp_end = min(len(lines), 25)
    head = "\n".join(lines[:imp_end])
    outline_block = "\n".join(outline)
    preserved_spans.sort()
    tail_parts = []
    covered_lines: set[int] = set(range(1, imp_end + 1))
    for s, e in preserved_spans:
        if any(ln in covered_lines for ln in range(s, e + 1)):
            s = max(s, imp_end + 1)
        if s <= e:
            tail_parts.append(f"# [L{s}-{e}]\n" + "\n".join(lines[s - 1:e]))
            covered_lines.update(range(s, e + 1))
    tail_block = "\n\n".join(tail_parts)
    if len(tail_block) > max_chars // 2:
        tail_block = tail_block[: max_chars // 2 - 25] + "\n# ... [entry truncated]"
    fixed_len = len(head) + len(outline_block) + len(tail_block) + 120
    middle_budget = max(0, max_chars - fixed_len)
    middle_lines = []
    used = 0
    for ln in range(imp_end + 1, len(lines) + 1):
        if ln in covered_lines:
            continue
        text_ln = lines[ln - 1]
        if used + len(text_ln) + 1 > middle_budget:
            middle_lines.append("# ... [remaining middle lines omitted] ...")
            break
        middle_lines.append(text_ln)
        used += len(text_ln) + 1
    parts = [head]
    if outline_block:
        parts.append("# Symbol index:\n" + outline_block)
    if middle_lines:
        parts.append("\n".join(middle_lines))
    if tail_block:
        parts.append("# Preserved entry points:\n" + tail_block)
    return "\n\n".join(parts)[:max_chars]


def plan_experiment(ctx, dispatch, parent, evidence):
    gateway = ctx.gateway
    ledger = gateway.ledger
    saved = ledger.get(dispatch.id, "experiment_plan", dispatch.signature)
    if saved:
        return saved
    proposal = dispatch.proposal
    visible = {row["operation_id"] for row in evidence}
    task = ctx.flags["task_facts"]
    parent_excerpt = _smart_source_excerpt(parent.code, 24000)
    names = sorted(python_symbols(parent.code))
    parent_out = trim_with(ctx.cfg, parent.output)[:4000] if getattr(parent, "output", None) else ""
    data_prev = (getattr(ctx, "data_preview", "") or "")[:3000]
    catalog = (getattr(ctx, "catalog_text", "") or "")[:3000]
    borrow_block = ""
    if proposal.action == "borrow":
        donor_id = next((r.get("node_id") for r in evidence if r.get("node_id") and r.get("node_id") != parent.id and r.get("node_id") in ctx.tree.by_id), None)
        if donor_id:
            donor = ctx.tree.by_id[donor_id]
            borrow_block = f"\nDonor method ({donor_id}): {donor.method_summary or donor.plan[:400]}\nDonor excerpt:\n{_smart_source_excerpt(donor.code, 4000)}\n"
    system = "Plan a single classification experiment choosing the most impactful component(s) among data_prep, model_def and train_eval. The split, required outputs and isolated reload interface are fixed."
    user = (
        f"Task and input manifest:\n{ctx.task_desc}\n{json.dumps(task, ensure_ascii=False)}\n"
        + (f"Data preview:\n{data_prev}\n" if data_prev else "")
        + (f"Available catalog methods/assets:\n{catalog}\n" if catalog else "")
        + f"Dispatched action: {proposal.action}\nHypothesis to test: {proposal.hypothesis}\n"
        + f"Observations: {json.dumps(evidence, ensure_ascii=False)}\n"
        + (f"Parent run output / diagnostics:\n{parent_out}\n" if parent_out else "")
        + borrow_block
        + f"Symbols: {', '.join(names)}\n"
        f"Source excerpt:\n{parent_excerpt}\n"
        "Use the exact Python symbol (or '__imports__' for pure import additions, or 'whole_script' only for a broad rewrite). "
        "Choose component(s) from data_prep, model_def, train_eval based on diagnostics. "
        "List separate changes when training and inference need separate edits. Cite the observation IDs used.\n"
        'Example JSON shape: {"schema_version": 1, "action": "refine", "evidence_ids": ["<op_id>"], '
        '"hypothesis": "...", "changes": [{"id": "c1", "component": "model_def", "target": "build_model", '
        '"edit": "...", "check": "..."}], "expected_observation": "...", "preserve": ["split", "submission schema"]}'
    )

    def check(plan, _last):
        return plan, validate_experiment_plan(plan, proposal.action, visible, parent.code)

    result = gateway.structured("planner.structured", system, user, EXPERIMENT_PLAN,
                                node_id=dispatch.id, extra_check=check, resume_key=dispatch.signature)
    ledger.put(dispatch.id, "experiment_plan", dispatch.signature, result)
    return result
