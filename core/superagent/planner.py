"""Structured planner: which pipeline components to change and how."""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

from .llm import Field
from .llm.gateway import LLMGateway

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


_KEY_ENTRYPOINTS = ("predict_only", "fit", "train", "evaluate", "save_model", "load_model", "main")


def _smart_source_excerpt(code: str, max_chars: int = 24000) -> str:
    if len(code) <= max_chars:
        return code
    lines = code.splitlines()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        half = max(100, (max_chars - 64) // 2)
        return code[:half] + "\n# ... [truncated] ...\n" + code[-half:]
    imp_end = 0
    main_blocks = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imp_end = max(imp_end, node.end_lineno or node.lineno)
        elif isinstance(node, ast.If):
            seg = "\n".join(lines[node.lineno - 1:(node.end_lineno or node.lineno)])
            if "__main__" in seg:
                main_blocks.append(f"# Lines {node.lineno}-{node.end_lineno}\n{seg}")
    symbols = python_symbols(code)
    index_lines = [f"- {name}: lines {sym.lineno}-{sym.end_lineno}" for name, sym in sorted(symbols.items(), key=lambda kv: kv[1].lineno)]
    parts = []
    if imp_end:
        parts.append("# Top-level imports\n" + "\n".join(lines[:imp_end]))
    if index_lines:
        parts.append("# Symbol index\n" + "\n".join(index_lines))
    added = set()
    key_blocks = []
    for sym_name, sym_node in symbols.items():
        short = sym_name.rsplit(".", 1)[-1]
        if short in _KEY_ENTRYPOINTS or any(k in short for k in ("predict_only", "main")):
            if sym_name not in added:
                added.add(sym_name)
                src = "\n".join(lines[sym_node.lineno - 1:(sym_node.end_lineno or sym_node.lineno)])
                key_blocks.append(f"# {sym_name} (lines {sym_node.lineno}-{sym_node.end_lineno})\n{src}")
    key_blocks.extend(main_blocks)
    head_block = "# Code head\n" + code[:3500]
    tail_block = "# Code tail\n" + code[-2500:]
    core = "\n\n".join(parts)
    for block in [*key_blocks, head_block, tail_block]:
        candidate = core + "\n\n" + block if core else block
        if len(candidate) <= max_chars:
            core = candidate
    if len(core) > max_chars:
        return core[:max_chars]
    return core


def plan_experiment(ctx, dispatch, parent, evidence):
    gateway = ctx.gateway
    ledger = gateway.ledger
    saved = ledger.get(dispatch.id, "experiment_plan", dispatch.signature)
    if saved:
        return saved
    proposal = dispatch.proposal
    visible = {row["operation_id"] for row in evidence}
    task = ctx.flags["task_facts"]
    compact_task = {k: task[k] for k in ("metric", "maximize", "capabilities", "eligible_methods") if k in task}
    parent_excerpt = _smart_source_excerpt(parent.code, 24000)
    names = sorted(python_symbols(parent.code))
    parent_output = (getattr(parent, "output", "") or "")[-4000:]
    data_preview = (getattr(ctx, "data_preview", "") or "")[:3000]
    catalog_text = (getattr(ctx, "catalog_text", "") or "")[:3000]
    donor_text = ""
    if proposal.action == "borrow":
        donor = next(
            (ctx.tree.by_id[eid] for eid in proposal.evidence_ids
             if eid in ctx.tree.by_id and ctx.tree.by_id[eid].id != parent.id and ctx.tree.by_id[eid].code),
            None,
        )
        if donor is not None:
            donor_text = (
                f"\nDonor branch ({donor.id}, score={donor.score}, method={donor.method_summary}):\n"
                f"{_smart_source_excerpt(donor.code, 4000)}"
            )
    system = "Plan a single classification experiment. The split, required outputs and isolated reload interface are fixed."
    user = (
        f"Task and input manifest:\n{ctx.task_desc}\n{json.dumps(compact_task, ensure_ascii=False)}\n"
        + (f"Data preview:\n{data_preview}\n" if data_preview else "")
        + (f"Catalog methods:\n{catalog_text}\n" if catalog_text else "")
        + f"Dispatched action: {proposal.action}\nHypothesis to test: {proposal.hypothesis}\n"
        + f"Observations: {json.dumps(evidence, ensure_ascii=False)}\nSymbols: {', '.join(names)}\n"
        + (f"Parent execution output (tail):\n{parent_output}\n" if parent_output else "")
        + (f"{donor_text}\n" if donor_text else "")
        + f"Source excerpt:\n{parent_excerpt}\n"
        + "Use the exact Python symbol for each change (or '__imports__' when adding only top-level imports; adding imports inside a symbol change is also allowed). "
        + "Use whole_script as the only target for a broad rewrite. "
        + "List separate changes when training and inference need separate edits. Cite the observation IDs used."
    )

    def check(plan, _last):
        return plan, validate_experiment_plan(plan, proposal.action, visible, parent.code)

    result = gateway.structured("planner.structured", system, user, EXPERIMENT_PLAN,
                                node_id=dispatch.id, extra_check=check, resume_key=dispatch.signature)
    ledger.put(dispatch.id, "experiment_plan", dispatch.signature, result)
    return result
