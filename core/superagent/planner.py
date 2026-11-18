"""Structured planner: which pipeline components to change and how."""

from __future__ import annotations

import logging
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
