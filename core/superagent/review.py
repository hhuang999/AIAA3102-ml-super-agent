"""Pre-run checks: LLM code review and the pretrained-policy scan."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .context import RunContext
from .errors import LLMUnavailableError, StructuredOutputError
from .llm import Field
from .patching import apply_patch
from .prompts import builders
from .tree import Node

log = logging.getLogger("superagent")

REVIEW_SCHEMA = {
    "verdict": Field("enum", "pass or revise", values=("pass", "revise")),
    "findings": Field("string", "2-4 sentences"),
    "patch": Field("string", "patch blocks when revising, otherwise null", nullable=True, required=False),
}


@dataclass
class ReviewOutcome:
    changed: bool
    verdict: str | None
    findings: str


def review(ctx: RunContext, node: Node) -> ReviewOutcome:
    rc = ctx.cfg["review"]
    if not rc["enabled"]:
        return ReviewOutcome(False, None, "review disabled")
    system, user = builders.reviewer(ctx.task_desc, node.code, ctx.mode, max(0.0, ctx.current_run_limit()))
    last = ReviewOutcome(False, None, "")
    for _ in range(int(rc["max_attempts"])):
        try:
            obj = ctx.gateway.structured("reviewer", system, user, REVIEW_SCHEMA, node_id=node.id)
        except (StructuredOutputError, LLMUnavailableError) as exc:
            last = ReviewOutcome(False, None, f"review call failed: {exc}")
            if ctx.should_abort_generation():
                break
            continue
        verdict = obj["verdict"]
        findings = str(obj.get("findings") or "")
        if verdict == "pass":
            return ReviewOutcome(False, "pass", findings)
        patch = obj.get("patch")
        if patch and str(patch).strip():
            new_code, applied = apply_patch(node.code, str(patch))
            if applied > 0 and new_code != node.code:
                node.code = new_code
                return ReviewOutcome(True, "revise", findings)
            return ReviewOutcome(False, "revise", findings + " (patch did not apply; code kept)")
        last = ReviewOutcome(False, "revise", findings + " (no patch given)")
    return last


def _strip_comment(line: str) -> str:
    """Drop a trailing # comment (outside of string literals, best effort)."""
    out = []
    quote: str | None = None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out)


def policy_scan(code: str, patterns: list[str]) -> list[tuple[int, str, str]]:
    """Return (line number, pattern, line) for every forbidden pattern found in non-comment code."""
    hits = []
    for i, line in enumerate(code.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        body = _strip_comment(line)
        for p in patterns:
            if p in body:
                hits.append((i, p, line.strip()))
    return hits


def apply_policy(ctx: RunContext, node: Node) -> bool:
    """Mark the node failed when the off-mode scan finds pretrained weights. Returns True if violated."""
    if ctx.mode != "off":
        return False
    hits = policy_scan(node.code, list(ctx.cfg["pretrained"]["forbidden_patterns"]))
    if not hits:
        return False
    lines = "\n".join(f"  line {ln}: found '{pat}' in: {txt}" for ln, pat, txt in hits[:20])
    node.failed = True
    node.error_type = "PolicyViolation"
    node.error_message = f"pretrained weights are not allowed ({len(hits)} match(es))"
    node.output = ("[superagent policy check] The script was NOT run because it uses pretrained weights, which are "
                   "forbidden in this run. Rule: do not use any pretrained weights; every model must be trained from "
                   "scratch on this task's data only.\nMatches:\n" + lines + "\n")
    node.summary = "POLICY_VIOLATION: the code loads pretrained weights, which are forbidden; all models must be trained from scratch."
    node.failure_reasons = ["policy_violation"]
    node.predictions_ok = None
    return True


__all__ = ["review", "policy_scan", "apply_policy", "ReviewOutcome"]
