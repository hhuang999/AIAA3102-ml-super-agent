"""Pre-run checks: LLM code review and the pretrained-policy scan."""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass

from .codegen import python_syntax_error, solution_error
from .context import RunContext
from .errors import StructuredOutputError
from .llm import Field
from .patching import apply_patch, parse_patch
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


SENSITIVE_SYMBOLS = frozenset({
    "predict_only", "load_data", "make_split", "split", "build_folds",
    "evaluate", "compute_metric", "score", "save_model", "load_model", "main",
})
_SENSITIVE_DIFF_RE = re.compile(
    r"\b(?:KFold|StratifiedKFold|GroupKFold|TimeSeriesSplit|train_test_split|"
    r"TargetEncoder|StandardScaler|MinMaxScaler|RobustScaler|TfidfVectorizer|CountVectorizer|"
    r"SMOTE|predict_only|val_predictions|val_labels|predictions\.csv|sample_predictions|"
    r"from_pretrained|read_csv|inference\.json|--predict-only)\b"
)
_HIGH_RISK_RE = re.compile(
    r"\b(?:split|kfold|stratified|fold|sampler|dataset|read_csv|from_pretrained|"
    r"predict_only|val_predictions|val_labels|predictions\.csv|sample_predictions|inference\.json)\b|threshold|blend|stack",
    re.I,
)


def _changed_symbol_names(before_code: str, after_code: str) -> set[str]:
    if not before_code.strip():
        return set()
    try:
        from .planner import python_symbols
        before = python_symbols(before_code)
        after = python_symbols(after_code)
    except SyntaxError:
        return set()
    changed_syms: set[str] = set()
    for name in set(before) | set(after):
        b, a = before.get(name), after.get(name)
        if b is None or a is None or ast.dump(b) != ast.dump(a):
            changed_syms.add(name)
            changed_syms.add(name.rsplit(".", 1)[-1])
    return changed_syms


def review(ctx: RunContext, node: Node) -> ReviewOutcome:
    rc = ctx.cfg["review"]
    if not rc["enabled"]:
        return ReviewOutcome(False, None, "review disabled")
    import difflib
    from .llm.ledger import fingerprint
    parent_code = node.parent.code if node.parent else ""
    diff = difflib.unified_diff(parent_code.splitlines(), node.code.splitlines(), n=3)
    changed_lines = [
        _strip_comment(x[1:]) for x in diff
        if x.startswith(("+", "-")) and not x.startswith(("+++", "---")) and not x[1:].lstrip().startswith("#")
    ]
    changed_code = "\n".join(changed_lines)
    changed_syms = _changed_symbol_names(parent_code, node.code)
    high_risk = (
        node.kind in ("draft", "synthesis")
        or not node.parent
        or bool(_HIGH_RISK_RE.search(changed_code))
        or bool(changed_syms & SENSITIVE_SYMBOLS)
    )
    if rc.get("mode") == "risk" and not high_risk:
        return ReviewOutcome(False, "pass", "low-risk diff: deterministic execution and output checks")
    ledger = getattr(ctx.gateway, "ledger", None)
    sensitive = (
        node.kind in ("draft", "synthesis")
        or not node.parent
        or bool(_SENSITIVE_DIFF_RE.search(changed_code))
        or bool(changed_syms & SENSITIVE_SYMBOLS)
    )
    effort = rc["sensitive_effort"] if sensitive else None
    sig=fingerprint([ctx.task_desc,node.code,effort])
    saved=ledger.get(node.id,"review",sig) if ledger else None
    system, user = builders.reviewer(ctx.task_desc, node.code, ctx.mode, max(0.0, ctx.current_run_limit()),
                                     declared_split=ctx.task_spec is not None)
    revised = node.code

    def check(obj, _last):
        nonlocal revised
        if obj["verdict"] == "pass":
            return obj, []
        patch = obj.get("patch") or ""
        parsed = parse_patch(patch)
        candidate, applied = apply_patch(node.code, patch)
        if not (applied > 0 and applied == parsed.n_start == parsed.n_end and candidate != node.code):
            return obj, ["a revise verdict requires complete patch blocks with unique matching SEARCH lines"]
        error = solution_error(candidate) if ctx.flags.get("formal") else python_syntax_error(candidate)
        if error:
            return obj, [f"the revised script does not compile: {error}"]
        revised = candidate
        return obj, []

    try:
        if saved and not check(saved, False)[1]:
            obj = saved
        else:
            obj = ctx.gateway.structured("reviewer", system, user, REVIEW_SCHEMA, node_id=node.id,
                                         initial_effort=effort, extra_check=check,
                                         expanded=effort not in (None, "off"),
                                         resume_key=sig,
                                         max_attempts=min(rc["max_attempts"], ctx.cfg["llm"]["json_max_attempts"]))
    except StructuredOutputError as exc:
        return ReviewOutcome(False, None, f"review call failed: {exc}")
    if ledger:
        ledger.put(node.id,"review",sig,obj)
    changed = revised != node.code
    node.code = revised
    return ReviewOutcome(changed, obj["verdict"], obj["findings"])


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
