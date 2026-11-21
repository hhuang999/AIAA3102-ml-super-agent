"""Pre-run checks: LLM code review and the pretrained-policy scan."""

from __future__ import annotations

import ast
import logging
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
    "predict_only", "load_data", "read_split", "split", "make_splits", "evaluate", "compute_metric",
})
_SENSITIVE_DIFF_RE = (
    r"\b(train_test_split|StratifiedKFold|KFold|GroupKFold|TimeSeriesSplit|"
    r"predict_only|val_predictions|val_labels|inference\.json|fit_transform|from_pretrained)\b"
)


def _changed_symbol_names(before: str, after: str) -> set[str]:
    def sym_dumps(src: str) -> dict[str, str]:
        tree = ast.parse(src)
        out: dict[str, str] = {}

        def walk(body, prefix=""):
            for n in body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    name = prefix + n.name
                    out[name] = ast.dump(n)
                    if isinstance(n, ast.ClassDef):
                        walk(n.body, name + ".")

        walk(tree.body)
        return out

    try:
        b = sym_dumps(before)
        a = sym_dumps(after)
    except SyntaxError:
        return {"__syntax_error__"}
    names = {k.rsplit(".", 1)[-1] for k in (set(b) ^ set(a)) | {k for k in (set(b) & set(a)) if b[k] != a[k]}}
    return names


def review(ctx: RunContext, node: Node) -> ReviewOutcome:
    rc = ctx.cfg["review"]
    if not rc["enabled"]:
        return ReviewOutcome(False, None, "review disabled")
    import difflib
    import re
    from .llm.ledger import fingerprint
    diff = difflib.unified_diff((node.parent.code if node.parent else "").splitlines(),node.code.splitlines(),n=3)
    changed = "\n".join(x[1:] for x in diff if x.startswith(("+","-")) and not x.startswith(("+++","---")))
    high_risk = node.kind in ("draft","synthesis") or not node.parent or bool(re.search(
        r"split|fold|label|target|metric|predict|valid|sampler|dataset|read_csv|from_pretrained",changed,re.I))
    if rc.get("mode")=="risk" and not high_risk:
        return ReviewOutcome(False,"pass","low-risk diff: deterministic execution and output checks")
    ledger=getattr(ctx.gateway,"ledger",None)
    changed_syms = _changed_symbol_names(node.parent.code, node.code) if node.parent and node.parent.code else set()
    sensitive = (
        node.kind in ("draft", "synthesis")
        or not node.parent
        or bool(re.search(_SENSITIVE_DIFF_RE, changed, re.I))
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
