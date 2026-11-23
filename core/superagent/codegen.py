"""Code generation: full-script mode, staged drafting and patch mode with cumulative retries."""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field
from .llm.ledger import fingerprint
from typing import Callable

from .llm.gateway import LLMGateway
from .patching import PatchBlock, apply_block, parse_patch, text_before_first_block

_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)
STAGES = ("data_prep", "model_def", "train_eval")


@dataclass
class Extracted:
    plan: str
    code: str


@dataclass
class FullScriptResult:
    extracted: Extracted | None
    last_reply: str
    attempts: int
    failure_reason: str | None = None


@dataclass
class PatchResult:
    code: str | None              # None when patching failed completely
    applied_blocks: int
    attempts: int
    partial: bool
    first_explanation: str
    notes: list[str] = field(default_factory=list)


def python_syntax_error(code: str) -> str | None:
    try:
        compile(code, "<generated code>", "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def extract_code(reply: str, truncated: bool = False) -> Extracted | None:
    """Validate the complete Python artifact, including all of its Python code blocks."""
    if truncated:
        return None
    blocks = [m for m in _FENCE_RE.finditer(reply or "") if m.group(1).lower() in ("", "py", "python", "python3")]
    if not blocks:
        return None
    plan = reply[:blocks[0].start()].strip()
    code = "\n\n".join(m.group(2).strip("\n") for m in blocks) + "\n"
    if not plan or not code.strip() or python_syntax_error(code):
        return None
    return Extracted(plan=plan, code=code)


def solution_error(code):
    """Static entry-point check for generated scripts. Smoke verifies actual behavior."""
    if error := python_syntax_error(code):
        return error
    tree = ast.parse(code)
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    entry = {"__main__", "--predict-only", "--model-dir", "--data-dir", "--output-dir"}
    filenames = {"val_predictions.csv", "predictions.csv", "inference.json"}
    basenames = {value.replace("\\", "/").rsplit("/", 1)[-1] for value in strings}
    missing = (entry - strings) | (filenames - basenames)
    if missing:
        return "code_incomplete: missing " + ", ".join(sorted(missing))
    return None


def _top_imports_dump(code: str) -> str:
    tree = ast.parse(code)
    return ast.dump(ast.Module(
        body=[n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))],
        type_ignores=[],
    ))


def _masked_without_target_and_imports(code: str, target: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    sentinel = lambda tag: ast.Expr(value=ast.Constant(value=f"__TARGET__:{tag}"))

    def mask_block(stmts: list[ast.stmt], prefix: str = "") -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for stmt in stmts:
            if not prefix and isinstance(stmt, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qname = prefix + stmt.name
                if qname == target:
                    out.append(sentinel(qname))
                    continue
                if isinstance(stmt, ast.ClassDef) and target.startswith(qname + "."):
                    stmt.body = mask_block(stmt.body, qname + ".")
            out.append(stmt)
        return out

    tree.body = mask_block(tree.body)
    return ast.dump(ast.fix_missing_locations(tree), include_attributes=False)


def _allowed_change_ast(before_code: str, after_code: str, target: str) -> bool:
    if python_syntax_error(before_code) or python_syntax_error(after_code):
        return False
    from .planner import python_symbols

    if target == "__imports__":
        return (
            _masked_without_target_and_imports(before_code, "__imports__")
            == _masked_without_target_and_imports(after_code, "__imports__")
            and _top_imports_dump(before_code) != _top_imports_dump(after_code)
        )
    before_sym = python_symbols(before_code).get(target)
    after_sym = python_symbols(after_code).get(target)
    if before_sym is None or after_sym is None or ast.dump(before_sym) == ast.dump(after_sym):
        return False
    return (
        _masked_without_target_and_imports(before_code, target)
        == _masked_without_target_and_imports(after_code, target)
    )


def plan_coverage(plan, original, candidate):
    from .planner import python_symbols

    if python_syntax_error(original) or python_syntax_error(candidate):
        return [change["id"] for change in plan["changes"]]
    before = python_symbols(original)
    after = python_symbols(candidate)
    missing = []
    for change in plan["changes"]:
        target = change["target"]
        if target == "whole_script":
            if ast.dump(ast.parse(original)) == ast.dump(ast.parse(candidate)):
                missing.append(change["id"])
        elif target == "__imports__":
            if _top_imports_dump(original) == _top_imports_dump(candidate):
                missing.append(change["id"])
        elif target not in before or target not in after or ast.dump(before[target]) == ast.dump(after[target]):
            missing.append(change["id"])
    return missing


_CHANGE = re.compile(r"^CHANGE\s+([A-Za-z0-9_-]+)\s*$", re.MULTILINE)


def _imports_excerpt(code: str) -> str:
    lines = code.splitlines()
    try:
        tree = ast.parse(code)
        imp_lines = [
            "\n".join(lines[n.lineno - 1:(n.end_lineno or n.lineno)])
            for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
        ]
        if imp_lines:
            return "\n".join(imp_lines)
    except SyntaxError:
        pass
    return "\n".join(lines[:30])


def apply_planned_reply(code, plan, reply, applied=()):
    """Apply complete, compile-valid blocks that edit their named symbol (plus optional top-level imports)."""
    from .planner import python_symbols

    done = set(applied)
    changes = {change["id"]: change for change in plan["changes"]}
    tags = list(_CHANGE.finditer(reply))
    notes = []
    for index, tag in enumerate(tags):
        ident = tag.group(1)
        if ident not in changes or ident in done:
            notes.append(f"unexpected or duplicate change ID {ident}")
            continue
        target = changes[ident]["target"]
        symbols = python_symbols(code)
        if target != "__imports__" and target not in symbols:
            notes.append(f"{ident}: unknown target {target}")
            continue
        segment = reply[tag.end(): tags[index + 1].start() if index + 1 < len(tags) else len(reply)]
        parsed = parse_patch(segment)
        if parsed.n_start < 1 or parsed.n_start != parsed.n_end or len(parsed.blocks) != parsed.n_start:
            notes.append(f"{ident}: needs complete SEARCH/REPLACE block(s) at {target}")
            continue
        if any(not b.search.strip() for b in parsed.blocks):
            notes.append(f"{ident}: SEARCH must locate existing code")
            continue
        result = code
        all_matched = True
        for block in parsed.blocks:
            result, matched = apply_block(result, block)
            if not matched:
                all_matched = False
                break
        if not all_matched or python_syntax_error(result):
            notes.append(f"{ident}: SEARCH must match uniquely and preserve valid Python")
            continue
        if not _allowed_change_ast(code, result, target):
            notes.append(f"{ident}: code diff must change only {target} (and top-level imports)")
            continue
        code = result
        done.add(ident)
    return code, sorted(done), notes


def planned_patch(gateway, operation, signature, plan, code, rules, *, role="patch_coder"):
    from .planner import python_symbols

    ledger = gateway.ledger
    saved = ledger.get(operation, "planned_patch", signature)
    current, applied, notes = (saved["code"], saved["applied"], saved["notes"]) if saved else (code, [], [])
    all_changes = plan["changes"]
    tries = saved["tries"] if saved else 0
    expanded = saved["expanded"] if saved else False
    system = "Implement the approved plan with precise, anchor-based Python edits."
    while tries < 2:
        pending = [change for change in all_changes if change["id"] not in applied]
        if not pending:
            return current
        symbols = python_symbols(current)
        lines = current.splitlines()
        excerpts = {}
        for item in pending:
            t = item["target"]
            if t == "__imports__":
                excerpts[t] = _imports_excerpt(current)
            elif t in symbols:
                excerpts[t] = "\n".join(lines[symbols[t].lineno - 1:symbols[t].end_lineno])
        user = (
            f"{rules}\nPlan: {plan['hypothesis']}\nRemaining changes: {pending}\nPreserve: {plan['preserve']}\n"
            "For each edit, write a line CHANGE followed by its exact ID, then one or more "
            "<<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks. Copy unique anchors verbatim. "
            "Top-level import additions may accompany a target symbol edit under the same CHANGE ID.\n"
            f"Top-level imports excerpt:\n{_imports_excerpt(current)}\n"
            f"Current target source: {excerpts}\n"
            f"Prior validation: {notes}"
        )
        response = gateway.text(role, system, user, node_id=operation, expanded=expanded)
        current, applied, notes = apply_planned_reply(current, plan, response.content, applied)
        if response.truncated:
            notes.append("The previous response ended before transport or generation completed.")
        expanded = expanded or response.body_limited
        tries += 1
        ledger.put(operation, "planned_patch", signature,
                   {"code": current, "applied": applied, "notes": notes, "tries": tries, "expanded": expanded})
    return current if not plan_coverage(plan, code, current) else None


class CodeGenerator:
    def __init__(self, gateway: LLMGateway, extract_attempts: int, patch_attempts: int) -> None:
        self.gw = gateway
        self.extract_attempts = int(extract_attempts)
        self.patch_attempts = int(patch_attempts)

    def full_script(self, role: str, system: str, user: str, node_id: str) -> FullScriptResult:
        ledger = getattr(self.gw, "ledger", None) if getattr(self.gw, "formal", False) else None
        signature = fingerprint([system, user])
        stage = f"{role}.code_retry"
        saved = ledger.get(node_id, stage, signature) if ledger else None
        last = ""
        reason = "code_incomplete"
        note = saved["note"] if saved else ""
        seen = {tuple(item) for item in saved["seen"]} if saved else set()
        expanded = saved["expanded"] if saved else False
        for i in range(len(seen), self.extract_attempts):
            prompt = user if not note else note + "\n\n" + user
            key = (prompt, self.gw.output_limit(role, expanded))
            if key in seen:
                break
            res = self.gw.text(role, system, prompt, node_id=node_id, expanded=expanded)
            seen.add(key)
            last = res.content
            ex = extract_code(res.content, res.truncated)
            malformed_fence = getattr(self.gw, "formal", False) and res.content.count("```") % 2
            error = None
            if ex and getattr(self.gw, "formal", False) and role in ("script_coder", "draft_integrator"):
                error = solution_error(ex.code)
            if ex is not None and not error and not malformed_fence:
                return FullScriptResult(ex, last, i + 1)
            reason = error or (
                "no_transport_terminator" if res.finish_reason == "no_transport_terminator" else
                "model_idle_timeout" if res.finish_reason == "model_idle_timeout" else
                "output_truncated" if res.finish_reason == "length" else
                "empty_response" if not res.content.strip() else "code_incomplete"
            )
            can_expand = not expanded and self.gw.output_limit(role, True) > self.gw.output_limit(role, False)
            if res.finish_reason == "length" and not res.content.strip() and not can_expand:
                break
            expanded = expanded or res.body_limited or (res.finish_reason == "length" and can_expand)
            note = ("# Note\nYour previous reply could not be used: " +
                    ("it was incomplete; return the complete script with all required behavior." if res.truncated else
                     "it must contain a short overview followed by one ```python block holding a complete script that parses as valid Python."))
            if error:
                note += "\n" + error
            if ledger:
                ledger.put(node_id, stage, signature, {"note": note, "seen": sorted(seen), "expanded": expanded})
        return FullScriptResult(None, last, len(seen), reason)

    def staged(
        self,
        build_stage: Callable[[str, list[tuple[str, str, str]]], tuple[str, str]],
        build_integrator: Callable[[list[tuple[str, str, str]]], tuple[str, str]],
        node_id: str,
    ) -> tuple[Extracted | None, str]:
        """Write data_prep, model_def, train_eval, then integrate. Returns (result, plan) or (None, '')."""
        done: list[tuple[str, str, str]] = []
        for stage in STAGES:
            ledger=getattr(self.gw,"ledger",None)
            saved=ledger.get(node_id,"draft."+stage,"accepted") if ledger else None
            if saved:
                done.append(tuple(saved))
                continue
            system, user = build_stage(stage, done)
            r = self.full_script(f"draft_writer.{stage}", system, user, node_id)
            if r.extracted is None:
                return None, ""
            done.append((stage, r.extracted.plan, r.extracted.code))
            if ledger:
                ledger.put(node_id,"draft."+stage,"accepted",done[-1])
        system, user = build_integrator(done)
        r = self.full_script("draft_integrator", system, user, node_id)
        if r.extracted is None:
            return None, ""
        plan = "\n\n".join(f"[{name}] {plan}" for name, plan, _code in done)
        return Extracted(plan=plan, code=r.extracted.code), plan

    def patch(
        self,
        role: str,
        parent_code: str,
        build_prompt: Callable[[str, str], tuple[str, str]],
        node_id: str,
    ) -> PatchResult:
        """Cumulative retry loop. ``build_prompt(current_code, retry_note)`` returns (system, user)."""
        ledger=getattr(self.gw,"ledger",None)
        sig=fingerprint(parent_code)
        saved=ledger.get(node_id,"patch",sig) if ledger else None
        state=PatchState(**saved) if saved else PatchState(parent_code=parent_code,current=parent_code)
        state.pending = [tuple(target) for target in state.pending]
        for _ in range(state.attempts,self.patch_attempts):
            if state.finished or state.exhausted:
                break
            system, user = build_prompt(state.current, state.retry_note)
            key = fingerprint([system,user,self.gw.output_limit(role,state.expanded)])
            if key in state.no_progress:
                break
            res = self.gw.text(role, system, user, node_id=node_id, expanded=state.expanded)
            before = state.current
            complete=patch_step(state,res.content,res.truncated)
            if state.current == before:
                state.no_progress.append(key)
            state.expanded = state.expanded or (res.body_limited and before == state.current)
            if res.finish_reason == "length" and not res.content.strip():
                state.exhausted = True
            if ledger:
                ledger.put(node_id,"patch",sig,asdict(state))
            if complete:
                break
        return patch_result(state)


@dataclass
class PatchState:
    """Progress of the cumulative patch loop between two replies."""

    parent_code: str
    current: str
    applied_total: int = 0
    attempts: int = 0
    retry_note: str = ""
    first_explanation: str = ""
    exhausted: bool = False
    expanded: bool = False
    no_progress: list[str] = field(default_factory=list)
    pending: list[tuple[str,str | None]] = field(default_factory=list)
    finished: bool = False
    notes: list[str] = field(default_factory=list)


NOTE_NO_BLOCKS = "# Retry note\nThe previous output contained no valid patch blocks. Output only complete patch blocks."
NOTE_NO_MATCH = ("# Retry note\nThe previous patch did not apply cleanly to the current code. Copy full SEARCH lines "
                 "from the current code and include enough surrounding lines to identify a unique location.")


def _changes_wrapped_search(search: str, block: PatchBlock) -> bool:
    """Recognize a retry that keeps extra anchor lines and revises the original span."""
    pending = search.splitlines()
    anchor = block.search.splitlines()
    replacement = block.replace.splitlines()
    width = len(pending)
    for i in range(len(anchor) - width + 1):
        if anchor[i:i + width] != pending:
            continue
        suffix = anchor[i + width:]
        end = len(replacement) - len(suffix)
        if end >= i and replacement[:i] == anchor[:i] and replacement[end:] == suffix:
            if replacement[i:end] != pending:
                return True
    return False


def patch_step(state: PatchState, reply: str, reply_truncated: bool = False) -> bool:
    """Process one reply of the patch writer. No LLM call happens here.

    Updates ``state`` (current code, applied block count, note for the next attempt) and
    returns True when the loop is finished with a complete patch.
    """
    state.attempts += 1
    if not state.first_explanation:
        state.first_explanation = text_before_first_block(reply)
    parsed = parse_patch(reply)
    if parsed.n_start == 0:
        state.retry_note = NOTE_NO_BLOCKS
        state.notes.append("no_blocks")
        return False
    truncated = parsed.truncated or reply_truncated
    new_code, applied = state.current, 0
    for block in parsed.blocks:
        updated, matched = apply_block(new_code, block)
        target = (block.search, block.replace)
        if matched:
            applied += 1
            if updated != new_code:
                state.pending = [(s,r) for s,r in state.pending if not (
                    (s.strip() == block.search.strip() and (r is None or r == block.replace))
                    or (r == block.replace and r.strip() and len([t for t in state.pending if t[1] == r]) == 1)
                    or _changes_wrapped_search(s, block)
                )]
                new_code = updated
        elif target not in state.pending:
            state.pending.append(target)
    for search in parsed.unfinished_searches:
        if (search,None) not in [tuple(t) for t in state.pending]:
            state.pending.append((search,None))
    if applied > 0 and new_code != state.current:
        state.current = new_code
        state.applied_total += applied
    syntax_error = None
    complete = (applied > 0 and applied == parsed.n_start == parsed.n_end
                and state.current != state.parent_code and not truncated and not state.pending)
    if complete:
        syntax_error = python_syntax_error(state.current)
    if complete and not syntax_error:
        state.finished = True
        return True
    if syntax_error:
        state.retry_note = f"# Retry note\nThe current code does not compile: {syntax_error}. Repair it using patch blocks."
        state.notes.append("syntax_error")
    elif truncated and state.applied_total > 0:
        state.retry_note = (f"# Retry note\nThe previous output was cut off. {state.applied_total} patch block(s) have already been "
                            "applied (the code below includes them). Give only the remaining patch blocks.")
        state.notes.append("truncated")
    else:
        state.retry_note = NOTE_NO_MATCH
        state.notes.append("no_match")
    if state.pending:
        state.retry_note += "\nOutstanding unapplied replacements (correct the SEARCH anchors):\n" + "\n".join(
            f"SEARCH:\n{s}\nREPLACE:\n{r if r is not None else '(unfinished; complete from the plan)'}" for s,r in state.pending
        )
    return False


def patch_result(state: PatchState) -> PatchResult:
    """The outcome once the loop has ended (finished, or attempts used up)."""
    changed = state.applied_total > 0 and state.current != state.parent_code
    if state.finished:
        return PatchResult(state.current, state.applied_total, state.attempts, False, state.first_explanation, state.notes)
    if changed:
        return PatchResult(state.current, state.applied_total, state.attempts, True, state.first_explanation, state.notes)
    return PatchResult(None, 0, state.attempts, False, state.first_explanation, state.notes)
