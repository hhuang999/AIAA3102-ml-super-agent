"""Code generation: full-script mode, staged drafting and patch mode with cumulative retries."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Callable

from .llm.gateway import LLMGateway
from .patching import apply_patch, parse_patch, text_before_first_block

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


@dataclass
class PatchResult:
    code: str | None              # None when patching failed completely
    applied_blocks: int
    attempts: int
    partial: bool
    first_explanation: str
    notes: list[str] = field(default_factory=list)


def _syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError):
        return False


def extract_code(reply: str, truncated: bool = False) -> Extracted | None:
    """Keep every fenced block that parses as Python, concatenated; the text before the first fence is the plan."""
    if truncated:
        return None
    blocks = [m.group(2) for m in _FENCE_RE.finditer(reply or "")]
    good = [b.strip("\n") for b in blocks if b.strip() and _syntax_ok(b)]
    if not good:
        return None
    first = reply.find("```")
    plan = reply[:first].strip() if first >= 0 else ""
    code = "\n\n".join(good) + "\n"
    if not plan or not code.strip():
        return None
    return Extracted(plan=plan, code=code)


class CodeGenerator:
    def __init__(self, gateway: LLMGateway, extract_attempts: int, patch_attempts: int) -> None:
        self.gw = gateway
        self.extract_attempts = int(extract_attempts)
        self.patch_attempts = int(patch_attempts)

    def full_script(self, role: str, system: str, user: str, node_id: str) -> FullScriptResult:
        last = ""
        note = ""
        for i in range(self.extract_attempts):
            prompt = user if not note else note + "\n\n" + user
            res = self.gw.text(role, system, prompt, node_id=node_id)
            last = res.content
            ex = extract_code(res.content, res.truncated)
            if ex is not None:
                return FullScriptResult(ex, last, i + 1)
            note = ("# Note\nYour previous reply could not be used: " +
                    ("it was cut off by the output length limit; be more concise." if res.truncated else
                     "it must contain a short overview followed by one ```python block holding a complete script that parses as valid Python."))
        return FullScriptResult(None, last, self.extract_attempts)

    def staged(
        self,
        build_stage: Callable[[str, list[tuple[str, str, str]]], tuple[str, str]],
        build_integrator: Callable[[list[tuple[str, str, str]]], tuple[str, str]],
        node_id: str,
    ) -> tuple[Extracted | None, str]:
        """Write data_prep, model_def, train_eval, then integrate. Returns (result, plan) or (None, '')."""
        done: list[tuple[str, str, str]] = []
        for stage in STAGES:
            system, user = build_stage(stage, done)
            r = self.full_script(f"draft_writer.{stage}", system, user, node_id)
            if r.extracted is None:
                return None, ""
            done.append((stage, r.extracted.plan, r.extracted.code))
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
        state = PatchState(parent_code=parent_code, current=parent_code)
        for _ in range(self.patch_attempts):
            system, user = build_prompt(state.current, state.retry_note)
            res = self.gw.text(role, system, user, node_id=node_id)
            if patch_step(state, res.content, res.truncated):
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
    finished: bool = False
    notes: list[str] = field(default_factory=list)


NOTE_NO_BLOCKS = "# Retry note\nThe previous output contained no valid patch blocks. Output only complete patch blocks."
NOTE_NO_MATCH = ("# Retry note\nThe previous patch did not apply cleanly to the current code. Generate minimal patch blocks "
                 "whose SEARCH parts are copied exactly from the current code below.")


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
    new_code, applied = apply_patch(state.current, reply)
    if applied > 0 and new_code != state.current:
        state.current = new_code
        state.applied_total += applied
    if state.applied_total > 0 and state.current != state.parent_code and not truncated:
        state.finished = True
        return True
    if truncated and state.applied_total > 0:
        state.retry_note = (f"# Retry note\nThe previous output was cut off. {state.applied_total} patch block(s) have already been "
                            "applied (the code below includes them). Give only the remaining patch blocks.")
        state.notes.append("truncated")
    else:
        state.retry_note = NOTE_NO_MATCH
        state.notes.append("no_match")
    return False


def patch_result(state: PatchState) -> PatchResult:
    """The outcome once the loop has ended (finished, or attempts used up)."""
    changed = state.applied_total > 0 and state.current != state.parent_code
    if state.finished:
        return PatchResult(state.current, state.applied_total, state.attempts, False, state.first_explanation, state.notes)
    if changed:
        return PatchResult(state.current, state.applied_total, state.attempts, True, state.first_explanation, state.notes)
    return PatchResult(None, 0, state.attempts, False, state.first_explanation, state.notes)
