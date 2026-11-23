"""Lenient JSON extraction from LLM replies."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.DOTALL)


def parse_complete_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    fenced = _FENCE_RE.fullmatch(stripped)
    try:
        value = json.loads(fenced.group(1) if fenced else stripped)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _escape_control_chars_in_strings(text: str) -> str:
    """Escape raw newlines, carriage returns and tabs that appear inside JSON strings."""
    out: list[str] = []
    in_str = False
    escaped = False
    for ch in text:
        if in_str:
            if escaped:
                escaped = False
                out.append(ch)
            elif ch == "\\":
                escaped = True
                out.append(ch)
            elif ch == '"':
                in_str = False
                out.append(ch)
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_str = True
            out.append(ch)
    return "".join(out)


def _first_object_span(text: str) -> str | None:
    """Return the first balanced ``{...}`` block, ignoring braces inside strings."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def _try_load(candidate: str) -> Any:
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_escape_control_chars_in_strings(candidate))
    except json.JSONDecodeError:
        return None


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object found in ``text``. Returns None when none can be parsed."""
    if not text:
        return None
    candidates: list[str] = []
    stripped = text.strip()
    candidates.append(stripped)
    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group(1).strip())
    for cand in list(candidates):
        span = _first_object_span(cand)
        if span:
            candidates.append(span)
    span = _first_object_span(_escape_control_chars_in_strings(text))
    if span:
        candidates.append(span)
    for cand in candidates:
        obj = _try_load(cand)
        if isinstance(obj, dict):
            return obj
    return None
