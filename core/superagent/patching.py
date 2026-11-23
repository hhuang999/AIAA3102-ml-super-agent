"""Search/replace patch blocks: parsing and applying them to code."""

from __future__ import annotations

import re
from dataclasses import dataclass

START = "<<<<<<< SEARCH"
MID = "======="
END = ">>>>>>> REPLACE"
_START_RE = re.compile(r"^<{5,9} ?SEARCH\s*$")
_MID_RE = re.compile(r"^={5,9}\s*$")
_END_RE = re.compile(r"^>{5,9} ?REPLACE\s*$")


@dataclass
class PatchBlock:
    search: str
    replace: str


@dataclass
class ParsedPatch:
    blocks: list[PatchBlock]
    n_start: int
    n_end: int
    unfinished_searches: list[str]

    @property
    def truncated(self) -> bool:
        return self.n_start > self.n_end


def parse_patch(text: str) -> ParsedPatch:
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    n_start = sum(1 for ln in lines if _START_RE.match(ln))
    n_end = sum(1 for ln in lines if _END_RE.match(ln))
    blocks: list[PatchBlock] = []
    unfinished = []
    i = 0
    while i < len(lines):
        if not _START_RE.match(lines[i]):
            i += 1
            continue
        j = i + 1
        search: list[str] = []
        while j < len(lines) and not _MID_RE.match(lines[j]) and not _START_RE.match(lines[j]):
            search.append(lines[j])
            j += 1
        if j >= len(lines) or not _MID_RE.match(lines[j]):
            i = j
            continue
        k = j + 1
        replace: list[str] = []
        while k < len(lines) and not _END_RE.match(lines[k]) and not _START_RE.match(lines[k]):
            replace.append(lines[k])
            k += 1
        if k >= len(lines) or not _END_RE.match(lines[k]):
            if search:
                unfinished.append("\n".join(search))
            i = k
            continue
        blocks.append(PatchBlock("\n".join(search), "\n".join(replace)))
        i = k + 1
    return ParsedPatch(blocks, n_start, n_end, unfinished)


def text_before_first_block(text: str) -> str:
    idx = text.find("<<<<<<<")
    return (text if idx < 0 else text[:idx]).strip()


def _indent_width(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _shift_lines(lines: list[str], shift: int) -> list[str]:
    out = []
    for ln in lines:
        if not ln.strip():
            out.append("")
            continue
        width = _indent_width(ln)
        new_width = max(0, width + shift)
        out.append(" " * new_width + ln.lstrip(" \t"))
    return out


def apply_block(code: str, block: PatchBlock) -> tuple[str, bool]:
    if not block.search.strip():
        sep = "" if code.endswith("\n") or not code else "\n"
        return code + sep + "\n" + block.replace + ("\n" if not block.replace.endswith("\n") else ""), True
    search_lines = block.search.split("\n")
    code_lines = code.split("\n")
    normalized = [line.rstrip() for line in code_lines]
    length = len(search_lines)
    exact = []
    shifted = []
    for i in range(len(code_lines) - length + 1):
        first = normalized[i]
        if first.strip() != search_lines[0].strip():
            continue
        window = normalized[i:i + length]
        if window == search_lines:
            exact.append((i, 0))
            continue
        shift = _indent_width(first) - _indent_width(search_lines[0])
        if window == _shift_lines(search_lines, shift):
            shifted.append((i, shift))
    matches = exact or shifted
    if len(matches) != 1:
        return code, False
    index, shift = matches[0]
    replacement = block.replace.split("\n") if block.replace else []
    if shift:
        replacement = _shift_lines(replacement, shift)
    code_lines[index:index + length] = replacement
    return "\n".join(code_lines), True


def apply_patch(code: str, patch_text: str) -> tuple[str, int]:
    """Apply every complete block in order. Returns (new code, number of applied blocks)."""
    parsed = parse_patch(patch_text)
    applied = 0
    for b in parsed.blocks:
        code, ok = apply_block(code, b)
        if ok:
            applied += 1
    return code, applied
