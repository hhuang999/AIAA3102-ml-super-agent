"""Text pieces assembled by code and placed into prompts."""

from __future__ import annotations

from typing import Mapping

from .checks import marker_lines
from .tree import Direction, Node

SEPARATOR = "-" * 40
MAX_MARKER_LINES_KEPT = 3


def trim_output(text: str, threshold: int = 5100, keep_each_side: int = 2500) -> str:
    """Shorten long run output, keeping both ends and any score markers from the middle."""
    if len(text) <= threshold:
        return text
    head = text[:keep_each_side]
    tail = text[-keep_each_side:]
    middle = text[keep_each_side:-keep_each_side]
    omitted = len(middle)
    markers = marker_lines(middle)[-MAX_MARKER_LINES_KEPT:]
    parts = [head, f"\n... [{omitted} characters omitted] ...\n"]
    if markers:
        parts.append("[score marker lines from the omitted part]\n" + "\n".join(markers) + "\n")
    parts.append(tail)
    return "".join(parts)


def trim_with(cfg: Mapping[str, Mapping[str, int]], text: str) -> str:
    t = cfg["output_trim"]
    return trim_output(text, int(t["threshold"]), int(t["keep_each_side"]))


def score_change(node: Node, direction: Direction) -> str:
    if node.score is None:
        return "n/a"
    p = node.parent
    if p is not None and p.ok and p.score is not None:
        g = direction.gain(node.score, p.score)
        arrow = "↑" if g > 0 else ("↓" if g < 0 else "→")
        return f"{p.score:.4f} → {node.score:.4f} ({g:+.4f} {arrow})"
    return f"{node.score:.4f}"


def _fmt_secs(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0f}s"


def attempts_summary(node: Node, direction: Direction) -> str:
    """What has already been tried as children of ``node``."""
    kids = [c for c in node.children if c.evaluated or c.kind == "draft"]
    if not kids:
        return ""
    ok = sorted([c for c in kids if c.ok and c.score is not None], key=lambda c: direction.sort_key(c.score))  # type: ignore[arg-type]
    failed = [c for c in kids if c.is_failed]
    pending = [c for c in kids if not c.evaluated]
    ordered = ok + failed + pending
    ran = len(ok) + len(failed)
    best = f"{ok[0].score:.4f}" if ok else "n/a"
    lines = [f"{len(ordered)} attempts so far: {len(pending)} waiting to run, {ran} already run, {len(ok)} successful; best score {best}.", SEPARATOR]
    for i, c in enumerate(ordered, 1):
        lines.append(f"Attempt {i}")
        lines.append(f"Design: {c.plan.strip() or '(no description)'}")
        if not c.evaluated:
            lines.append("Status: code generated, waiting to run.")
        elif c.is_failed:
            lines.append("Result: the implementation of this design had errors. A different implementation approach may avoid the same error.")
        else:
            lines.append(f"Result: {c.summary.strip()}")
            lines.append(f"Validation score: {score_change(c, direction)}")
            lines.append(f"Run time: {_fmt_secs(c.exec_seconds)}")
        lines.append(SEPARATOR)
    return "\n".join(lines)


def node_card(node: Node, direction: Direction, with_code: bool = False) -> str:
    lines = [f"Branch: {node.branch_id}", f"Node type: {node.kind}", f"Design: {node.plan.strip() or '(no description)'}"]
    if with_code and node.code:
        lines.append("Code:\n```python\n" + node.code.rstrip() + "\n```")
    if not node.evaluated:
        lines.append("Result: not run yet.")
    elif node.is_failed:
        lines.append(f"Result: the implementation had errors. {node.summary.strip()}")
    else:
        lines.append(f"Result: {node.summary.strip()}")
        lines.append(f"Validation score: {score_change(node, direction)}")
        lines.append(f"Run time: {_fmt_secs(node.exec_seconds)}")
    return "\n".join(lines)


def trajectory_nodes(node: Node, max_steps: int) -> list[Node]:
    path: list[Node] = []
    x: Node | None = node
    while x is not None and not x.is_root:
        path.append(x)
        x = x.parent
    path = path[:max_steps]
    return list(reversed(path))


def trajectory(node: Node, direction: Direction, max_steps: int) -> str:
    nodes = trajectory_nodes(node, max_steps)
    parts = []
    for k, n in enumerate(nodes, 1):
        parts.append(f"Step {k}\n{node_card(n, direction)}")
    return f"\n{SEPARATOR}\n".join(parts)
