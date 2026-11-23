"""One table of token use across MLEvolve and SuperAgent tasks.

Looks for ``api_usage.json`` (preferred) and ``logs/llm_calls.jsonl``.
Rebuilds a missing bill from the call log with the MLEvolve off-peak card.

    python -m superagent.usage_board
    python -m superagent.usage_board --serve 8766
    python -m superagent.usage_board --root /path/to/MLEvolve --root /path/to/Super_agent_core --out board.html
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .billing import snapshot_from_calls
from .usage_page import render_board_html
from .utils import read_jsonl

SKIP_DIRS = {
    ".git", "__pycache__", "pretrained_cache", "node_modules", "site-packages",
    ".venv", "venv", "tokenizer", "model", "out", "artifacts",
}


def default_roots() -> list[Path]:
    course = Path(__file__).resolve().parents[3]
    roots = []
    for name in ("MLEvolve", "Super_agent_core", "tasks"):
        path = course / name
        if path.is_dir():
            roots.append(path)
    return roots


def _guess_project(path: Path) -> str:
    text = str(path)
    if "MLEvolve" in text:
        return "MLEvolve"
    if "Super_agent" in text or "superagent" in text:
        return "SuperAgent"
    return "unknown"


_GENERIC_DIRS = {"logs", "superagent_out", "core_out", "outs", "output"}


def _task_name(path: Path) -> str:
    """A readable task label. A run directory called ``superagent_out`` is named after its task folder."""
    parent = path.parent
    run_dir = parent.parent if parent.name == "logs" else parent
    if run_dir.name in _GENERIC_DIRS or run_dir.name.endswith("_out"):
        if run_dir.parent.is_dir() and run_dir.parent.name not in _GENERIC_DIRS:
            return f"{run_dir.parent.name}/{run_dir.name}"
    return run_dir.name


def _load_api_usage(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not int(data.get("calls") or 0):
        return None
    history = data.get("history") or []
    project = str(data.get("project") or _guess_project(path))
    stored = str(data.get("task") or "")
    task = _task_name(path)
    if stored and stored not in _GENERIC_DIRS and not stored.endswith("_out"):
        task = stored
    data["task"] = task
    if history and (not data.get("by_part") or not data.get("by_interval")):
        rebuilt = snapshot_from_calls(history, task=task, project=project)
        for key in ("by_part", "by_interval", "top_part", "top_interval", "total_tokens", "cost_rmb"):
            data[key] = rebuilt[key]
    elif not data.get("by_part") and data.get("by_kind"):
        # Older bills stored only generate/query totals, with no per-call history.
        model = next(iter(data.get("by_model") or {"deepseek-v4.1-flash": None}))
        calls = []
        for name, bucket in data["by_kind"].items():
            if not isinstance(bucket, dict):
                continue
            calls.append({
                "part": name,
                "kind": name,
                "model": model,
                "input_tokens": int(bucket.get("input_tokens") or 0),
                "output_tokens": int(bucket.get("output_tokens") or 0),
                "cache_tokens": int(bucket.get("cache_tokens") or 0),
                "reasoning_tokens": int(bucket.get("reasoning_tokens") or 0),
            })
        rebuilt = snapshot_from_calls(calls, task=task, project=project)
        data["by_part"] = rebuilt["by_part"]
        data["top_part"] = rebuilt["top_part"]
        data["by_interval"] = []
        data["top_interval"] = None
    data.setdefault("project", _guess_project(path))
    data.setdefault("task", _task_name(path))
    data.setdefault("total_tokens", int(data.get("input_tokens") or 0) + int(data.get("output_tokens") or 0) + int(data.get("cache_tokens") or 0))
    data["source"] = str(path)
    data["mtime"] = _mtime(path)
    return data


def _load_call_log(path: Path) -> dict[str, Any] | None:
    rows = read_jsonl(path)
    if not rows:
        return None
    data = snapshot_from_calls(rows, task=_task_name(path), project=_guess_project(path))
    if not data["calls"]:
        return None
    data["source"] = str(path)
    data["mtime"] = _mtime(path)
    return data


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _matches(item: Mapping[str, Any], only: list[str] | None) -> bool:
    if not only:
        return True
    label = f"{item.get('project') or ''} · {item.get('task') or ''}".lower()
    return any(text.lower() in label for text in only)


def collect_tasks(roots: list[Path], only: list[str] | None = None) -> list[dict[str, Any]]:
    """Every non-empty task bill under ``roots``. A directory with both files counts once."""
    tasks: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in _walk(root):
            directory = Path(dirpath)
            if "api_usage.json" in filenames:
                loaded = _load_api_usage(directory / "api_usage.json")
                seen.add(directory)
                if loaded and _matches(loaded, only):
                    tasks.append(loaded)
            elif "llm_calls.jsonl" in filenames and directory not in seen:
                loaded = _load_call_log(directory / "llm_calls.jsonl")
                if loaded and _matches(loaded, only):
                    tasks.append(loaded)
    return tasks


def _walk(root: Path):
    import os

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS and not name.startswith(".")]
        yield dirpath, dirnames, filenames


def write_board(roots: list[Path], out: Path, selected: str | None = None, only: list[str] | None = None) -> Path:
    tasks = collect_tasks(roots, only)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_board_html(tasks, selected), encoding="utf-8")
    return out


def serve_board(roots: list[Path], port: int, only: list[str] | None = None) -> None:
    from urllib.parse import parse_qs, urlparse

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path not in ("/", "/usage_board.html"):
                self.send_error(404)
                return
            query = parse_qs(parsed.query)
            selected = (query.get("task") or [None])[0]
            body = render_board_html(collect_tasks(roots, only), selected).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:
            return

    print(f"usage board http://0.0.0.0:{port}/  roots={[str(p) for p in roots]} only={only or 'all tasks'}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Token usage board for MLEvolve and SuperAgent tasks")
    parser.add_argument("--root", action="append", type=Path, default=None, help="Directory to scan. Repeat to add more. Default: the two project checkouts.")
    parser.add_argument("--out", type=Path, default=None, help="HTML file to write. Default: <Super_agent_core>/usage_board.html")
    parser.add_argument("--serve", type=int, default=None, metavar="PORT", help="Serve a live page that rebuilds on each refresh.")
    parser.add_argument("--task", default=None, help="Task to show on the written page: name, index, or a substring of project · task.")
    parser.add_argument("--only", action="append", default=None, metavar="TEXT",
                        help="Keep only tasks whose project · task contains TEXT. Repeat to allow several. Default: every task found.")
    args = parser.parse_args(argv)
    roots = args.root or default_roots()
    only = args.only
    if args.serve is not None:
        serve_board(roots, args.serve, only)
        return
    out = args.out or (Path(__file__).resolve().parents[2] / "usage_board.html")
    path = write_board(roots, out, args.task, only)
    print(f"wrote {path} ({len(collect_tasks(roots, only))} tasks)")


if __name__ == "__main__":
    main()
