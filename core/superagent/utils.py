"""Small shared helpers: time stamps, ids, atomic file writes, JSONL appends."""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

_JSONL_LOCK = threading.Lock()
_UMASK = os.umask(0)
os.umask(_UMASK)


def now_iso() -> str:
    """Current local time as ISO 8601 with time zone offset."""
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def new_id() -> str:
    """A unique identifier made only of letters and digits."""
    return uuid.uuid4().hex[:16]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    try:
        import numpy as np  # local import keeps startup light

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(obj)


def sanitize_json(obj: Any) -> Any:
    """Replace non-finite floats (which JSON cannot hold) with None, recursively."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    return obj


def dumps(obj: Any, indent: int | None = None) -> str:
    return json.dumps(sanitize_json(obj), ensure_ascii=False, indent=indent, default=_json_default)


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    """Write text to a temporary file next to ``path`` and rename it over ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(tmp, 0o666 & ~_UMASK)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: str | os.PathLike[str], obj: Any) -> None:
    atomic_write_text(path, dumps(obj, indent=2) + "\n")


def append_jsonl(path: str | os.PathLike[str], obj: Any) -> None:
    """Append one JSON object as a line. Thread-safe within the process."""
    line = dumps(obj) + "\n"
    with _JSONL_LOCK:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"  # pragma: no cover


def is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def load_env_file(path: str | os.PathLike[str], environ: dict[str, str] | None = None) -> list[str]:
    """Read ``KEY=VALUE`` lines from a ``.env`` file into the environment.

    Variables that are already set keep their value. Blank lines and lines starting with ``#``
    are ignored; an optional leading ``export`` and matching quotes around the value are removed.
    Returns the names that were set. A missing file is not an error.
    """
    env = os.environ if environ is None else environ
    p = Path(path)
    if not p.is_file():
        return []
    added: list[str] = []
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not name or name in env:
            continue
        env[name] = value
        added.append(name)
    return added
