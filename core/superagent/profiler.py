"""Data profiler: a text preview of the prepared data and a modality guess.

Only code is used here; no LLM is called.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .utils import human_bytes

SHALLOW_DEPTH_MAX_FILES = 15
DEEP_DIR_MANY_THRESHOLD = 30
DEEP_DIR_MANY_SHOW = 5
DEEP_DIR_FEW_SHOW = 8
MAX_SUBDIRS_SHOWN = 30
FULL_TEXT_MAX_LINES = 30
HEAD_LINES = 15
HEAD_LINE_MAX_CHARS = 300
BRIEF_MAX_COLUMNS = 15
FEW_DISTINCT_VALUES = 10
TOP_STRING_VALUES = 4
DETECT_ROWS = 2000
STATS_MAX_FILE_BYTES = 200 * 1024 * 1024
STATS_SAMPLE_ROWS = 100_000
JSON_SAMPLE_BYTES = 2 * 1024 * 1024
JSONL_SAMPLE_LINES = 50
MIN_TXT_FILES_FOR_TEXT = 20
ID_SHORT_STRING_MAX_CHARS = 40
PATH_COLUMN_MAJORITY = 0.5
MAX_FILE_NAMES_INDEXED = 500_000

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"})
TABLE_EXTS = frozenset({".csv", ".tsv"})
TFRECORD_EXTS = frozenset({".tfrec", ".tfrecord", ".tfrecords"})
CODE_EXTS = frozenset({".py", ".ipynb", ".js", ".ts", ".sh", ".r", ".java", ".c", ".cpp", ".h", ".go", ".rs", ".m"})
COMMON_FILE_EXTS = IMAGE_EXTS | frozenset({
    ".txt", ".wav", ".mp3", ".flac", ".npy", ".npz", ".json", ".csv", ".pdf", ".mp4", ".avi", ".dcm", ".nii", ".gz", ".pt", ".pkl", ".html", ".xml",
})
SAMPLE_NAME = "sample_predictions.csv"
PREVIEW_HARD_CAP_FACTOR = 3
PREVIEW_FORMAT_NOTE = (
    "Note: the column names of the prediction file must follow sample_predictions.csv "
    "(when present); it is the final authority on the output format."
)


@dataclass
class ProfileResult:
    preview: str
    modalities: list[str]


# --------------------------------------------------------------------------- helpers

def _is_text_file(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(8192)
    except OSError:
        return False
    if b"\x00" in chunk:
        return False
    try:
        chunk.decode("utf-8")
    except UnicodeDecodeError:
        # A multi-byte char may be cut at the chunk end.
        try:
            chunk[:-4].decode("utf-8")
        except UnicodeDecodeError:
            return False
    return True


def _count_lines(path: Path) -> int:
    n = 0
    last = b""
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(1 << 20)
            if not buf:
                break
            n += buf.count(b"\n")
            last = buf[-1:]
    if last and last != b"\n":
        n += 1
    return n


def _sorted_entries(d: Path) -> tuple[list[Path], list[Path]]:
    try:
        entries = sorted(d.iterdir(), key=lambda p: p.name)
    except OSError:
        return [], []
    dirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if not e.is_dir()]
    return dirs, files


# --------------------------------------------------------------------------- tree

def build_tree(root: Path) -> tuple[str, list[Path]]:
    """Return the directory tree text and the list of files shown in it."""
    lines: list[str] = []
    shown: list[Path] = []

    def size_label(p: Path) -> str:
        try:
            if _is_text_file(p):
                return f"{_count_lines(p)} lines"
            return human_bytes(p.stat().st_size)
        except OSError:
            return "unreadable"

    def walk(d: Path, depth: int) -> None:
        dirs, files = _sorted_entries(d)
        indent = "  " * depth
        if depth <= 1:
            limit = SHALLOW_DEPTH_MAX_FILES
        else:
            limit = DEEP_DIR_MANY_SHOW if len(files) > DEEP_DIR_MANY_THRESHOLD else DEEP_DIR_FEW_SHOW
        for f in files[:limit]:
            lines.append(f"{indent}{f.name} ({size_label(f)})")
            shown.append(f)
        if len(files) > limit:
            lines.append(f"{indent}... and {len(files) - limit} more files")
        for sub in dirs[:MAX_SUBDIRS_SHOWN]:
            lines.append(f"{indent}{sub.name}/")
            walk(sub, depth + 1)
        if len(dirs) > MAX_SUBDIRS_SHOWN:
            lines.append(f"{indent}... and {len(dirs) - MAX_SUBDIRS_SHOWN} more directories")

    lines.append(f"{root.name}/  (available to scripts as ./data)")
    walk(root, 1)
    return "\n".join(lines), shown


# --------------------------------------------------------------------------- previews

def _read_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep, nrows=nrows, low_memory=False)


def _fmt_num(x: Any) -> str:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if xf.is_integer() and abs(xf) < 1e15:
        return str(int(xf))
    return f"{xf:.6g}"


def _column_info(s: pd.Series) -> str:
    dtype = str(s.dtype)
    nunique = int(s.nunique(dropna=True))
    missing = int(s.isna().sum())
    if pd.api.types.is_bool_dtype(s):
        ratio = float(s.mean()) if len(s) else 0.0
        return f"{dtype}, true ratio {ratio:.3f}"
    if nunique < FEW_DISTINCT_VALUES:
        vals = [_fmt_num(v) if pd.api.types.is_numeric_dtype(s) else repr(str(v)) for v in sorted(s.dropna().unique(), key=str)]
        extra = f", {missing} missing" if missing else ""
        return f"{dtype}, {nunique} distinct values: {', '.join(vals)}{extra}"
    if pd.api.types.is_numeric_dtype(s):
        return f"{dtype}, min {_fmt_num(s.min())}, max {_fmt_num(s.max())}, {missing} missing"
    top = s.astype(str).value_counts().head(TOP_STRING_VALUES)
    tops = ", ".join(f"{repr(v[:60])} ({c})" for v, c in top.items())
    extra = f", {missing} missing" if missing else ""
    return f"{dtype}, {nunique} distinct values; most common: {tops}{extra}"


def preview_csv(path: Path, detailed: bool) -> str:
    try:
        big = path.stat().st_size > STATS_MAX_FILE_BYTES
        df = _read_table(path, nrows=STATS_SAMPLE_ROWS if big else None)
        n_rows = (_count_lines(path) - 1) if big else len(df)
    except Exception as exc:  # noqa: BLE001 - any parser failure is reported in the preview
        return f"(could not be parsed as a table: {type(exc).__name__}: {str(exc)[:200]})"
    cols = list(df.columns)
    head = f"{n_rows} rows, {len(cols)} columns"
    if big:
        head += f" (column statistics from the first {STATS_SAMPLE_ROWS} rows)"
    if not detailed:
        shown = ", ".join(map(str, cols[:BRIEF_MAX_COLUMNS]))
        more = f", ... ({len(cols) - BRIEF_MAX_COLUMNS} more)" if len(cols) > BRIEF_MAX_COLUMNS else ""
        return f"{head}. Columns: {shown}{more}"
    lines = [head + ":"]
    for c in cols:
        lines.append(f"  - {c}: {_column_info(df[c])}")
    return "\n".join(lines)


def _json_structure(obj: Any, depth: int = 0) -> str:
    if depth > 3:
        return "..."
    if isinstance(obj, dict):
        items = list(obj.items())[:20]
        inner = ", ".join(f"{json.dumps(k)}: {_json_structure(v, depth + 1)}" for k, v in items)
        more = ", ..." if len(obj) > 20 else ""
        return "{" + inner + more + "}"
    if isinstance(obj, list):
        if not obj:
            return "[]"
        return f"[{_json_structure(obj[0], depth + 1)}, ...] (length {len(obj)})"
    if isinstance(obj, bool):
        return "bool"
    if isinstance(obj, int):
        return "int"
    if isinstance(obj, float):
        return "float"
    if isinstance(obj, str):
        return "str"
    return "null"


def preview_json(path: Path) -> str:
    try:
        if path.suffix.lower() == ".jsonl":
            records = []
            with open(path, encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh):
                    if i >= JSONL_SAMPLE_LINES:
                        break
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            if not records:
                return "(empty)"
            return f"JSON lines; each record looks like: {_json_structure(records[0])}"
        if path.stat().st_size > JSON_SAMPLE_BYTES:
            return "(large JSON file; structure not inferred)"
        with open(path, encoding="utf-8", errors="replace") as fh:
            obj = json.load(fh)
        return f"JSON structure: {_json_structure(obj)}"
    except Exception as exc:  # noqa: BLE001
        return f"(could not be parsed as JSON: {type(exc).__name__})"


def _looks_comma_table(lines: list[str]) -> bool:
    sample = [ln for ln in lines[:10] if ln.strip()]
    if len(sample) < 2:
        return False
    counts = [ln.count(",") for ln in sample]
    return counts[0] >= 1 and all(c == counts[0] for c in counts)


def preview_text(path: Path) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None
    if path.suffix.lower() == ".txt" and _looks_comma_table(lines):
        head = "\n".join(ln[:HEAD_LINE_MAX_CHARS] for ln in lines[:HEAD_LINES])
        return f"(comma-separated table stored as .txt; parse it with a CSV reader)\n{head}"
    if len(lines) <= FULL_TEXT_MAX_LINES:
        return "\n".join(lines)
    if path.suffix.lower() in CODE_EXTS:
        return None
    head = "\n".join(ln[:HEAD_LINE_MAX_CHARS] for ln in lines[:HEAD_LINES])
    return f"(first {HEAD_LINES} of {len(lines)} lines)\n{head}"


def preview_tfrecord(path: Path) -> str:
    from google.protobuf.message import DecodeError

    from .contract import image_features, records

    try:
        features = next(records(path), {})
    except (OSError, ValueError, DecodeError) as exc:
        return f"TFRecord preview unavailable: {exc}"
    images = set(image_features(features))
    fields = []
    kinds = {"bytes_list": "bytes", "float_list": "float32", "int64_list": "int64"}
    for name in sorted(features)[:BRIEF_MAX_COLUMNS]:
        feature = features[name]
        kind = kinds.get(feature.WhichOneof("kind"), "unset")
        if name in images:
            encoding = "JPEG" if feature.bytes_list.value[0].startswith(b"\xff\xd8\xff") else "PNG"
            kind += f" ({encoding})"
        fields.append(f"{name}: {kind}")
    if len(features) > BRIEF_MAX_COLUMNS:
        fields.append(f"({len(features) - BRIEF_MAX_COLUMNS} more features)")
    return "TFRecord first Example features: " + (", ".join(fields) or "(none)")


def file_preview(path: Path, detailed: bool) -> str | None:
    ext = path.suffix.lower()
    if ext in TABLE_EXTS:
        text = preview_csv(path, detailed)
        if path.name == SAMPLE_NAME:
            try:
                cols = list(_read_table(path, nrows=5).columns)
            except Exception:  # noqa: BLE001
                cols = []
            note = (
                "THIS IS THE REQUIRED FORMAT OF THE PREDICTION FILE. Follow it exactly. "
                f"Exact column names, in order: {cols}. "
                "If the task description disagrees with this file about the format, this file wins."
            )
            text = note + "\n" + text
        return text
    if ext in {".json", ".jsonl"}:
        return preview_json(path)
    if ext in TFRECORD_EXTS:
        return preview_tfrecord(path)
    if ext in IMAGE_EXTS:
        return None
    if _is_text_file(path):
        return preview_text(path)
    return None


def build_preview(root: Path, detailed: bool) -> str:
    tree, shown = build_tree(root)
    parts = ["## Directory tree", tree, "", "## File previews"]
    record_dirs = set()
    for f in shown:
        if f.suffix.lower() in TFRECORD_EXTS:
            if f.parent in record_dirs:
                continue
            record_dirs.add(f.parent)
        prev = file_preview(f, detailed)
        if prev is None:
            continue
        rel = f.relative_to(root)
        parts.append(f"### ./data/{rel}")
        parts.append(prev)
        parts.append("")
    parts.append(PREVIEW_FORMAT_NOTE)
    return "\n".join(parts)


# --------------------------------------------------------------------------- modality detection

def _iter_files(root: Path) -> Iterable[Path]:
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        for f in filenames:
            yield Path(dirpath) / f


def _is_path_like_value(v: str, file_names: set[str]) -> bool:
    v = v.strip()
    if not v:
        return False
    low = v.lower()
    if any(low.endswith(ext) for ext in COMMON_FILE_EXTS):
        return True
    base = os.path.basename(v)
    return base in file_names


def _classify_columns(df: pd.DataFrame, file_names: set[str], cfg: Mapping[str, Any]) -> dict[str, str]:
    """Map each column to one of: id, text, path, other."""
    kinds: dict[str, str] = {}
    min_mean = float(cfg.get("text_min_mean_chars", 30))
    min_unique = float(cfg.get("text_min_unique_ratio", 0.5))
    n = len(df)
    for c in df.columns:
        s = df[c].dropna()
        if s.empty:
            kinds[c] = "other"
            continue
        is_str = (pd.api.types.is_string_dtype(s) or s.dtype == object) and not pd.api.types.is_bool_dtype(s)
        if is_str:
            sv = s.astype(str)
            path_ratio = sum(_is_path_like_value(v, file_names) for v in sv.head(500)) / max(1, min(len(sv), 500))
            if path_ratio > PATH_COLUMN_MAJORITY:
                kinds[c] = "path"
                continue
            mean_len = float(sv.str.len().mean())
            uniq = sv.nunique() / max(1, len(sv))
            if mean_len >= min_mean and uniq >= min_unique:
                kinds[c] = "text"
                continue
            if sv.nunique() == n and mean_len <= ID_SHORT_STRING_MAX_CHARS:
                kinds[c] = "id"
                continue
        elif pd.api.types.is_integer_dtype(s) and s.nunique() == n and n > 1:
            kinds[c] = "id"
            continue
        kinds[c] = "other"
    return kinds


def detect_modalities(root: Path, cfg: Mapping[str, Any]) -> list[str]:
    min_images = int(cfg.get("min_image_files", 20))
    n_images = 0
    n_txt = 0
    tables: list[Path] = []
    file_names: set[str] = set()
    for p in _iter_files(root):
        ext = p.suffix.lower()
        if len(file_names) < MAX_FILE_NAMES_INDEXED:
            file_names.add(p.name)
        if ext in TFRECORD_EXTS and n_images < min_images:
            from .contract import records, image_features
            if image_features(next(records(p), {})):
                n_images = min_images
        if ext in IMAGE_EXTS:
            n_images += 1
        elif ext == ".txt":
            n_txt += 1
        elif ext in TABLE_EXTS and p.name != SAMPLE_NAME:
            tables.append(p)
    found: set[str] = set()
    if n_images >= min_images:
        found.add("image")
    if n_txt >= MIN_TXT_FILES_FOR_TEXT:
        found.add("text")
    for t in tables[:50]:
        try:
            df = _read_table(t, nrows=DETECT_ROWS)
        except Exception:  # noqa: BLE001
            continue
        kinds = _classify_columns(df, file_names, cfg)
        if any(k == "text" for k in kinds.values()):
            found.add("text")
        rest = [c for c, k in kinds.items() if k == "other"]
        if len(rest) >= 2:
            found.add("tabular")
    order = ["tabular", "text", "image"]
    return [m for m in order if m in found]


def profile(root: str | os.PathLike[str], cfg: Mapping[str, Any]) -> ProfileResult:
    root = Path(root)
    max_chars = int(cfg.get("max_preview_chars", 6000))
    text = build_preview(root, detailed=True)
    if len(text) > max_chars:
        text = build_preview(root, detailed=False)
    hard_cap = max_chars * PREVIEW_HARD_CAP_FACTOR
    if len(text) > hard_cap:
        # Even the brief form can grow with very many files; never let it flood the prompts.
        text = text[:hard_cap] + "\n... [data preview truncated]\n" + PREVIEW_FORMAT_NOTE
    return ProfileResult(preview=text, modalities=detect_modalities(root, cfg))
