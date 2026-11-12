"""Input checks for the task and output directories, and the prepared data view."""

from __future__ import annotations

import os
import shutil
import stat
import zipfile
from pathlib import Path

from .errors import InvalidInputError

JUNK_DIR_NAMES = frozenset({"__MACOSX"})
JUNK_FILE_NAMES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})
JUNK_FILE_PREFIXES = ("._",)


def check_output_dir(output_dir: str | os.PathLike[str]) -> Path:
    """Create the output directory. Refuse (without writing anything) if it holds files."""
    out = Path(output_dir).expanduser().resolve()
    if out.exists():
        if not out.is_dir():
            raise InvalidInputError(f"output dir exists and is not a directory: {out}", write_status=False)
        if any(out.iterdir()):
            raise InvalidInputError(
                f"output dir is not empty: {out} (refusing to overwrite earlier results)",
                write_status=False,
            )
    out.mkdir(parents=True, exist_ok=True)
    return out


def check_task_dir(task_dir: str | os.PathLike[str]) -> Path:
    task = Path(task_dir).expanduser().resolve()
    if not task.is_dir():
        raise InvalidInputError(f"task dir does not exist: {task}")
    desc = task / "description.md"
    if not desc.is_file():
        raise InvalidInputError(f"description.md is missing in task dir: {task}")
    if not desc.read_text(encoding="utf-8", errors="replace").strip():
        raise InvalidInputError("description.md is empty")
    data = task / "data"
    if not data.is_dir():
        raise InvalidInputError(f"data/ directory is missing in task dir: {task}")
    if not any(data.iterdir()):
        raise InvalidInputError("data/ directory is empty")
    return task


def _is_junk(name: str) -> bool:
    return name in JUNK_FILE_NAMES or name in JUNK_DIR_NAMES or name.startswith(JUNK_FILE_PREFIXES)


def _remove_junk(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        for d in list(dirnames):
            if d in JUNK_DIR_NAMES:
                shutil.rmtree(Path(dirpath) / d, ignore_errors=True)
                dirnames.remove(d)
        for f in filenames:
            if _is_junk(f):
                try:
                    (Path(dirpath) / f).unlink()
                except OSError:
                    pass


def _make_read_only(root: Path) -> None:
    """Best effort: drop write permission from extracted files and directories."""
    mask = ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        for f in filenames:
            p = Path(dirpath) / f
            try:
                if not p.is_symlink():
                    p.chmod(p.stat().st_mode & mask)
            except OSError:
                pass
        try:
            Path(dirpath).chmod(Path(dirpath).stat().st_mode & mask)
        except OSError:
            pass


def _extract_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            # Refuse path traversal.
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(dest.resolve()):
                continue
            zf.extract(member, dest)
    _remove_junk(dest)
    entries = [e for e in dest.iterdir() if not _is_junk(e.name)]
    if len(entries) == 1 and entries[0].is_dir() and entries[0].name == dest.name:
        inner = entries[0]
        tmp = dest.parent / f".{dest.name}.lift"
        inner.rename(tmp)
        for e in dest.iterdir():  # only junk could remain
            if e.is_dir():
                shutil.rmtree(e, ignore_errors=True)
            else:
                e.unlink()
        dest.rmdir()
        tmp.rename(dest)


def prepare_data(data_dir: str | os.PathLike[str], prepared_dir: str | os.PathLike[str]) -> Path:
    """Build a read-only view of ``data_dir``: symlinks to entries, extracted archives."""
    src = Path(data_dir).resolve()
    dst = Path(prepared_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for entry in sorted(src.iterdir()):
        if _is_junk(entry.name):
            continue
        if entry.is_file() and entry.suffix.lower() == ".zip":
            target = dst / entry.stem
            if target.exists():
                continue
            try:
                _extract_zip(entry, target)
            except zipfile.BadZipFile:
                os.symlink(entry, dst / entry.name)
                continue
            _make_read_only(target)
        else:
            link = dst / entry.name
            if not link.exists() and not link.is_symlink():
                os.symlink(entry, link)
    return dst
