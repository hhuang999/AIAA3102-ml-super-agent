"""Publish a complete directory through one atomic POSIX symlink replacement."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from .utils import atomic_write_json, new_id


def version(root):
    versions = Path(root) / "submissions"
    versions.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="version-", dir=versions))


def publish(root, directory):
    if (
        not (directory / "predictions.csv").is_file()
        or not (directory / "source.json").is_file()
    ):
        raise ValueError("incomplete submission version")
    link = Path(root) / (".publish-" + new_id())
    try:
        link.symlink_to(directory.relative_to(root), target_is_directory=True)
        os.replace(link, Path(root) / "final")
    finally:
        link.unlink(missing_ok=True)


def incumbent(ctx, node):
    dest = version(ctx.paths.root)
    shutil.copyfile(
        ctx.executor.run_dir(node.id) / "out/predictions.csv", dest / "predictions.csv"
    )
    spec = getattr(ctx, "task_spec", None)
    atomic_write_json(
        dest / "source.json",
        {
            "available": True,
            "kind": "single",
            "method": "single",
            "node_id": node.id,
            "score": node.score,
            "score_source": node.score_source,
            "metric": spec.metric if spec else None,
        },
    )
    publish(ctx.paths.root, dest)


def latest(root):
    path = Path(root) / "final"
    if (path / "predictions.csv").is_file() and (path / "source.json").is_file():
        return json.loads((path / "source.json").read_text(encoding="utf-8"))
    return None
