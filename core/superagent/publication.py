"""Publish a complete directory through one atomic POSIX symlink replacement."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from .artifacts import payload_hash
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


def publish_if_better(ctx, node):
    """Commit a verified bundle. The atomic link is the authority after a crash."""
    receipt = node.artifact_verification or {}
    if not node.ok or node.score is None or not receipt.get("passed"):
        raise ValueError("single-model publication requires independent scoring and reload")
    root = ctx.paths.root
    spec = ctx.task_spec
    previous = latest(root)
    if previous and not ctx.direction.better(node.score, previous["score"]):
        return None
    source = ctx.executor.run_dir(node.id)
    dest = version(root)
    linked = False
    try:
        shutil.copyfile(source / "solution.py", dest / "solution.py")
        for name in ("predictions.csv", "val_predictions.csv"):
            shutil.copyfile(source / "out" / name, dest / name)
        shutil.copytree(source / "out/model", dest / "model", symlinks=True)
        expected = {"solution.py": receipt["code_sha256"], **receipt["predictions_sha256"]}
        expected.update({f"model/{k}": v for k, v in receipt["package_sha256"].items()})
        actual = {
            p.relative_to(dest).as_posix(): payload_hash(p)
            for p in dest.rglob("*") if p.is_file() and not p.is_symlink()
        }
        if expected != actual or any(p.is_symlink() for p in dest.rglob("*")):
            raise ValueError("source, predictions or model changed after verification")
        info = {
            "schema_version": 1, "available": True, "kind": "single", "method": "single",
            "version_id": dest.name, "node_id": node.id, "experiment_id": node.id,
            "metric": spec.metric, "score": node.score, "score_source": "task_contract",
            "task_contract_hash": spec.signature, "split_id": spec.signature,
            "protocol": "isolated-single-model-v1",
            "validation": {"score": node.score, "rows": len(spec.truth)},
            "submission_sha256": actual["predictions.csv"],
        }
        atomic_write_json(dest / "source.json", info)
        atomic_write_json(dest / "verification.json", receipt)
        atomic_write_json(dest / "inference_inputs.json", ctx.flags["input_manifest"].payload(inference=True))
        atomic_write_json(dest / "environment.json", ctx.flags["runtime_environment"])
        manifest = {
            p.relative_to(dest).as_posix(): {"sha256": payload_hash(p), "bytes": p.stat().st_size}
            for p in dest.rglob("*") if p.is_file()
        }
        atomic_write_json(dest / "manifest.json", {"schema_version": 1, "files": manifest})
        with ctx.flags["publish_lock"]:
            if ctx.stop_event.is_set():
                return None
            previous = latest(root)
            if previous:
                if (previous["task_contract_hash"], previous["split_id"], previous["protocol"]) != (spec.signature, spec.signature, info["protocol"]):
                    raise ValueError("incumbent uses a different task, split or verification protocol")
                if not ctx.direction.better(node.score, previous["score"]):
                    return None
            from .utils import now_iso

            publish(root, dest)
            linked = True
            atomic_write_json(root / "logs/publication.json", {**info, "switched_at": now_iso()})
            if previous:
                older = root / "logs/previous_version.json"
                if older.is_file():
                    old = root / "submissions" / json.loads(older.read_text(encoding="utf-8"))["id"]
                    for name in ("model",):
                        if (old / name).is_dir():
                            shutil.rmtree(old / name)
                atomic_write_json(older, {"id": previous["version_id"]})
        return info
    finally:
        if not linked:
            shutil.rmtree(dest)
