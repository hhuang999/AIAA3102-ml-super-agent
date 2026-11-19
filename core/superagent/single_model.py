"""The supplied-baseline and generated-candidate delivery path."""

from __future__ import annotations

import ast
import logging
import shutil
import stat
from itertools import chain
from pathlib import Path

from .artifacts import package_files, same_predictions
from .assessment import assess
from .bookkeeping import after_assessment_files, update_rankings, write_search_tree, write_status
from .publication import publish_if_better
from .utils import atomic_write_json

log = logging.getLogger("superagent")


def inference_only_change(before, after):
    """Allow changes solely inside the body of one top-level predict_only."""
    def masked(source):
        tree = ast.parse(source)
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "predict_only"]
        if len(funcs) != 1:
            raise ValueError("missing unique predict_only")
        body = funcs[0].body
        # Replacing a line shared with a signature or another statement is ambiguous.
        if body[0].lineno <= funcs[0].lineno:
            raise ValueError("predict_only must have a separate body")
        lines = source.splitlines(keepends=True)
        funcs[0].body = [ast.Pass()]
        return (ast.dump(tree), lines[:body[0].lineno - 1], lines[body[-1].end_lineno:])

    try:
        return before != after and masked(before) == masked(after)
    except (SyntaxError, ValueError):
        return False


def _regular_outputs(output):
    # Trusted readers should only open regular files from a candidate.
    for path in chain((output,), output.rglob("*")):
        mode = path.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(f"unsupported script output: {path.name}")


def _run_repaired_predictions(ctx, node):
    from tempfile import TemporaryDirectory

    executor = ctx.executor
    parent = node.parent
    if not parent or parent.id != ctx.flags.get("repairable_id") or not inference_only_change(parent.code, node.code):
        return False
    old = executor.run_dir(parent.id) / "out"
    try:
        package, _ = package_files(old / "model")
    except (ValueError, OSError):
        return False
    if package != (parent.artifact_verification or {}).get("package_sha256"):
        return False
    slot = executor.acquire_slot(ctx.stop_event)
    if slot is None:
        return False
    try:
        run = executor.prepare_run_dir(node.id, node.code)
        with TemporaryDirectory(prefix="repair-", dir=run) as temp:
            fresh = Path(temp)
            (fresh / "out").mkdir()
            shutil.copyfile(run / "solution.py", fresh / "solution.py")
            shutil.copytree(old / "model", fresh / "model")
            result = executor._run_in_slot(fresh, slot, ctx.cfg["verification"]["timeout_seconds"], ctx.stop_event,
                argv=["solution.py", "--predict-only", "--model-dir", "model", "--data-dir", "data", "--output-dir", "out"])
            if result.error_type or result.exit_code != 0:
                return False
            try:
                _regular_outputs(fresh / "out")
                equivalent = same_predictions(old, fresh / "out", ctx.task_spec.probability)
            except (OSError, ValueError):
                return False
            if not equivalent:
                return False
            output = run / "out"
            for name in ("predictions.csv", "val_predictions.csv"):
                shutil.copyfile(fresh / "out" / name, output / name)
            shutil.copytree(fresh / "model", output / "model")
            node.exec_seconds = result.exec_seconds
            node.output = result.output
            return True
    finally:
        executor.release_slot(slot)


def evaluate(ctx, node, run_candidate):
    from .executor import ExecResult

    reuse = node.kind == "repair" and _run_repaired_predictions(ctx, node)
    if reuse:
        res = ExecResult(node.output, node.exec_seconds, None, None, False, 0,
                         ctx.cfg["verification"]["timeout_seconds"])
    else:
        res = run_candidate(node)
    output = ctx.executor.run_dir(node.id) / "out"
    if res.exit_code == 0 and not res.error_type:
        try:
            _regular_outputs(output)
        except (OSError, ValueError) as exc:
            res.error_type, res.error_message = "OutputError", str(exc)
    info = assess(ctx, node, res)
    if node.ok:
        node.artifact_verification = ctx.executor.verify_bundle(
            node.id, ctx.task_spec, ctx.cfg["verification"]["timeout_seconds"], ctx.stop_event,
        )
        node.artifact_verification["reused_parent_weights"] = reuse
    node.has_artifacts = bool((node.artifact_verification or {}).get("package_sha256"))
    info.update(
        score_verified=node.ok,
        deliverable=bool((node.artifact_verification or {}).get("passed")),
        reused_parent_weights=reuse,
    )
    return info


def finish(ctx, node, info, record=None):
    if info.get("deliverable"):
        try:
            info["publication"] = "published" if publish_if_better(ctx, node) else "retained_incumbent"
        except (OSError, ValueError) as exc:
            info["publication"] = "failed"
            info["publication_error"] = str(exc)
            log.error("could not publish %s: %s", node.id, exc)
    with ctx.io_lock:
        if info.get("score_verified") and not info.get("deliverable"):
            old_id = ctx.flags.get("repairable_id")
            incumbent = ctx.tree.by_id.get(old_id)
            if node.has_artifacts and (not incumbent or ctx.direction.better(node.score, incumbent.score)):
                ctx.flags["repairable_id"] = node.id
                ctx.flags["artifact_leases"].add(node.id)
                if old_id and old_id not in ctx.flags.get("active_parent_leases", ()):
                    ctx.flags["artifact_leases"].discard(old_id)
            else:
                node.has_artifacts = False
            node.failed = True
            node.failure_reasons.append("artifact_contract")
            node.summary = (node.artifact_verification or {}).get("error", "artifact unavailable")
        after_assessment_files(ctx, node)
        if record is not None:
            record()
        with ctx.tree.lock:
            ctx.tree.append(node)
            if node.ok:
                ctx.tree.register_ok(node)
        update_rankings(ctx, node if info.get("deliverable") else None)
        write_search_tree(ctx)
        write_status(ctx)
