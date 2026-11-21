"""Bookkeeping: best solution, top list, model-file retention, status and tree files."""

from __future__ import annotations

import logging
import os
import shutil
import statistics
from pathlib import Path
from typing import Any

from .context import RunContext
from .experience import Record, effect_label
from .textfmt import trim_with
from .tree import Node
from .utils import atomic_write_json, atomic_write_text, now_iso

log = logging.getLogger("superagent")

INTERFACE_VERSION = "1.1"
VAL_FILES = ("val_predictions.csv", "val_labels.csv")


# --------------------------------------------------------------------------- experience

def save_experience(ctx: RunContext, node: Node) -> None:
    if ctx.experience is None:
        return
    diagnostic=not node.ok and bool(set(node.failure_reasons) & {"task_contract","timeout","runtime_error","data_leak","policy_violation"})
    if not node.ok and not diagnostic:
        return
    parent = node.parent
    parent_score = parent.score if parent is not None and parent.ok else None
    error_text = None
    if node.kind == "repair" and parent is not None:
        error_text = trim_with(ctx.cfg, parent.output)
    ref_score = None
    if node.kind in ("draft", "synthesis") and node.ok:
        with ctx.tree.lock:
            others = [n.score for n in ctx.tree.nodes if n is not node and n.ok and n.score is not None]
        if len(others) >= 2:
            ref_score = statistics.median(others)
    rec = Record(
        record_id=node.id,
        kind="diagnostic" if diagnostic else node.kind,
        plan=node.plan,
        method_summary=node.method_summary or node.plan[:500],
        effect=-1 if diagnostic else effect_label(
            node.kind, node.score, parent_score, ctx.direction.better, reference_score=ref_score,  # type: ignore[arg-type]
        ),
        score=node.score,
        parent_score=parent_score,
        exec_seconds=node.exec_seconds,
        error_text=(node.summary or node.error_message) if diagnostic else error_text,
        created_at=now_iso(),
    )
    ctx.experience.add(rec)


# --------------------------------------------------------------------------- serialisation

def node_json(ctx: RunContext, n: Node) -> dict[str, Any]:
    return {
        "id": n.id,
        "step": n.step,
        "kind": n.kind,
        "parent_id": n.parent.id if n.parent is not None else None,
        "branch_id": n.branch_id,
        "plan": n.plan,
        "structured_plan": n.structured_plan,
        "code": n.code,
        "output": trim_with(ctx.cfg, n.output) if n.output else "",
        "exec_seconds": n.exec_seconds,
        "error_type": n.error_type,
        "error_message": n.error_message,
        "summary": n.summary,
        "method_summary": n.method_summary,
        "score": n.score,
        "parent_gain":n.parent_gain,"global_gain":n.global_gain,"complementarity":n.complementarity,
        "trial_id":n.trial_id, "rung":n.rung, "checkpoint":n.checkpoint,
        "artifact_verification": n.artifact_verification,
        "continuation_verification": n.continuation_verification,
        "score_source": n.score_source,
        "failed": n.failed,
        "predictions_ok": n.predictions_ok,
        "val_files_ok": n.val_files_ok,
        "smoke": n.smoke,
        "failure_reasons": list(n.failure_reasons),
        "visits": n.visits,
        "reward_sum": n.reward_sum,
        "dead_end": n.dead_end,
        "chain_best_id": n.chain_best.id if n.chain_best is not None else None,
        "via_elite": n.via_elite,
        "created_at": n.created_at,
        "finished_at": n.finished_at,
    }


def write_search_tree(ctx: RunContext) -> None:
    with ctx.tree.lock:
        data = {
            "interface_version": INTERFACE_VERSION,
            "higher_is_better": ctx.direction.higher_is_better,
            "nodes": [node_json(ctx, n) for n in ctx.tree.nodes],
        }
    atomic_write_json(ctx.paths.search_tree, data)


def score_json(ctx: RunContext, n: Node) -> dict[str, Any]:
    return {
        "node_id": n.id,
        "score": n.score,
        "higher_is_better": ctx.direction.higher_is_better,
        "kind": n.kind,
        "branch_id": n.branch_id,
        "via_elite": n.via_elite,
        "exec_seconds": n.exec_seconds,
        "created_at": n.created_at,
        "has_artifacts": n.has_artifacts,
    }


def status_payload(ctx: RunContext, status: str, error: str | None = None, ensemble_available: bool = False,
                   final: dict[str, Any] | None = None) -> dict[str, Any]:
    from .publication import latest

    if final is None:
        final = latest(ctx.paths.root)
    tree = ctx.tree
    formal = ctx.flags.get("formal")
    identity = {
        "run_id": ctx.flags["task_facts"]["run_id"],
        "claimed_operations": ctx.flags.get("claimed_steps", 0),
        "final_version_id": final.get("version_id") if final else None,
    } if formal else {}
    with tree.lock:
        sol = [n for n in tree.nodes if not n.is_root]
        best = tree.best
        branches = {n.branch_id for n in sol if n.branch_id is not None}
        return {
            "interface_version": INTERFACE_VERSION,
            **identity,
            "status": status,
            "stop_reason": ctx.stop_reason if status != "running" else None,
            "started_at": ctx.started_at,
            "updated_at": now_iso(),
            "elapsed_secs": round(ctx.elapsed(), 1),
            "higher_is_better": ctx.direction.higher_is_better,
            "direction_source": ctx.direction_source,
            "num_nodes": len(sol),
            "num_ok_nodes": sum(1 for n in sol if n.ok),
            "num_branches": len(branches),
            "best_node_id": best.id if best else None,
            "best_score": best.score if best else None,
            "best_has_artifacts": bool(best.has_artifacts) if best else False,
            "ensemble_available": ensemble_available,
            "final_available": bool(final and final.get("available")),
            "final_kind": final.get("kind") if final else None,
            "final_method": final.get("method") if final else None,
            "final_metric": final.get("metric") if final else None,
            "final_validation_score": (final.get("validation") or {}).get("score", final.get("score")) if final else None,
            "error": error,
        }


def write_status(ctx: RunContext, status: str = "running", error: str | None = None, ensemble_available: bool = False,
                 final: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = status_payload(ctx, status, error, ensemble_available, final)
    atomic_write_json(ctx.paths.status, payload)
    return payload


# --------------------------------------------------------------------------- best / top

def _model_dir(ctx: RunContext, n: Node) -> Path:
    return ctx.executor.run_dir(n.id) / "out" / "model"


def _pred_file(ctx: RunContext, n: Node) -> Path:
    return ctx.executor.run_dir(n.id) / "out" / "predictions.csv"


def compute_top(ctx: RunContext) -> list[Node]:
    s = ctx.cfg["search"]
    per_branch = int(s["top_per_branch"])
    size = int(s["top_list_size"])
    tree = ctx.tree
    with tree.lock:
        counts: dict[int, int] = {}
        out: list[Node] = []
        for n in tree.sorted_ok():
            b = n.branch_id or 0
            if counts.get(b, 0) >= per_branch:
                continue
            counts[b] = counts.get(b, 0) + 1
            out.append(n)
        return out[:size]


def _write_solution_dir(ctx: RunContext, d: Path, n: Node) -> None:
    d.mkdir(parents=True, exist_ok=True)
    atomic_write_text(d / "solution.py", n.code)
    pf = _pred_file(ctx, n)
    if pf.exists():
        shutil.copyfile(pf, d / "predictions.csv")
    for name in VAL_FILES:
        vf = pf.parent / name
        if vf.exists():
            shutil.copyfile(vf, d / name)
    atomic_write_json(d / "score.json", score_json(ctx, n))


def write_best(ctx: RunContext, n: Node) -> None:
    d = ctx.paths.best
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    _write_solution_dir(ctx, d, n)
    art = d / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    md = _model_dir(ctx, n)
    if md.is_dir():
        shutil.copytree(md, art, dirs_exist_ok=True, symlinks=False)


def write_top(ctx: RunContext, top: list[Node]) -> None:
    root = ctx.paths.top
    root.mkdir(parents=True, exist_ok=True)
    k = retention_size(ctx.cfg)
    for i, n in enumerate(top, 1):
        d = root / f"rank_{i:02d}"
        if d.is_symlink() or d.is_file():
            d.unlink()
        elif d.exists():
            shutil.rmtree(d, ignore_errors=True)
        _write_solution_dir(ctx, d, n)
        md = _model_dir(ctx, n)
        if i <= k and n.has_artifacts and md.is_dir():
            os.symlink(os.path.relpath(md, d), d / "artifacts")
    for extra in sorted(root.glob("rank_*")):
        try:
            idx = int(extra.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if idx > len(top):
            shutil.rmtree(extra, ignore_errors=True)


def retention_size(cfg: dict[str, Any]) -> int:
    """Number of top solutions that keep their model files (ensemble members of final/ need theirs)."""
    k = int(cfg["execution"]["keep_artifacts_top_k"])
    if cfg["final"]["enabled"] and cfg["final"]["allow_ensemble"]:
        k = max(k, int(cfg["ensemble"]["max_candidates"]))
    return k


def apply_retention(ctx: RunContext, top: list[Node]) -> None:
    k = retention_size(ctx.cfg)
    keep = {n.id for n in top[:k]}
    if ctx.tree.best is not None:
        keep.add(ctx.tree.best.id)
    with ctx.tree.lock:
        evaluated = [n for n in ctx.tree.nodes if not n.is_root and n.evaluated]
    for n in evaluated:
        if n.id in keep or n.id in ctx.flags.get("artifact_leases", ()) or not n.has_artifacts:
            continue
        md = _model_dir(ctx, n)
        if md.exists():
            shutil.rmtree(md, ignore_errors=True)
        n.has_artifacts = False


def after_assessment_files(ctx: RunContext, n: Node) -> None:
    """Run-directory housekeeping right after a node is assessed."""
    out = ctx.executor.run_dir(n.id) / "out"
    if n.is_failed and n.id != ctx.flags.get("repairable_id"):
        shutil.rmtree(out, ignore_errors=True)
        n.has_artifacts = False
        return
    md = out / "model"
    n.has_artifacts = md.is_dir() and any(md.rglob("*"))


def update_rankings(ctx: RunContext, n: Node | None) -> tuple[bool, bool]:
    """Update the global best and the top list (after ``n`` was added) and rewrite best/ and top/.

    Everything runs under ``io_lock`` and the written state is re-read from the tree, so concurrent
    workers can never write an older best or top list over a newer one.
    """
    tree = ctx.tree
    with ctx.io_lock:
        if n is not None and ctx.flags.get("finalized"):
            return False, False
        best_changed = False
        with tree.lock:
            if n is not None and n.in_tree and n.ok and n.score is not None and (tree.best is None or ctx.direction.better(n.score, tree.best.score)):
                tree.best = n
                best_changed = True
                tree.score_timeline.append({
                    "step": n.step,
                    "elapsed_secs": round(ctx.elapsed(), 1),
                    "node_id": n.id,
                    "score": n.score,
                    "kind": n.kind,
                    "branch_id": n.branch_id,
                    "plan": n.plan[:300],
                })
            new_top = compute_top(ctx)
            tree.top_list = new_top
            best = tree.best
        if best_changed and n is not None:
            ctx.add_event("new_global_best", n.id, f"score {n.score}", step=n.step)
        written_best = ctx.flags.get("written_best_id")
        if best is not None and written_best != best.id:
            from .publication import incumbent
            import time
            started=time.monotonic()
            if not ctx.flags.get("formal"):
                incumbent(ctx, best)
            if ctx.control is not None and ctx.control.name=="adaptive":
                with ctx.state_lock:
                    ctx.control.publish_seconds.append(time.monotonic()-started)
            write_best(ctx, best)
            ctx.flags["written_best_id"] = best.id  # type: ignore[assignment]
        top_ids = [x.id for x in new_top]
        top_changed = top_ids != ctx.flags.get("written_top_ids")
        # Retention runs every time: a node that did not enter the top list must lose its model files too.
        apply_retention(ctx, new_top)
        if top_changed:
            write_top(ctx, new_top)
            ctx.flags["written_top_ids"] = top_ids  # type: ignore[assignment]
        return best_changed, top_changed
