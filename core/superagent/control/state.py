"""Read-only snapshot of a running search, handed to controllers."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:  # pragma: no cover
    from ..context import RunContext


@dataclass(frozen=True)
class StepTiming:
    """Timing of one finished step (generation, review, execution and assessment)."""

    step: int | None
    node_id: str
    kind: str
    ok: bool
    step_seconds: float
    exec_seconds: float | None

    @property
    def other_seconds(self) -> float:
        """Time spent outside the script itself (mainly LLM calls)."""
        return max(0.0, self.step_seconds - (self.exec_seconds or 0.0))


@dataclass(frozen=True)
class GpuInfo:
    """GPUs detected on the machine. ``memory_mb`` has one entry per GPU."""

    count: int
    memory_mb: tuple[float, ...] = ()
    names: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunState:
    elapsed_secs: float
    time_limit_secs: float
    time_left_secs: float
    reserve_secs: float
    progress: float | None
    parallel: int
    active_tasks: int
    max_steps: int
    nodes_total: int
    nodes_ok: int
    nodes_failed: int
    nodes_by_kind: Mapping[str, int]
    step_timings: tuple[StepTiming, ...]
    cost: float
    currency: str
    prompt_tokens: int
    completion_tokens: int
    best_score: float | None
    best_step: int | None
    higher_is_better: bool
    configured_gpus: int
    gpu: GpuInfo | None = None
    taken_at: float = field(default_factory=time.monotonic)

    @property
    def usable_secs(self) -> float:
        """Time left before the finalize reserve starts."""
        return max(0.0, self.time_left_secs - self.reserve_secs)

    @property
    def steps_since_best(self) -> int | None:
        if self.best_step is None:
            return None
        return max(0, self.nodes_total - self.best_step)

    def recent(self, k: int) -> tuple[StepTiming, ...]:
        return self.step_timings[-k:] if k > 0 else ()

    def median_step_seconds(self, k: int) -> float | None:
        vals = [t.step_seconds for t in self.recent(k)]
        return statistics.median(vals) if vals else None

    def median_other_seconds(self, k: int) -> float | None:
        vals = [t.other_seconds for t in self.recent(k) if t.exec_seconds is not None]
        return statistics.median(vals) if vals else None

    def median_ok_script_seconds(self) -> float | None:
        vals = [t.exec_seconds for t in self.step_timings if t.ok and t.exec_seconds is not None]
        return statistics.median(vals) if vals else None


def build_state(ctx: "RunContext") -> RunState:
    """Snapshot of ``ctx``. Takes the tree lock briefly."""
    tree = ctx.tree
    with tree.lock:
        sol = [n for n in tree.nodes if not n.is_root]
        by_kind: dict[str, int] = {}
        for n in sol:
            by_kind[n.kind] = by_kind.get(n.kind, 0) + 1
        best = tree.best
        best_score = best.score if best is not None else None
        best_step = best.step if best is not None else None
        n_ok = sum(1 for n in sol if n.ok)
        n_failed = sum(1 for n in sol if n.is_failed)
    with ctx.state_lock:
        timings = tuple(ctx.step_timings)
    gw = ctx.gateway
    gw_lock = getattr(gw, "_lock", None)
    if gw_lock is not None:
        with gw_lock:
            totals = dict(getattr(gw, "totals", None) or {})
    else:
        totals = getattr(gw, "totals", None) or {}
    cost = float(totals.get("cost", 0.0) or 0.0)
    exp = ctx.experience
    embedder = getattr(exp, "embedder", None) if exp is not None else None
    if embedder is not None:
        cost += float(getattr(embedder, "totals", {}).get("cost", 0.0) or 0.0)
    return RunState(
        elapsed_secs=ctx.elapsed(),
        time_limit_secs=ctx.time_limit,
        time_left_secs=ctx.time_left(),
        reserve_secs=ctx.reserve,
        progress=ctx.progress(),
        parallel=int(ctx.cfg["resources"]["parallel"]),
        active_tasks=int(ctx.active_tasks),
        max_steps=int(ctx.cfg["run"]["max_steps"]),
        nodes_total=len(sol),
        nodes_ok=n_ok,
        nodes_failed=n_failed,
        nodes_by_kind=by_kind,
        step_timings=timings,
        cost=cost,
        currency=str(ctx.cfg["llm"].get("currency", "")),
        prompt_tokens=int(totals.get("prompt_tokens", 0) or 0),
        completion_tokens=int(totals.get("completion_tokens", 0) or 0),
        best_score=best_score,
        best_step=best_step,
        higher_is_better=ctx.direction.higher_is_better,
        configured_gpus=int(ctx.cfg["resources"]["num_gpus"]),
        gpu=ctx.gpu_info,
    )
