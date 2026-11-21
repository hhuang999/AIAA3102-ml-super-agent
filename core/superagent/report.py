"""Run summary report (JSON and Markdown), built by code from the tree and the logs."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any

from .bookkeeping import INTERFACE_VERSION
from .context import RunContext
from .scheduling import attempts_since_branch_best
from .selection import explore_weight
from .utils import atomic_write_json, atomic_write_text, now_iso, read_jsonl

OPERATORS = ("draft", "refine", "repair", "retrospect", "borrow", "synthesis")
PLAN_PREVIEW = 300
EFFECTIVE_BRANCH_WINDOW = 10
TRIGGERED_OPERATORS = ("retrospect", "borrow", "synthesis")


def _llm_usage(ctx: RunContext) -> dict[str, Any]:
    formal = getattr(ctx, "flags", {}).get("formal")
    if formal:
        ledger = ctx.gateway.ledger
        with ledger.db() as db:
            raw = db.execute("SELECT role, status, usage FROM attempts ORDER BY started").fetchall()
        import json

        rows = []
        for role, status, payload in raw:
            usage = json.loads(payload) if payload else {}
            rows.append({"agent": role, "model": usage.get("model") or ctx.cfg["llm"]["generation"]["model"],
                         "ok": status == "received", "status": status, "prompt_tokens": usage.get("input_tokens"),
                         "completion_tokens": usage.get("output_tokens"), "reasoning_tokens": usage.get("reasoning_tokens")})
    else:
        rows = read_jsonl(ctx.paths.llm_calls)
    late = {r["attempt_id"]: r for r in read_jsonl(ctx.paths.llm_calls.with_name("late_usage.jsonl"))}

    def empty() -> dict[str, Any]:
        return dict(calls=0, failed_calls=0, prompt_tokens=0, completion_tokens=0, reasoning_tokens=0,
                    cost=0.0, unknown_usage=0)

    total = empty()
    total["currency"] = ctx.cfg["llm"].get("currency", "")
    by_agent = defaultdict(empty)
    by_model = defaultdict(empty)
    no_billing = {"not_sent", "rate_limited", "overloaded", "http_rejected"}
    for r in rows:
        usage = late.get(r.get("attempt_id"), r)
        for bucket in (total, by_agent[str(r.get("agent"))], by_model[str(r.get("model"))]):
            bucket["calls"] += 1
            if not r.get("ok"):
                bucket["failed_calls"] += 1
            bucket["prompt_tokens"] += usage.get("prompt_tokens") or 0
            bucket["completion_tokens"] += usage.get("completion_tokens") or 0
            bucket["reasoning_tokens"] += usage.get("reasoning_tokens") or 0
            bucket["cost"] += usage.get("cost") or 0.0
            if formal:
                unknown = (
                    r.get("status") not in no_billing
                    and (usage.get("prompt_tokens") is None or usage.get("completion_tokens") is None)
                )
            else:
                unknown = usage.get("cost") is None
            bucket["unknown_usage"] += int(unknown)
    for bucket in [total, *by_agent.values(), *by_model.values()]:
        bucket["cost"] = None if formal else round(bucket["cost"], 6)
    if formal:
        total["accounting"] = "adapter"
    total["by_agent"] = dict(by_agent)
    total["by_model"] = dict(by_model)
    return total


def _embedding_usage(ctx: RunContext) -> dict[str, Any]:
    rows = read_jsonl(ctx.paths.embedding_calls)
    return {
        "calls": len(rows),
        "tokens": sum(int(r.get("tokens") or 0) for r in rows),
        "cost": round(sum(float(r.get("cost") or 0.0) for r in rows), 6),
        "currency": ctx.cfg["llm"].get("currency", ""),
    }


def _final_view(final: dict[str, Any] | None) -> dict[str, Any] | None:
    if final is None:
        return None
    keys = ("available", "kind", "node_id", "score", "method", "metric", "encoding", "reason", "validation", "members", "candidates",
            "metric_identification", "decided_by_tie_break")
    return {k: final.get(k) for k in keys if k in final}


def effective_branches(branch_seq: list[int], window: int = EFFECTIVE_BRANCH_WINDOW) -> list[float]:
    """exp(entropy) of the branch distribution in a sliding window over the node sequence."""
    out = []
    for i in range(len(branch_seq)):
        win = branch_seq[max(0, i - window + 1): i + 1]
        counts = Counter(win)
        n = len(win)
        h = -sum((c / n) * math.log(c / n) for c in counts.values())
        out.append(round(math.exp(h), 3))
    return out


def _diagnostics(ctx: RunContext) -> dict[str, Any]:
    cfg = ctx.cfg
    rows = read_jsonl(ctx.paths.decisions)
    modes = Counter(str(r.get("selection_mode")) for r in rows if r.get("selection_mode"))
    reasons = Counter(str(r.get("reason")) for r in rows if r.get("reason"))
    left = [r for r in rows if (r.get("details") or {}).get("left_chain_at")]
    left_chain = sum(1 for r in left if r.get("selection_mode") != "trigger")
    left_for_synthesis = len(left) - left_chain
    switch_start = float(cfg["search"]["explore"]["switch_start"])
    late = [r for r in rows if r.get("progress") is not None and float(r["progress"]) >= switch_start]
    late_modes = Counter(str(r.get("selection_mode")) for r in late)
    tree = ctx.tree
    with tree.lock:
        sol = [n for n in tree.nodes if not n.is_root]
        kinds = Counter(n.kind for n in sol)
        branch_seq = [n.branch_id or 0 for n in sol]
        since = {b: attempts_since_branch_best(tree, b) for b in tree.branches()}
        smoke = Counter(n.smoke or "none" for n in sol if n.exec_seconds is not None or n.smoke)
        val_ok = Counter("ok" if n.val_files_ok else ("missing_or_invalid" if n.val_files_ok is False else "not_checked")
                         for n in sol if n.ok)
    with ctx.state_lock:
        timings = list(ctx.step_timings)
    elapsed = max(1e-9, ctx.elapsed())
    search_secs = elapsed
    if tree.search_start is not None:
        search_secs = max(1e-9, ctx.elapsed() - (tree.search_start - ctx.process_start))
    parallel = int(cfg["resources"]["parallel"])
    step_total = sum(t.step_seconds for t in timings)
    eff = effective_branches(branch_seq)
    half = len(eff) // 2
    window = int(cfg["plateau"]["branch_window"])
    max_since = max([v for v in since.values() if v is not None], default=None)
    blockers = dict(ctx.flags.get("synthesis_blockers", {}))
    not_triggered: dict[str, str] = {}
    if kinds.get("retrospect", 0) == 0 and kinds.get("borrow", 0) == 0:
        not_triggered["retrospect_borrow"] = (f"no branch plateau was detected: the largest number of attempts after a branch best "
                                              f"was {max_since} (plateau window {window})")
    if kinds.get("synthesis", 0) == 0:
        top = max(blockers.items(), key=lambda kv: kv[1])[0] if blockers else "no check recorded"
        not_triggered["synthesis"] = f"synthesis conditions were not met; most frequent unmet condition: {top}"
    if modes.get("elite_pick", 0) == 0:
        draws = len(late)
        not_triggered["elite_pick"] = (f"{draws} steps started after the exploit phase began (progress >= {switch_start}); "
                                       "none drew elite_pick")
    p = ctx.progress()
    return {
        "throughput": {
            "nodes": len(sol),
            "nodes_per_hour": round(len(sol) / (search_secs / 3600.0), 2),
            "mean_step_seconds": round(step_total / len(timings), 1) if timings else None,
            "median_step_seconds": round(statistics.median([t.step_seconds for t in timings]), 1) if timings else None,
            "median_script_seconds": round(statistics.median([t.exec_seconds for t in timings if t.exec_seconds is not None]), 1)
            if any(t.exec_seconds is not None for t in timings) else None,
            "slot_utilization": round(step_total / (parallel * search_secs), 3),
        },
        "selection_modes": dict(modes),
        "selection_modes_after_switch_start": dict(late_modes),
        "chain_share": round(modes.get("chain", 0) / max(1, sum(modes.values())), 3),
        "chains_left_for_elite": left_chain,
        "chains_left_for_synthesis": left_for_synthesis,
        "dispatch_reasons": dict(reasons),
        "operators": {k: kinds.get(k, 0) for k in OPERATORS},
        "not_triggered": not_triggered,
        "synthesis_checks": blockers,
        "attempts_since_branch_best": {str(b): v for b, v in since.items()},
        "effective_branches": {"window": EFFECTIVE_BRANCH_WINDOW, "series": eff,
                               "mean_first_half": round(statistics.mean(eff[:half]), 3) if half else None,
                               "mean_second_half": round(statistics.mean(eff[half:]), 3) if eff[half:] else None},
        "smoke_runs": dict(smoke),
        "validation_files": dict(val_ok),
        "explore_weight_now": round(explore_weight(p, cfg), 3) if p is not None else None,
    }


def _control_summary(ctx: RunContext) -> dict[str, Any] | None:
    if ctx.control is None:
        return None
    try:
        return ctx.control.summary()
    except Exception as exc:  # noqa: BLE001 - a broken controller must not break the report
        return {"controller": getattr(ctx.control, "name", "unknown"), "error": f"{type(exc).__name__}: {exc}"}


def build_summary(ctx: RunContext, ensemble: dict[str, Any] | None = None, final: dict[str, Any] | None = None) -> dict[str, Any]:
    tree = ctx.tree
    d = ctx.direction
    with tree.lock:
        sol = [n for n in tree.nodes if not n.is_root]
        best = tree.best
        best_block: dict[str, Any] | None = None
        if best is not None:
            path = []
            x = best
            while x is not None and not x.is_root:
                path.append({"node_id": x.id, "kind": x.kind, "score": x.score, "plan": x.plan[:PLAN_PREVIEW]})
                x = x.parent
            best_block = {
                "node_id": best.id, "score": best.score, "kind": best.kind, "branch_id": best.branch_id, "plan": best.plan,
                "method_summary": best.method_summary, "has_artifacts": best.has_artifacts, "path": list(reversed(path)),
            }
        branches = []
        for b in sorted({n.branch_id for n in sol if n.branch_id is not None}):
            nodes_b = [n for n in sol if n.branch_id == b]
            origin = next((n for n in nodes_b if n.kind in ("draft", "synthesis") and n.parent is tree.root), nodes_b[0])
            ok_b = [n for n in nodes_b if n.ok and n.score is not None]
            bb = tree.sorted_ok(ok_b)[0] if ok_b else None
            branches.append({
                "branch_id": b, "origin_kind": origin.kind, "origin_plan": origin.plan[:PLAN_PREVIEW], "num_nodes": len(nodes_b),
                "num_ok_nodes": len(ok_b), "best_score": bb.score if bb else None, "best_node_id": bb.id if bb else None,
            })
        actions = {}
        for op in OPERATORS:
            ns = [n for n in sol if n.kind == op]
            ok = [n for n in ns if n.ok]
            gains = [d.gain(n.score, n.parent.score) for n in ok
                     if n.score is not None and n.parent is not None and n.parent.ok and n.parent.score is not None]
            actions[op] = {
                "count": len(ns), "ok_count": len(ok), "ok_rate": round(len(ok) / len(ns), 4) if ns else None,
                "improved_count": sum(1 for g in gains if g > 0),
                "mean_gain": round(sum(gains) / len(gains), 6) if gains else None,
            }
        # Each failed node counts once, under its first (root) reason; all_reasons counts every reason recorded.
        by_reason: dict[str, int] = defaultdict(int)
        all_reasons: dict[str, int] = defaultdict(int)
        examples: dict[str, dict[str, Any]] = {}
        for n in sol:
            if n.failure_reasons:
                first = n.failure_reasons[0]
                by_reason[first] += 1
                examples.setdefault(first, {"node_id": n.id, "summary": n.summary[:PLAN_PREVIEW],
                                            "all_reasons": list(n.failure_reasons)})
            for r in dict.fromkeys(n.failure_reasons):
                all_reasons[r] += 1
        top = [{"rank": i, "node_id": n.id, "score": n.score, "branch_id": n.branch_id, "kind": n.kind,
                "method_summary": n.method_summary, "has_artifacts": n.has_artifacts} for i, n in enumerate(tree.top_list, 1)]
        timeline = list(tree.score_timeline)
        num_ok = sum(1 for n in sol if n.ok)
        num_failed = sum(1 for n in sol if n.is_failed)
        n_runs = sum(1 for n in sol if n.exec_seconds is not None)
        n_timeouts = sum(1 for n in sol if n.error_type == "TimeoutError")
        script_secs = round(sum(n.exec_seconds or 0.0 for n in sol), 1)
    events = sorted(list(ctx.key_events), key=lambda e: (e.get("elapsed_secs") or 0))
    return {
        "interface_version": INTERFACE_VERSION,
        "task": {
            "higher_is_better": d.higher_is_better,
            "direction_source": ctx.direction_source,
            "detected_modalities": list(ctx.modalities),
            "pretrained_mode": ctx.mode,
            "has_prediction_template": ctx.sample_path is not None,
            "data_preview": ctx.data_preview,
        },
        "run": {
            "started_at": ctx.started_at, "updated_at": now_iso(), "elapsed_secs": round(ctx.elapsed(), 1),
            "stop_reason": ctx.stop_reason, "time_limit_secs": ctx.time_limit, "max_steps": int(ctx.cfg["run"]["max_steps"]),
            "num_nodes": len(sol), "num_ok_nodes": num_ok, "num_failed_nodes": num_failed, "num_branches": len(branches),
        },
        "best": best_block,
        "score_timeline": timeline,
        "branches": branches,
        "actions": actions,
        "key_events": events,
        "top_solutions": top,
        "ensemble": ensemble,
        "final": _final_view(final),
        "failures": {"failed_nodes": sum(by_reason.values()), "by_reason": dict(by_reason), "all_reasons": dict(all_reasons),
                     "examples": examples},
        "control": _control_summary(ctx),
        "diagnostics": _diagnostics(ctx),
        "resources": {
            "llm": _llm_usage(ctx),
            "embedding": _embedding_usage(ctx),
            "exec": {"num_runs": n_runs, "num_timeouts": n_timeouts, "total_script_seconds": script_secs},
        },
    }


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _md_cell(text: Any) -> str:
    return str(text if text is not None else "-").replace("|", "\\|").replace("\n", " ")


def render_markdown(s: dict[str, Any]) -> str:
    run, task, res = s["run"], s["task"], s["resources"]
    lines = ["# SuperAgent run summary", "", "## Overview", "",
             f"- Started: {run['started_at']}; updated: {run['updated_at']}; elapsed: {run['elapsed_secs']} s",
             f"- Stop reason: {run['stop_reason'] or 'running'}",
             f"- Metric direction: {'higher' if task['higher_is_better'] else 'lower'} is better (source: {task['direction_source']})",
             f"- Detected modalities: {', '.join(task['detected_modalities']) or 'none'}; pretrained mode: {task['pretrained_mode']}",
             f"- Nodes: {run['num_nodes']} ({run['num_ok_nodes']} ok, {run['num_failed_nodes']} failed) in {run['num_branches']} branches",
             ""]
    b = s.get("best")
    lines += ["## Best solution", ""]
    if b:
        lines += [f"- Node `{b['node_id']}` ({b['kind']}, branch {b['branch_id']}), score **{_fmt(b['score'])}**", "",
                  "Improvement path:", "", "| # | node | kind | score | plan |", "|---|---|---|---|---|"]
        for i, p in enumerate(b["path"], 1):
            lines.append(f"| {i} | `{p['node_id']}` | {p['kind']} | {_fmt(p['score'])} | {_md_cell(p['plan'][:120])} |")
    else:
        lines.append("No valid solution yet.")
    lines += ["", "## Score timeline", "", "| step | elapsed (s) | node | kind | branch | score |", "|---|---|---|---|---|---|"]
    for t in s["score_timeline"]:
        lines.append(f"| {t['step']} | {t['elapsed_secs']} | `{t['node_id']}` | {t['kind']} | {t['branch_id']} | {_fmt(t['score'])} |")
    lines += ["", "## Branches", "", "| branch | origin | nodes | ok | best score | origin plan |", "|---|---|---|---|---|---|"]
    for br in s["branches"]:
        lines.append(f"| {br['branch_id']} | {br['origin_kind']} | {br['num_nodes']} | {br['num_ok_nodes']} | {_fmt(br['best_score'])} | {_md_cell(br['origin_plan'][:120])} |")
    lines += ["", "## Operators", "", "| operator | count | ok | ok rate | improved | mean gain |", "|---|---|---|---|---|---|"]
    for op, a in s["actions"].items():
        lines.append(f"| {op} | {a['count']} | {a['ok_count']} | {_fmt(a['ok_rate'], 2)} | {a['improved_count']} | {_fmt(a['mean_gain'])} |")
    lines += ["", "## Key events", ""]
    for e in s["key_events"]:
        lines.append(f"- [{e['elapsed_secs']} s] {e['type']} (node {e['node_id']}): {e['detail']}")
    lines += ["", "## Top solutions", "", "| rank | node | score | branch | kind | artifacts |", "|---|---|---|---|---|---|"]
    for t in s["top_solutions"]:
        lines.append(f"| {t['rank']} | `{t['node_id']}` | {_fmt(t['score'])} | {t['branch_id']} | {t['kind']} | {t['has_artifacts']} |")
    ens = s.get("ensemble")
    if ens:
        lines += ["", "## Ensemble", "", f"- Available: {ens.get('available')}; primary size: {ens.get('primary_size')}; strategy: {ens.get('strategy')}"]
    fin = s.get("final")
    if fin:
        val = fin.get("validation") or {}
        lines += ["", "## Final result", "",
                  f"- Kind: {fin.get('kind')}; method: {fin.get('method')}; metric: {fin.get('metric')}",
                  f"- Validation score: {_fmt(val.get('score'))} (single best: {_fmt(val.get('single_full_score'))}; "
                  f"shared validation samples: {val.get('rows', '-')})",
                  f"- Reason: {fin.get('reason')}"]
        for m in fin.get("members") or []:
            lines.append(f"  - rank {m['rank']} `{m['node_id']}` weight {_fmt(m['weight'])}")
    dg = s.get("diagnostics")
    if dg:
        th = dg["throughput"]
        eb = dg["effective_branches"]
        lines += ["", "## Diagnostics", "",
                  f"- Nodes per hour: {th['nodes_per_hour']}; median step {th['median_step_seconds']} s; median script "
                  f"{th['median_script_seconds']} s; slot utilization {th['slot_utilization']}",
                  f"- Selection modes: {dg['selection_modes']}; after the exploit phase began: {dg['selection_modes_after_switch_start']}",
                  f"- Chain share: {dg['chain_share']}; chains left for an elite node: {dg['chains_left_for_elite']}; "
                  f"for synthesis: {dg['chains_left_for_synthesis']}",
                  f"- Operators: {dg['operators']}",
                  f"- Effective branches (window {eb['window']}): first half {eb['mean_first_half']}, second half {eb['mean_second_half']}",
                  f"- Smoke runs: {dg['smoke_runs']}; validation files: {dg['validation_files']}"]
        for name, why in dg["not_triggered"].items():
            lines.append(f"- Not triggered ({name}): {why}")
    ctl = s.get("control")
    if ctl and ctl.get("cost_cap"):
        cc = ctl["cost_cap"]
        curve = cc.get("cost_curve") or []
        last = curve[-1]["cost"] if curve else None
        lines += ["", "## Control", "", f"- Controller: {ctl.get('controller')}; rules: {ctl.get('rules')}",
                  f"- Cost (cap currency {cc.get('currency')}): {last}; limit {cc.get('limit')}; events: {cc.get('events')}"]
        ta = (ctl.get("time_allocation") or {}).get("run_limits")
        if ta:
            lines.append(f"- Script time limits applied: min {ta['min']} s, median {ta['median']} s, max {ta['max']} s")
    lines += ["", "## Failures", ""]
    for r, n in s["failures"]["by_reason"].items():
        lines.append(f"- {r}: {n}")
    llm = res["llm"]
    amount = "price using Adapter cost" if llm.get("accounting") == "adapter" else f"known cost {llm['cost']} {llm['currency']}"
    lines += ["", "## Resource usage", "",
              f"- LLM HTTP attempts: {llm['calls']} ({llm['failed_calls']} failed); prompt tokens {llm['prompt_tokens']}, "
              f"completion tokens {llm['completion_tokens']} (reasoning {llm['reasoning_tokens']}); "
              f"{amount}; attempts without usage {llm['unknown_usage']}",
              f"- Embedding: {res['embedding']['calls']} calls, {res['embedding']['tokens']} tokens, cost {res['embedding']['cost']} {res['embedding']['currency']}",
              f"- Script runs: {res['exec']['num_runs']} ({res['exec']['num_timeouts']} timeouts), total {res['exec']['total_script_seconds']} s",
              "", "| agent | calls | prompt tokens | completion tokens | cost |", "|---|---|---|---|---|"]
    for agent, u in sorted(llm["by_agent"].items()):
        lines.append(f"| {agent} | {u['calls']} | {u['prompt_tokens']} | {u['completion_tokens']} | {u['cost']} |")
    return "\n".join(lines) + "\n"


def write_summary(ctx: RunContext, ensemble: dict[str, Any] | None = None, final: dict[str, Any] | None = None) -> dict[str, Any]:
    s = build_summary(ctx, ensemble, final)
    atomic_write_json(ctx.paths.report / "run_summary.json", s)
    atomic_write_text(ctx.paths.report / "run_summary.md", render_markdown(s))
    return s
