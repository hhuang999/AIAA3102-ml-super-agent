"""Summarise a Core output directory (and, optionally, its grades) for evaluation reports. Reads files only."""

from __future__ import annotations

import collections
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

OPERATORS = ("draft", "refine", "repair", "retrospect", "borrow", "synthesis")
LOAD_ERROR_TYPES = ("OSError", "LocalEntryNotFoundError", "HFValidationError", "FileNotFoundError", "EnvironmentError")
LOAD_ERROR_RE = re.compile(r"(LocalEntryNotFoundError|Can't load|couldn't connect|HFValidationError|is not a local folder|"
                           r"does not appear to have a file|No such file or directory)", re.I)


def _jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def _medal(rep: dict[str, Any]) -> str:
    for m in ("gold", "silver", "bronze"):
        if rep.get(f"{m}_medal"):
            return m
    return "above median" if rep.get("above_median") else "none"


def analyze(core_out: Path, grades: Path | None = None, catalog: Path | None = None) -> dict[str, Any]:
    o = Path(core_out)
    tree = json.loads((o / "logs" / "search_tree.json").read_text())
    nodes = [n for n in tree["nodes"] if n["kind"] != "root"]
    by_id = {n["id"]: n for n in tree["nodes"]}
    dec = _jsonl(o / "logs" / "decisions.jsonl")
    summ = json.loads((o / "report" / "run_summary.json").read_text())
    status = json.loads((o / "status.json").read_text())
    hib = tree["higher_is_better"]
    res: dict[str, Any] = {"status": status}

    ops = {}
    for k in OPERATORS:
        ns = [n for n in nodes if n["kind"] == k]
        ok = [n for n in ns if n["failed"] is False]
        ops[k] = {"count": len(ns), "ok": len(ok), "ok_rate": round(len(ok) / len(ns), 3) if ns else None}
    res["operators"] = ops
    res["selection_modes"] = dict(collections.Counter(d.get("selection_mode") for d in dec))
    res["dispatch_reasons"] = dict(collections.Counter(d.get("reason") for d in dec))
    res["trigger_evidence"] = {
        "branch_plateau_decisions": sum(1 for d in dec if d.get("reason") == "branch_plateau"),
        "synthesis_decisions": sum(1 for d in dec if d.get("action") == "synthesis"),
        "chains_left_for_elite": sum(1 for d in dec if (d.get("details") or {}).get("left_chain_at")),
        "ok_per_branch": dict(collections.Counter(n["branch_id"] for n in nodes if n["failed"] is False)),
        "key_event_types": dict(collections.Counter(e["type"] for e in summ.get("key_events", []))),
    }
    best = by_id.get(status.get("best_node_id"))
    res["best"] = None if best is None else {
        "node_id": best["id"], "kind": best["kind"], "score": best["score"], "step": best["step"],
        "path": [(p["kind"], p["score"]) for p in (summ.get("best") or {}).get("path", [])],
        "method_summary": best.get("method_summary")}
    res["score_timeline"] = summ.get("score_timeline", [])

    gains = []
    for n in nodes:
        p = by_id.get(n["parent_id"])
        if n["failed"] is False and p and p["kind"] != "root" and p["failed"] is False and n["score"] is not None and p["score"] is not None:
            gains.append((n["score"] - p["score"]) if hib else (p["score"] - n["score"]))
    imp = [g for g in gains if g > 0]
    res["improvements_over_parent"] = {"comparable_children": len(gains), "improved": len(imp),
                                       "mean_gain_of_improved": round(sum(imp) / len(imp), 6) if imp else None,
                                       "mean_gain_all": round(sum(gains) / len(gains), 6) if gains else None}
    # Each failed node counts once under its first (root) reason; all_reasons counts every reason recorded.
    first = collections.Counter(n["failure_reasons"][0] for n in nodes if n["failure_reasons"])
    reasons = collections.Counter(r for n in nodes for r in dict.fromkeys(n["failure_reasons"]))
    res["failures"] = {"failed_nodes": sum(first.values()), "by_reason": dict(first), "all_reasons": dict(reasons),
                       "internal_error": reasons.get("internal_error", 0),
                       "codegen_error": reasons.get("codegen_error", 0), "timeouts": reasons.get("timeout", 0),
                       "discarded_steps": sum(1 for d in dec if (d.get("details") or {}).get("discarded")),
                       "error_types": dict(collections.Counter(n["error_type"] for n in nodes if n["error_type"]))}

    if grades is not None and Path(grades).exists():
        g = json.loads(Path(grades).read_text())
        res["grades"] = {k: {"score": v["score"], "medal": _medal(v)} for k, v in g["grades"].items()}
        res["primary"] = g.get("primary")
        first = next(iter(g["grades"].values()))
        res["thresholds"] = {k: first.get(k) for k in ("gold_threshold", "silver_threshold", "bronze_threshold", "median_threshold")}
        rows = []
        for name, rep in g["grades"].items():
            if not name.startswith("top_rank_"):
                continue
            sj = o / "top" / name.replace("top_", "") / "score.json"
            if sj.exists():
                s = json.loads(sj.read_text())
                rows.append({"rank": name.replace("top_", ""), "node_id": s.get("node_id"), "validation": s.get("score"),
                             "test": rep["score"]})
        if rows:
            res["validation_vs_test"] = {"rows": rows, "spearman": _spearman([r["validation"] for r in rows], [r["test"] for r in rows])}

    llm = summ["resources"]["llm"]
    emb = summ["resources"]["embedding"]
    res["usage"] = {"llm_calls": llm["calls"], "failed_calls": llm["failed_calls"], "prompt_tokens": llm["prompt_tokens"],
                    "completion_tokens": llm["completion_tokens"], "reasoning_tokens": llm["reasoning_tokens"],
                    "llm_cost": llm["cost"], "currency": llm["currency"], "embedding": emb, "exec": summ["resources"]["exec"],
                    "cost_per_node": round((llm["cost"] + emb["cost"]) / max(1, len(nodes)), 6),
                    "by_agent": {k: {"calls": v["calls"], "cost": v["cost"]} for k, v in llm["by_agent"].items()}}
    res["pretrained"] = _pretrained(nodes, catalog)
    for key in ("final", "diagnostics", "control"):
        if summ.get(key) is not None:
            res[key] = summ[key]
    return res


def _spearman(a: list[float], b: list[float]) -> float | None:
    n = len(a)
    if n < 3:
        return None

    def ranks(x: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: x[i])
        r = [0.0] * n
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r

    ra, rb = ranks(a), ranks(b)
    return round(1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1)), 4)


def _pretrained(nodes: list[dict[str, Any]], catalog: Path | None) -> dict[str, Any]:
    hub: dict[str, list[str]] = {}
    if catalog is not None and Path(catalog).exists():
        for e in yaml.safe_load(Path(catalog).read_text()) or []:
            if e.get("pretrained") and e.get("hub_id"):
                hub[e["id"]] = [e["hub_id"], e["hub_id"].split("/")[-1]]
    used: collections.Counter[str] = collections.Counter()
    used_ok: collections.Counter[str] = collections.Counter()
    for n in nodes:
        for eid, keys in hub.items():
            if any(k in (n["code"] or "") for k in keys):
                used[eid] += 1
                if n["failed"] is False:
                    used_ok[eid] += 1
    literals = collections.Counter(m for n in nodes for m in re.findall(r"from_pretrained\(\s*['\"]([^'\"]+)['\"]", n["code"] or ""))
    load_fail = [n["id"] for n in nodes if n["failed"] and ((n["error_type"] or "") in LOAD_ERROR_TYPES
                                                             or LOAD_ERROR_RE.search(n["error_message"] or ""))]
    return {"nodes_mentioning": dict(used), "successful_nodes_mentioning": dict(used_ok), "from_pretrained_literals": dict(literals),
            "nodes_failed_with_model_loading_errors": load_fail}


def plot_timeline(core_out: Path, out_png: Path) -> None:
    """Validation score of every successful node and the best so far, over hours since the start."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    o = Path(core_out)
    tree = json.loads((o / "logs" / "search_tree.json").read_text())
    status = json.loads((o / "status.json").read_text())
    summ = json.loads((o / "report" / "run_summary.json").read_text())
    hib = tree["higher_is_better"]
    ok = [n for n in tree["nodes"] if n["kind"] != "root" and n["failed"] is False and n["score"] is not None and n.get("finished_at")]
    t0 = datetime.fromisoformat(status["started_at"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter([(datetime.fromisoformat(n["finished_at"]) - t0).total_seconds() / 3600 for n in ok], [n["score"] for n in ok],
               s=12, alpha=0.4, label="successful node (validation)")
    tl = summ.get("score_timeline", [])
    if tl:
        ax.step([t["elapsed_secs"] / 3600 for t in tl] + [status["elapsed_secs"] / 3600], [t["score"] for t in tl] + [tl[-1]["score"]],
                where="post", color="C3", label="best so far")
    ax.set_xlabel("hours since start")
    ax.set_ylabel("validation score (" + ("higher" if hib else "lower") + " is better)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Score over time")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
