"""Command-line entry point and the ``run_task`` function.

    python -m superagent.run --task-dir TASK --output-dir OUT [options]
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import random
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from . import report
from .bookkeeping import INTERFACE_VERSION, update_rankings, write_search_tree, write_status
from .catalog import build_catalog, offline_env
from .contract import load_spec
from .publication import latest
from .config import build_config, dump_config
from .context import LockedRandom, OutputPaths, RunContext
from .control import build_state, detect_gpus, load_controller_class, make_controller
from .control.hardware import cuda_mask
from .errors import InvalidInputError, LLMUnavailableError, StructuredOutputError
from .executor import Executor
from .experience import ExperienceStore, make_embedder
from .llm import Field
from .llm.gateway import ASSESSMENT_ROLES, LLMGateway, OpenAITransport, abort_decision
from .profiler import SAMPLE_NAME, profile
from .prompts import builders
from .prompts.blocks import TASK_ALIGNMENT_RULES
from .search import SearchController
from .task_io import check_output_dir, check_task_dir, prepare_data
from .tree import Direction, SearchTree
from .utils import atomic_write_json, atomic_write_text, load_env_file, new_id, now_iso

log = logging.getLogger("superagent")

ENV_FILE_NAME = ".env"
DIRECTION_ATTEMPTS = 3
EXIT_OK, EXIT_FAILED, EXIT_NO_SOLUTION, EXIT_INVALID = 0, 1, 2, 3
DIRECTION_SCHEMA = {
    "goal": Field("enum", "minimize or maximize", values=("minimize", "maximize")),
    "explanation": Field("string", "short reason"),
}


class _Interrupt:
    """Shared interrupt state so signal handlers can reach the running task."""

    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.hard_event = threading.Event()
        self.ctx: RunContext | None = None
        self.executor: Executor | None = None

    def trigger(self) -> None:
        if self.ctx is not None:
            self.ctx.stop_reason = "interrupted"
        self.interrupted = True
        self.stop_event.set()
        self.hard_event.set()
        if self.executor is not None:
            self.executor.kill_all()

    interrupted = False


def _setup_logging(path: Path) -> list[logging.Handler]:
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(message)s")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    log.propagate = False
    return [fh, sh]


def _bare_status(out: Path, status: str, error: str | None, started_at: str, elapsed: float) -> dict[str, Any]:
    payload = {
        "interface_version": INTERFACE_VERSION, "status": status,
        "stop_reason": "error" if status == "failed" else None,
        "started_at": started_at, "updated_at": now_iso(), "elapsed_secs": round(elapsed, 1), "higher_is_better": None,
        "direction_source": None, "num_nodes": 0, "num_ok_nodes": 0, "num_branches": 0, "best_node_id": None,
        "best_score": None, "best_has_artifacts": False, "ensemble_available": False, "final_available": False,
        "final_kind": None, "final_method": None, "final_metric": None, "final_validation_score": None, "error": error,
    }
    atomic_write_json(out / "status.json", payload)
    return payload


def _clean_description(gw: LLMGateway, raw: str) -> str:
    system, user = builders.desc_cleaner(raw)
    try:
        res = gw.text("desc_cleaner", system, user)
        cleaned = res.content.strip()
        if not cleaned or res.truncated:
            raise ValueError("empty or truncated cleaned description")
    except Exception as exc:  # noqa: BLE001
        log.warning("description cleaning failed (%s); using the original text", exc)
        cleaned = raw.strip()
    return cleaned + "\n\n" + TASK_ALIGNMENT_RULES


def _judge_direction(gw: LLMGateway, task_desc: str) -> bool:
    system, user = builders.direction_judge(task_desc)
    for _ in range(DIRECTION_ATTEMPTS):
        try:
            obj = gw.structured("direction_judge", system, user, DIRECTION_SCHEMA)
            log.info("metric direction judged by the LLM: %s (%s)", obj["goal"], str(obj["explanation"])[:200])
            return obj["goal"] == "maximize"
        except (StructuredOutputError, LLMUnavailableError) as exc:
            log.warning("direction judgement failed: %s", exc)
    log.warning("could not determine the metric direction; assuming higher is better")
    return True


def run_task(
    task_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    config: str | os.PathLike[str] | None = None,
    time_limit_secs: int | None = None,
    max_steps: int | None = None,
    higher_is_better: bool | None = None,
    num_gpus: int | None = None,
    num_cpus: int | None = None,
    parallel: int | None = None,
    keep_artifacts_top_k: int | None = None,
    pretrained_mode: str | None = None,
    seed: int | None = None,
    _interrupt: _Interrupt | None = None,
) -> dict[str, Any]:
    """Run one task to completion. Returns the final ``status.json`` content.

    Raises :class:`InvalidInputError` only for unusable input; search failures are reported in the result.
    """
    process_start = time.monotonic()
    started_at = now_iso()
    out = check_output_dir(output_dir)  # raises without writing anything
    paths = OutputPaths(out)
    paths.make_dirs()
    handlers = _setup_logging(paths.run_log)
    intr = _interrupt or _Interrupt()
    ctx: RunContext | None = None
    async_transport = None
    try:
        try:
            load_env_file(Path.cwd() / ENV_FILE_NAME)
            cli = {"time_limit_secs": time_limit_secs, "max_steps": max_steps, "seed": seed, "num_gpus": num_gpus,
                   "num_cpus": num_cpus, "parallel": parallel, "keep_artifacts_top_k": keep_artifacts_top_k,
                   "pretrained_mode": pretrained_mode}
            cfg = build_config(config, cli_overrides=cli)
            formal = cfg["verification"]["runner"] == "bubblewrap"
            mask = cuda_mask()
            if mask is not None and cfg["resources"]["num_gpus"] > len(mask):
                log.warning("limiting resources.num_gpus to the %d device(s) in CUDA_VISIBLE_DEVICES", len(mask))
                cfg["resources"]["num_gpus"] = len(mask)
            task = check_task_dir(task_dir)
            key_env = str(cfg["llm"]["api_key_env"])
            api_key = os.environ.get(key_env, "")
            if not api_key and not formal:
                raise InvalidInputError(f"environment variable {key_env} is not set (it can also be put in a .env file)")
            if int(cfg["resources"]["num_cpus"]) < int(cfg["resources"]["parallel"]):
                raise InvalidInputError("resources.num_cpus must be at least resources.parallel")
            load_controller_class(str(cfg["control"]["controller"]))
        except InvalidInputError as exc:
            _bare_status(out, "invalid_input", str(exc), started_at, time.monotonic() - process_start)
            log.error("invalid input: %s", exc)
            raise
        seed_v = int(cfg["run"]["seed"])
        random.seed(seed_v)
        rng = LockedRandom(seed_v)
        _bare_status(out, "running", None, started_at, time.monotonic() - process_start)
        atomic_write_text(paths.config_used, dump_config(cfg))
        log.info("task %s -> %s", task, out)

        prepare_data(task / "data", paths.prepared)
        deadline = process_start + float(cfg["run"]["time_limit_seconds"])
        reserve = float(cfg["run"]["finalize_reserve_seconds"])
        if cfg["control"]["controller"]=="adaptive":
            reserve=min(reserve,.1*float(cfg["run"]["time_limit_seconds"]))

        def should_abort(role: str) -> bool:
            if formal:
                return intr.stop_event.is_set()
            if ctx is not None and role not in ASSESSMENT_ROLES and ctx.generation_window_expired():
                return True
            return abort_decision(role, deadline - time.monotonic(), ctx.reserve if ctx is not None else reserve, intr.interrupted, intr.hard_event.is_set())

        runs_dir = paths.runs

        def prompt_dir(node_id: str) -> Path:
            d = runs_dir / node_id
            d.mkdir(parents=True, exist_ok=True)
            return d

        run_id = new_id() if formal else None
        if formal:
            from .llm.async_transport import AsyncDeepSeekTransport

            slots = cfg["llm"]["max_in_flight"] or cfg["resources"]["parallel"]
            async_transport = AsyncDeepSeekTransport(cfg["llm"]["base_url"], api_key, cfg["llm"]["request_limits"], slots)
            tr = async_transport
        else:
            tr = OpenAITransport(cfg["llm"]["base_url"], api_key, float(cfg["llm"]["request_timeout_seconds"]))
        gateway = LLMGateway(cfg["llm"], paths.llm_calls, tr, should_abort=should_abort, prompt_log_dir=prompt_dir, run_id=run_id)

        cap=cfg["control"]["cost_cap"]
        if not formal and cap["mode"]=="enforce" and cap["limit"]>0:
            gateway.ledger.limit=cap["limit"]/cap["rate_to_cap_currency"]
        raw_desc = (task / "description.md").read_text(encoding="utf-8", errors="replace")
        spec = load_spec(task, paths.prepared)
        visible = paths.prepared
        manifest = None
        if formal:
            if spec is None:
                raise InvalidInputError("single-model runs require task.yaml v1 and declared validation truth")
            from .inputs import prepare_inputs

            manifest = prepare_inputs(task, paths.prepared, spec, out / "input_views")
            visible = out / "input_views/training"
        if manifest:
            import json

            task_desc = (
                raw_desc + "\nUse data/input_manifest.json and this task-owned contract:\n"
                + json.dumps({"metric": spec.metric, "classes": spec.classes, "columns": spec.columns,
                              "probability": spec.probability, "input_manifest": manifest.payload()}, ensure_ascii=False)
                + "\nEmit out/val_predictions.csv and out/predictions.csv. "
                "Validation IDs must cover the fixed split and test rows must follow the sample order. "
                "The independent scorer owns the validation truth. "
                "Your --predict-only path receives only bound val/test inputs and the fitted model package."
            )
        else:
            task_desc = raw_desc + "\n" + spec.prompt if spec else _clean_description(gateway, raw_desc)
        if spec is not None:
            if not spec.inference_inputs and not formal:
                raise InvalidInputError("task.yaml must declare inference_inputs for independent artifact verification")
            if higher_is_better is not None and higher_is_better != spec.maximize:
                raise InvalidInputError("CLI metric direction contradicts task.yaml")
            hib, source = spec.maximize, "task_contract"
            atomic_write_json(paths.logs / "task_contract.json", {"sha256":spec.signature, "metric":spec.metric,
                              "validation_rows":len(spec.truth), "test_rows":len(spec.test_ids)})
        elif higher_is_better is not None:
            hib, source = bool(higher_is_better), "cli"
        else:
            hib, source = _judge_direction(gateway, task_desc), "llm"
        direction = Direction(hib)

        prof = profile(visible, cfg["profiler"])
        sample = paths.prepared / SAMPLE_NAME
        sample_path = sample if sample.exists() and spec is None else None
        if sample_path is None and spec is None:
            log.warning("data/%s not found: prediction file format checks are skipped", SAMPLE_NAME)
        mode = str(cfg["pretrained"]["mode"])
        local_dir = str(cfg["pretrained"]["local_dir"])
        cat = build_catalog("catalog" if formal and mode != "off" else mode, local_dir, prof.modalities)
        from .knowledge import read_cards
        if not formal:
            cards = read_cards(task, prof.modalities)
            if cards:
                cat.text += "\n\nTask method cards:\n" + cards
        log.info("detected modalities: %s; catalog entries shown: %d", prof.modalities or "none", len(cat.entries))

        experience = None
        if cfg["experience"]["enabled"]:
            embedder = make_embedder(cfg["experience"], local_dir, paths.embedding_calls)
            experience = ExperienceStore(cfg["experience"], paths.records, embedder, background=True)
        tree = SearchTree(direction)
        executor = Executor(paths.runs, paths.prepared, int(cfg["resources"]["parallel"]), int(cfg["resources"]["num_cpus"]),
                            int(cfg["resources"]["num_gpus"]), offline_env(mode, local_dir, str(cfg["pretrained"]["hub_endpoint"])))
        intr.executor = executor
        ctx = RunContext(cfg=cfg, paths=paths, task_dir=task, direction=direction, direction_source=source, tree=tree,
                         gateway=gateway, executor=executor, experience=experience, task_desc=task_desc,
                         data_preview=prof.preview, catalog_text=cat.text, modalities=prof.modalities,
                         sample_path=sample_path, rng=rng, process_start=process_start, started_at=started_at,
                         stop_event=intr.stop_event, hard_event=intr.hard_event)
        if intr.interrupted:
            ctx.stop_reason = "interrupted"
        ctx.task_spec = spec
        from .llm.ledger import fingerprint
        if formal:
            from importlib.metadata import PackageNotFoundError, version
            from dataclasses import asdict

            from .catalog import match_methods, render, stage_method_assets
            from .inputs import build_task_facts
            from .isolation import Bubblewrap
            from .knowledge import load_cards
            from .training import digest

            versions = {}
            for package in ("superagent", "numpy", "pandas", "scikit-learn", "Pillow", "tfrecord", "joblib",
                            "torch", "torchvision", "timm", "transformers", "sentence-transformers",
                            "lightgbm", "xgboost", "catboost", "tabicl"):
                try:
                    versions[package] = version(package)
                except PackageNotFoundError:
                    continue
            facts = build_task_facts(spec, manifest, {"modalities": prof.modalities, "dependencies": versions}, run_id)
            worktree = Path(__file__).parent
            runtime_sha = fingerprint([(file.relative_to(worktree).as_posix(), digest(file))
                for file in sorted(worktree.rglob("*")) if file.is_file() and file.suffix in (".py", ".yaml")])
            facts["runtime_sha256"] = runtime_sha
            facts["catalog_sha256"] = digest(worktree / "resources/method_catalog.yaml")
            matches = match_methods(facts, cat, load_cards(task, prof.modalities), local_dir=local_dir)
            eligible = [match for match in matches if match.eligible]
            drafts = eligible[:3]
            assets = stage_method_assets(drafts, out / "method_assets", local_dir)
            methods = [
                {**{key: value for key, value in asdict(m).items() if key != "entry"},
                 "weights_sha256": m.entry.get("weight_hashes") if m.eligible else None}
                for m in matches
            ]
            facts["methods"] = methods
            atomic_write_json(paths.logs / "method_matches.json", methods)
            ctx.catalog_text = render([m.entry for m in drafts], "catalog" if mode != "off" else mode)
            task_notes = [m for m in eligible if m.entry.get("method")]
            if task_notes:
                ctx.catalog_text += "\nTask methods:\n" + "\n".join(
                    f"{m.key}: {m.hypothesis}. Avoid: {m.entry['counterexample']}. Source: {m.source}" for m in task_notes[:3]
                )
            ctx.flags.update(formal=True, input_manifest=manifest, task_facts=facts,
                runtime_environment={"schema_version": 1, "python": platform.python_version(), "packages": versions,
                                     "runtime_sha256": runtime_sha, "catalog_sha256": facts["catalog_sha256"]},
                publish_lock=threading.Lock(), artifact_leases=set(), draft_methods=drafts)
            atomic_write_json(paths.logs / "task_facts.json", facts)
            executor.isolation = Bubblewrap(visible, out / "input_views/inference", task / "data", assets=assets)
        # The prepared dataset is stable for the lifetime of this run; promotions never cross runs.
        ctx.flags["trial_environment"]=fingerprint([str(out),spec.signature if spec else raw_desc,
            mode,[(e["id"],e.get("example"),e.get("weight_hashes")) for e in cat.entries],cfg["run"]["seed"]])
        intr.ctx = ctx
        ctx.gpu_info = detect_gpus()
        ctx.control = make_controller(ctx)
        if not formal:
            ctx.control.start(build_state(ctx))
        log.info("controller: %s", ctx.control.name)
        atomic_write_text(paths.logs / "task_description_clean.md", task_desc)
        write_search_tree(ctx)
        write_status(ctx, "running")

        controller = SearchController(ctx)
        try:
            controller.run()
        except Exception:  # noqa: BLE001
            log.error("search loop crashed:\n%s", traceback.format_exc())
            ctx.stop_reason = ctx.stop_reason or "error"
        ctx.stop_reason = controller.stop_reason() if ctx.stop_reason is None else ctx.stop_reason
        return _finalize(ctx)
    except InvalidInputError as exc:
        _bare_status(out, "invalid_input", str(exc), started_at, time.monotonic() - process_start)
        raise
    except Exception as exc:  # noqa: BLE001
        log.error("fatal error: %s\n%s", exc, traceback.format_exc())
        try:
            if ctx is not None:
                ctx.stop_reason = "error"
                return write_status(ctx, "failed", f"{type(exc).__name__}: {exc}")
            return _bare_status(out, "failed", f"{type(exc).__name__}: {exc}", started_at, time.monotonic() - process_start)
        except Exception:  # noqa: BLE001
            return {"status": "failed", "error": str(exc)}
    finally:
        if async_transport is not None:
            async_transport.close()
        if ctx is not None and ctx.experience is not None:
            ctx.experience.close()
        for h in handlers:
            log.removeHandler(h)
            h.close()


def _finalize(ctx: RunContext) -> dict[str, Any]:
    ens_block: dict[str, Any] | None = None
    ctx.flags["finalizing"] = True
    try:
        ctx.executor.kill_all()
    except Exception as exc:  # noqa: BLE001
        log.warning("could not stop running scripts: %s", exc)
    try:
        update_rankings(ctx, None)  # make best/ and top/ match the final tree
    except Exception as exc:  # noqa: BLE001
        log.error("could not refresh the rankings: %s", exc)
    with ctx.io_lock:
        ctx.flags["finalized"] = True  # workers still running no longer write any file
    try:
        if ctx.cfg["ensemble"]["enabled"] and getattr(ctx.control,"name",None)!="adaptive":
            from .ensemble import run_ensemble

            with ctx.tree.lock:
                top = [{"rank": i, "node_id": n.id, "score": n.score} for i, n in enumerate(ctx.tree.top_list, 1)]
            ens_block = run_ensemble(ctx.paths.root, top, ctx.direction.higher_is_better, ctx.cfg["ensemble"])
        else:
            ens_block = {"available": False, "primary_size": None, "strategy": None, "members": []}
    except Exception as exc:  # noqa: BLE001
        log.error("ensembling failed: %s", exc)
        ens_block = {"available": False, "primary_size": None, "strategy": None, "members": [], "error": str(exc)}
    final_block: dict[str, Any] | None = None
    try:
        if ctx.flags.get("formal"):
            final_block = latest(ctx.paths.root)
            receipt = ctx.paths.logs / "publication.json"
            if final_block and (not receipt.exists() or latest_receipt_id(receipt) != final_block["version_id"]):
                atomic_write_json(receipt, {**final_block, "recovered": True})
        elif ctx.cfg["final"]["enabled"] and ctx.tree.best is not None:
            final_block = _write_final(ctx)
    except Exception as exc:  # noqa: BLE001
        log.error("could not write final/: %s", exc)
        final_block = {"available": False, "kind": None, "reason": f"{type(exc).__name__}: {exc}"}
    try:
        write_search_tree(ctx)
    except Exception as exc:  # noqa: BLE001
        log.error("could not write the search tree: %s", exc)
    try:
        report.write_summary(ctx, ens_block, final_block)
    except Exception as exc:  # noqa: BLE001
        log.error("could not write the run summary: %s", exc)
    ledger=getattr(ctx.gateway,"ledger",None)
    if ledger and not ctx.flags.get("formal"):
        accounting = ledger.summary()
        accounting.update(ctx.executor.utilization(ctx.process_start, time.monotonic()))
        with ctx.tree.lock:
            accounting["unfinished_nodes"] = [{"id":n.id,"kind":n.kind,"parent":n.parent.id,"rung":n.rung}
                for n in ctx.tree.by_id.values() if not n.is_root and not n.in_tree]
            accounting["unfinished_steps"] = list(ctx.flags.get("unfinished_steps", []))
            accounting["active_leases"] = sorted(getattr(ctx.control,"leases",()))
        atomic_write_json(ctx.paths.logs / "accounting.json",accounting)
    final_block = latest(ctx.paths.root) or final_block
    if final_block and final_block.get("kind")=="ensemble":
        ens_block={"available":True,"strategy":final_block.get("method")}
    status = ("success" if final_block and final_block.get("available") else
              "failed" if ctx.tree.best is not None else "no_valid_solution")
    payload = write_status(ctx, status, None, bool(ens_block and ens_block.get("available")), final_block)
    log.info("finished: status=%s stop_reason=%s best=%s", status, ctx.stop_reason,
             ctx.tree.best.score if ctx.tree.best else None)
    return payload


def latest_receipt_id(path):
    import json

    return json.loads(path.read_text(encoding="utf-8")).get("version_id")


def _write_final(ctx: RunContext) -> dict[str, Any]:
    from .final import Candidate, build_final

    with ctx.tree.lock:
        top = list(ctx.tree.top_list)
    limit = int(ctx.cfg["ensemble"]["max_candidates"])
    cands = [Candidate(rank=i, node_id=n.id, score=float(n.score), out_dir=ctx.executor.run_dir(n.id) / "out", code=n.code)
             for i, n in enumerate(top[:limit], 1) if n.score is not None]
    if getattr(ctx.control,"name",None)=="adaptive":
        import multiprocessing
        # Heavy ensemble search and file assembly must not overrun the hard deadline.
        # The child only switches final/ after completing the entire artifact.
        proc=multiprocessing.get_context("spawn").Process(target=build_final,
            args=(ctx.paths.root,cands,ctx.direction.higher_is_better,ctx.cfg["final"]),
            kwargs={"spec":ctx.task_spec,"deadline":ctx.deadline})
        if ctx.time_left()>1 and cands:
            proc.start()
            proc.join(timeout=max(0,ctx.time_left()-.25))
            if proc.is_alive():
                proc.kill()
                proc.join()
            elif proc.exitcode:
                log.warning("ensemble worker exited with code %s; keeping published result",proc.exitcode)
        info=latest(ctx.paths.root) or {"available":False}
    else:
        info = build_final(ctx.paths.root, cands, ctx.direction.higher_is_better, ctx.cfg["final"], spec=ctx.task_spec)
    log.info("final result: %s (%s); %s", info.get("kind"), info.get("method"), info.get("reason"))
    return info


def exit_code_for(status: dict[str, Any]) -> int:
    s = status.get("status")
    if s == "success":
        return EXIT_OK
    if s == "no_valid_solution":
        return EXIT_NO_SOLUTION
    if s == "invalid_input":
        return EXIT_INVALID
    return EXIT_FAILED


def _bool(v: str) -> bool:
    low = v.strip().lower()
    if low in ("true", "1", "yes"):
        return True
    if low in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m superagent.run", description="Autonomous classification modelling agent.")
    p.add_argument("--task-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config")
    p.add_argument("--time-limit-secs", type=int)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--higher-is-better", type=_bool)
    p.add_argument("--num-gpus", type=int)
    p.add_argument("--num-cpus", type=int)
    p.add_argument("--parallel", type=int)
    p.add_argument("--keep-artifacts-top-k", type=int)
    p.add_argument("--pretrained-mode", choices=["catalog", "open", "off"])
    p.add_argument("--seed", type=int)
    return p


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        return EXIT_INVALID if exc.code else EXIT_OK
    intr = _Interrupt()
    previous: dict[int, Any] = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda *_a: intr.trigger())
    try:
        status = run_task(
            task_dir=args.task_dir, output_dir=args.output_dir, config=args.config, time_limit_secs=args.time_limit_secs,
            max_steps=args.max_steps, higher_is_better=args.higher_is_better, num_gpus=args.num_gpus, num_cpus=args.num_cpus,
            parallel=args.parallel, keep_artifacts_top_k=args.keep_artifacts_top_k, pretrained_mode=args.pretrained_mode,
            seed=args.seed, _interrupt=intr,
        )
    except InvalidInputError as exc:
        print(f"superagent: invalid input: {exc}", file=sys.stderr)
        return EXIT_INVALID
    finally:
        for sig, h in previous.items():
            signal.signal(sig, h)
    return exit_code_for(status)


if __name__ == "__main__":
    sys.exit(main())
