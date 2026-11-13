# SuperAgent

SuperAgent is an autonomous agent that develops classification models. Given a task directory (a task
description, training data and unlabeled test data), it writes complete modelling scripts, runs them, reads
the results, decides what to try next, and returns the best solution together with a full record of every
decision. It handles binary and multi-class classification on tabular, text and image data.

The repository has two independent projects:

| Project | Purpose |
|---|---|
| [`core/`](core/) | The agent (`superagent`): search, code generation, execution, assessment, resource control, final result |
| [`adapter/`](adapter/) | Connects the agent to task sources and evaluation (`superagent_adapter`): task directories, runs, grading, batch runs, cost recomputation, run analysis, Colab launchers, course delivery and Report Agent |

The adapter uses the agent only through its command line.

## Data flow

```
external task (e.g. an MLE-bench competition)
   │  adapter: prepare            public files → task directory; isolation check against the answers
   ▼
task_dir/ ─ description.md, data/ (training data, unlabeled test data, sample_predictions.csv)
   │  adapter: run  →  python -m superagent.run --task-dir ... --output-dir ...
   ▼
core
   ├─ startup     prepare data, clean the description, metric direction, data preview, method catalog,
   │              hardware detection, controller
   ├─ search      worker threads share one search tree; each step:
   │                select a node (tree_walk / elite_pick / synthesis trigger) → choose an operator
   │                (draft, refine, repair, retrospect, borrow, synthesis) → step limits from the controller
   │                → plan and code (LLM) → code review → smoke run → full run → assessment
   │                (score marker, prediction format and content, validation files, leak check)
   │                → per-step reward → bookkeeping
   └─ finalize    ensembles of top predictions → metric identified from validation files →
                  primary result chosen on validation data → report with diagnostics → status
   ▼
output_dir/ ─ final/ (primary result), best/, top/, ensemble/, logs/, report/, status.json, runs/
   │  adapter: grade / analyze / cost
   ▼
scores against held-out answers, run analysis, recomputed cost
```

## Modules

| Module | Role |
|---|---|
| `core/superagent/run.py` | Command line, startup, finalize |
| `core/superagent/search.py`, `selection.py`, `scheduling.py`, `chain.py` | Search loop, node selection, operator choice, plateaus, synthesis trigger, chains and rewards |
| `core/superagent/operators.py`, `planner.py`, `codegen.py`, `patching.py`, `prompts/` | The six operators, structured plans, code generation, patches, prompt wording |
| `core/superagent/review.py`, `executor.py`, `assessment.py`, `checks.py` | Pre-run review, script execution with smoke runs, result assessment |
| `core/superagent/metrics.py`, `final.py`, `ensemble.py` | Validation files, metric identification, validated primary result, ensembles |
| `core/superagent/control/` | Controller: script time budgets, step admission, cost cap, GPU memory hints, control log |
| `core/superagent/experience.py`, `bookkeeping.py`, `report.py` | Experience store, best/top/status files, run summary and diagnostics |
| `core/superagent/llm/` | LLM gateway: retries, structured output, call log, cost |
| `adapter/superagent_adapter/common/` | Task directory and isolation check, Core command line, submissions, cost recomputation |
| `adapter/superagent_adapter/mlebench/` | Grading with the MLE-bench grader |
| `adapter/superagent_adapter/batch.py`, `analysis.py`, `cli.py` | Batch runs, run analysis, command line |
| `adapter/launchers/colab/` | Running on Google Colab |

## Quick start

```bash
cd core && pip install -r requirements.txt && pip install -e .
```

```bash
python examples/make_example_tasks.py --out example_tasks
```

```bash
python -m superagent.run --task-dir example_tasks/tabular/task --output-dir outputs/tabular --time-limit-secs 1800 --higher-is-better true
```

The LLM key is read from the environment or a `.env` file in the start directory (see `core/.env.example`).
See [`core/README.md`](core/README.md) and [`adapter/README.md`](adapter/README.md) for details.

## License

MIT, see [LICENSE](LICENSE).

## Course integration

The course workflow is integrated into the existing Adapter package, with explicit Core 1.1 support
and legacy 0.1 regression coverage. See [course instructions](adapter/docs/COURSE_ADAPTER.md)
and [offline verification](adapter/docs/COURSE_ACCEPTANCE.md). The Core implementation is unchanged.
