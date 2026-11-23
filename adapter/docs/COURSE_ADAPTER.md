# Course Adapter — Core 1.1

**Scope update, 2026-11-13:** the team writes the course report. Report Agent development is
paused. The normal course workflow produces evidence, results and file locations without
generating a report or invoking a report model. Existing report commands remain opt-in for
compatibility; they are not a required delivery step.

This module integrates the previously verified standalone course Adapter into the existing
`superagent-adapter` distribution (0.3.1). The Core implementation is unchanged. New course
specifications select Core interface **1.1**; explicitly declared **0.1** specifications retain
their legacy behavior and regression tests. Package version and interface version are different.

## Ownership and entry points

- `superagent_adapter.course.prepare`: CSV/NPY conversion, exact row IDs and input hashes.
- `runtime` / `runner`: read-only preflight, source/config snapshots and Core subprocess lifecycle.
- `core11` / `collect`: primary output validation, self-contained delivery and final evaluation.
- `consistency` / `reporting` / `report_llm`: evidence projection, report drafting, optional writer.
- `outcome`: separate Core/evaluation/report states and report-to-evidence binding.

The existing MLE-bench, batch, cost and analysis commands retain their behavior. Course accuracy
evaluation is separate from the MLE-bench grader. Core owns search, autonomous solution selection
and resource control; the course module does not implement a second scheduler or budget manager.

Install the Adapter with the Python environment intended for the Core:

```bash
python -m pip install -e ./adapter
python -m superagent_adapter.cli course --help
```

Both `python -m superagent_adapter.cli course ...` and
`python -m superagent_adapter.course ...` invoke the same course workflow.

## Prepare and inspect

The two released course specifications are `adapter/configs/course/headlines.json` and
`adapter/configs/course/coupon.json`. Both request interface 1.1 and `final_choice: final`.
They retain the original feature order and declared class-label accuracy metric.

```bash
python -m superagent_adapter.cli course prepare \
  --spec adapter/configs/course/headlines.json --source-dir /absolute/raw/headlines_ctr_clean \
  --task-dir /absolute/prepared/tasks/headlines_ctr_clean \
  --private-dir /absolute/prepared/private/headlines_ctr_clean

python -m superagent_adapter.cli course check \
  --task-dir /absolute/prepared/tasks/headlines_ctr_clean \
  --private-dir /absolute/prepared/private/headlines_ctr_clean \
  --run-dir /absolute/runs/new-run
```

Output locations must be new and separate from the raw input and private labels. Core 1.1 receives
`description.md` and `data/` with train/val/test CSVs and `sample_predictions.csv`.
Generated candidate scripts read `./data` and write `./out/predictions.csv`.
`scripts/prepare_course.py` can prepare both tasks and compare every converted row with its source;
it does not run models or calculate benchmark scores.

Preflight checks hashes, paths, config readability, Python >=3.11 and a statically discoverable
`superagent.run`. If using a source checkout without installing the Core, use an **absolute**
`PYTHONPATH` containing the repository's `core` directory. The child runs from a new run directory,
so relative import paths are not reliable. The checks never import or start the Core. A successful
check does not prove model availability, dependency completeness or authentication.

## Run, collect and report

`course run` starts the real Core and therefore needs a separately agreed model/compute plan.
It accepts the same task/private/run directories, plus `--core-config` and `--seed`.
It writes `adapter/outcome.json` with result states and a `files` index. It does not generate a
report by default. `--generate-report` explicitly requests the existing deterministic draft;
`--report-writer-config` also explicitly enables report generation and may call the configured model.
Credentials are inherited through the environment; the launcher does not copy a secret `.env`
file into the new run directory. The Core also supports its own cwd-based `.env` loading.

`course collect` can collect an already finished run; it refuses an active run or a reused delivery
directory. `course evaluate-final` explicitly evaluates the sealed selected predictions against
private labels. Evaluation is not automatic unless `course run --evaluate-final` is requested.
Formal evaluation remains pending team approval. `formal_evaluation_eligible` stays false; the
module does not certify a run as satisfying all course requirements.

The `files` index locates the selected prediction CSV, selected member scripts, combination
script, selection metadata, execution logs, runtime snapshot, collection evidence and (only
after evaluation) final evaluation JSON. Invalid solutions have no deliverable prediction/script
entries. These are absolute paths for the current run location, not a portable replay command.
After moving a run, regenerate the outcome from its evidence to refresh locations.

Evaluation holds an exclusive `adapter/final-evaluation.lock` and commits the score record before
updating the evidence. If only the evidence write failed, a retry verifies the record's task,
collection and prediction hashes and restores the evidence without recalculating the score.
Malformed or mismatched existing records are never overwritten. An abrupt process kill can leave
a lock: verify that no evaluator is active before manually removing that exact lock and retrying.
This is crash recovery and same-account coordination, not protection against deliberate file edits.

```bash
python -m superagent_adapter.cli course report \
  --run-dir /absolute/runs/completed-run --output-dir /absolute/reports/new-report
```

An optional writer configuration names `base_url`, `model`, `api_key_env` and `timeout`. Without
it, no report model is called. Writer errors fall back to a deterministic draft and are recorded.
An actual writer must use the team's agreed gateway; cost control is owned by that team module.
Writer call/usage records remain separate evidence and must be included in final accounting.

## Core 1.1 result contract

The Adapter validates `status.json` (`best_score`, final fields), `best/score.json`, and the
selected `final/source.json`, prediction files and member scripts. It preserves raw Core status
and summary fields rather than pretending they follow the old `best_metric` schema.

- `final/` is the autonomous Core selection. The Adapter does not compare candidates on test labels.
- Single final results must match the best single node, score, script and predictions.
- Whenever an ensemble includes the best single node, that member's script and predictions must
  also match `best/`; matching node IDs and scores alone is insufficient.
- Ensemble results bind member identities, ranks, scores, weights, scripts and class-label outputs.
  The recorded combination is independently checked using Core's raw-string voting and member-order
  tie rule. Its exact format must have normalization disabled. This is a file check, not script execution.
- Missing `final/` permits `best_partial` only for an incomplete run whose status declares no final
  result. A successful run with missing final output, or a malformed final output, cannot silently
  become a different solution.
- Core's documented member `artifacts` link may point only to that node's own
  `runs/<node>/out/model` directory. The delivery materializes it into ordinary files. Other links,
  linked ancestors and nested links are rejected. Raw per-run results are not required to keep the
  collected delivery usable.
- The full member scripts, `combine.py` and metadata are retained. Recombining saved predictions,
  rerunning full member scripts and inference-only prediction are different capabilities. Collection
  does not claim that any of these has been execution-tested for a real run.

Delivery files are hashed before final evaluation. Changes invalidate scoring. `outcome.json`
separately reports Core, evaluation and report state. A new score makes an older report stale;
a failed report retry retains the last successful report and records the failed attempt.

## Report evidence and provisional decisions

The Report Agent consumes collected facts, verified selection metadata and a bounded projection
of the Core summary, including `final`, `control` and `diagnostics`. It separates best single
validation score, final-selection validation score and final test accuracy. Missing scores remain
unknown. Conflicting summaries are retained in the raw archive but excluded from writing context.
Raw `task.data_preview`, code and sample-level fields are excluded from the report context.
Filtering field names is not a guarantee that arbitrary free text is fully sanitized.

The following remain marked provisional in manifests and reports:

1. Original train/val splits stay separate and both are development data visible to Core. The Adapter
   does not merge or re-split; Core uses its default internal validation behavior. Team agreement
   may change this later.
2. Private test labels are outside the input task directory on Assignment 1. Same-account path
   separation is not operating-system-enforced isolation; deployment policy remains to be agreed.
3. Full scripts are accepted. An available model file does not itself establish inference-only support.
4. Report writing belongs to the team. The optional historical Markdown/JSON generator is retained
   but disabled by default; Adapter delivery consists of checked artifacts, file locations and results.

## Offline checks

Install only the Adapter test dependencies; full model-training dependencies are unnecessary:

```bash
python -m pip install -e './adapter[test]'
# Linux; use ';' between paths in Windows PYTHONPATH.
export PYTHONPATH="$PWD/adapter:$PWD/core"
python -m pytest adapter/tests -q
python adapter/scripts/smoke_core11.py --output-dir /absolute/new-offline-smoke
```

The Core 1.1 fixture calls real bookkeeping, final selection and report builders with handmade
nodes and synthetic arrays, a null gateway and a network guard. It never runs search or trains
a model. The legacy smoke script remains for explicit 0.1 regression checks. See
`COURSE_ACCEPTANCE.md` for the verified baseline and exact outcomes.
The smoke script also skips the optional Report Agent by default; its historical mock-writer
coverage can be explicitly requested with `--generate-report`.
