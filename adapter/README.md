# superagent-adapter

Connects the SuperAgent Core to task sources and evaluation. The Core is used only through its command line
(`python -m superagent.run`); this project never imports Core modules.

## Layout

| Module | Purpose |
|---|---|
| `superagent_adapter/common/task_dir.py` | Builds a Core task directory from the public part of a split and checks that no file from the private part (the answers) reached it |
| `superagent_adapter/common/core_runner.py` | Builds the Core command line and runs it as a child process |
| `superagent_adapter/common/submission.py` | Lists the prediction files of a Core output (primary result first) and converts them into submission files |
| `superagent_adapter/common/costs.py` | Recomputes the model cost of a run from its token logs with a price table |
| `superagent_adapter/mlebench/grader.py` | Grades prediction files with the official MLE-bench grader (`mlebench grade-sample`) |
| `superagent_adapter/batch.py` | Runs several tasks or seeds from a plan file, grades them and summarises the scores |
| `superagent_adapter/analysis.py` | Summarises a Core output (operators, selection modes, failures, usage, grades, diagnostics) |
| `superagent_adapter/cli.py` | Command-line entry |
| `launchers/colab/` | Scripts to run the Core on Google Colab and keep the session alive |
| `examples/` | Example price table and batch plan |

## Primary result

The primary result is `final/predictions.csv` of the Core output (interface 1.1). For outputs without
`final/` (interface 1.0) it is `best/predictions.csv`. `best/`, `ensemble/` and, on request, `top/` are graded
for comparison only.

## Commands

```bash
pip install -e .
```

```bash
python -m superagent_adapter.cli prepare --public <prepared>/public --private <prepared>/private --task-dir task
```

```bash
python -m superagent_adapter.cli run --task-dir task --output-dir core_out --config config.yaml \
  --core-python <python with superagent> --cwd <dir with .env> --time-limit-secs 28800 \
  --higher-is-better false --num-gpus 1 --num-cpus 12 --parallel 2 --pretrained-mode catalog
```

```bash
python -m superagent_adapter.cli grade --core-output core_out --sample task/data/sample_predictions.csv \
  --data-dir <mlebench data dir> --results results --competition <competition id> --mlebench-bin <mlebench>
```

```bash
python -m superagent_adapter.cli batch --plan examples/batch_plan.example.yaml --results batch_results.json
```

```bash
python -m superagent_adapter.cli cost --core-output core_out --prices examples/prices.example.yaml
```

```bash
python -m superagent_adapter.cli analyze --core-output core_out --grades results/grades.json --out analysis
```

`prepare` and `grade` can run on a different machine from `run`, so the answers never need to be on the
machine that runs the Core.

## Tests

```bash
pytest
```

## Course tasks and delivery

The course-specific workflow lives in `superagent_adapter.course` and is available through
`python -m superagent_adapter.cli course ...` (or `python -m superagent_adapter.course ...`).
It adds CSV/NPY preparation, immutable input checks, Core 1.1 final-result collection, private-label
evaluation, run provenance, unified outcome records and a file-location index for handoff.
The team writes the course report; Report Agent development is paused. `course run` does not
generate a report by default. The existing report entry remains available only when explicitly requested.
The existing MLE-bench commands retain their behavior.

See [course integration and usage](docs/COURSE_ADAPTER.md),
[historical report design](docs/COURSE_REPORT_AGENT.md) and [offline acceptance](docs/COURSE_ACCEPTANCE.md).
Real model calls, training and official scoring were not run during this integration.
