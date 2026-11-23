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
| `superagent_adapter/course/front/` | Course input: a spec plus a raw folder becomes a Core task directory. No model call |
| `superagent_adapter/course/back/` | Course result: check `final/`, score private labels on request, write `outcome.json` |
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

## Course tasks

`course prepare` reads `configs/course/headlines.json` or `coupon.json` and a raw data folder. It writes a new task directory and keeps test labels in a separate private directory. `course check` only verifies that this Core can be launched. `course collect` and `course evaluate-final` read a finished Core output. `course run` is the command that starts the Core and therefore calls the model.

```bash
python -m superagent_adapter.cli course prepare \
  --spec configs/course/headlines.json --source-dir /absolute/raw/headlines_ctr_clean \
  --task-dir /absolute/prepared/headlines/task --private-dir /absolute/prepared/headlines/private
```

New input handling goes in `course/front/`. New result checks and scores go in `course/back/`.

### Interactive page

```bash
adapter/scripts/course_ui.sh                      # drag a data folder into the terminal when asked
adapter/scripts/course_ui.sh --folder /abs/headlines_ctr_clean --time-limit-secs 3600
```

The script uses `../MLEvolve/env_superagent/bin/python` (override with `SUPERAGENT_PYTHON`) and runs
`python -m superagent_adapter.cli course ui`. The page scans the folder, writes `spec.json` with the Core's
`judgement` model (endpoint, model and key variable from the Core configuration, keys from the repository
`.env`), shows it for acceptance or editing, prepares the task, runs preflight, asks for time and resources,
and starts the Core only after a separate confirmation. It then collects the result, optionally scores the
sealed test labels, and lists the prediction CSV, scripts, logs and `outcome.json`.

Each run gets `adapter/outputs/course/<folder>/<timestamp>/` with `spec.json`, `task/`, `private/`, `run/`
and `front/logs/llm_calls.jsonl` (the spec writer's own calls). Test files contribute only headers, row counts
and shapes to the model prompt. `--no-llm` writes the spec by rules alone; `--yes` accepts every default.

## Tests

```bash
pytest
```
