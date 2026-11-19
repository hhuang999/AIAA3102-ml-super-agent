# superagent-adapter

Connects the SuperAgent Core to task sources and evaluation. The Core is used only through its command line
(`python -m superagent.run`); this project never imports Core modules.

## Layout

| Module | Purpose |
|---|---|
| `superagent_adapter/common/task_dir.py` | Builds a Core task directory from the public part of a split and checks that no file from the private part (the answers) reached it |
| `superagent_adapter/common/core_runner.py` | Builds the Core command line and runs it as a child process |
| `superagent_adapter/common/submission.py` | Lists the prediction files of a Core output (primary result first) and converts them into submission files |
| `superagent_adapter/common/costs.py` | Reads transactional usage receipts and prices them with a caller-provided table |
| `superagent_adapter/mlebench/grader.py` | Grades prediction files with the official MLE-bench grader (`mlebench grade-sample`) |
| `superagent_adapter/batch.py` | Runs several tasks or seeds from a plan file, grades them and summarises the scores |
| `superagent_adapter/analysis.py` | Summarises a Core output (operators, selection modes, failures, usage, grades, diagnostics) |
| `superagent_adapter/cli.py` | Command-line entry |
| `launchers/colab/` | Scripts to run the Core on Google Colab and keep the session alive |
| `examples/prices.example.yaml` | Price table for cost queries |

## Primary result

The primary result is `final/predictions.csv`. Adapter pins one resolved publication version so the
prediction and source metadata agree. Legacy 1.0 runs can use `best/predictions.csv` as their primary result.
Submission conversion matches the sample by the first column (ID), preserves ID and label strings, and
rejects missing, unexpected, or duplicate IDs. A Core run that ends without a submission exits with code `2`
through the adapter and Colab wrapper. On POSIX, the adapter forwards SIGINT and SIGTERM to Core to allow
it to finalize its latest valid submission.

## Commands

Run the commands from the repository root.

```bash
python -m pip install -r requirements.txt
```

```bash
python -m superagent_adapter.cli prepare --public <prepared>/public --private <prepared>/private --task-dir task
```

When using `--contract`, the adapter rewrites references to the renamed sample in `sample` and
`inference_inputs`. An isolation check requires a private split with files to compare.

```bash
python -m superagent_adapter.cli run --task-dir task --output-dir core_out \
  --config core/configs/single_model.yaml --max-steps 5
python -m superagent_adapter.cli submit --core-output core_out \
  --sample task/data/sample_predictions.csv --output submission.csv
```

```bash
python -m superagent_adapter.cli grade --core-output core_out --sample task/data/sample_predictions.csv \
  --data-dir <mlebench data dir> --results results --competition <competition id> --mlebench-bin <mlebench>
```

```bash
python -m superagent_adapter.cli batch --plan "$PLAN" --results batch_results.json
```

Set `PLAN` to your YAML batch plan. [Batch plan format](superagent_adapter/batch.py)

```bash
python -m superagent_adapter.cli cost --core-output core_out --prices adapter/examples/prices.example.yaml
```

The report includes the latest cursor, attempt states, known token counts and the known subtotal in
each priced currency. Pass `--after-cursor RUN_ID:RECEIPT_SEQ` to list changed attempt IDs since that
cursor while retaining totals for the run. Pass `--process-exited` only after confirming Core has
stopped if it could not record a final status. Adapter reports unresolved usage or prices as unknown.
The included table applies the course's standard input price to all input tokens, including cache hits. Use a separate dated price table for provider reconciliation.

```bash
python -m superagent_adapter.cli analyze --core-output core_out --grades results/grades.json --out analysis
```

Put `DEEPSEEK_API_KEY` in the launch directory's `.env` for the official chat API. For optional
semantic retrieval, set the same `SUPERAGENT_PRETRAIN_DIR` for setup and launch. [Local retrieval setup](../README.md#enable-deepseek-and-local-retrieval)

`prepare` and `grade` can run on a different machine from `run`, so the answers never need to be on the
machine that runs the Core.
