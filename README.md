# SuperAgent

SuperAgent develops single-model classifiers from a task directory. The isolated workflow independently scores each candidate, tests model reload and updates the published submission whenever a strictly better verified model is ready.

- [Core](core/) specifies task inputs, candidate behavior and model packages.
- [Adapter](adapter/) launches Core, exports submissions and prices API receipts.
- [Colab launchers](adapter/launchers/colab/) package a task and run it through Adapter.

## Install

Use Linux or WSL, Python 3.11 or newer and bubblewrap with user namespaces enabled. On Ubuntu or Debian, install the OS package with `sudo apt-get install bubblewrap`. Core checks the isolation boundary before running a task.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m superagent.run --help
python -m superagent_adapter.cli --help
```

The standard installation supports tabular, text and classical image classifiers. Install any additional libraries required by your selected methods in the same environment. The [full runner requirements](core/requirements.txt) list optional model and local embedding libraries.

## Run a course task

Provide a task directory with `description.md`, a `task.yaml` contract and the data files named by the contract. Use the task's announced validation split and metric. If the task supplies no validation split, set aside a fixed validation partition from the labeled training data before launching. Keep test answers outside the task and reserve them for final evaluation. The isolated scorer supports `accuracy` and `log_loss`. The contract describes CSV or TFRecord inputs and the sample submission schema. [Task contract](core/README.md#task-directory)

Run these commands from the repository root after setting `DEEPSEEK_API_KEY` in the shell or in a local `.env`. Point `SAMPLE` to the sample named in your task contract. Use a new, empty output directory for each run.

```bash
TASK=/path/to/course-task
OUT=/path/to/course-run
SAMPLE="$TASK/data/sample_predictions.csv"
python -m superagent_adapter.cli run --task-dir "$TASK" --output-dir "$OUT" \
  --config core/configs/single_model.yaml
python -m superagent_adapter.cli submit --core-output "$OUT" \
  --sample "$SAMPLE" --output "$OUT/submission.csv"
python -m superagent_adapter.cli cost --core-output "$OUT" \
  --prices adapter/examples/prices.example.yaml
```

Adapter delegates to `python -m superagent.run`, which accepts the same task, output and config arguments. Pass `core/configs/single_model.yaml` to either entry point to select independent scoring and isolated reload. The installed Core defaults to the legacy workflow when this profile is omitted. `status.json` reports the run outcome and published version. Exit code `2` means the run produced no valid submission.

## Reload and retain the result

After the run stops, resolve `final/` once and retain that complete version and `input_views/inference/`. The version contains the model and preprocessing, solution code, validation and test predictions, and the verification and file manifests. Reload in the runner environment:

```bash
VERSION=$(realpath "$OUT/final")
mkdir -p "$OUT/reloaded"
python "$VERSION/solution.py" --predict-only --model-dir "$VERSION/model" \
  --data-dir "$OUT/input_views/inference" --output-dir "$OUT/reloaded"
```

## Enable DeepSeek and local retrieval

Copy the environment template to `.env` in the launch directory and set `DEEPSEEK_API_KEY`. The configured route uses `deepseek-flash` at `https://api.deepseek.com/chat/completions`. Use a course-approved model and the official standard rates required by the course. [Environment template](core/.env.example)

Optional local semantic retrieval uses Qwen3 embeddings. Install the encoder and prepare its weights in the runner environment. Keyword retrieval remains available when the encoder is absent. [Qwen3 embedding model](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

```bash
python -m pip install './core[embedding]'
python -m superagent.fetch_models --only experience_embedding
```

Keep `logs/decisions.jsonl`, `report/run_summary.json` and the API receipts in `logs/operations.sqlite` with the course result. Use Adapter `cost` during and after the run to track priced and unresolved usage. The caller controls the course's CNY 20 limit per task and the task deadline. Send SIGINT or SIGTERM to stop the run and retain the last complete publication. Core applies limits to individual scripts and API requests.

[MIT license](LICENSE)
