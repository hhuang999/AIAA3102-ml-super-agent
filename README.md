# SuperAgent

SuperAgent generates, trains and improves Python models for classification tasks. It scores declared tasks against fixed validation labels and publishes a submission as soon as it has a verified candidate. The default controller can compare a new draft, a code modification and continuation from an eligible checkpoint.

| Directory | Start here |
|---|---|
| [Core](core/) | Task contract, generation, runner, scoring and search |
| [Adapter](adapter/) | Task preparation, Core launch, submission checks, grading and cost reports |
| [Colab launchers](adapter/launchers/colab/) | Optional upload, entry point and snapshots |

## Install on Linux or WSL

Use Python 3.11 or newer. The CPU example installs without a GPU framework.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e './core[demo]' -e ./adapter
```

For generated scripts that need additional model libraries, install the relevant dependencies from [core/requirements.txt](core/requirements.txt) in the same runner environment. Supply the appropriate PyTorch build for a CUDA task. Generated scripts run under the account launching the agent. Use an isolated account or container for untrusted jobs.

## Check the installation

Check both command-line entry points before configuring credentials.

```bash
python -m superagent.run --help
python -m superagent_adapter.cli --help
```

## Configure DeepSeek chat and local Qwen embeddings

Copy [core/.env.example](core/.env.example) to `.env` at the repository root and set `DEEPSEEK_API_KEY` to a key from DeepSeek. Generation and judgement use `deepseek-flash` at `https://api.deepseek.com/chat/completions`. Leave `SUPERAGENT_LLM_BASE_URL` unset for the official route. [DeepSeek API documentation](https://api-docs.deepseek.com/)

From the repository root, install the local encoder and prepare its weights once in the runner environment:

```bash
python -m pip install -e './core[embedding]'
python -m superagent.fetch_models --only experience_embedding
```

The setup command downloads `Qwen/Qwen3-Embedding-0.6B` and checks a local forward pass. During a run, experience retrieval encodes text on the CPU using the cached weights. Set `SUPERAGENT_PRETRAIN_DIR` to share a cache between runs. If the weights or encoder are unavailable, Core logs a warning and uses keyword retrieval. [Qwen model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

## Run the Iris classification example

Run from the same directory and keep task test labels outside the task directory.

```bash
python core/examples/make_example_tasks.py --out example_tasks --tasks tabular_small --contract
python -m superagent.run --task-dir example_tasks/tabular_small/task \
  --output-dir outputs/iris --config core/configs/quickstart.yaml \
  --time-limit-secs 600 --max-steps 2
python core/examples/score_predictions.py outputs/iris/final/predictions.csv \
  example_tasks/tabular_small/test_labels.csv --metric accuracy
```

To launch through Adapter, replace the Core command with `python -m superagent_adapter.cli run` and use the same arguments with a fresh output directory. The quickstart config uses one CPU worker, a single initial script, and single-model delivery. A live run's first valid result depends on the provider, environment and the code generated.

## Read a completed run

- `final/predictions.csv` is the submission and `final/source.json` identifies its source. The pointer updates when a verified candidate is published. On Linux preserve symlinks when copying the complete run directory, or export the CSV via Adapter.
- `best/` identifies the best single model. `status.json` reports the outcome and whether a final file exists. Exit `2` means the run has no valid submission.
- `logs/decisions.jsonl`, `logs/llm_calls.jsonl` and `logs/accounting.json` record decisions, HTTP attempts and known cost with unknown exposure. `report/run_summary.json` summarizes the run. A late receipt in `logs/late_usage.jsonl` can arrive after a report was written.
- Use the [Core reference](core/README.md) for task and model artifact formats, and the [Adapter guide](adapter/README.md) for prepared datasets, submission and grading.

[MIT license](LICENSE)
