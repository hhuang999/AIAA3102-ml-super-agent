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
python -m pip install './core[demo]' ./adapter
python -m superagent.run --help
python -m superagent_adapter.cli --help
```

Install any extra model libraries required by your chosen methods in the same environment. The [full runner requirements](core/requirements.txt) list optional model and local embedding libraries.

## Run, export and reload the Iris example

The packaged `tab_linear@1` recipe trains a scikit-learn pipeline on Iris. These commands use one candidate slot and zero API requests. Use a new output directory on each run.

```bash
python core/examples/make_example_tasks.py --out example_tasks --tasks tabular_small --contract
python -m superagent.run --task-dir example_tasks/tabular_small/task \
  --output-dir outputs/iris --config core/configs/single_model.yaml --max-steps 1
python -m superagent_adapter.cli submit --core-output outputs/iris \
  --sample example_tasks/tabular_small/task/data/sample_predictions.csv --output outputs/iris-submission.csv
python -m superagent_adapter.cli cost --core-output outputs/iris \
  --prices adapter/examples/prices.example.yaml
python core/examples/score_predictions.py outputs/iris-submission.csv \
  example_tasks/tabular_small/test_labels.csv --metric accuracy
python outputs/iris/final/solution.py --predict-only --model-dir outputs/iris/final/model \
  --data-dir outputs/iris/input_views/inference --output-dir outputs/iris-reloaded
```

The reload command for this bundled recipe uses only the released model package and the projected inference inputs. `final/` points to one complete version under `submissions/`. Preserve symlinks when moving the run, or copy the resolved version together with its inference inputs. The source, manifest and verification receipt are included in that version.

To launch through Adapter, use `python -m superagent_adapter.cli run` with the same Core arguments and a fresh output directory. `status.json` reports the latest published version and the run stop reason. Exit code `2` means no valid submission was published.

## Enable DeepSeek and local retrieval

Copy [the env example](core/.env.example) to `.env` in the launch directory and set `DEEPSEEK_API_KEY`. The official route is `deepseek-flash` at `https://api.deepseek.com/chat/completions`. To check the generated-script path on Iris, copy `single_model.yaml`, set `search.local_recipes: false` and `codegen.staged_drafting: false`, and pass the new configuration with `--max-steps 1`. [DeepSeek Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)

Optional local semantic retrieval uses Qwen3 embeddings. Install the encoder and prepare its weights in the runner environment. Keyword retrieval remains available when the encoder is absent. [Qwen3 embedding model](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

```bash
python -m pip install './core[embedding]'
python -m superagent.fetch_models --only experience_embedding
```

Use `logs/decisions.jsonl` to inspect candidate choices, `report/run_summary.json` for outcomes, and Adapter `cost` for prices and unknown usage. A caller can stop a job with SIGINT or SIGTERM. Core completes or preserves the last atomic publication and applies limits to individual scripts and API requests.

[MIT license](LICENSE)
