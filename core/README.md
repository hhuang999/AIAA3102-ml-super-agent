# SuperAgent Core

The [root README](../README.md) gives installation, generation, evaluation and reload commands. Use `python -m superagent.run --config core/configs/single_model.yaml` with a task and output directory for isolated single-model delivery.

## Task directory

```text
task/
  description.md
  task.yaml
  knowledge.yaml       # optional method cards
  baseline.py          # optional starting solution
  data/
    train.csv
    val.csv
    val_labels.csv
    test.csv
    sample_predictions.csv
```

Example classification contract. Paths are relative to prepared `data/`. The sample determines the exact prediction columns and test ID order.

```yaml
version: 1
metric: accuracy
classes: [setosa, versicolor, virginica]
sample: sample_predictions.csv
inference_inputs: [val.csv, test.csv, sample_predictions.csv]
validation:
  format: csv
  path: val_labels.csv
  id: id
  label: species
input_bindings:
  train: {format: csv, path: train.csv, id: id, label: species, features: [sepal_length, sepal_width]}
  validation: {format: csv, path: val.csv, id: id, features: [sepal_length, sepal_width]}
  test: {format: csv, path: test.csv, id: id, features: [sepal_length, sepal_width]}
```

Bind all features you want available to the solution. Each split specifies `path` or `glob`. Only the training binding has a `label`. For CSV columns containing relative paths to image or text files, list them under `file_features`. Core copies the named assets into the appropriate view. Set `shared_id_domain: true` when train and validation use the same ID domain.

For TFRecord Example shards, specify `format: tfrecord` and the feature names. The `validation` truth locator has its own `glob` and `label`. Omit the validation ID to use `relative/shard:zero_based_index`. Core projects fields without changing shard names or record order. The bundled Iris data and the conventional flower `train/val/test` TFRecord layout can infer bindings from `task.yaml` and the prepared data.

Use `encoding: probability` and sample columns in the exact `classes` order for probability outputs. `accuracy` accepts labels or probabilities. `log_loss` requires probabilities. The task's sample controls the output columns and test order. Keep test answers outside the task.

## Generation and runtime

- Core provides projected `data/` views and a manifest under `input_views/`. Training can read labelled training records and the public inference features. Reload can read the public inference features and the model. Core scores inside the trusted process.
- The controller prioritizes a supplied `baseline.py` and chooses eligible recipes or method hypotheses for other openings. Search considers repair, refinement, retrospection, borrowing and checkpoint promotion from committed evidence. `logs/task_facts.json` and `logs/method_matches.json` describe eligible inputs and methods. `logs/decisions.jsonl` tracks selected candidates and the evidence used.
- Merge settings through `--config`, environment values and CLI overrides. `run.max_steps` counts claimed operations, including failures. A resumed operation reuses its slot. The caller controls task-wide budgets and stops a run with SIGINT or SIGTERM. Use a fresh output directory to start a new search. The isolated profile rejects old task-wide time and cost options. Existing legacy profiles continue to read older task formats. [Configuration reference](superagent/resources/default_config.yaml)
- `llm.role_policy` and `llm.request_limits` specify thinking, response budgets and request timeouts. All recovery sends share the operation reserve. The gateway keeps completed stages when a route cools down, and handles known responses and missing usage independently. Adapter prices the transaction receipts in `logs/operations.sqlite`.

## Inference and training artifacts

A candidate must write `out/predictions.csv`, `out/val_predictions.csv` and a portable `out/model/` package. The package contains everything necessary to reconstruct the model, preprocessing and class order. Declare real weight paths in `out/model/inference.json`, for example `{"version":1,"weight_files":["weights.npz"]}`. The file names depend on the framework.

Support `--predict-only --model-dir ... --data-dir ... --output-dir ...` in the solution. Core scores the exported predictions first, then checks that isolation can reproduce both files and corrupt copies of the declared weights cause reload to fail. The score and reload verdict are separate. Core can reuse fitted state when a verified-score candidate needs a change confined to the body of `predict_only` and the repaired path reproduces the original predictions.

Each published `final/` version contains `solution.py`, `model/`, both prediction files, `source.json`, `verification.json`, `inference_inputs.json` and a content manifest. Use the resolved version and the inference view for replay. Publish history keeps the two latest full model versions.

For resumable trials, see `superagent.training.save_torch` and `restore_torch` or `save_checkpoint` and `resume_path` for other frameworks. Call `restore_torch` before the first optimizer update so updates can be counted. Persist the completed epoch, optimizer state and cumulative steps, and report measured times for the next epoch and for validation and prediction. Each promotion requires evidence that state was restored from the selected parent and that training advanced. A verified prediction remains a candidate when its optional checkpoint is unusable. [Synthetic image reference implementation](examples/flower_baseline.py)

## Pretrained models and experience

Use `pretrained.mode: catalog` for eligible local models or `off` for training from scratch. Run `python -m superagent.fetch_models --only MODEL_ID` with the same interpreter used by the runner. Isolated runs stage selected snapshots with a local forward receipt and matching weight hashes. The solution includes any assets it still needs in its model package for offline reload. [Method catalog](superagent/resources/method_catalog.yaml)

Experiment observations persist by operation ID throughout the run. Optional local semantic retrieval uses Qwen3 embeddings. Set `experience.vector.enabled: false` for keyword-only retrieval. `logs/embedding_calls.jsonl` records local calls, tokens and compute time. [Setup instructions](../README.md#enable-deepseek-and-local-retrieval)

Task-owned `knowledge.yaml` can supply cards with `id`, `version`, `modality`, `method`, `counterexample` and `source`. Use cards for methods that generalize across tasks and describe when each method stops being useful. Keep dataset facts in the contract and description. Learned scores and failures belong in the run records.
