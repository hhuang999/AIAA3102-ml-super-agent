# SuperAgent Core

The [root README](../README.md) gives installation, smoke, test and live example commands. Core runs as `python -m superagent.run`. The main loop passes from `run.py` to `search.py`, then to generation and review, `executor.py`, independent assessment and publication.

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
```

Use `encoding: probability` with class names as columns to submit probability vectors. `accuracy` and `log_loss` are supported for declared tasks. For TFRecord Example shards, set `validation.format: tfrecord`, `glob`, `label` and optionally `id`. Without `id`, validation keys are `relative/shard:zero_based_index`. The profiler lists representative feature names and types, including JPEG/PNG when detected. Task labels and sample IDs are validated independently of an agent's reported score. Legacy tasks without `task.yaml` use legacy grading.

## Generation and runtime

- Keep public training data, the fixed validation split and unlabeled test data in the task. Store private test labels separately. `inference_inputs` should contain only inference inputs. Core reconstructs a clean inference directory for its replay. Use OS-level isolation for generated code that you do not trust.
- The draft can use a task-provided `baseline.py` or generate code. Search can refine, repair, explore a new branch or continue a verified trial. Review is enabled by default. The adaptive controller protects an estimated execution window before the publication reserve. `logs/decisions.jsonl` records estimates and rejected actions.
- Use `--config` to apply a partial YAML file. [Default settings](superagent/resources/default_config.yaml) describe supported keys. Defaults are merged with the file, then environment overrides and CLI overrides. Relative cache paths resolve from the launch directory. The process environment takes precedence over `.env`. Use distinct output directories for runs and a scheduler-provided `CUDA_VISIBLE_DEVICES` to restrict visible devices.
- `llm.role_policy` controls per-role thinking and output budgets. Defaults use a 16,384-token structured planner, 32,768-token patch and repair, 65,536-token whole script, 8,192-token ordinary review and 16,384-token sensitive review. Thinking counts toward the output budget. Truncated results, blank JSON and transport failures use different recovery paths. Recovery can reuse verified stages. 429/524 cool the route, and explicit authentication or balance failure stops paid sends.
- Chat cost caps are optional. Set `control.cost_cap.mode: enforce`, the limit and currency conversion for a capped run. Reservations estimate maximum request exposure. Missing usage stays unknown, including requests the client stopped waiting for. Review the late usage log after background responses complete.

## Inference and training artifacts

A candidate must write `out/predictions.csv`, `out/val_predictions.csv` and a portable `out/model/` package. The package contains everything necessary to reconstruct the model, preprocessing and class order. Declare real weight paths in `out/model/inference.json`, for example `{"version":1,"weight_files":["weights.npz"]}`. The file names depend on the framework.

Support `--predict-only --model-dir ... --data-dir ... --output-dir ...` in the solution. This path loads the saved model and writes the two prediction files using only declared inputs. Core compares predictions with the training run and checks that corrupt copies of the declared weights fail. Validation truth comes from the contract and is read by Core.

For resumable trials, see `superagent.training.save_torch` and `restore_torch` or `save_checkpoint` and `resume_path` for other frameworks. Call `restore_torch` before the first optimizer update so updates can be counted. Persist the completed epoch, optimizer state and cumulative steps, and report measured times for the next epoch and for validation and prediction. Each promotion requires evidence that state was restored from the selected parent and that training advanced. A verified prediction remains a candidate when its optional checkpoint is unusable. [Synthetic image reference implementation](examples/flower_baseline.py)

## Pretrained models and experience

Use `pretrained.mode: catalog` for verified local models, `open` for models that may download at runtime, or `off` for training from scratch. Run `python -m superagent.fetch_models --only MODEL_ID` with the same interpreter used by the runner. Snapshot-backed entries require an offline forward receipt and matching local weight hashes. Use the [method catalog](superagent/resources/method_catalog.yaml) to find IDs and runtime dependencies.

Experience retrieval loads Qwen3-Embedding-0.6B locally. Follow the [setup instructions](../README.md#configure-deepseek-chat-and-local-qwen-embeddings) before the first run. Set `experience.vector.device` to `cuda` to use an available GPU, or `experience.vector.enabled` to `false` for keyword-only retrieval. `logs/embedding_calls.jsonl` records local calls, tokens and compute time.

Task-owned `knowledge.yaml` can supply cards with `id`, `version`, `modality`, `method`, `counterexample` and `source`. Use cards for methods that generalize across tasks and describe when each method stops being useful. Keep dataset facts in the contract and description. Learned scores and failures belong in the run records.
