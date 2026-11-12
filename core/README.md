# SuperAgent

SuperAgent is an autonomous agent that develops classification models. You give it a task directory (a
task description plus training data and unlabeled test data). Within a time budget, and without any human
input, it:

1. analyses the task and the data,
2. writes several different complete modelling solutions (each one a Python script),
3. runs every solution and checks the results independently,
4. decides what to do next: repair a broken solution, refine a working one, review the history of a branch and
   change direction, borrow a technique from another branch, or synthesise a new solution from the best ones,
5. writes out the primary result (the best solution, or an ensemble of top solutions when validation shows it
   is better), the best single solution, the trained model files, and a complete record of every decision,
   LLM call and token used.

It supports binary and multi-class classification on tabular, text and image data.

## How it works

```
task dir ──► startup: prepare data, clean description, metric direction, data preview, method catalog
               │
               ▼
          search loop (several worker threads sharing one search tree)
            ① select a node      tree_walk (upper confidence bound) or elite_pick (weighted top-k);
                                 synthesis starts directly when its conditions hold
            ② choose an operator draft | refine | repair | retrospect | borrow | synthesis
            ③ step limits        the controller sets the script time budget and GPU memory hint
            ④ generate           plan (structured planner) → code (staged / patch / full script)
            ⑤ pre-run checks     LLM code review, pretrained-policy scan
            ⑥ run                smoke run on a tiny subset, then the full run (own run directory,
                                 CPU/GPU slot, time limit)
            ⑦ assess             LLM reads the log → code re-checks score marker, file format, content,
                                 validation files, metric direction, zero score, suspicious scores (leak check)
            ⑧ reward and chain   per-step reward on the path; continue the chain or return to the root
            ⑨ bookkeeping        best solution, top list, status, logs, experience store
               │
               ▼
          finalize: ensembles of the top predictions → metric identified from the validation files →
                    primary result in final/ → run summary with diagnostics → final status

          controller (any time): may a new step start, should the search finish early (cost cap)
```

Design principles:

- **Code controls the process, the LLM provides content.** Which node to expand and which operator to use are
  decided by fixed rules (with seeded randomness). The LLM only writes plans and code and reads logs.
- **LLM self-reports are never trusted blindly.** Scores come from the `VALIDATION_SCORE:` line the script
  prints; prediction files are checked by code against `sample_predictions.csv`.
- **Every decision is logged** (`logs/decisions.jsonl`, `logs/llm_calls.jsonl`, `logs/search_tree.json`).
- **Decisions use task content only.** Methods are never chosen from task names or dataset identities; the
  method catalog is organised by data modality only.
- **One controller owns resource decisions.** Script time limits, whether a step may start, early finishing
  and the GPU memory hint all come from `superagent.control`, which can be replaced by a custom class.

## Installation

Python 3.11 or newer.

```bash
pip install -r requirements.txt
pip install -e .
```

For CPU-only machines you can install the CPU build of PyTorch first:
`pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision`.

Set the LLM key (OpenAI-compatible endpoint; the default is DeepSeek). Either export it or put it in a
`.env` file in the directory you start the run from (see `.env.example`; `.env` is git-ignored):

```bash
export DEEPSEEK_API_KEY=...                  # required, never written to any file
export SUPERAGENT_LLM_BASE_URL=...           # optional, overrides llm.base_url
```

Variables that are already set take precedence over `.env`. The name of the key variable is
configurable (`llm.api_key_env`). `AIHUBMIX_API_KEY` is only needed when the embedding model is called
through an API (`experience.vector.backend: api`, see below).

## Preparing a task directory

```
task_dir/
├── description.md              required: goal, data files, metric and its direction, submission format
└── data/                       required: training data and unlabeled test data, any layout
    ├── sample_predictions.csv  strongly recommended: first column = all test ids, other columns = predictions
    └── ...                     .zip archives are extracted automatically
```

`data/` is treated as read-only. `sample_predictions.csv` is used to check every prediction file (column names
and order, row count, id set, empty cells); without it these checks are skipped with a warning.

`examples/make_example_tasks.py` builds four small example tasks (tabular, tabular_small, image, text) from
datasets that ship with scikit-learn; the test labels are written outside the task directory, and
`examples/score_predictions.py` scores a prediction file against them.

```bash
python examples/make_example_tasks.py --out example_tasks
```

## Running

```bash
python -m superagent.run \
  --task-dir   example_tasks/tabular/task \
  --output-dir outputs/tabular \
  [--config my_config.yaml] [--time-limit-secs 43200] [--max-steps 500] \
  [--higher-is-better true|false] [--num-gpus 1] [--num-cpus 8] [--parallel 2] \
  [--keep-artifacts-top-k 5] [--pretrained-mode catalog|open|off] [--seed 42]
```

or from Python:

```python
from superagent import run_task
status = run_task(task_dir="...", output_dir="...", time_limit_secs=3600, higher_is_better=True)
```

The output directory must not exist or must be empty. Exit codes: `0` finished with at least one valid
solution, `2` finished without a valid solution, `1` internal error, `3` invalid input (bad configuration,
missing files, non-empty output directory, missing API key, `num_cpus < parallel`, unknown controller).
SIGINT/SIGTERM stop all running scripts, finalize the outputs and exit with `0` or `2`
(`stop_reason` = `interrupted`). Other stop reasons: `time_limit`, `max_steps`, `cost_limit` (cost cap in
enforce mode), `error`.

The run stops by itself before `--time-limit-secs`: no script runs into `run.finalize_reserve_seconds`, and
the last part of the budget is used to write the final results.

## Configuration

`superagent/resources/default_config.yaml` holds every option with its default and a comment. A user config
only lists the keys it changes, for example:

```yaml
run:
  time_limit_seconds: 7200
llm:
  generation:
    model: deepseek-flash
    reasoning_effort: low
search:
  initial_drafts: 2
experience:
  vector:
    device: cpu
```

Precedence: command-line options and environment variables > user config > defaults. Unknown keys or values of
the wrong type stop the run with exit code 3. The effective configuration is saved to `logs/config_used.yaml`.

Main groups: `run` (time, steps, seed, reserve), `resources` (GPUs, CPUs, parallel scripts), `execution`
(script time limit, artefact retention), `llm` (endpoint, two parameter groups `generation`/`judgement`
with model, thinking mode, reasoning effort, max output tokens and prices, retries), `search`
(drafts, capacities, chain control, selection), `plateau`, `operators` (retrospect, borrow, synthesis,
escalation), `codegen` (staged drafting, patch mode), `review`, `validation_rule` (shared split seed and
fraction), `assessment`, `output_trim`, `profiler`, `experience` (keyword + vector retrieval),
`pretrained`, `ensemble`, `final` (primary result), `control` (controller and its rules).

Ready-made config files in `configs/`:

- `aihubmix.yaml` runs the LLM and the embedding model through AIHubMix.
- `static.yaml` switches off every search and control mechanism that has a switch (fixed script time limit,
  rewards at chain end, no chain switch, no synthesis trigger, no smoke run, single best solution as the
  primary result).
- `cost_cap_enforced.yaml` makes the cost cap stop the search.

### Token usage and cost

Every LLM request is logged in `logs/llm_calls.jsonl` with its role, model, input/output/reasoning tokens,
latency and `cost`. The cost uses the per-million-token prices configured in `llm.generation` and
`llm.judgement` (`input_price_per_million`, `output_price_per_million`, currency `llm.currency`); no cache
discount is applied. Embedding calls of the experience store (either backend) are logged in
`logs/embedding_calls.jsonl`, priced with `experience.vector.price_per_million`. `report/run_summary.json` (`resources`) sums everything up,
by role and by model. Reasoning tokens are the largest part of the output tokens: the default effort is
`high`; `low` is clearly cheaper and may give weaker plans and code. A reply that is empty because the
reasoning used the whole output limit is retried once with the next lower effort.

## Resource control

`superagent/control/` decides every resource question of the search through one controller object:

| hook | question | default (`control.controller: rules`) |
|---|---|---|
| `step_limits` | script time budget, estimated steps left, GPU memory hint for this step | time allocation rule + GPU memory rule |
| `may_start_step` | may a new step start now | time allocation rule (short steps near the end) |
| `should_finish` | finish the search early | cost cap rule |
| `search_config` | search parameters for selection, dispatch and chain control | the configuration |
| `on_step_finished`, `start`, `summary` | notifications and the summary for the report | control log |

Each rule has a mode: `off` (not evaluated), `observe` (proposals are written to `logs/control.jsonl` but not
applied) or `enforce` (applied). Defaults: time allocation `enforce`, GPU memory hints `enforce`, cost cap
`observe` (limit 20 in the cap currency).

- **Time allocation.** The time one execution slot has left is shared among the steps still planned for it
  (`control.time_allocation.target_nodes` for the whole run). A script receives that share minus the measured
  generation time per step, at least `min_script_seconds` while enough time is left and at most
  `execution.script_timeout_seconds`. The prompt states this budget and an estimate of the remaining steps.
  Near the end a step still starts when `tail_min_script_seconds` of script time remain.
- **Cost cap.** The recorded cost (LLM and embeddings, times `rate_to_cap_currency`) plus the expected cost of
  the running steps is compared with the limit. In observe mode the moment the limit would have stopped the
  search, and the best score at that moment, are logged.
- **GPU memory.** The GPUs are detected with `nvidia-smi`; each script is told how much memory it can count on
  when several scripts share one GPU.

`control.controller: static` reproduces fixed rules (script limit = time left, a step starts when
`run.late_start_guard` times the median script time fits). A custom controller is a subclass of
`superagent.control.Controller` (or `RuleController`) named as `module:Class`; `examples/controllers/`
contains one.

## Pretrained models

`superagent/resources/method_catalog.yaml` lists general methods and pretrained models per data modality
(tabular, text, image), each with usage notes and a minimal example. The `pretrained.mode` policy decides
what the generated solutions may use:

| mode | catalog shown to the LLM | network during script runs |
|---|---|---|
| `catalog` (default) | non-pretrained methods + pretrained models downloaded locally | offline |
| `open` | everything; other public models allowed | online |
| `off` | non-pretrained methods only; pretrained weights forbidden (scripts using them are not run) | offline |

Download the pretrained models once, before running tasks (uses `pretrained.local_dir`, default
`./pretrained_cache`, or `SUPERAGENT_PRETRAIN_DIR`; `HF_TOKEN` is used if set):

```bash
python -m superagent.fetch_models [--config my_config.yaml] [--only img_dinov2_small_features ...]
```

It writes `manifest.json`; in `catalog` mode only entries downloaded successfully are shown.

### Embedding model of the experience store

The experience store ranks earlier attempts with keyword search plus an embedding model
(Qwen3-Embedding). `experience.vector.backend` chooses how the model is called:

| backend | what happens | needs |
|---|---|---|
| `local` (default) | the model (`experience.vector.model`) runs on this machine; `fetch_models` downloads it | GPU memory shared with the scripts, or `device: cpu` |
| `api` | an OpenAI-compatible embeddings endpoint is called (`api_base_url`, `api_model`) | the key named by `api_key_env` (default `AIHUBMIX_API_KEY`) |

If the model cannot be set up, or a request fails, that lookup uses keywords only; after five failures in a
row vector search is switched off for the rest of the run. `experience.vector.enabled: false` uses keywords only.

## Output directory

```
output_dir/
├── status.json                 run status (updated after every node; final at the end)
├── report/run_summary.json|md  summary: best path, score timeline, branches, operators, events, failures,
│                               usage, final result, control, diagnostics
├── final/                      primary result: predictions.csv, source.json, combine.py,
│                               members/<rank>_<node>/ (solution.py, predictions, validation files, artifacts link)
├── best/                       best single solution: solution.py, predictions.csv, validation files,
│                               score.json, artifacts/ (model files)
├── top/rank_01 … rank_20/      top solutions (per-branch limit); artifacts link for the first k
├── ensemble/                   predictions.csv (+ predictions_top{k}.csv), members.json, ensemble.py
├── logs/                       search_tree.json, decisions.jsonl, llm_calls.jsonl, embedding_calls.jsonl,
│                               control.jsonl, superagent.log, config_used.yaml, task_description_clean.md
├── experience/records.json     experience records of successful nodes
├── prepared_data/              read-only view of the task data (archives extracted)
└── runs/<node_id>/             solution.py, stdout.log, smoke_stdout.log, out/ (predictions, validation
                                files, model), prompts.jsonl
```

Every solution script runs in its own `runs/<node_id>/` directory and only uses `./data` (read-only),
`./work` (temporary, deleted after the run) and `./out` (`predictions.csv`, `val_predictions.csv`,
`val_labels.csv` and `model/`). The validation files hold the predictions and true targets of the validation
samples in the encoding of the prediction file.

At the end the task metric is identified by recomputing each solution's reported validation score from its
validation files with a library of classification metrics. An ensemble of top solutions becomes the primary
result in `final/` only if it beats the best single solution on both halves of the shared validation samples,
each half scored with weights chosen on the other half; otherwise `final/` holds the best single solution.
When the task metric ties on a half (for example both reach its best value), probability predictions are
compared by log loss instead. Label predictions are combined by an equal-weight vote (ties go to the
higher-ranked solution), so that a majority of members can overrule the best one.
`final/combine.py` recomputes `final/predictions.csv` from the members' prediction files, and each member's
`solution.py` with its model files is a complete, re-runnable solution including preprocessing.

## Tests

```bash
pytest -m "not llm"                 # tests of the code-only logic (no API key, no cost)
pytest                              # with DEEPSEEK_API_KEY set (or in .env): also the tests marked `llm`
pytest -m embed_api                 # with AIHUBMIX_API_KEY set: the embeddings endpoint test
pytest -m "not slow"                # skip the slow end-to-end tests
```

LLM replies are not simulated, except in the tests of the gateway's retry logic, which use a scripted
transport. Logic that needs no LLM (parsing, validation, selection, scheduling, rewards, patching,
assessment rules, retrieval fusion, ensembling, metric identification, control rules) is written as plain
functions and tested with direct inputs. Everything that needs a reply calls the real endpoint; those
tests are marked `llm`, need the API key and cost a little. Because real replies vary, they check
structure and invariants rather than exact text.

```bash
pytest -m llm -k gateway            # quick check that the endpoint and the key work
```

## Project layout

```
superagent/
├── run.py            CLI, run_task, startup and finalize
├── search.py         worker threads, task queue, chain switch, synthesis trigger, decision log
├── selection.py      tree_walk / elite_pick
├── scheduling.py     operator choice, plateau detection, synthesis condition
├── chain.py          chain control, per-step rewards, backpropagation
├── operators.py      draft, refine, repair, retrospect, borrow, synthesis
├── planner.py        structured plans
├── codegen.py        full-script, staged and patch code generation
├── patching.py       search/replace patch parsing and application
├── review.py         pre-run code review, pretrained-policy scan
├── executor.py       run directories, execution slots, smoke runs, time limits
├── assessment.py     result reading and independent checks
├── checks.py         score marker, prediction format and content checks
├── bookkeeping.py    best / top / retention / status / tree files
├── experience.py     experience store with BM25 + embedding retrieval
├── ensemble.py       ensembling of top predictions
├── metrics.py        validation files, metric library, metric identification
├── final.py          validated choice of the primary result (final/)
├── control/          controller interface, rules (time, cost, GPU memory), run state, hardware detection
├── report.py         run summary
├── profiler.py       data preview and modality detection
├── catalog.py        method catalog and pretrained policy
├── llm/              gateway (retries, JSON validation, call log), lenient JSON parsing, schemas
├── prompts/          all prompt wording
└── resources/        default_config.yaml, method_catalog.yaml
configs/              ready-made config files (aihubmix.yaml, static.yaml, cost_cap_enforced.yaml)
examples/             example task generator, scorer, example controller
```
