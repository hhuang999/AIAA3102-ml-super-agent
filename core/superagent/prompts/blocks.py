"""Reusable prompt blocks. All prompt wording lives in this package."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .policy import RESOURCE_LINE

ROLE_LINE = "You are an experienced machine learning engineer."

TASK_ALIGNMENT_RULES = """## Task and metric alignment rules
- Every solution must target the official task and the official metric stated above.
- Local validation must evaluate the same task as the final prediction.
- The printed validation score must be computed with the official metric's definition, or an equivalent implementation.
- Do not simplify the task into an easier proxy objective to obtain a higher local score.
- Do not use a proxy metric as the main score for comparing solutions."""

PACKAGES = ["numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "catboost", "torch", "torchvision", "timm",
            "transformers", "sentence-transformers", "opencv-python", "Pillow", "tabicl"]
PRETRAINED_ONLY_PACKAGES = {"sentence-transformers", "tabicl"}


@dataclass
class Resources:
    time_left_secs: float
    steps_left: int
    run_limit_secs: float
    cpus_per_run: int
    num_gpus: int
    mode: str
    steps_left_estimated: bool = False
    budget_text: bool = False
    gpu_memory_gb: float | None = None
    gpu_sharing: int = 1
    smoke_seconds: float = 0.0
    external_stop: bool = False


def _fmt_dur(secs: float) -> str:
    secs = max(0, int(secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _gpu_line(res: Resources) -> str:
    if res.num_gpus <= 0:
        return "No GPU is available: everything runs on the CPU, so choose methods that train fast on a CPU."
    if res.gpu_memory_gb is not None:
        shared = (f" The GPU is shared by up to {res.gpu_sharing} scripts running at the same time." if res.gpu_sharing > 1 else "")
        return (f"One GPU is visible to this script; about {res.gpu_memory_gb:g} GB of GPU memory is available to it.{shared} "
                "Choose batch sizes and model sizes that fit into this memory.")
    return (f"{res.num_gpus} GPU(s) may be visible; another solution may be using the same GPU at the same time, "
            "so keep GPU memory use moderate.")


def _time_lines(res: Resources) -> str:
    if res.external_stop:
        return (f"- Candidate slots left: {res.steps_left}. The caller controls when the task stops.\n"
                f"- Finish this script within {_fmt_dur(res.run_limit_secs)} including training and predictions.")
    if not res.budget_text:
        return (f"- Total time left for the whole search: {_fmt_dur(res.time_left_secs)}. Steps left: {res.steps_left}.\n"
                f"- Time limit for this script: {_fmt_dur(res.run_limit_secs)} ({int(res.run_limit_secs)} seconds). "
                "The script must finish within it; a script that times out is discarded. Within that limit, aim for the best result.")
    steps = (f"Estimated number of further steps at the current pace: about {res.steps_left}."
             if res.steps_left_estimated else f"Steps left: {res.steps_left}.")
    return (f"- Total time left for the whole search: {_fmt_dur(res.time_left_secs)}. {steps}\n"
            f"- Time budget for this script: {_fmt_dur(res.run_limit_secs)} ({int(res.run_limit_secs)} seconds), including "
            "data loading, training, validation and prediction. The script is stopped and discarded when it exceeds the "
            "budget, so plan the work to finish with a safety margin (for example by choosing the number of epochs, "
            "folds and models accordingly).")


def smoke_mode_rule(seconds: float) -> str:
    return f"""10. Smoke mode. When the environment variable SUPERAGENT_SMOKE equals "1", run the complete pipeline on a tiny subset so that it finishes well within {int(seconds)} seconds: at most 200 training samples (keep at least two samples of every class where possible), at most 200 test samples, one epoch or a few iterations, no hyperparameter search. Write all output files as usual. Without this variable, use all data. The agent runs every script in smoke mode first and skips the full run when the smoke run fails.
"""


GPU_HABITS = """
- Mixed precision: prefer bf16 autocast (torch.autocast("cuda", dtype=torch.bfloat16)) when torch.cuda.is_bf16_supported() is true; fp16 needs a GradScaler. Keep the model weights in float32 under autocast and do not call .half() on a model or its heads. If the loss becomes NaN, train in float32.
- GPU memory: start with a moderate batch size; on a CUDA out-of-memory error lower the batch size and use gradient accumulation. Before training another model in the same script, release the previous one (del model; torch.cuda.empty_cache())."""


def implementation_rules(res: Resources, split_seed: int, val_fraction: float, *, declared_split=False, include_torch: bool = True) -> str:
    from ..helper_api import reference
    gpu_habits = GPU_HABITS if res.num_gpus > 0 else ""
    smoke_rule = smoke_mode_rule(res.smoke_seconds) if res.smoke_seconds > 0 else ""
    split = ("Use the validation partition, ID rules and metric in the authoritative task contract. Never substitute a generated split." if declared_split else f"Use a stratified split by label with validation fraction {val_fraction}.")
    score_rule = (
        "The agent scores validation predictions independently. Training may report its validation score for diagnostics. "
        "Predict-only writes predictions without reading validation labels or computing a score."
        if declared_split else
        "The very last line printed by the script must be `VALIDATION_SCORE: <number>`, computed on held-out validation data with the official task metric."
    )
    validation_files = (
        """Write ./out/val_predictions.csv with the sample prediction columns and the exact validation IDs defined by the task contract. The evaluator supplies validation truth."""
        if declared_split else
        """Also write ./out/val_predictions.csv and ./out/val_labels.csv for the validation samples behind the printed score.
   - Both files have exactly the columns of the prediction file, in the same order.
   - The first column identifies each validation sample: its id from the training data, or its 0-based row number in the training file when there is no id column. Both files use the same identifiers.
   - val_predictions.csv holds the predictions for these samples, produced and encoded exactly like the test predictions.
   - val_labels.csv holds the true targets in the same encoding: the true class label where the prediction is a class label; 1 for the true class and 0 for the others where there is one probability column per class; the true 0 or 1 where there is a single probability column.
   - With cross-validation, write the out-of-fold predictions of all training samples."""
    )
    validation_check = (
        "- Does predict-only recreate both prediction files using only the model package and inference inputs?"
        if declared_split else
        "- Are val_predictions.csv and val_labels.csv written with the prediction file's columns?\n- Is the last printed line the score marker?"
    )
    continuation_line = (
        "Training continuation: use superagent.training.remaining_train_seconds() before each epoch to leave time for validation and full test predictions. Call restore_torch(model, optimizer, scheduler=...) before training and save_torch(..., epoch=..., next_seconds=..., delivery_seconds=...) after a resumable epoch. Save sampler/generator/scaler if used. Other frameworks use resume_path() and save_checkpoint(...). Always emit predictions at each rung. For a verified frozen extractor with deterministic preprocessing, superagent.features.frozen_features can cache features across rungs: bind split, ordered row_ids, weights_sha256, transform and extractor revision. Never reuse for random augmentations or changing weights."
        if include_torch else
        "Training continuation: use superagent.training.remaining_train_seconds() to leave time for validation and full test predictions. Iterative models can use resume_path() and save_checkpoint(...). Always emit predictions at each rung."
    )
    return f"""## Implementation rules
Resources:
{_time_lines(res)}
- CPU cores available to this script: {res.cpus_per_run}. {_gpu_line(res)}
- {RESOURCE_LINE[res.mode]}

Hard requirements:
{reference(include_torch=include_torch)}
1. Real inference. Every validation and test prediction must come from inference with the trained model. Never use constants, placeholders, random numbers or statistics as predictions. Never fake or simplify the metric function.
2. Prediction file. Predict every test sample and write ./out/predictions.csv, formatted exactly like sample_predictions.csv (when present) and the task description.
3. Model files. Save everything needed to predict again later into ./out/model/: model weights, fitted preprocessing objects (encoders, tokenizers, scalers, ...) and the label mapping.
4. Validation scoring. {score_rule}
5. Validation equals test. Validation and test data must go through exactly the same preprocessing, inference and post-processing.
6. Shared validation split.
   - Use the fixed random seed {split_seed}.
   - {split}
   - With cross-validation, use stratified K-fold with the same seed and report the mean score over the folds.
   - If the data has a group structure or a time order, split by group or by time instead and print why.
   - If the task description prescribes a split, follow it.
   - When modifying an existing solution, never change its split.
7. Auditable. Print one line describing the validation split and one line naming the metric used.
8. Task semantics. Local validation must evaluate the same task as the final prediction. Do not silently switch to an easier proxy task, and do not use a proxy metric as the main score.
9. Validation files. {validation_files}
{smoke_rule}
{continuation_line}
Directories:
- ./data: read-only input data. Never write there.
- ./work: temporary files and caches. Deleted after the run.
- ./out: outputs only: predictions.csv, val_predictions.csv, val_labels.csv and model/.
- Do not read or write outside these three directories.

Common API pitfalls:
- LightGBM early stopping is set with callbacks (lgb.early_stopping), not as a fit() argument.
- XGBoost early_stopping_rounds is a constructor argument, not a fit() argument.
- AdamW is imported from torch.optim.

Running habits:{gpu_habits}
- No progress bars. Do not enable per-batch logging of libraries.
- Print at most one line per training epoch.
- Use multiple worker processes for data loading, at most the number of available CPU cores.

Self-check before giving the code (go through each item):
- Do the predictions pass through the weights learned by the model?
- Is the prediction file written to the right path and does it contain every test sample?
{validation_check}
- Are the model files saved?
- Outside smoke mode, is the full training data used, not a small subset?
- Does local validation keep the original task semantics and the official metric?"""


LEAKAGE_RULES = """## Preventing data leakage
- Principle: validation and test data must be completely invisible during training.
- Split the data first, then do any processing.
- Fit every stateful transform (scaling, encoding, imputation, tokenizers, dimensionality reduction) on the training part only, then apply it to the validation and test parts.
- Compute global statistics (means, variances, vocabularies) from the training part only.
- Never use label information from the validation part (for example target encoding)."""

ROBUSTNESS_ADVICE = """## Robustness advice (suggestions, not hard requirements)
- Match the model to the data structure: CNNs or vision transformers for images, transformers or recurrent networks for sequences, gradient boosted trees first for tabular data.
- When the data has several modalities, try to use all of them.
- Very long or large inputs can be processed in windows or chunks.
- Regularisation: dropout, normalisation layers, weight decay, label smoothing.
- Look at the class distribution before choosing the loss (weighted loss, focal loss, ...).
- Learning-rate schedules: cosine annealing, reduce-on-plateau on the validation metric, warm-up.
- Use data augmentation suited to the modality.
- Monitor the validation metric and use early stopping.
- Prefer models that capture the intrinsic structure of the data over simply making the model bigger."""


def environment_block(mode: str, rng: random.Random | None = None) -> str:
    # Keep deterministic package order so prompt prefix caching stays effective across calls.
    pkgs = [p for p in PACKAGES if not (mode == "off" and p in PRETRAINED_ONLY_PACKAGES)]
    return f"""## Environment
- Installed packages include: {", ".join(pkgs)}.
- PyTorch is recommended for neural networks.
- Never install or upgrade packages in the script."""


FULL_SCRIPT_FORMAT = """## Output format
Your reply has exactly two parts: first a short natural-language overview of the solution, then exactly one ```python code block containing the complete, self-contained, runnable script. Nothing else."""


def section(title: str, body: str) -> str:
    body = (body or "").strip()
    if not body:
        return ""
    return f"# {title}\n{body}"


def code_block(code: str) -> str:
    return "```python\n" + code.rstrip() + "\n```"


def join_sections(*parts: str) -> str:
    return "\n\n".join(p for p in parts if p and p.strip())
