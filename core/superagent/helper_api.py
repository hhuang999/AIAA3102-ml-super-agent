"""Render the installed training interface without importing a training framework."""

import inspect

from . import training


def reference():
    names = (
        "remaining_train_seconds",
        "resume_path",
        "save_checkpoint",
        "save_torch",
        "restore_torch",
        "state_fingerprint",
    )
    signatures = "\n".join(
        f"{name}{inspect.signature(getattr(training, name))}" for name in names
    )
    return f"""## Installed training helper API (authoritative)
From superagent.training:
{signatures}

remaining_train_seconds() returns float seconds before the training cutoff; reserve time for full validation, prediction and artifact verification.
resume_path() returns pathlib.Path or None, never a dictionary. It locates a committed payload; the caller must deserialize the correct framework format.
restore_torch(model, optimizer, ...) loads into existing objects and returns (completed_epoch: int, sampler_state), including (0, None) on a fresh run. Both positional arguments are required. Optional scheduler/scaler/generator must match those saved.
save_torch takes only model and optimizer positionally; epoch and all remaining arguments are keyword-only. It saves training state including optimizer and RNG, not a deployable inference package.
save_checkpoint takes an existing file path, not a dict. epoch must be an integer >=1; next_seconds and delivery_seconds must be finite and strictly positive. Use max(.001, measured_seconds) for subsecond operations.
save_torch automatically records actual optimizer steps and model/optimizer state fingerprints. For custom payloads, pass optimizer_steps, model_state and optimizer_state to save_checkpoint after loading and updating real state; do not report a guessed epoch as proof. Checkpoints without this evidence can be scored but cannot be automatically promoted. state_fingerprint hashes arrays/tensors and nested state deterministically.\ncan_continue means that additional training actually remains, not merely that this process has wall time left. Distinguish the per-invocation allowance from the trial's total training target. When changing model framework, remove or migrate ALL old checkpoint readers (e.g. np.load cannot restore a torch payload).
Initialize model and optimizer before calling restore_torch, even on a fresh run (this attaches optimizer step observation); restore before the first update. Preserve last-epoch training state for continuation separately from best-validation inference weights. Never report a prediction-only rerun as more training.

## Persistent inference contract
Save weights, architecture/config, tokenizers, label mapping and fitted preprocessing inside ./out/model/. ./work is deleted after execution, and /method_assets is NOT mounted during standalone --predict-only verification. Any pretrained backbone config/tokenizer/weights needed at inference time must be saved into ./out/model/ so --predict-only loads exclusively from --model-dir. A training checkpoint is not a replacement for this package.
Write ./out/model/inference.json with {{"version": 1, "weight_files": ["weights.pt"]}} listing actual weight payloads relative to model/. Use the correct filename(s) for the framework, not a fixed .pt extension.
The same solution.py must support:
python solution.py --predict-only --model-dir ABS_MODEL_DIR --data-dir ABS_DATA_DIR --output-dir ABS_OUTPUT_DIR
This mode must load only the inference package and declared inference inputs, recreate predictions.csv and val_predictions.csv, and never train or create an optimizer. It runs in a fresh directory without train data, old predictions, work or checkpoints. Create output-dir if needed. Use if __name__ == '__main__' to guard execution.
"""
