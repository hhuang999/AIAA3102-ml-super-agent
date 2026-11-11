#!/bin/bash
#SBATCH --job-name=aiaa3102-env-check
#SBATCH --partition=debug
#SBATCH --qos=debug
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

# Submit from the project root, with --output pointing into its logs/ directory.
cd "${SLURM_SUBMIT_DIR:?Submit this script from the project root}"
project_root="$PWD"
result_dir="$project_root/runs/hpc-smoke/${SLURM_JOB_ID:?}"
mkdir -p "$result_dir" "$project_root/cache/pip" "$project_root/cache/tmp"
export PIP_CACHE_DIR="$project_root/cache/pip"
export TMPDIR="$project_root/cache/tmp"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HF_HOME="$project_root/cache/huggingface"

date -Iseconds
hostname
printf 'SLURM_JOB_ID=%s CUDA_VISIBLE_DEVICES=%s\n' "$SLURM_JOB_ID" "${CUDA_VISIBLE_DEVICES-}"
nvidia-smi

# This cluster can expose one GPU as NVML index 0 while Slurm exports its
# node-wide index (for example 7). UUIDs identify the allocated device in both
# namespaces; only replace the mask when NVML exposes exactly one device.
export SCHEDULER_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"
mapfile -t visible_gpu_uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
if [[ ${#visible_gpu_uuids[@]} -eq 1 ]]; then
    export CUDA_VISIBLE_DEVICES="${visible_gpu_uuids[0]}"
fi
printf 'GPU binding: scheduler=%s runtime=%s\n' "$SCHEDULER_CUDA_VISIBLE_DEVICES" "${CUDA_VISIBLE_DEVICES-}"

source /etc/profile.d/modules.sh
module load anaconda3
python --version
if [[ ! -x "$project_root/.venv/bin/python" ]]; then
    python -m venv "$project_root/.venv"
fi
python_bin="$project_root/.venv/bin/python"
"$python_bin" -c 'import sys; print(sys.executable); assert sys.version_info[:2] == (3, 11), sys.version'
requirements_file=requirements-hpc.txt
if [[ -f requirements-hpc.lock.txt ]]; then
    requirements_file=requirements-hpc.lock.txt
fi
"$python_bin" -m pip install --disable-pip-version-check --progress-bar off -r "$requirements_file"
"$python_bin" -m pip check
"$python_bin" -m pip freeze > "$result_dir/pip-freeze.txt"
"$python_bin" scripts/hpc/smoke_train.py --output-dir "$result_dir"
du -sh "$project_root/.venv" "$project_root/cache"
date -Iseconds
printf 'HPC_VALIDATION_PASSED job=%s\n' "$SLURM_JOB_ID"
