#!/bin/bash
# Packs the Core, the adapter and a task directory for upload:
#   pack.sh CORE_DIR ADAPTER_DIR TASK_DIR OUT_DIR
# OUT_DIR receives core.tar.gz, adapter.tar.gz and task.tar.gz (the task directory is stored as "task/").
set -euo pipefail
CORE=$1; ADAPTER=$2; TASK=$3; OUT=$4
mkdir -p "$OUT"
EXCL=(--exclude=.git --exclude=.venv --exclude=__pycache__ --exclude=.pytest_cache
      --exclude=.ruff_cache --exclude='*.egg-info' --exclude=build --exclude=dist
      --exclude=tests --exclude=.env)
tar czf "$OUT/core.tar.gz" "${EXCL[@]}" -C "$CORE" .
tar czf "$OUT/adapter.tar.gz" "${EXCL[@]}" -C "$ADAPTER" .
tar czf "$OUT/task.tar.gz" -C "$(dirname "$TASK")" --transform "s,^$(basename "$TASK"),task," "$(basename "$TASK")"
ls -la "$OUT"
