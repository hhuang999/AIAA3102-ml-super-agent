#!/bin/bash
# Packs the Core, the adapter and a task directory for upload:
#   pack.sh CORE_DIR ADAPTER_DIR TASK_DIR OUT_DIR
# OUT_DIR receives core.tar.gz, adapter.tar.gz and task.tar.gz (the task directory is stored as "task/").
set -euo pipefail
CORE=$1; ADAPTER=$2; TASK=$3; OUT=$4
mkdir -p "$OUT"
OUT=$(realpath "$OUT")

pack_source() (
    cd "$1"
    archive=$2
    package=$3
    shift 3
    {
        printf '%s\0' pyproject.toml README.md "$@"
        find "$package" -type d \( -name tests -o -name __pycache__ \) -prune -o \
            -type f -name '*.py' ! -name 'test_*.py' ! -name conftest.py -print0
    } | tar -czf "$archive" --null -T -
)

pack_source "$CORE" "$OUT/core.tar.gz" superagent requirements.txt .env.example \
    configs/single_model.yaml superagent/resources/default_config.yaml superagent/resources/method_catalog.yaml
pack_source "$ADAPTER" "$OUT/adapter.tar.gz" superagent_adapter examples/prices.example.yaml
tar -czf "$OUT/task.tar.gz" -C "$TASK" --transform 'flags=r;s,^\.,task,' .
ls -la "$OUT"
