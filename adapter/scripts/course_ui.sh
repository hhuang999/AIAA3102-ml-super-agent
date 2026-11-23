#!/usr/bin/env bash
# Start the interactive course page with the Core's Python environment.
# SUPERAGENT_PYTHON overrides the interpreter; extra arguments go to "course ui".
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${SUPERAGENT_PYTHON:-$REPO/../MLEvolve/env_superagent/bin/python}"
export PYTHONPATH="$REPO/core:$REPO/adapter${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m superagent_adapter.cli course ui "$@"
