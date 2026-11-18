#!/bin/bash
# Creates a Python 3.11 environment in $WORKDIR/venv, installs the Core and the adapter from the uploaded
# archives, downloads the pretrained models and runs the Core's offline tests. Log: $WORKDIR/setup.log
set -u
WORKDIR=${WORKDIR:-/content}
cd "$WORKDIR" || exit 1
exec >>"$WORKDIR/setup.log" 2>&1
echo "== start $(date -Is)"
pip install -q uv
uv python install 3.11 && uv venv -q -p 3.11 "$WORKDIR/venv" && echo "== venv ok"
PY="$WORKDIR/venv/bin/python"
rm -rf core adapter task && mkdir core adapter
tar xzf core.tar.gz -C core && tar xzf adapter.tar.gz -C adapter && tar xzf task.tar.gz -C "$WORKDIR"
uv pip install --python "$PY" -r core/requirements.txt && uv pip install --python "$PY" -e core \
  && uv pip install --python "$PY" -e adapter && uv pip install --python "$PY" pytest && echo "== install ok"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
export SUPERAGENT_PRETRAIN_DIR="$WORKDIR/pretrained_cache"
"$PY" -m superagent.fetch_models > "$WORKDIR/fetch.log" 2>&1; echo "== fetch exit $?"
(cd core && "$PY" -m pytest -q -m "not llm" -p no:cacheprovider > "$WORKDIR/core_tests.log" 2>&1); echo "== core tests exit $?"
tail -3 "$WORKDIR/core_tests.log"
echo "== SETUP DONE $(date -Is)"
