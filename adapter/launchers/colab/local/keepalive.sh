#!/bin/bash
# Keeps a Colab session busy while a run is in progress and downloads snapshots of its output.
# An idle session can be reclaimed by Colab, so the session is contacted every INTERVAL seconds.
#
#   SESSION=<colab session name> LOCAL_DIR=<download dir> keepalive.sh
#
# Optional: INTERVAL (default 180 s), SNAPSHOT_EVERY (default 7 contacts), WORKDIR (default /content),
#           LAUNCH_DIR (default $WORKDIR/launch), COLAB (default "colab --auth oauth2").
# Writes LOCAL_DIR/keepalive.log; stops after "RUN FINISHED" or "SESSION LOST" (three failed contacts).
set -u
SESSION=${SESSION:?set SESSION}
LOCAL_DIR=${LOCAL_DIR:?set LOCAL_DIR}
INTERVAL=${INTERVAL:-180}
SNAPSHOT_EVERY=${SNAPSHOT_EVERY:-7}
WORKDIR=${WORKDIR:-/content}
COLAB=${COLAB:-colab --auth oauth2}
HERE=$(cd "$(dirname "$0")/.." && pwd)
LOG="$LOCAL_DIR/keepalive.log"
mkdir -p "$LOCAL_DIR"
stamp() { date -u +%FT%TZ; }
i=0; fails=0
while true; do
  i=$((i + 1))
  if [ $((i % SNAPSHOT_EVERY)) -eq 0 ]; then
    out=$(timeout 600 $COLAB exec -s "$SESSION" --timeout 0 -f "$HERE/remote/snapshot.py" 2>&1)
    echo "$(stamp) $(echo "$out" | grep -E 'STATUS|run end|not found' | tr '\n' ' ')" >> "$LOG"
    if timeout 900 $COLAB download -s "$SESSION" "$WORKDIR/snapshot.tar.gz" "$LOCAL_DIR/snapshot_new.tar.gz" >/dev/null 2>&1 \
       && tar tzf "$LOCAL_DIR/snapshot_new.tar.gz" >/dev/null 2>&1; then
      [ -f "$LOCAL_DIR/snapshot_latest.tar.gz" ] && mv "$LOCAL_DIR/snapshot_latest.tar.gz" "$LOCAL_DIR/snapshot_prev.tar.gz"
      mv "$LOCAL_DIR/snapshot_new.tar.gz" "$LOCAL_DIR/snapshot_latest.tar.gz"
      echo "$(stamp) downloaded $(du -h "$LOCAL_DIR/snapshot_latest.tar.gz" | cut -f1)" >> "$LOG"
    fi
  else
    out=$(timeout 180 $COLAB exec -s "$SESSION" -f "$HERE/remote/ping.py" 2>&1)
    echo "$(stamp) $(echo "$out" | grep -E 'ALIVE|not found|Error' | head -1 | cut -c1-200)" >> "$LOG"
  fi
  if echo "$out" | grep -q "not found"; then fails=$((fails + 1)); else fails=0; fi
  if [ "$fails" -ge 3 ]; then echo "$(stamp) SESSION LOST" >> "$LOG"; break; fi
  if echo "$out" | grep -q "run end"; then echo "$(stamp) RUN FINISHED" >> "$LOG"; break; fi
  sleep "$INTERVAL"
done
