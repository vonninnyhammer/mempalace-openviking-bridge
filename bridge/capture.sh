#!/usr/bin/env bash
# Capture cycle: OpenCode sessions -> MemPalace palace -> importance ranking.
#
# Safe to run concurrently with itself and with opencode: the exporter opens
# opencode.db read-only, and mempalace takes a per-palace file lock. Each stage
# is incremental, so an idle cycle costs a few seconds and does no embedding.
set -euo pipefail

BRIDGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$BRIDGE")"
VENV="${MEMPALACE_VENV:-$HOME/.mempalace-venv}"
INGEST="${MEMPALACE_INGEST:-$ROOT/ingest/opencode}"
LOG="${MEMPALACE_CAPTURE_LOG:-$ROOT/logs/capture.log}"

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

# Serialize cycles. The timer, a manual run, and the post-boot catch-up can all
# fire at once; without this they would each re-mine the same transcripts.
exec 9>/run/user/$(id -u)/mempalace-capture.lock
if ! flock -n 9; then
    echo "SKIP: another capture cycle is already running"
    exit 0
fi

echo "=== capture $(date -Is) ==="

"$VENV/bin/python" "$BRIDGE/opencode_exporter.py"

# Mining is the only stage that can be deferred without losing work, because
# the exporter already left the new transcripts in the dropbox. Two things
# block it:
#
#   * the local embedder being down, and
#   * the palace lock, which the mempalace MCP server holds for its entire
#     lifetime. Any cycle overlapping a connected agent session would abort
#     here, and under `set -e` that killed the script before ranking ran.
#
# Lock contention is expected and transient, so it retries and then defers to
# the next cycle instead of failing. A non-lock failure is still fatal, so a
# genuinely broken mine is never mistaken for contention.
mine_deferred=""

if ! curl -fsS --max-time 10 http://127.0.0.1:11434/v1/models >/dev/null 2>&1; then
    mine_deferred="embedding endpoint 127.0.0.1:11434 unreachable"
else
    for attempt in 1 2 3; do
        mine_log=$(mktemp)
        if "$VENV/bin/mempalace" mine "$INGEST" --mode convos --wing opencode >"$mine_log" 2>&1; then
            cat "$mine_log"
            rm -f "$mine_log"
            mine_deferred=""
            break
        fi
        locked=0
        if grep -q "held by PID" "$mine_log"; then
            locked=1
        fi
        cat "$mine_log"
        rm -f "$mine_log"
        if [ "$locked" -eq 0 ]; then
            echo "ERROR: mine failed for a non-lock reason — aborting cycle"
            echo "=== done $(date -Is) ==="
            exit 1
        fi
        if [ "$attempt" -lt 3 ]; then
            mine_deferred="palace lock held (attempt $attempt/3)"
            sleep 20
        else
            mine_deferred="palace lock still held after 3 attempts"
        fi
    done
fi

if [ -n "$mine_deferred" ]; then
    echo "SKIP: mining deferred — $mine_deferred"
    echo "      unmined transcripts remain in the dropbox; the next cycle picks them up"
fi

# Ranking talks to the palace directly and takes no CLI lock, so it runs on
# every cycle even when mining was deferred. Previously an early exit above
# skipped it too, which let drawer importance go stale.
"$VENV/bin/python" "$BRIDGE/rank_drawers.py" --wing opencode

echo "=== done $(date -Is) ==="
