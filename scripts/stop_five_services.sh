#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$ROOT/.launcher.pid" ]]; then
  echo 'Five-service launcher is not running'
  exit 0
fi
PID="$(tr -cd '0-9' < "$ROOT/.launcher.pid")"
COMMAND="$(ps -p "$PID" -o command= || true)"
if [[ "$COMMAND" != *"run_five_services.sh"* ]]; then
  echo 'PID is stale or belongs to another process; nothing stopped' >&2
  exit 1
fi
kill -TERM "$PID"
