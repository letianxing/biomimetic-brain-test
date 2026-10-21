#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NO_PROXY="localhost,127.0.0.1,::1,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"
export SESSION_ID="${SESSION_ID:-mac-first-test}"
echo "$$" > "$ROOT/.launcher.pid"
ATTENTION_S2S_ENABLED=0 bash "$ROOT/../robot-attention-perception/scripts/run_mac_first_test.sh" &
STACK_PID=$!
cleanup() {
  if [[ -n "${BRAIN_PID:-}" ]]; then
    kill -TERM "$BRAIN_PID" 2>/dev/null || true
    wait "$BRAIN_PID" 2>/dev/null || true
  fi
  kill -TERM "$STACK_PID" 2>/dev/null || true
  wait "$STACK_PID" 2>/dev/null || true
  python3 -c 'import pathlib,sys; pathlib.Path(sys.argv[1]).unlink(missing_ok=True)' "$ROOT/.launcher.pid"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for port in 8080 8090 8092 8788 9090; do
  ready=0
  for _ in $(seq 1 180); do
    kill -0 "$STACK_PID" 2>/dev/null || exit 1
    if (echo >/dev/tcp/127.0.0.1/${port}) >/dev/null 2>&1; then ready=1; break; fi
    sleep 1
  done
  if [[ "$ready" == "0" ]]; then echo "Port $port readiness timeout" >&2; exit 1; fi
done
cd "${ROOT}"
python3 -m brain.dashboard --host 127.0.0.1 --port 8094 &
BRAIN_PID=$!
while kill -0 "$STACK_PID" 2>/dev/null && kill -0 "$BRAIN_PID" 2>/dev/null; do sleep 1; done
