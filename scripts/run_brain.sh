#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export BRAIN_LLM_BASE_URL="${BRAIN_LLM_BASE_URL:-http://172.16.60.162:18050/v1/chat/completions}"
export BRAIN_LLM_MODEL="${BRAIN_LLM_MODEL:-qwen3.5-9b}"
exec python3 -m brain.dashboard --host 127.0.0.1 --port 8094
