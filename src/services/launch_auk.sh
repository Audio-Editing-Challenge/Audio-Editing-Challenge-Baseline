#!/usr/bin/env bash
# Launch the AuK generative-edit service (PE local: SenseVoice ASR + DeepSeek LLM) on port 8310.
#
# Runs with AuK's OWN .venv (torch 2.7, funasr/modelscope from the .[gradio] extra).
# Needs ~1 A800 for AuK; SenseVoice shares the card by default. Free a GPU with
#   grabgpu kill <id>   if all cards are occupied.
#
# Requires DEEPSEEK_API_KEY (PE's LLM). SenseVoice weights come from ModelScope on first
# run unless SENSEVOICE_MODEL points to a local dir (see services/fetch_sensevoice.sh).
#
# Usage:  services/launch_auk.sh [gpu_id]
set -euo pipefail
HERE_SVC="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
. "$HERE_SVC/_common.sh"

AUK_REPO="${AUK_REPO:-$PWD/AuK}"
PORT="${AUK_PORT:-8310}"
PY="$AUK_REPO/.venv/bin/python"

[ -x "$PY" ] || { echo "[launch-auk] missing AuK venv python: $PY" >&2; exit 1; }
[ -n "${DEEPSEEK_API_KEY:-}${AUK_PE_LLM_API_KEY:-}" ] || { echo "[launch-auk] set DEEPSEEK_API_KEY (PE LLM)" >&2; exit 1; }

# ensure the web deps are present in the AuK venv (idempotent; uses the Tencent mirror)
if ! "$PY" -c "import fastapi, uvicorn, multipart" >/dev/null 2>&1; then
  echo "[launch-auk] installing fastapi/uvicorn/python-multipart into the AuK venv..."
  uv pip install --python "$PY" \
      fastapi "uvicorn[standard]" python-multipart
fi

GPU="${1:-${CUDA_VISIBLE_DEVICES:-}}"
[ -n "$GPU" ] || GPU="$(pick_empty_gpu)" || { echo "[launch-auk] no idle GPU. Free one with: grabgpu kill <id>" >&2; exit 2; }
echo "[launch-auk] using GPU $GPU, port $PORT"

if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then echo "[launch-auk] port ${PORT} already in use" >&2; exit 3; fi

export AUK_REPO
export AUK_PORT="$PORT"
export CUDA_VISIBLE_DEVICES="$GPU"
export SENSEVOICE_DEVICE="${SENSEVOICE_DEVICE:-cuda}"
strip_compat_ld   # avoid CUDA error 803 from a mismatched /compat libcuda stub
# PE LLM defaults to DeepSeek hosted API (deepseek-v4-flash); override AUK_PE_LLM_* to swap.
exec "$PY" "$HERE_SVC/auk_server.py"
