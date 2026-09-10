#!/usr/bin/env bash
# Launch the SAM-Audio-Large separation service (Meta SAM-Audio) on port 8305.
#
# Runs THIS repo's bundled server (src/services/sam_audio_server.py) against an environment that has
# the official `sam_audio` package installed (github.com/facebookresearch/sam-audio). Point it at
# your install via env vars:
#
#   SAM_AUDIO_PYTHON   (required) python of the env with `sam_audio` installed
#   SAM_AUDIO_HOME     HF_HOME weights cache (facebook/sam-audio-* + full-mode aux); default: HF cache
#   SAM_AUDIO_LIB      dir with libav*.so for torchcodec; default: <env-prefix>/lib
#   SAM_AUDIO_WORKDIR  cwd holding ./.checkpoints/imagebind_huge.pth (full mode only); default: cwd
#   SAM_AUDIO_MODEL    model id (default facebook/sam-audio-large)
#   SAM_AUDIO_MODE     full | lean  (default full; lean is text-only, ~15 GB)
#   SAM_AUDIO_PORT     default 8305
#
# Full mode needs >= ~20 GB VRAM. Free a card with `grabgpu kill <id>` if all are busy.
#
# Usage:  services/launch_sam_audio.sh [gpu_id]
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
. "$HERE/_common.sh"

SAM_PY="${SAM_AUDIO_PYTHON:-}"
{ [ -n "$SAM_PY" ] && [ -x "$SAM_PY" ]; } || {
  echo "[launch-sam] set SAM_AUDIO_PYTHON to a python whose env has the official 'sam_audio' package" >&2
  echo "             (github.com/facebookresearch/sam-audio); see README." >&2; exit 1; }
[ -f "$HERE/sam_audio_server.py" ] || { echo "[launch-sam] missing bundled server: $HERE/sam_audio_server.py" >&2; exit 1; }

SAM_ENV="$(cd "$(dirname "$SAM_PY")/.." && pwd)"          # env prefix (.../bin/python -> env)
SAM_HOME="${SAM_AUDIO_HOME:-}"
SAM_LIB="${SAM_AUDIO_LIB:-$SAM_ENV/lib}"
SAM_WORKDIR="${SAM_AUDIO_WORKDIR:-$PWD}"
SAM_MODEL="${SAM_AUDIO_MODEL:-facebook/sam-audio-large}"
SAM_MODE="${SAM_AUDIO_MODE:-full}"
PORT="${SAM_AUDIO_PORT:-8305}"

GPU="${1:-${CUDA_VISIBLE_DEVICES:-}}"
[ -n "$GPU" ] || GPU="$(pick_empty_gpu)" || { echo "[launch-sam] no idle GPU. Free one with: grabgpu kill <id>" >&2; exit 2; }
echo "[launch-sam] GPU $GPU, port $PORT, $SAM_MODE mode, model $SAM_MODEL"

if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then echo "[launch-sam] port ${PORT} already in use" >&2; exit 3; fi

MODE_FLAG="--full"; [ "$SAM_MODE" = "lean" ] && MODE_FLAG="--lean"
cd "$SAM_WORKDIR"                                 # full mode reads ./.checkpoints/imagebind_huge.pth
[ -n "$SAM_HOME" ] && export HF_HOME="$SAM_HOME"
export LD_LIBRARY_PATH="$SAM_LIB"                 # torchcodec needs libav*.so from the SAM-Audio env
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$SAM_PY" "$HERE/sam_audio_server.py" --model "$SAM_MODEL" --port "$PORT" $MODE_FLAG
