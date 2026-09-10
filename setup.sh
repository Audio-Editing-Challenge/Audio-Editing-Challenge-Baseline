#!/usr/bin/env bash
# Set up the environment for a chosen track. Mirrors run.sh: pick the track.
#
#   ./setup.sh single   # Track 1: AuK env (core, PE off) + weights check
#   ./setup.sh agent    # Track 2: router venv + AuK env (.[gradio]/PE) + SenseVoice + SAM check
#   ./setup.sh both     # everything
#
# Idempotent: it verifies what already exists and installs ONLY what is missing; it never
# force-reinstalls. It does NOT build the SAM-Audio env — install Meta SAM-Audio separately
# (github.com/facebookresearch/sam-audio) and point SAM_AUDIO_PYTHON at it.
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$HERE"

AUK_REPO="${AUK_REPO:-$HERE/AuK}"
SENSEVOICE="${SENSEVOICE_MODEL:-$HERE/ckpts/SenseVoiceSmall}"

setup_router_venv() {
  if [ ! -x .venv/bin/python ]; then
    echo "[setup] creating router venv (.venv)..."
    uv venv --python 3.10 .venv
  fi
  echo "[setup] installing/refreshing router deps (openai/librosa/soundfile/requests/...)"
  uv pip install --python .venv/bin/python -e .
  echo "[setup] router venv OK"
}

# $1 = core | gradio   (gradio = superset with the Prompt Enhancer / SenseVoice deps)
setup_auk_env() {
  local need="$1" py="$AUK_REPO/.venv/bin/python"
  [ -d "$AUK_REPO" ] || { echo "[setup][error] AuK checkout not found at $AUK_REPO (set AUK_REPO)"; return 1; }
  if [ ! -x "$py" ]; then
    echo "[setup] creating AuK venv at $AUK_REPO/.venv..."
    uv venv --python 3.10 "$AUK_REPO/.venv"
  fi
  if ! "$py" -c "import auk" >/dev/null 2>&1; then
    echo "[setup] installing AuK core into its venv..."
    uv pip install --python "$py" -e "$AUK_REPO"
  fi
  if [ "$need" = "gradio" ] && ! "$py" -c "import funasr, modelscope" >/dev/null 2>&1; then
    echo "[setup] installing AuK Prompt-Enhancer deps (.[gradio]: funasr/modelscope/silero-vad/...)..."
    uv pip install --python "$py" -e "${AUK_REPO}[gradio]"
  fi
  for w in "ckpts/AuK/auk_base.safetensors" "ckpts/Qwen2.5-Omni-3B"; do
    if [ -e "$AUK_REPO/$w" ]; then echo "[setup] AuK weight OK: $w"
    else echo "[setup][warn] missing AuK weight $AUK_REPO/$w  (hf download tencent/AuK ; hf download Qwen/Qwen2.5-Omni-3B)"; fi
  done
  echo "[setup] AuK env ($need) OK: $py"
}

setup_sensevoice() {
  if [ -f "$SENSEVOICE/model.pt" ]; then echo "[setup] SenseVoice present: $SENSEVOICE"
  else echo "[setup] fetching SenseVoice -> $SENSEVOICE"; src/services/fetch_sensevoice.sh "$SENSEVOICE"; fi
}

check_sam_env() {
  local py="${SAM_AUDIO_PYTHON:-}"
  if [ -n "$py" ] && [ -x "$py" ] && "$py" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('sam_audio') else 1)" >/dev/null 2>&1; then
    echo "[setup] SAM-Audio env OK: $py"
  else
    echo "[setup][warn] SAM-Audio (github.com/facebookresearch/sam-audio) not found."
    echo "              Install it in its own Python 3.12 / torch cu128 env + gated HF weights,"
    echo "              then export SAM_AUDIO_PYTHON / SAM_AUDIO_HOME (see README). Not auto-built."
  fi
}

case "${1:-}" in
  single)
    echo "== Track 1 (Single Model) environment =="
    setup_auk_env core
    echo "[setup] done. Run:  ./run.sh single --meta <MMAE-meta.json> --output-dir ./outputs/single"
    ;;
  agent)
    echo "== Track 2 (Agent) environment =="
    setup_router_venv
    setup_auk_env gradio
    setup_sensevoice
    check_sam_env
    [ -n "${DEEPSEEK_API_KEY:-}" ] && echo "[setup] DEEPSEEK_API_KEY set" || echo "[setup][warn] DEEPSEEK_API_KEY not set (router + PE need it)"
    echo "[setup] done. Run:  ./run.sh agent --meta <MMAE-meta.json> --output-dir ./outputs/agent"
    ;;
  both)
    "$0" single
    "$0" agent
    ;;
  *)
    echo "usage: ./setup.sh {single|agent|both}" >&2
    exit 2
    ;;
esac
