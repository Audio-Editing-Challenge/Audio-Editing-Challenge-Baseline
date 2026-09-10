#!/usr/bin/env bash
# Unified one-click for BOTH baselines. Same UX; pick the track.
#
#   ./run.sh single --meta <MMAE-meta.json> --output-dir <dir> [options]
#   ./run.sh agent  --meta <MMAE-meta.json> --output-dir <dir> [options]
#
# single : AuK base model (Prompt Enhancer OFF), no services. Runs in AuK's .venv on an
#          idle GPU. Default --complexity single (official Track-1 scope).
# agent  : boots the SAM-Audio + AuK services on idle GPUs, waits until healthy, runs the
#          DeepSeek-routed agent, then stops the services it started. Default --complexity all.
#
# Both write the SAME outputs under --output-dir:
#   audio/<id>.wav  predictions.json  submission.jsonl  run_meta.json   (agent also: traces/)
#
# Common options: --complexity {single,all}  --limit N  --wav-root DIR
# single options: --gpu ID   (also honours AUK_CKPT / AUK_QWEN_PATH / AUK_REPO env)
# agent  options: --sam-gpu ID --auk-gpu ID --sensevoice DIR --keep-services
# Unknown options are passed through to the underlying runner.
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$HERE"
. "$HERE/src/services/_common.sh"

TRACK="${1:-}"; shift || true
case "$TRACK" in
  single|agent) ;;
  -h|--help|"") grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "[run] unknown track '$TRACK' (expected: single | agent)" >&2; exit 2 ;;
esac

META="" OUT="" COMPLEXITY="" LIMIT="" WAVROOT="" SINGLE_GPU="" SAM_GPU="" AUK_GPU="" KEEP=0
SENSEVOICE="${SENSEVOICE_MODEL:-$HERE/ckpts/SenseVoiceSmall}"
AUK_REPO="${AUK_REPO:-$HERE/AuK}"
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --meta) META="$2"; shift 2 ;;
    --output-dir) OUT="$2"; shift 2 ;;
    --complexity) COMPLEXITY="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --wav-root) WAVROOT="$2"; shift 2 ;;
    --gpu) SINGLE_GPU="$2"; shift 2 ;;
    --sam-gpu) SAM_GPU="$2"; shift 2 ;;
    --auk-gpu) AUK_GPU="$2"; shift 2 ;;
    --sensevoice) SENSEVOICE="$2"; shift 2 ;;
    --keep-services) KEEP=1; shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
[ -n "$META" ] && [ -f "$META" ] || { echo "[run] --meta <MMAE-meta.json> required and must exist" >&2; exit 2; }
[ -n "$OUT" ] || { echo "[run] --output-dir required" >&2; exit 2; }
mkdir -p logs

# ------------------------------------------------------------------ single-model
if [ "$TRACK" = "single" ]; then
  PY="$AUK_REPO/.venv/bin/python"
  [ -x "$PY" ] || { echo "[run] missing AuK venv: $PY (see README Agent Track setup)" >&2; exit 1; }
  GPU="${SINGLE_GPU:-${CUDA_VISIBLE_DEVICES:-}}"
  [ -n "$GPU" ] || GPU="$(pick_empty_gpu)" || { echo "[run] no idle GPU (grabgpu kill <id>)" >&2; exit 3; }
  strip_compat_ld
  export CUDA_VISIBLE_DEVICES="$GPU"
  ARGS=(--auk-repo "$AUK_REPO"
        --ckpt "${AUK_CKPT:-$AUK_REPO/ckpts/AuK/auk_base.safetensors}"
        --qwen-path "${AUK_QWEN_PATH:-$AUK_REPO/ckpts/Qwen2.5-Omni-3B}"
        --meta "$META" --output-dir "$OUT" --complexity "${COMPLEXITY:-single}")
  [ -n "$LIMIT" ] && ARGS+=(--limit "$LIMIT")
  [ -n "$WAVROOT" ] && ARGS+=(--wav-root "$WAVROOT")
  [ ${#EXTRA[@]} -gt 0 ] && ARGS+=("${EXTRA[@]}")
  echo "[run] single-model (AuK base, PE off) on GPU $GPU"
  exec "$PY" "$HERE/run_single_mmae.py" "${ARGS[@]}"
fi

# ------------------------------------------------------------------ agent
[ -n "${DEEPSEEK_API_KEY:-}" ] || { echo "[run] DEEPSEEK_API_KEY not set (router + PE)" >&2; exit 2; }

if [ ! -x .venv/bin/python ]; then
  echo "[run] creating router venv (.venv)..."
  uv venv --python 3.10 .venv
  uv pip install --python .venv/bin/python -e .
fi
if [ ! -f "$SENSEVOICE/model.pt" ]; then
  echo "[run] SenseVoice not found at $SENSEVOICE; fetching..."
  src/services/fetch_sensevoice.sh "$SENSEVOICE"
fi

STARTED_SAM=0 STARTED_AUK=0 SAM_PID="" AUK_PID=""
cleanup() {
  if [ "$KEEP" = 1 ]; then echo "[run] --keep-services: leaving SAM :8305 / AuK :8310 up"; return 0; fi
  [ "$STARTED_SAM" = 1 ] && { kill "$SAM_PID" 2>/dev/null || true; pkill -f sam_audio_server.py 2>/dev/null || true; }
  [ "$STARTED_AUK" = 1 ] && { kill "$AUK_PID" 2>/dev/null || true; pkill -f auk_server.py 2>/dev/null || true; }
  { [ "$STARTED_SAM" = 1 ] || [ "$STARTED_AUK" = 1 ]; } && echo "[run] stopped services it started."
  return 0
}
trap cleanup EXIT

if curl -s --max-time 4 http://127.0.0.1:8305/health 2>/dev/null | grep -q '"status":"ok"'; then
  echo "[run] reusing SAM-Audio already up on :8305"
else
  [ -n "$SAM_GPU" ] || SAM_GPU="$(pick_empty_gpu "$AUK_GPU")" || { echo "[run] no idle GPU for SAM-Audio" >&2; exit 3; }
  echo "[run] launching SAM-Audio on GPU $SAM_GPU..."
  src/services/launch_sam_audio.sh "$SAM_GPU" > logs/sam.log 2>&1 &
  SAM_PID=$!; STARTED_SAM=1
fi
if curl -s --max-time 4 http://127.0.0.1:8310/health 2>/dev/null | grep -q '"status":"ok"'; then
  echo "[run] reusing AuK already up on :8310"
else
  [ -n "$AUK_GPU" ] || AUK_GPU="$(pick_empty_gpu "$SAM_GPU")" || { echo "[run] no idle GPU for AuK" >&2; exit 3; }
  echo "[run] launching AuK on GPU $AUK_GPU..."
  SENSEVOICE_MODEL="$SENSEVOICE" src/services/launch_auk.sh "$AUK_GPU" > logs/auk.log 2>&1 &
  AUK_PID=$!; STARTED_AUK=1
fi

wait_health http://127.0.0.1:8305/health SAM-Audio "$SAM_PID" logs/sam.log
wait_health http://127.0.0.1:8310/health AuK "$AUK_PID" logs/auk.log

ARGS=(--meta "$META" --output-dir "$OUT" --complexity "${COMPLEXITY:-all}"
      --sam-audio-url http://127.0.0.1:8305 --auk-url http://127.0.0.1:8310)
[ -n "$LIMIT" ] && ARGS+=(--limit "$LIMIT")
[ -n "$WAVROOT" ] && ARGS+=(--wav-root "$WAVROOT")
[ ${#EXTRA[@]} -gt 0 ] && ARGS+=("${EXTRA[@]}")
echo "[run] agent inference (complexity=${COMPLEXITY:-all})..."
.venv/bin/python run_agent_mmae.py "${ARGS[@]}"
echo "[run] DONE. Outputs under: $OUT"
echo "         audio/  predictions.json  submission.jsonl  run_meta.json  traces/"
