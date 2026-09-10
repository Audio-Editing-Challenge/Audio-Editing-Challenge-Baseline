#!/usr/bin/env bash
# Download the local SenseVoiceSmall ASR weights for the AuK Prompt Enhancer
# (so PE never calls a cloud ASR). Uses the AuK venv's modelscope.
#
# Usage:  services/fetch_sensevoice.sh [dest_dir]
#   dest_dir  default: <repo>/ckpts/SenseVoiceSmall
# Afterwards, launch the AuK service with:  SENSEVOICE_MODEL=<dest_dir> services/launch_auk.sh
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
AUK_REPO="${AUK_REPO:-$HERE/AuK}"
DEST="${1:-$HERE/ckpts/SenseVoiceSmall}"
PY="$AUK_REPO/.venv/bin/python"

[ -x "$PY" ] || { echo "[fetch] missing AuK venv python: $PY" >&2; exit 1; }
mkdir -p "$DEST"
echo "[fetch] downloading iic/SenseVoiceSmall -> $DEST (via modelscope)"
"$PY" - "$DEST" <<'PY'
import sys
from modelscope import snapshot_download

path = snapshot_download("iic/SenseVoiceSmall", local_dir=sys.argv[1])
print("[fetch] SenseVoice snapshot at:", path)
PY
echo "[fetch] done. Export:  SENSEVOICE_MODEL=$DEST"
