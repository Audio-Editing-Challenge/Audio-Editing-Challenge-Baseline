#!/usr/bin/env bash
# Shared shell helpers for the baseline launch / one-click scripts.
# Source it (don't exec):  . "$(dirname "$0")/_common.sh"

# pick_empty_gpu [exclude_id] [thresh_mib]
# Print the first GPU whose memory.used < thresh (default 1000 MiB), skipping exclude_id.
# Picks only essentially-idle cards, so a partially-used GPU (someone else's job) is never grabbed.
pick_empty_gpu() {
  local excl="${1:-}" thresh="${2:-1000}" idx used
  while read -r idx used; do
    [ "$idx" = "$excl" ] && continue
    [ "$used" -lt "$thresh" ] && { echo "$idx"; return 0; }
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  return 1
}

# strip_compat_ld
# Remove any CUDA forward-compat dir (.../compat) from LD_LIBRARY_PATH. On some images the
# inherited path carries a mismatched libcuda stub (e.g. 550 vs a 580 driver) that makes
# torch fail with CUDA error 803 (cudaErrorSystemDriverMismatch).
strip_compat_ld() {
  if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    LD_LIBRARY_PATH="$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/compat' | paste -sd: -)"
    export LD_LIBRARY_PATH
  fi
}

# wait_health <url> <name> <pid-or-empty> <logfile> [max_tries=120] [sleep=5]
# Poll an HTTP /health endpoint until it reports {"status":"ok"}. Fail fast on
# {"status":"error"} or a dead pid; time out after max_tries*sleep seconds.
wait_health() {
  local url="$1" name="$2" pid="${3:-}" log="${4:-/dev/null}" tries="${5:-120}" nap="${6:-5}" resp i
  for ((i = 0; i < tries; i++)); do
    resp="$(curl -s --max-time 4 "$url" 2>/dev/null || true)"
    case "$resp" in
      *'"status":"ok"'*) echo "[health] $name ready"; return 0 ;;
      *'"status":"error"'*) echo "[health] $name failed to load:"; echo "$resp" | head -c 600; echo; return 1 ;;
    esac
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      echo "[health] $name process died; log tail:"; tail -25 "$log" 2>/dev/null; return 1
    fi
    sleep "$nap"
  done
  echo "[health] $name health-check timed out; log tail:"; tail -25 "$log" 2>/dev/null; return 1
}
