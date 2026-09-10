#!/usr/bin/env python3
"""Smoke test for the SAM-Audio service (run in the router env, service already up).

Usage:  .venv/bin/python services/smoke_sam_audio.py <mixture.wav> [description]
"""

import os
import sys

from agent.audio_utils import audio_info
from agent.clients.sam_audio_client import SAMAudioClient


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    mix = sys.argv[1]
    desc = sys.argv[2] if len(sys.argv) > 2 else "music"
    url = os.environ.get("SAM_AUDIO_URL", "http://127.0.0.1:8305")

    client = SAMAudioClient(url)
    if not client.is_ready():
        print(f"[smoke] SAM-Audio service not ready at {url}", file=sys.stderr)
        return 1
    print("[smoke] /health:", client.health())

    target = client.separate(mix, "/tmp/sam_target.wav", desc, stem="target")
    residual = client.separate(mix, "/tmp/sam_residual.wav", desc, stem="residual")
    for path in (target, residual):
        dur, sr, ch = audio_info(path)
        print(f"[smoke] {path}: {dur:.2f}s sr={sr} ch={ch}")
        assert sr == 48000, f"expected 48 kHz output, got {sr}"
    print("[smoke] SAM-Audio OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
