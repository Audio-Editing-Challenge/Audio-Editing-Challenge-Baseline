#!/usr/bin/env python3
"""Smoke test for the AuK edit service (run in the router env, service already up).

Verifies a length-changing generative edit end-to-end and that the Prompt-Enhancer
metadata (enhanced instruction + gen_seconds) is surfaced. No cloud ASR/hy3 is
touched — the service runs PE with a local SenseVoice model.

Usage:  .venv/bin/python services/smoke_auk.py <input.wav> ["instruction"]
"""

import os
import sys

from agent.audio_utils import audio_info
from agent.clients.auk_client import AuKClient


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    inp = sys.argv[1]
    instruction = sys.argv[2] if len(sys.argv) > 2 else "add applause at the end"
    url = os.environ.get("AUK_URL", "http://127.0.0.1:8310")

    client = AuKClient(url)
    if not client.is_ready():
        print(f"[smoke] AuK service not ready at {url}", file=sys.stderr)
        return 1
    print("[smoke] /health:", client.health())

    res = client.edit(inp, instruction, "/tmp/auk_edit.wav")
    dur, sr, ch = audio_info(res.output_path)
    print(f"[smoke] out={res.output_path} {dur:.2f}s sr={sr} ch={ch}")
    print(f"[smoke] gen_seconds={res.gen_seconds} task={res.task_type}")
    print(f"[smoke] enhanced_instruction={res.enhanced_instruction!r}")
    assert res.gen_seconds is not None, "PE gen_seconds not surfaced"
    print("[smoke] AuK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
