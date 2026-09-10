"""Client for the AuK generative-edit service (`services/auk_server.py`).

Wire contract: `POST /edit` multipart (audio_file + instruction [+ gen_seconds, seed]);
the response body is the edited WAV and the Prompt-Enhancer metadata rides in
headers (URL-encoded to stay HTTP-safe).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import quote, unquote

import requests


class AuKError(RuntimeError):
    pass


@dataclass
class AuKEditResult:
    output_path: str
    enhanced_instruction: str
    gen_seconds: float | None
    task_type: str | None


class AuKClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8310", timeout: float = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()

    def is_ready(self) -> bool:
        try:
            return self.health().get("status") == "ok"
        except Exception:
            return False

    def edit(
        self,
        input_audio: str,
        instruction: str,
        output_path: str,
        gen_seconds: float | None = None,
        seed: int | None = None,
    ) -> AuKEditResult:
        if input_audio and not os.path.isfile(input_audio):
            raise AuKError(f"input audio not found: {input_audio}")

        data = {"instruction": instruction}
        if gen_seconds is not None:
            data["gen_seconds"] = str(gen_seconds)
        if seed is not None:
            data["seed"] = str(seed)

        files = None
        fh = None
        try:
            if input_audio:
                fh = open(input_audio, "rb")
                files = {
                    "audio_file": (
                        os.path.basename(input_audio),
                        fh,
                        "application/octet-stream",
                    )
                }
            resp = requests.post(
                f"{self.base_url}/edit", files=files, data=data, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise AuKError(f"AuK request failed: {exc}") from exc
        finally:
            if fh is not None:
                fh.close()

        if resp.status_code != 200:
            raise AuKError(f"AuK server {resp.status_code}: {resp.text[:300]}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as out:
            out.write(resp.content)

        gs = resp.headers.get("X-Gen-Seconds")
        return AuKEditResult(
            output_path=output_path,
            enhanced_instruction=unquote(
                resp.headers.get("X-Enhanced-Instruction", "")
            ),
            gen_seconds=float(gs) if gs not in (None, "", "None") else None,
            task_type=resp.headers.get("X-Task-Type") or None,
        )

    @staticmethod
    def encode_meta_header(value: str) -> str:
        """Helper for the server: URL-encode metadata so it is HTTP-header safe."""
        return quote(value or "")


GENERATIVE_EDIT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "generative_edit",
        "description": (
            "Generatively edit the current audio with the AuK model + Prompt Enhancer. "
            "Use for edits that CREATE or REPLACE content and cannot be done by DSP or "
            "separation: add a sound/word, remove a segment by re-synthesis, replace one "
            "sound with another, change speaker emotion, denoise, or any length-changing "
            "edit. The Prompt Enhancer predicts the target duration automatically."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "The natural-language edit instruction to apply to the current audio.",
                },
                "gen_seconds": {
                    "type": ["number", "null"],
                    "description": "Optional target duration override in seconds; omit/null to let the Prompt Enhancer decide.",
                },
            },
            "required": ["instruction"],
        },
    },
}
