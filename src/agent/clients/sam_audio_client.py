"""Client for the SAM-Audio-Large text-prompted separation service.

Self-contained (no dependency on the Audio-Oscar-Dev `song` package): it speaks
the same multipart wire contract as `servers/sam_audio_server.py`
(`POST /separate`, field `audio_file` + form fields) and returns a 48 kHz WAV.
"""

from __future__ import annotations

import os

import requests


class SAMAudioError(RuntimeError):
    pass


class SAMAudioClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8305", timeout: float = 600):
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

    def separate(
        self,
        input_audio: str,
        output_path: str,
        description: str,
        stem: str = "target",
        predict_spans: bool = False,
        reranking_candidates: int = 1,
        span: str | None = None,
    ) -> str:
        """Separate `description` from `input_audio` into `output_path`; returns output_path.

        stem="target" -> the described sound; stem="residual" -> everything else.
        Raises SAMAudioError on any failure (so the router can catch and fall back).
        """
        if stem not in ("target", "residual"):
            raise SAMAudioError(f"stem must be 'target' or 'residual', got {stem!r}")
        if not os.path.isfile(input_audio):
            raise SAMAudioError(f"input audio not found: {input_audio}")

        data = {
            "description": description,
            "stem": stem,
            "predict_spans": str(predict_spans),
            "reranking_candidates": str(reranking_candidates),
        }
        if span:
            data["span"] = span

        try:
            with open(input_audio, "rb") as f:
                files = {
                    "audio_file": (
                        os.path.basename(input_audio),
                        f,
                        "application/octet-stream",
                    )
                }
                resp = requests.post(
                    f"{self.base_url}/separate",
                    files=files,
                    data=data,
                    timeout=self.timeout,
                )
        except requests.RequestException as exc:
            raise SAMAudioError(f"SAM-Audio request failed: {exc}") from exc

        if resp.status_code != 200:
            raise SAMAudioError(
                f"SAM-Audio server {resp.status_code}: {resp.text[:300]}"
            )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as out:
            out.write(resp.content)
        return output_path


# OpenAI-style tool schema advertised to the router LLM.
SEPARATE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "separate",
        "description": (
            "Isolate or remove a described sound in the current audio using text-prompted "
            "source separation. stem='target' keeps ONLY the described sound (use for "
            "'extract/isolate/keep only the X'); stem='residual' removes the described "
            "sound and keeps everything else (use for 'remove/delete/get rid of the X')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Short lowercase noun/verb phrase for the sound, e.g. 'dog barking', 'music', 'man speaking'.",
                },
                "stem": {
                    "type": "string",
                    "enum": ["target", "residual"],
                    "description": "'target' = keep only the described sound; 'residual' = remove it, keep the rest.",
                },
            },
            "required": ["description", "stem"],
        },
    },
}
