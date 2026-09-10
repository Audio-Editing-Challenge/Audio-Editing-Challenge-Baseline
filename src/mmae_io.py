"""Shared MMAE I/O for both baselines (Single-Model and Agent).

Stdlib-only (json/os/copy) so it imports cleanly under every environment — AuK's
`.venv`, the SAM-Audio conda env, and the light router `.venv`. Both runners use
these helpers so the input selection and the output contract
(`predictions.json` / `submission.jsonl` / `run_meta.json`) are byte-identical
across tracks.
"""

from __future__ import annotations

import copy
import json
import os


def select_samples(meta: list, complexity: str) -> list:
    """Select MMAE samples by complexity. 'single' = Track-1 scope; anything else = all."""
    if complexity == "single":
        return [r for r in meta if r.get("complexity") == "single"]
    return list(meta)


def extract_instruction_and_audios(messages: list) -> tuple[str, list[str]]:
    """Return (combined user instruction, ordered list of user audio_urls)."""
    texts, audios = [], []
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if not isinstance(c, dict):
                continue
            if c.get("type") == "text" and c.get("text"):
                texts.append(c["text"].strip())
            elif c.get("type") == "audio":
                url = c.get("audio_url") or c.get("audio")
                if url:
                    audios.append(url)
    return " ".join(t for t in texts if t), audios


def abs_audio(url: str, wav_root: str) -> str:
    return url if os.path.isabs(url) else os.path.abspath(os.path.join(wav_root, url))


def rewrite_messages_absolute(messages: list, wav_root: str) -> list:
    """Deep-copy `messages` and rewrite every user audio_url to an absolute path, so the
    MMAE evaluator resolves inputs regardless of its working directory."""
    out = copy.deepcopy(messages)
    for m in out:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if isinstance(c, dict) and c.get("type") == "audio":
                url = c.get("audio_url") or c.get("audio")
                if url:
                    c["audio_url"] = abs_audio(url, wav_root)
                    c.pop("audio", None)
    return out


def build_manifests(
    samples: list, audio_dir: str, wav_root: str
) -> tuple[list, list, list]:
    """Build (predictions, submission, present_ids) from the wavs present on disk."""
    predictions, submission, present = [], [], []
    for rec in samples:
        sid = rec["id"]
        wav = os.path.join(audio_dir, f"{sid}.wav")
        if not os.path.isfile(wav):
            continue
        present.append(sid)
        abs_wav = os.path.abspath(wav)
        msgs = rewrite_messages_absolute(rec.get("messages", []), wav_root)
        # assistant turn content MUST be a list (the evaluator iterates message["content"])
        msgs.append(
            {"role": "assistant", "content": [{"type": "audio", "audio_url": abs_wav}]}
        )
        predictions.append({"id": sid, "messages": msgs})
        submission.append({"id": sid, "audio_path": f"audio/{sid}.wav"})
    return predictions, submission, present


def write_manifests(
    output_dir: str, predictions: list, submission: list, run_meta: dict
) -> None:
    """Write predictions.json / submission.jsonl / run_meta.json ATOMICALLY (tmp + os.replace).

    Atomicity matters for sharded runs: a half-written predictions.json would break the
    MMAE evaluator. The on-disk schema is unchanged from the original single-model writer.
    """

    def _atomic(name: str, write_fn) -> None:
        path = os.path.join(output_dir, name)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            write_fn(f)
        os.replace(tmp, path)

    def _write_submission(f) -> None:
        for row in submission:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _atomic(
        "predictions.json",
        lambda f: json.dump(predictions, f, ensure_ascii=False, indent=1),
    )
    _atomic("submission.jsonl", _write_submission)
    _atomic(
        "run_meta.json", lambda f: json.dump(run_meta, f, ensure_ascii=False, indent=1)
    )
