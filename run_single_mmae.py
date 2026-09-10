#!/usr/bin/env python3
"""AuK base-model (Prompt Enhancer OFF) inference over the MMAE benchmark.

Single-model baseline for the Audio Editing Challenge @ ICASSP 2027 (Track 1).
For each selected MMAE sample it edits the input audio with the AuK base model
and writes:

  <output-dir>/audio/<id>.wav   the edited audio (24 kHz mono wav)
  <output-dir>/predictions.json  MMAE-evaluator input (chatml `messages`, absolute audio_urls)
  <output-dir>/submission.jsonl  official leaderboard manifest ({"id", "audio_path"})
  <output-dir>/run_meta.json     reproducibility metadata + counts + best-effort reductions

Duration policy: EQUAL-LENGTH. We call AukInfer.generate(messages) with no
`gen_seconds`, so AuK matches the source-clip length. No Prompt Enhancer, no
ASR/LLM/VAD, no second learned model -> a valid Track 1 single-model system.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

# src/ layout: make the bare `mmae_io` module importable (works in AuK's env too — it is
# stdlib-only and does NOT pull in the router package).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from mmae_io import (  # noqa: E402
    abs_audio,
    build_manifests,
    extract_instruction_and_audios,
    select_samples,
    write_manifests,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AuK base (no Prompt Enhancer) inference over MMAE -> evaluator + leaderboard artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- model / repo ---
    p.add_argument(
        "--auk-repo",
        default=None,
        help="Path to the AuK repo. If given, <auk-repo>/src is added to sys.path so `import auk` works "
        "even without `pip install -e .`.",
    )
    p.add_argument(
        "--ckpt",
        required=True,
        help="Path to the AuK base checkpoint (auk_base.safetensors).",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml. Defaults to <ckpt-dir>/config.yaml. vae.safetensors auto-loads "
        "from the ckpt directory.",
    )
    p.add_argument(
        "--qwen-path",
        required=True,
        help="Absolute path to the Qwen2.5-Omni-3B text encoder (config's text_encoder_path is relative).",
    )
    p.add_argument(
        "--device", default="cuda", help="Torch device, e.g. cuda, cuda:0, cpu."
    )
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    # --- data ---
    p.add_argument("--meta", required=True, help="Path to MMAE-meta.json.")
    p.add_argument(
        "--wav-root",
        default=None,
        help="Root that MMAE relative audio_urls (wav/<id>/audioN.wav) resolve against. "
        "Defaults to the directory of --meta.",
    )
    p.add_argument(
        "--output-dir", required=True, help="Where to write audio/ + manifests."
    )
    p.add_argument(
        "--complexity",
        default="single",
        choices=["single", "all"],
        help="Which MMAE samples to process. 'single' = official Track 1 scope; 'all' = full benchmark.",
    )
    # --- sampling (base defaults) ---
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--nfe", type=int, default=32)
    p.add_argument("--cfg-strength", type=float, default=2.0)
    p.add_argument("--sway-coef", type=float, default=-1.0)
    # --- run control ---
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N selected samples (dry-run).",
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split generation across N processes; each writes into the same audio/ dir.",
    )
    p.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="This process's shard index in [0, num-shards).",
    )
    p.add_argument(
        "--manifests-only",
        action="store_true",
        help="Skip generation; just (re)build predictions.json / submission.jsonl / run_meta.json "
        "from the wavs already present in <output-dir>/audio.",
    )
    return p.parse_args()


def die(msg: str, code: int = 2) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(code)


# --------------------------------------------------------------------------- MMAE I/O
# select_samples / extract_instruction_and_audios / abs_audio / rewrite_messages_absolute /
# build_manifests / write_manifests now live in mmae_io.py, shared with run_agent_mmae.py.


# --------------------------------------------------------------------------- main


def main() -> None:
    args = parse_args()

    # ---- resolve paths ----
    config_path = args.config or os.path.join(
        os.path.dirname(os.path.abspath(args.ckpt)), "config.yaml"
    )
    wav_root = args.wav_root or os.path.dirname(os.path.abspath(args.meta))
    audio_dir = os.path.join(args.output_dir, "audio")

    # ---- fail-fast prerequisite checks (before loading the model) ----
    if args.auk_repo:
        src = os.path.join(args.auk_repo, "src")
        if not os.path.isdir(src):
            die(f"--auk-repo/src not found: {src}")
        sys.path.insert(0, src)
    for label, path in [
        ("--ckpt", args.ckpt),
        ("--config", config_path),
        ("--qwen-path", args.qwen_path),
        ("--meta", args.meta),
    ]:
        if not os.path.exists(path):
            die(f"missing {label}: {path}")
    if not os.path.isdir(wav_root):
        die(f"--wav-root is not a directory: {wav_root}")
    if not (0 <= args.shard_id < args.num_shards):
        die(f"--shard-id must be in [0, {args.num_shards}); got {args.shard_id}")

    os.makedirs(audio_dir, exist_ok=True)

    with open(args.meta, encoding="utf-8") as f:
        meta = json.load(f)
    samples = select_samples(meta, args.complexity)
    if args.limit is not None:
        samples = samples[: args.limit]
    print(
        f"[info] complexity={args.complexity} -> {len(samples)} samples "
        f"(shard {args.shard_id}/{args.num_shards})"
    )

    run_meta = {
        "track": "single-model",
        "model": "AuK-base (Prompt Enhancer OFF)",
        "ckpt": os.path.abspath(args.ckpt),
        "config": os.path.abspath(config_path),
        "qwen_path": os.path.abspath(args.qwen_path),
        "duration_policy": "equal-length (no gen_seconds)",
        "seed": args.seed,
        "sampling": {
            "nfe": args.nfe,
            "cfg_strength": args.cfg_strength,
            "sway_sampling_coef": args.sway_coef,
        },
        "complexity": args.complexity,
        "num_selected": len(samples),
        "counts": {"generated": 0, "skipped_existing": 0, "failed": 0},
        "failed_ids": [],
        "reductions": {},  # id -> best-effort reduction note (multi-audio / multi-round in `all` mode)
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    if not args.manifests_only:
        try:
            from tqdm import tqdm  # optional progress bar
        except Exception:

            def tqdm(x, **_):
                return x

        try:
            from auk.infer.infer_auk import AukInfer, save_audio
        except Exception as exc:
            die(
                f"cannot import auk ({exc!r}). Install AuK core deps (pip install -e . in the AuK repo, "
                f"NOT the [gradio] extra) or pass --auk-repo."
            )

        engine = AukInfer(
            config_path,
            args.ckpt,
            device=args.device,
            dtype=args.dtype,
            qwen_path=args.qwen_path,
        )

        for idx, rec in enumerate(tqdm(samples, desc="AuK-base MMAE", unit="clip")):
            if idx % args.num_shards != args.shard_id:
                continue
            sid = rec["id"]
            out_wav = os.path.join(audio_dir, f"{sid}.wav")
            if os.path.isfile(out_wav):
                run_meta["counts"]["skipped_existing"] += 1
                continue

            instruction, audio_urls = extract_instruction_and_audios(
                rec.get("messages", [])
            )
            if (
                len(audio_urls) > 1
                or len([m for m in rec.get("messages", []) if m.get("role") == "user"])
                > 1
            ):
                note = f"reduced: {len(audio_urls)} input audio(s), combined instruction -> use audio1 + joined text"
                run_meta["reductions"][sid] = note
                print(f"[reduce] {sid}: {note}")
            if not audio_urls:
                run_meta["counts"]["failed"] += 1
                run_meta["failed_ids"].append(sid)
                print(f"[fail] {sid}: no input audio in messages", file=sys.stderr)
                continue

            in_wav = abs_audio(audio_urls[0], wav_root)
            if not os.path.isfile(in_wav):
                run_meta["counts"]["failed"] += 1
                run_meta["failed_ids"].append(sid)
                print(f"[fail] {sid}: input audio not found: {in_wav}", file=sys.stderr)
                continue

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "audio", "audio": in_wav},
                    ],
                }
            ]

            try:
                # EQUAL-LENGTH: no gen_seconds -> AuK matches the source clip length.
                audio_out, sr = engine.generate(
                    messages,
                    gen_seconds=None,
                    nfe=args.nfe,
                    cfg_strength=args.cfg_strength,
                    sway_sampling_coef=args.sway_coef,
                    seed=args.seed,
                )
                # atomic write: temp file in the same dir, then os.replace
                tmp = os.path.join(audio_dir, f".{sid}.tmp.wav")
                save_audio(audio_out, sr, tmp)
                os.replace(tmp, out_wav)
                run_meta["counts"]["generated"] += 1
            except Exception:  # per-sample isolation: log, skip, continue
                run_meta["counts"]["failed"] += 1
                run_meta["failed_ids"].append(sid)
                print(
                    f"[fail] {sid}: generation error\n{traceback.format_exc()}",
                    file=sys.stderr,
                )
                continue

    # ---- (re)build manifests from whatever wavs are present, so partial/sharded runs stay consistent ----
    predictions, submission, present = build_manifests(samples, audio_dir, wav_root)
    run_meta["counts"]["present_on_disk"] = len(present)
    write_manifests(args.output_dir, predictions, submission, run_meta)

    print(
        f"[done] generated={run_meta['counts']['generated']} "
        f"skipped={run_meta['counts']['skipped_existing']} "
        f"failed={run_meta['counts']['failed']} "
        f"present={len(present)}/{len(samples)}"
    )
    print(
        f"[done] wrote predictions.json / submission.jsonl / run_meta.json under {args.output_dir}"
    )


if __name__ == "__main__":
    main()
