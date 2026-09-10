#!/usr/bin/env python3
"""Agent-track MMAE runner.

Consumes MMAE-meta.json, routes each sample through the DeepSeek-driven agent
(agent.router.Router -> DSP tools / SAM-Audio / AuK services), and writes the SAME
artifacts as the Single Model baseline so both tracks submit/score identically:

  <output-dir>/audio/<id>.wav     edited audio, one per sample
  <output-dir>/predictions.json   MMAE-evaluator input (chatml messages, absolute audio_urls)
  <output-dir>/submission.jsonl   leaderboard manifest ({"id","audio_path"})
  <output-dir>/traces/<id>.json   per-sample agent trace (routing + tool calls)
  <output-dir>/run_meta.json      reproducibility metadata + counts + agent config

Resumable (skips samples whose audio/<id>.wav exists) and shardable across GPUs.
Reuses select_samples / build_manifests / write_manifests from the Single Model script.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback

# src/ layout: put src on sys.path so `mmae_io` and the `agent` package resolve.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from mmae_io import build_manifests, select_samples, write_manifests  # noqa: E402

from agent.config import RouterConfig  # noqa: E402
from agent.router import Router  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Agent-track MMAE runner (DeepSeek router over DSP/SAM-Audio/AuK) -> evaluator + leaderboard artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--meta", required=True, help="Path to MMAE-meta.json.")
    p.add_argument(
        "--wav-root",
        default=None,
        help="Root for relative MMAE audio_urls. Defaults to the dir of --meta.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Where to write audio/ + manifests + traces.",
    )
    p.add_argument(
        "--complexity",
        default="all",
        choices=["single", "all"],
        help="Which MMAE samples to process. Agent track covers 'all' complexities.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N selected samples (dry-run).",
    )
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument(
        "--manifests-only",
        action="store_true",
        help="Skip routing; rebuild manifests from the wavs already in <output-dir>/audio.",
    )
    # service + LLM overrides (else env / defaults from agent.config)
    p.add_argument("--sam-audio-url", default=None)
    p.add_argument("--auk-url", default=None)
    p.add_argument("--llm-model", default=None)
    p.add_argument("--llm-base-url", default=None)
    p.add_argument("--max-tool-calls", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> RouterConfig:
    cfg = RouterConfig()
    if args.sam_audio_url:
        cfg.services.sam_audio_url = args.sam_audio_url
    if args.auk_url:
        cfg.services.auk_url = args.auk_url
    if args.llm_model:
        cfg.llm.model = args.llm_model
    if args.llm_base_url:
        cfg.llm.base_url = args.llm_base_url
    if args.max_tool_calls is not None:
        cfg.max_tool_calls = args.max_tool_calls
    if args.seed is not None:
        cfg.seed = args.seed
    return cfg


def main() -> None:
    args = parse_args()
    wav_root = args.wav_root or os.path.dirname(os.path.abspath(args.meta))
    audio_dir = os.path.join(args.output_dir, "audio")
    work_root = os.path.join(args.output_dir, "work")
    traces_dir = os.path.join(args.output_dir, "traces")
    for d in (audio_dir, traces_dir):
        os.makedirs(d, exist_ok=True)

    if not os.path.isfile(args.meta):
        print(f"[FATAL] missing --meta: {args.meta}", file=sys.stderr)
        sys.exit(2)
    if not (0 <= args.shard_id < args.num_shards):
        print(f"[FATAL] --shard-id must be in [0,{args.num_shards})", file=sys.stderr)
        sys.exit(2)

    with open(args.meta, encoding="utf-8") as f:
        meta = json.load(f)
    samples = select_samples(meta, args.complexity)
    if args.limit is not None:
        samples = samples[: args.limit]
    print(
        f"[info] complexity={args.complexity} -> {len(samples)} samples (shard {args.shard_id}/{args.num_shards})"
    )

    cfg = build_config(args)
    run_meta = {
        "track": "agent",
        "agent": {
            "llm_model": cfg.llm.model,
            "llm_base_url": cfg.llm.base_url,
            "api_key_env": cfg.llm.api_key_env,
            "tools": ["speed", "volume", "pitch", "separate", "generative_edit"],
            "services": {
                "sam_audio": cfg.services.sam_audio_url,
                "auk": cfg.services.auk_url,
            },
            "max_tool_calls": cfg.max_tool_calls,
            "seed": cfg.seed,
            "compliance_note": (
                "Router uses the DeepSeek hosted API; Track-2 rule #1 bans hosted APIs. "
                "For a compliant submission, point --llm-base-url at a locally-served open-weights LLM."
            ),
        },
        "complexity": args.complexity,
        "num_selected": len(samples),
        "counts": {"generated": 0, "skipped_existing": 0, "failed": 0},
        "failed_ids": [],
        "reductions": {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    if not args.manifests_only:
        try:
            from tqdm import tqdm
        except Exception:

            def tqdm(x, **_):
                return x

        router = Router(cfg)
        for idx, rec in enumerate(tqdm(samples, desc="agent-MMAE", unit="clip")):
            if idx % args.num_shards != args.shard_id:
                continue
            sid = rec["id"]
            out_wav = os.path.join(audio_dir, f"{sid}.wav")
            if os.path.isfile(out_wav):
                run_meta["counts"]["skipped_existing"] += 1
                continue

            work_dir = os.path.join(work_root, str(sid))
            try:
                result = router.route(rec, wav_root, work_dir)
                # persist the trace regardless of outcome
                with open(
                    os.path.join(traces_dir, f"{sid}.json"), "w", encoding="utf-8"
                ) as tf:
                    json.dump(result.trace, tf, ensure_ascii=False, indent=1)
                if result.trace.get("reduction"):
                    run_meta["reductions"][sid] = result.trace["reduction"]

                if not result.output_path or not os.path.isfile(result.output_path):
                    run_meta["counts"]["failed"] += 1
                    run_meta["failed_ids"].append(sid)
                    print(
                        f"[fail] {sid}: {result.trace.get('error') or 'no output audio'}",
                        file=sys.stderr,
                    )
                    continue

                tmp = os.path.join(audio_dir, f".{sid}.tmp.wav")
                shutil.copyfile(result.output_path, tmp)
                os.replace(tmp, out_wav)
                run_meta["counts"]["generated"] += 1
            except Exception:
                run_meta["counts"]["failed"] += 1
                run_meta["failed_ids"].append(sid)
                print(
                    f"[fail] {sid}: routing error\n{traceback.format_exc()}",
                    file=sys.stderr,
                )
                continue

    # Manifests. In a multi-shard GENERATION run, do NOT write from every shard —
    # concurrent writers would race and can corrupt predictions.json. Write only when
    # single-process or in the dedicated --manifests-only assembly pass.
    predictions, submission, present = build_manifests(samples, audio_dir, wav_root)
    run_meta["counts"]["present_on_disk"] = len(present)
    wrote_manifests = args.manifests_only or args.num_shards == 1
    if wrote_manifests:
        write_manifests(args.output_dir, predictions, submission, run_meta)

    c = run_meta["counts"]
    print(
        f"[done] generated={c['generated']} skipped={c['skipped_existing']} failed={c['failed']} "
        f"present={len(present)}/{len(samples)}"
    )
    if wrote_manifests:
        print(
            f"[done] wrote predictions.json / submission.jsonl / run_meta.json + traces/ under {args.output_dir}"
        )
    else:
        print(
            f"[note] shard {args.shard_id}/{args.num_shards} wrote audio + traces only. "
            f"After ALL shards finish, run once with --manifests-only to assemble the manifests."
        )


if __name__ == "__main__":
    main()
