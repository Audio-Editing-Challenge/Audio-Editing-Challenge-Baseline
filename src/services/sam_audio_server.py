"""HTTP /separate server for Meta SAM-Audio — vendored into this repo so the Agent baseline is
self-contained: it needs only an environment with the official `sam_audio` package installed
(github.com/facebookresearch/sam-audio) plus its Hugging Face weights, not any private checkout.

SAM-Audio is a text/visual/span-prompted UNIVERSAL source separator (DiT + flow-matching,
DAC-VAE latent). Given a mixture + a natural-language `description` (lowercase NP/VP, e.g.
"man speaking", "car honking"), it returns the isolated `target` and the `residual` (everything
else) as 48 kHz mono WAV.

Wire contract: multipart POST /separate with `audio_file` (the mix) + `description` (required text
prompt) + `stem` in {target, residual} + optional `predict_spans` / `reranking_candidates` / `span`.

Full mode loads the CLAP+Judge text rankers, the ImageBind visual ranker, and the PE-A-frame span
predictor (enables predict_spans / reranking_candidates>1); it reads `./.checkpoints/imagebind_huge.pth`
relative to the CWD. Lean mode strips them (text-only, ~15 GB, reranking=1 / predict_spans=False).

Launched by services/launch_sam_audio.sh, which sets HF_HOME (weights), LD_LIBRARY_PATH (torchcodec
libav), CUDA_VISIBLE_DEVICES, and the CWD. To run by hand:

    CUDA_VISIBLE_DEVICES=<idx> HF_HOME=/path/to/sam-audio-weights \
        LD_LIBRARY_PATH=/path/to/sam-audio-env/lib \
        /path/to/sam-audio-env/bin/python src/services/sam_audio_server.py \
            --model facebook/sam-audio-large --port 8305   # add --lean for the text-only load
"""
from __future__ import annotations

import argparse
import os
import tempfile
import threading

import torch
import torchaudio
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from sam_audio import SAMAudio, SAMAudioProcessor

app = FastAPI()
STATE = {}
LOCK = threading.Lock()


def _parse_span(span: str):
    """"start,end" -> [[("+", start, end)]] ; empty -> None (text-only, no temporal cond)."""
    span = (span or "").strip()
    if not span:
        return None
    parts = [p for p in span.replace(";", ",").split(",") if p.strip()]
    if len(parts) < 2:
        raise ValueError(f"span must be 'start,end' seconds; got {span!r}")
    start, end = float(parts[0]), float(parts[1])
    return [[("+", start, end)]]


@app.get("/health")
def health():
    proc = STATE.get("processor")
    return {
        "status": "ok",
        "model": STATE.get("model_name"),
        "sample_rate": getattr(proc, "audio_sampling_rate", None),
        "lean": STATE.get("lean"),
        "prompting": ["text", "span", "predict_spans", "reranking"],
    }


@app.post("/separate")
async def separate(
    audio_file: UploadFile = File(...),
    description: str = Form(...),
    stem: str = Form("target"),
    predict_spans: bool = Form(False),
    reranking_candidates: int = Form(1),
    span: str = Form(""),
):
    if stem not in ("target", "residual"):
        return JSONResponse(
            {"error": f"unknown stem '{stem}'; available: ['target', 'residual']"}, status_code=400
        )
    if STATE.get("lean") and (predict_spans or reranking_candidates > 1):
        return JSONResponse(
            {"error": "server started with --lean (rankers/span predictor stripped); "
                      "predict_spans and reranking_candidates>1 are unavailable. "
                      "Restart in full mode to use them."},
            status_code=400,
        )
    try:
        anchors = _parse_span(span)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    suffix = os.path.splitext(audio_file.filename or "in.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(await audio_file.read())
        in_path = tf.name
    out_path = tempfile.mktemp(suffix=".wav")
    try:
        model: SAMAudio = STATE["model"]
        processor: SAMAudioProcessor = STATE["processor"]
        batch = processor(audios=[in_path], descriptions=[description], anchors=anchors).to("cuda")
        with LOCK, torch.inference_mode():
            result = model.separate(
                batch, predict_spans=predict_spans, reranking_candidates=reranking_candidates
            )
        # result.target / result.residual are LISTS of per-item 1-D tensors; index [0].
        wav = (result.target if stem == "target" else result.residual)[0].unsqueeze(0).cpu()
        torchaudio.save(out_path, wav, processor.audio_sampling_rate)
        return FileResponse(out_path, media_type="audio/wav", filename=f"{stem}.wav")
    finally:
        if os.path.exists(in_path):
            os.remove(in_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/sam-audio-large")
    p.add_argument("--port", type=int, default=8305)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--lean", dest="lean", action="store_true",
                   help="strip rankers + span predictor (text-only, ~15 GB, reranking=1 only)")
    p.add_argument("--full", dest="lean", action="store_false",
                   help="load rankers + span predictor (default; enables predict_spans / reranking>1)")
    p.set_defaults(lean=False)
    args = p.parse_args()

    kw = dict(visual_ranker=None, text_ranker=None, span_predictor=None) if args.lean else {}
    STATE["model_name"] = args.model
    STATE["lean"] = args.lean
    STATE["model"] = SAMAudio.from_pretrained(args.model, **kw).eval().cuda()
    STATE["processor"] = SAMAudioProcessor.from_pretrained(args.model)
    uvicorn.run(app, host=args.host, port=args.port)
