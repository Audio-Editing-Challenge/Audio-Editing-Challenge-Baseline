#!/usr/bin/env python3
"""FastAPI service wrapping AuK generative editing with a LOCAL Prompt Enhancer.

Load-once server. On startup it builds `AukInfer` and a `PromptEnhancer` whose ASR
is a local `SenseVoiceSmallASR` (no Tencent hy3 / cloud ASR). Per request it runs
PE (instruction + audio -> enhanced instruction + target duration), then AuK
inference, and returns the edited WAV with the PE metadata in response headers.

Run it with AuK's OWN environment (see services/launch_auk.sh):
    AuK/.venv/bin/python services/auk_server.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import traceback
import uuid
from contextlib import asynccontextmanager
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


AUK_REPO = _env("AUK_REPO", "AuK")
AUK_CKPT = _env(
    "AUK_CKPT", os.path.join(AUK_REPO, "ckpts", "AuK", "auk_base.safetensors")
)
AUK_CONFIG = _env("AUK_CONFIG", os.path.join(os.path.dirname(AUK_CKPT), "config.yaml"))
AUK_QWEN_PATH = _env(
    "AUK_QWEN_PATH", os.path.join(AUK_REPO, "ckpts", "Qwen2.5-Omni-3B")
)
AUK_DEVICE = _env("AUK_DEVICE", "cuda")
AUK_DTYPE = _env("AUK_DTYPE", "bf16")

SENSEVOICE_MODEL = _env(
    "SENSEVOICE_MODEL", "iic/SenseVoiceSmall"
)  # local dir OR modelscope id
SENSEVOICE_DEVICE = _env("SENSEVOICE_DEVICE", "cuda")
SENSEVOICE_NCPU = int(_env("SENSEVOICE_NCPU", "4"))

# PE's LLM (OpenAI-compatible) — defaults to DeepSeek hosted API, key from DEEPSEEK_API_KEY.
PE_LLM_API_KEY = _env("AUK_PE_LLM_API_KEY", os.environ.get("DEEPSEEK_API_KEY"))
PE_LLM_BASE_URL = _env("AUK_PE_LLM_BASE_URL", "https://api.deepseek.com")
PE_LLM_MODEL = _env("AUK_PE_LLM_MODEL", "deepseek-v4-flash")

AUK_NFE = int(_env("AUK_NFE", "32"))
AUK_CFG = float(_env("AUK_CFG", "2.0"))
AUK_SWAY = float(_env("AUK_SWAY", "-1.0"))
AUK_SEED = int(_env("AUK_SEED", "44"))

AUK_HOST = _env("AUK_HOST", "0.0.0.0")
AUK_PORT = int(_env("AUK_PORT", "8310"))
WORK_DIR = _env("AUK_WORK_DIR", os.path.join(tempfile.gettempdir(), "auk_server"))

STATE: dict = {
    "ready": False,
    "error": None,
    "engine": None,
    "pe": None,
    "save_audio": None,
}
LOCK = threading.Lock()


def _load_models() -> None:
    # belt-and-suspenders: make `auk` importable even without an editable install
    src = os.path.join(AUK_REPO or "", "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)

    for label, path in [
        ("AUK_CKPT", AUK_CKPT),
        ("AUK_CONFIG", AUK_CONFIG),
        ("AUK_QWEN_PATH", AUK_QWEN_PATH),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing {label}: {path}")
    if not PE_LLM_API_KEY:
        raise RuntimeError(
            "no LLM API key: set DEEPSEEK_API_KEY (or AUK_PE_LLM_API_KEY) for the Prompt Enhancer"
        )

    from auk.infer.infer_auk import AukInfer, save_audio
    from auk.infer.pe import PromptEnhancer, SenseVoiceSmallASR

    asr = SenseVoiceSmallASR(
        model=SENSEVOICE_MODEL, device=SENSEVOICE_DEVICE, ncpu=SENSEVOICE_NCPU
    )
    # Fail fast if the local SenseVoice weights are missing/broken. The provider is
    # lazy (loads on first transcribe), so force the load now — otherwise a broken
    # deployment would report /health ok and only fail deep inside the first /edit.
    try:
        asr._get_model()
    except Exception as exc:
        raise RuntimeError(
            f"failed to load local SenseVoice model '{SENSEVOICE_MODEL}': {exc}. "
            "Set SENSEVOICE_MODEL to a valid local dir (see services/fetch_sensevoice.sh) "
            "or ensure ModelScope access."
        ) from exc
    pe = PromptEnhancer(
        llm_api_key=PE_LLM_API_KEY,
        llm_base_url=PE_LLM_BASE_URL,
        llm_model=PE_LLM_MODEL,
        asr_provider=asr,  # forces local ASR, bypasses all cloud ASR
    )
    engine = AukInfer(
        AUK_CONFIG,
        AUK_CKPT,
        device=AUK_DEVICE,
        dtype=AUK_DTYPE,
        qwen_path=AUK_QWEN_PATH,
    )

    STATE.update(engine=engine, pe=pe, save_audio=save_audio, ready=True, error=None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(WORK_DIR, exist_ok=True)
    try:
        _load_models()
    except (
        Exception
    ) as exc:  # keep serving /health with the error rather than crash silently
        STATE["error"] = f"{exc}\n{traceback.format_exc()}"
        print(f"[FATAL] AuK service failed to load models: {exc}", file=sys.stderr)
    yield


app = FastAPI(title="AuK edit service (PE local)", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok"
        if STATE["ready"]
        else ("error" if STATE["error"] else "loading"),
        "pe_mode": "local",
        "asr": f"SenseVoiceSmall ({SENSEVOICE_MODEL})",
        "model": "AuK-base (PE on)",
        "llm_model": PE_LLM_MODEL,
        "error": STATE["error"],
    }


@app.post("/edit")
def edit(
    instruction: str = Form(...),
    audio_file: UploadFile | None = File(None),
    gen_seconds: str | None = Form(None),
    seed: str | None = Form(None),
):
    if not STATE["ready"]:
        return JSONResponse(
            {"error": STATE["error"] or "model still loading"}, status_code=503
        )

    req = uuid.uuid4().hex[:12]
    in_path = None
    if audio_file is not None:
        in_path = os.path.join(
            WORK_DIR, f"in_{req}_{os.path.basename(audio_file.filename or 'audio.wav')}"
        )
        with open(in_path, "wb") as f:
            f.write(audio_file.file.read())

    td = None
    if gen_seconds not in (None, "", "None"):
        try:
            td = float(gen_seconds)
        except ValueError:
            return JSONResponse(
                {"error": f"invalid gen_seconds: {gen_seconds!r}"}, status_code=400
            )
    use_seed = int(seed) if seed not in (None, "", "None") else AUK_SEED

    out_path = os.path.join(WORK_DIR, f"out_{req}.wav")
    try:
        with LOCK:
            pe_out = STATE["pe"].prepare(
                instruction, audio_path=in_path, target_duration=td
            )
            model_audio = pe_out.audio or in_path
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": pe_out.instruction},
                        *(
                            [{"type": "audio", "audio": model_audio}]
                            if model_audio
                            else []
                        ),
                    ],
                }
            ]
            audio_out, sr = STATE["engine"].generate(
                messages,
                audio=model_audio,
                gen_seconds=pe_out.gen_seconds,
                nfe=AUK_NFE,
                cfg_strength=AUK_CFG,
                sway_sampling_coef=AUK_SWAY,
                seed=use_seed,
            )
            STATE["save_audio"](audio_out, sr, out_path)
    except Exception as exc:
        return JSONResponse(
            {"error": f"{exc}", "trace": traceback.format_exc()[-1500:]},
            status_code=500,
        )

    headers = {
        "X-Enhanced-Instruction": quote(pe_out.instruction or ""),
        "X-Gen-Seconds": str(pe_out.gen_seconds),
        "X-Task-Type": str(getattr(pe_out, "task_type", "") or ""),
    }

    def _cleanup():
        for p in (in_path, out_path):
            try:
                if p and os.path.isfile(p):
                    os.remove(p)
            except OSError:
                pass

    return FileResponse(
        out_path,
        media_type="audio/wav",
        filename="edited.wav",
        headers=headers,
        background=BackgroundTask(_cleanup),
    )


if __name__ == "__main__":
    uvicorn.run(app, host=AUK_HOST, port=AUK_PORT, log_level="info")
