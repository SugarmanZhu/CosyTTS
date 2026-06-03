import base64
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import re

from app.audio import tensor_chunks_to_wav_bytes, tensor_to_pcm16_bytes
from app.config import (
    CORS_ORIGINS,
    DEFAULT_VOICE,
    HOST,
    MODEL_DIR,
    PORT,
    SAMPLE_RATE,
    VERIFY_ATTEMPTS_CAP,
    VERIFY_DEFAULT,
    VERIFY_MAX_ATTEMPTS,
    VERIFY_THRESHOLD,
)
from app.schemas import (
    HealthResponse,
    TTSJsonResponse,
    TTSRequest,
    VerificationInfo,
    VoiceCreated,
    VoiceInfo,
)
from app.voices import VoiceRegistry

# Silence noisy 3rd-party loggers
for noisy in ("numba", "matplotlib", "modelscope", "urllib3", "httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lazy-import engine so /docs is reachable even if model files missing.
    from app.engine import CosyVoiceEngine

    if not MODEL_DIR.exists():
        raise RuntimeError(
            f"Model directory not found: {MODEL_DIR}. "
            "Run `python scripts/download_model.py` first."
        )
    state["engine"] = CosyVoiceEngine()
    state["voices"] = VoiceRegistry()
    yield
    state.clear()


app = FastAPI(
    title="CosyVoice 2 TTS Service",
    description="Local HTTP TTS service with zero-shot voice cloning",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow browser-side clients on the tailnet to call us.
# `allow_credentials=False` because we don't use cookies/sessions; this also
# lets us safely keep allow_origins=["*"] (the two are incompatible per spec).
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Length", "Content-Type"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    engine = state.get("engine")
    info = engine.gpu_info() if engine else {
        "cuda_available": False, "gpu": None, "vram_used_mb": None
    }
    return HealthResponse(
        status="ok",
        model_loaded=engine is not None,
        **info,
    )


@app.post("/voices", response_model=VoiceCreated, status_code=201)
async def create_voice(
    name: Annotated[str, Form(min_length=1, max_length=120)],
    prompt_text: Annotated[str, Form(min_length=1, max_length=2000)],
    audio: Annotated[UploadFile, File(description="Reference audio, any format")],
):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "empty audio upload")
    try:
        meta = state["voices"].register(name, prompt_text, audio_bytes)
    except Exception as e:
        raise HTTPException(400, f"failed to register voice: {e}")
    return VoiceCreated(
        voice_id=meta["voice_id"],
        name=meta["name"],
        sample_rate=meta["sample_rate"],
        duration_s=meta["duration_s"],
    )


@app.get("/voices", response_model=list[VoiceInfo])
async def list_voices():
    return [VoiceInfo(**m) for m in state["voices"].list()]


@app.delete("/voices/{voice_id}", status_code=204)
async def delete_voice(voice_id: str):
    if not state["voices"].delete(voice_id):
        raise HTTPException(404, "voice not found")
    return Response(status_code=204)


def _resolve_voice(speaker: str | None) -> dict:
    """Pick a voice from the registry. Priority:
       1. explicit `speaker`
       2. config.DEFAULT_VOICE
       3. first voice in registry (whatever was created earliest)
    Raises 404 if none of those exist.
    """
    registry: VoiceRegistry = state["voices"]
    candidates = [speaker, DEFAULT_VOICE]
    for vid in candidates:
        if not vid:
            continue
        v = registry.get(vid)
        if v:
            return v
    voices = registry.list()
    if voices:
        return registry.get(voices[0]["voice_id"])
    raise HTTPException(
        404,
        f"no voice available (tried speaker={speaker!r}, default={DEFAULT_VOICE!r}, "
        "registry is empty)",
    )


# Any run of 2+ space-separated ASCII-letter words. Lowercasing these makes
# CosyVoice read them as natural connected speech instead of emphasizing /
# question-intoning capitalized title words ("You Belong With Me" tended to
# come out like a question). Single tokens are NOT matched, so standalone
# acronyms (RTX, USA, NYC) and "RTX 5090" (word+digit) keep their case.
_ENG_PHRASE_RE = re.compile(r"[A-Za-z]+(?:[ \t]+[A-Za-z]+)+")


def _preprocess_text(text: str, simplify_chinese: bool) -> str:
    """Normalize text before it reaches CosyVoice.
    - Traditional → Simplified (model is trained on Simplified)
    - Multi-word English phrases → lowercase (smoother prosody; avoids
      letter-by-letter spelling and questioning intonation on Title Case).
    """
    if simplify_chinese:
        import zhconv
        text = zhconv.convert(text, "zh-cn")
    text = _ENG_PHRASE_RE.sub(lambda m: m.group(0).lower(), text)
    return text


@app.post("/tts")
async def tts(req: TTSRequest):
    voice = _resolve_voice(req.speaker)

    if req.timestamps and req.stream:
        raise HTTPException(400, "timestamps and stream cannot both be true")

    engine = state["engine"]
    prompt_text = voice.get("prompt_text") if req.mode == "zero_shot" else None
    text = _preprocess_text(req.text, req.simplify_chinese)

    def _generate_once() -> list:
        try:
            return list(engine.synth(
                text=text,
                ref_wav_path=voice["ref_path"],
                prompt_text=prompt_text,
                mode=req.mode,
                instruct_text=req.instruct_text,
                stream=req.stream,
                speed=req.speed,
            ))
        except ValueError as e:
            raise HTTPException(400, str(e))

    # --- Streaming path: can't verify (audio leaves before we can judge it) ---
    if req.stream:
        gen = (t for t in _generate_once())
        return StreamingResponse(
            (tensor_to_pcm16_bytes(t) for t in gen),
            media_type=f"audio/L16; rate={SAMPLE_RATE}; channels=1",
        )

    # --- Non-stream path: optional rejection sampling ---
    do_verify = VERIFY_DEFAULT if req.verify is None else req.verify
    max_attempts = req.max_attempts or VERIFY_MAX_ATTEMPTS
    max_attempts = max(1, min(max_attempts, VERIFY_ATTEMPTS_CAP))
    threshold = req.min_score if req.min_score is not None else VERIFY_THRESHOLD
    if not do_verify:
        max_attempts = 1

    best = None  # (score, wav_bytes, heard)
    attempts_used = 0
    for attempt in range(max_attempts):
        attempts_used = attempt + 1
        chunks = _generate_once()
        if not chunks:
            raise HTTPException(500, "no audio generated")
        wav_bytes = tensor_chunks_to_wav_bytes(chunks, SAMPLE_RATE)

        if not do_verify:
            best = (1.0, wav_bytes, "")
            break

        from app.verify import score_pronunciation
        # Judge against the preprocessed text (what we asked it to say)
        score, heard = score_pronunciation(wav_bytes, text, language="zh")
        if best is None or score > best[0]:
            best = (score, wav_bytes, heard)
        if score >= threshold:
            break

    score, wav_bytes, heard = best
    passed = (not do_verify) or score >= threshold
    verification = None
    if do_verify:
        verification = VerificationInfo(
            attempts=attempts_used, score=round(score, 4),
            passed=passed, heard=heard,
        )

    headers = {}
    if do_verify:
        headers["X-TTS-Attempts"] = str(attempts_used)
        headers["X-TTS-Score"] = f"{score:.4f}"
        headers["X-TTS-Passed"] = "1" if passed else "0"

    if req.timestamps:
        from app.alignment import align_chars
        # Remap to the ORIGINAL request text so timestamp.char shows what the
        # caller sent (handles 萧→肖 homophones, case, T/S).
        timestamps = align_chars(wav_bytes, req.text, language="zh")
        duration_s = (len(wav_bytes) - 44) / (SAMPLE_RATE * 2)
        return TTSJsonResponse(
            audio_b64=base64.b64encode(wav_bytes).decode("ascii"),
            sample_rate=SAMPLE_RATE,
            duration_s=round(duration_s, 3),
            speaker=voice["voice_id"],
            timestamps=timestamps,
            verification=verification,
        )

    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)
