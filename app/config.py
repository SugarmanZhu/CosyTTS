from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COSYVOICE_REPO = ROOT / "CosyVoice"
MATCHA_TTS = COSYVOICE_REPO / "third_party" / "Matcha-TTS"
MODEL_DIR = ROOT / "pretrained_models" / "CosyVoice2-0.5B"

VOICES_DIR = ROOT / "data" / "voices"
VOICES_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 24000
REF_SAMPLE_RATE = 16000

import os

# 0.0.0.0 accepts LAN/loopback. Set TTS_HOST=127.0.0.1 to restrict to local.
HOST = os.environ.get("TTS_HOST", "0.0.0.0")
PORT = int(os.environ.get("TTS_PORT", "8765"))

# Used when /tts request omits speaker/voice_id. Falls back to first
# voice in registry if this one doesn't exist.
DEFAULT_VOICE = os.environ.get("TTS_DEFAULT_VOICE", "dj")

# Whisper model used for forced-alignment timestamps. Loaded lazily.
WHISPER_MODEL = os.environ.get("TTS_WHISPER_MODEL", "large-v3")

# Pronunciation verification (rejection sampling). When enabled, generated
# audio is judged by Whisper and regenerated if it scores below threshold.
VERIFY_DEFAULT = os.environ.get("TTS_VERIFY_DEFAULT", "false").lower() == "true"
VERIFY_THRESHOLD = float(os.environ.get("TTS_VERIFY_THRESHOLD", "0.90"))
VERIFY_MAX_ATTEMPTS = int(os.environ.get("TTS_VERIFY_MAX_ATTEMPTS", "3"))
VERIFY_ATTEMPTS_CAP = 6  # hard ceiling regardless of request

# CORS — comma-separated origins. Default "*" lets any web frontend call us
# (fine for a Tailscale-only service since the tailnet is the trust boundary).
# Lock down with e.g. TTS_CORS_ORIGINS="https://app.example.com,http://localhost:3000"
CORS_ORIGINS = [
    o.strip() for o in os.environ.get("TTS_CORS_ORIGINS", "*").split(",") if o.strip()
]

FP16 = False
LOAD_JIT = False
LOAD_TRT = False
