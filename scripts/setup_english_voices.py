"""Download two public-domain LibriVox recordings and register them as
`english_female` and `english_male` voices.

Public domain (LibriVox), so free to clone:
  - english_female: "Benigna Machiavelli" read by Winnifred Assmann
  - english_male:   "The Legend of Sleepy Hollow" read by Bob Neufeld

Requires the TTS service to be running (see run.ps1) and ffmpeg on PATH
(run.ps1 puts the bundled one there). Run from the project root:

    python scripts/setup_english_voices.py
"""
import subprocess
import sys
import urllib.request
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8765"

# (voice name, source URL, extract start sec, duration sec)
VOICES = [
    ("english_female",
     "https://archive.org/download/benigna_machiavelli_2208_librivox/benignamachiavelli_01_gilman.mp3",
     60, 11),
    ("english_male",
     "https://archive.org/download/sleepyhollow_1206_librivox/legendofsleepyhollow_01_irving.mp3",
     90, 11),
]


def ffmpeg() -> str:
    local = ROOT / ".venv" / "Library" / "bin" / "ffmpeg.exe"
    return str(local) if local.exists() else "ffmpeg"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import whisper, torch
    print("Loading Whisper (for reference transcripts)...", file=sys.stderr)
    model = whisper.load_model("large-v3",
                               device="cuda" if torch.cuda.is_available() else "cpu")

    for name, url, start, dur in VOICES:
        src = ROOT / f"{name}_src.mp3"
        seg = ROOT / f"{name}_ref.wav"
        print(f"\n[{name}] downloading {url}")
        urllib.request.urlretrieve(url, src)

        print(f"[{name}] extracting {dur}s from {start}s, 16k mono")
        subprocess.run([ffmpeg(), "-y", "-v", "error", "-ss", str(start),
                        "-t", str(dur), "-i", str(src), "-ac", "1", "-ar", "16000",
                        str(seg)], check=True)

        prompt_text = model.transcribe(str(seg), language="en").get("text", "").strip()
        print(f"[{name}] transcript: {prompt_text!r}")

        requests.delete(f"{BASE}/voices/{name}")
        with open(seg, "rb") as f:
            r = requests.post(f"{BASE}/voices",
                              files={"audio": (seg.name, f, "audio/wav")},
                              data={"name": name, "prompt_text": prompt_text},
                              timeout=60)
        r.raise_for_status()
        print(f"[{name}] registered: {r.json()}")
        src.unlink(missing_ok=True)
        seg.unlink(missing_ok=True)

    print("\nDone. Try:  curl -X POST", f"{BASE}/tts",
          '-H "Content-Type: application/json"',
          '-d \'{"text":"Hello world.","speaker":"english_male"}\' -o en.wav')


if __name__ == "__main__":
    main()
