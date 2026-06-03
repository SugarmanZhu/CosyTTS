"""Example: how another project calls this TTS service.

Run the service first:
    uvicorn app.main:app --host 127.0.0.1 --port 8765

Then run this script with a reference WAV file:
    python examples/client.py path/to/reference.wav "transcript of reference audio"
"""
import sys
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8765"


def register_voice(ref_path: str, prompt_text: str, name: str = "demo") -> str:
    with open(ref_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/voices",
            files={"audio": (Path(ref_path).name, f, "audio/wav")},
            data={"name": name, "prompt_text": prompt_text},
            timeout=30,
        )
    r.raise_for_status()
    return r.json()["voice_id"]


def synthesize(text: str, voice_id: str, out_path: str = "out.wav"):
    r = requests.post(
        f"{BASE_URL}/tts",
        json={"text": text, "voice_id": voice_id, "mode": "zero_shot"},
        timeout=120,
    )
    r.raise_for_status()
    Path(out_path).write_bytes(r.content)
    return out_path


def main():
    if len(sys.argv) < 3:
        print("usage: python client.py <ref_wav> <ref_transcript> [text_to_synth]")
        sys.exit(1)
    ref_wav = sys.argv[1]
    prompt_text = sys.argv[2]
    text = sys.argv[3] if len(sys.argv) > 3 else "你好,这是一段用克隆音色合成的语音。"

    print(f"Registering voice from {ref_wav}...")
    voice_id = register_voice(ref_wav, prompt_text)
    print(f"voice_id = {voice_id}")

    print(f"Synthesizing: {text!r}")
    out = synthesize(text, voice_id, "out.wav")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
