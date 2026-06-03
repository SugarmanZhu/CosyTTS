"""Transcribe an audio file with Whisper. Used to obtain prompt_text for voice cloning."""
import sys
from pathlib import Path

import whisper


def main():
    if len(sys.argv) < 2:
        print("usage: python transcribe.py <audio_file> [model_name=large-v3]")
        sys.exit(1)
    path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "large-v3"

    sys.stdout.reconfigure(encoding="utf-8")
    print(f"Loading whisper {model_name}...", file=sys.stderr)
    model = whisper.load_model(model_name, device="cuda")

    print(f"Transcribing {path}...", file=sys.stderr)
    result = model.transcribe(
        path,
        language="zh",
        initial_prompt="以下是普通话的句子,请使用简体中文输出。",
    )

    print("---TRANSCRIPT---")
    print(result["text"].strip())
    print("---SEGMENTS---")
    for s in result["segments"]:
        print(f"{s['start']:.2f}-{s['end']:.2f}: {s['text'].strip()}")


if __name__ == "__main__":
    main()
