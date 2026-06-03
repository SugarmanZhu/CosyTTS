"""Streaming client — measures time-to-first-byte and saves output as WAV.

Usage:
    python examples/stream_client.py <voice_id> "<text to synthesize>"

If voice_id doesn't exist, register one first via /voices.
"""
import io
import struct
import sys
import time
import wave
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8765"
SAMPLE_RATE = 24000  # CosyVoice2 output SR
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # int16


def stream_synth(text: str, voice_id: str, out_path: str = "stream_out.wav"):
    t0 = time.perf_counter()
    req_body = {"text": text, "voice_id": voice_id, "mode": "zero_shot", "stream": True}

    print(f"POST /tts (stream=true) — voice_id={voice_id}, text={text!r}")

    with requests.post(f"{BASE_URL}/tts", json=req_body, stream=True, timeout=120) as r:
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        print(f"Content-Type: {ct}")

        pcm_buf = bytearray()
        first_byte_t = None
        chunk_count = 0
        chunk_sizes: list[int] = []

        for chunk in r.iter_content(chunk_size=None):  # None = native chunks
            if not chunk:
                continue
            if first_byte_t is None:
                first_byte_t = time.perf_counter()
                print(f"  TTFB: {(first_byte_t - t0) * 1000:.1f} ms")
            pcm_buf.extend(chunk)
            chunk_count += 1
            chunk_sizes.append(len(chunk))

        t_end = time.perf_counter()

    if not pcm_buf:
        print("ERROR: empty stream", file=sys.stderr)
        sys.exit(1)

    # Wrap raw PCM in a WAV header so out_path is playable
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH_BYTES)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(bytes(pcm_buf))

    audio_seconds = len(pcm_buf) / (SAMPLE_RATE * SAMPLE_WIDTH_BYTES * CHANNELS)
    wall_seconds = t_end - t0
    ttfb_ms = (first_byte_t - t0) * 1000
    rtf = wall_seconds / audio_seconds if audio_seconds else float("inf")

    print(
        f"\n--- streaming stats ---\n"
        f"  audio duration   : {audio_seconds:.2f} s\n"
        f"  total wall time  : {wall_seconds:.2f} s\n"
        f"  TTFB             : {ttfb_ms:.1f} ms\n"
        f"  RTF              : {rtf:.3f}  (lower = faster than realtime)\n"
        f"  chunks received  : {chunk_count}\n"
        f"  avg chunk size   : {sum(chunk_sizes) // max(1, chunk_count)} B\n"
        f"  output file      : {Path(out_path).resolve()}  ({len(pcm_buf) + 44} bytes)\n"
    )


def main():
    if len(sys.argv) < 2:
        print('usage: python stream_client.py <voice_id> ["text to synthesize"]')
        sys.exit(1)
    voice_id = sys.argv[1]
    text = sys.argv[2] if len(sys.argv) > 2 else (
        "流式合成的延迟测试,看一下第一个字节从请求发出到收到的时间。"
    )
    stream_synth(text, voice_id, "stream_out.wav")


if __name__ == "__main__":
    main()
