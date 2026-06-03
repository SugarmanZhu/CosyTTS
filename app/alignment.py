"""Forced alignment: derive per-character timestamps from a synthesized WAV
using Whisper. Loaded lazily so the service doesn't pay the cost on startup.
"""
import io
import threading
from typing import Optional

import numpy as np
import soundfile as sf

from app.config import WHISPER_MODEL


_lock = threading.Lock()
_model = None  # type: ignore[var-annotated]


def _get_model():
    """Load Whisper once, lazily. Returns the loaded model."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                import whisper
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _model = whisper.load_model(WHISPER_MODEL, device=device)
    return _model


def align_chars(
    wav_bytes: bytes,
    text: str,
    language: str = "zh",
) -> list[dict]:
    """Run Whisper on ``wav_bytes`` with word timestamps and split into chars.

    Returns ``[{"char": str, "start": float, "end": float}, ...]`` with times
    in seconds from the start of the audio.

    Strategy:
    - Whisper with ``word_timestamps=True`` returns per-word timing.
    - For Chinese, each "word" is typically 1–3 characters; we distribute the
      word duration evenly across its characters.
    - English/punctuation chars in the source text are mixed in by matching
      Whisper's word output back against the original ``text`` greedily.
    """
    model = _get_model()

    # Decode WAV bytes to float32 mono 16k array for whisper
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    if sr != 16000:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)

    result = model.transcribe(
        data.astype(np.float32),
        language=language,
        word_timestamps=True,
        initial_prompt=text[:200],  # bias decoding toward the known text
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        verbose=None,
    )

    out: list[dict] = []
    # "Pending glue": when Whisper emits a lone apostrophe / hyphen / curly
    # quote, we hold it here and attach it to the next Latin token so
    # "'97" -> ['97], "rock-n-roll" -> [rock-n-roll], etc.
    pending: Optional[dict] = None

    def flush_pending():
        nonlocal pending
        if pending is not None:
            out.append(pending)
            pending = None

    for segment in result.get("segments", []):
        for word in segment.get("words", []) or []:
            w_raw = word.get("word", "")
            w = w_raw.strip()
            start = float(word.get("start", 0.0))
            end = float(word.get("end", start))
            if not w:
                continue
            # Whisper occasionally returns 0-duration tokens (e.g. "97"
            # at a segment boundary). Don't drop them — give them a sliver
            # so downstream still sees the token.
            if end <= start:
                end = start + 0.01

            has_latin = any(_is_latin_or_digit(c) for c in w)
            is_glue = all(c in _GLUE_CHARS for c in w)

            # 1. Standalone glue (' - – ’) -> hold for the next Latin token
            if is_glue and not has_latin:
                # If glue follows a Latin token with no leading space, attach
                # directly to it (e.g. "Eminem" + "'s" boundary).
                if (
                    bool(out)
                    and _is_latin_token(out[-1]["char"])
                    and not (w_raw != w_raw.lstrip(" "))
                ):
                    out[-1]["char"] += w
                    out[-1]["end"] = round(end, 3)
                    continue
                flush_pending()
                pending = {
                    "char": w,
                    "start": round(start, 3),
                    "end": round(end, 3),
                }
                continue

            # 2. Whisper's BPE continuation fragment -> merge into previous
            if (
                has_latin
                and bool(out)
                and _is_latin_token(out[-1]["char"])
                and not (w_raw != w_raw.lstrip(" "))
            ):
                out[-1]["char"] += w
                out[-1]["end"] = round(end, 3)
                pending = None
                continue

            # 3. New Latin token; absorb any pending glue ('97 etc.)
            if has_latin:
                if pending is not None:
                    out.append({
                        "char": pending["char"] + w,
                        "start": pending["start"],
                        "end": round(end, 3),
                    })
                    pending = None
                else:
                    out.append({
                        "char": w,
                        "start": round(start, 3),
                        "end": round(end, 3),
                    })
                continue

            # 4. Pure CJK (+ Chinese punctuation, maybe glue at edges).
            # Edge case: Whisper sometimes merges a Chinese title-opener with
            # a following ASCII apostrophe into one token like "《'" — peel
            # any trailing glue off and hold it as pending for the next Latin.
            flush_pending()
            chars = [c for c in w if c.strip()]
            if not chars:
                continue

            trailing_glue: list[str] = []
            while chars and chars[-1] in _GLUE_CHARS:
                trailing_glue.append(chars.pop())
            trailing_glue.reverse()

            duration = max(end - start, 0.001)
            total_units = len(chars) + len(trailing_glue)
            per_unit = duration / total_units
            for i, c in enumerate(chars):
                out.append({
                    "char": c,
                    "start": round(start + i * per_unit, 3),
                    "end": round(start + (i + 1) * per_unit, 3),
                })
            if trailing_glue:
                glue_start = start + len(chars) * per_unit
                pending = {
                    "char": "".join(trailing_glue),
                    "start": round(glue_start, 3),
                    "end": round(end, 3),
                }
    flush_pending()
    # Remap whisper's transcribed chars back to the original input chars
    # (preserves rare CJK characters like 萧 that whisper confuses with
    # high-frequency homophones like 肖).
    out = _remap_to_input(out, text)
    return out


def _remap_to_input(timestamps: list[dict], input_text: str) -> list[dict]:
    """Replace each unit's ``char`` with the corresponding slice of the
    original input text, keeping Whisper's timing.

    Strategy: strip whitespace from both Whisper's reconstructed text and
    the input. If their lengths match (the common case), walk positionally.
    Otherwise fall back to difflib for fuzzy alignment.
    """
    if not timestamps:
        return timestamps

    import re
    from difflib import SequenceMatcher

    whisper_text = "".join(ts["char"] for ts in timestamps)
    whisper_compact = re.sub(r"\s+", "", whisper_text)
    input_compact = re.sub(r"\s+", "", input_text)

    if len(whisper_compact) == len(input_compact):
        # Fast path: 1:1 char alignment.
        out: list[dict] = []
        idx = 0
        for ts in timestamps:
            unit = ts["char"]
            unit_compact = re.sub(r"\s+", "", unit)
            n = len(unit_compact)
            new_char = input_compact[idx : idx + n]
            idx += n
            out.append({**ts, "char": new_char if new_char else unit})
        return out

    # Slow path: align via SequenceMatcher on the compacted strings.
    matcher = SequenceMatcher(None, whisper_compact, input_compact, autojunk=False)
    char_map: dict[int, int] = {}
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        wlen = i2 - i1
        ilen = j2 - j1
        if op == "equal":
            for k in range(wlen):
                char_map[i1 + k] = j1 + k
        elif op == "replace":
            n = max(wlen, ilen)
            for k in range(n):
                wi = min(i1 + k, i2 - 1) if wlen else None
                ji = min(j1 + k, j2 - 1) if ilen else None
                if wi is not None and ji is not None:
                    char_map[wi] = ji
        # 'insert' / 'delete': leave whisper chars without a mapping

    out = []
    cursor = 0
    for ts in timestamps:
        unit_compact = re.sub(r"\s+", "", ts["char"])
        n = len(unit_compact)
        mapped = [char_map.get(cursor + k) for k in range(n)]
        mapped = [m for m in mapped if m is not None]
        if mapped:
            new_char = input_compact[min(mapped) : max(mapped) + 1]
        else:
            new_char = ts["char"]
        cursor += n
        out.append({**ts, "char": new_char})
    return out


# Characters that act as word-internal "glue" — should not break an English
# token. Includes straight apostrophe, curly apostrophe, ASCII hyphen, en/em
# dashes. Period is NOT here (sentence boundary semantics).
_GLUE_CHARS = frozenset("'-’–—")


def _is_latin_token(token: str) -> bool:
    """A previously-emitted unit that is purely Latin/digit (a complete
    English word/number we might still want to extend)."""
    return any(_is_latin_or_digit(c) for c in token) and all(
        _is_latin_or_digit(c) or not c.strip() for c in token
    )


def _is_latin_or_digit(c: str) -> bool:
    """Letters A-Z/a-z and digits 0-9 — anything Whisper would tokenize as a
    multi-char English word/number rather than per-char CJK."""
    return (
        ("a" <= c <= "z")
        or ("A" <= c <= "Z")
        or ("0" <= c <= "9")
    )


def is_loaded() -> bool:
    return _model is not None
