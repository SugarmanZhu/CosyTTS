"""Pronunciation verification for rejection-sampling.

CosyVoice 2 is non-deterministic — the same text yields different audio each
call, and occasionally mispronounces a word (e.g. a cloned voice saying
"bel long" instead of "belong", or "Taylor" coming out as "Filler").

We use Whisper as an impartial judge: transcribe the generated audio WITHOUT
feeding it the expected text (no initial_prompt — otherwise Whisper "cheats"
and writes the expected words regardless of how they actually sound), then
compare phonetically against the reference. A low score means the synthesis
garbled something; the caller can regenerate (rejection sampling).

Phonetic (not literal) comparison is essential so we don't false-flag:
  - Whisper homophone substitution (萧→肖, both pinyin "xiao") — must pass
  - punctuation / spacing / bracket differences — ignored
  - English case — ignored
while still catching real errors (叙事 xùshì → 事实 shìshí, Taylor → Filler).
"""
import io
import re
from difflib import SequenceMatcher
from typing import Optional

import numpy as np
import soundfile as sf

_PUNCT_RE = re.compile(r"[^a-z0-9]+")


def phonetic_key(text: str) -> str:
    """Reduce text to a comparable phonetic string: CJK→pinyin (toneless),
    ASCII letters/digits kept lowercase, everything else dropped."""
    from pypinyin import lazy_pinyin, Style

    # lazy_pinyin keeps non-Chinese runs (English words, numbers) intact.
    toks = lazy_pinyin(text, style=Style.NORMAL)
    out = []
    for t in toks:
        t = _PUNCT_RE.sub("", t.lower())
        if t:
            out.append(t)
    return "".join(out)


def similarity(reference: str, hypothesis: str) -> float:
    """Phonetic similarity in [0, 1]. 1.0 = identical pronunciation."""
    a = phonetic_key(reference)
    b = phonetic_key(hypothesis)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _wav_bytes_to_array(wav_bytes: bytes) -> np.ndarray:
    arr, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    arr = arr.mean(axis=1) if arr.shape[1] > 1 else arr[:, 0]
    if sr != 16000:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
    return arr.astype(np.float32)


def transcribe_unbiased(wav_bytes: bytes, language: str = "zh") -> str:
    """Transcribe WITHOUT initial_prompt so mispronunciations surface as text."""
    from app.alignment import _get_model

    model = _get_model()
    arr = _wav_bytes_to_array(wav_bytes)
    result = model.transcribe(
        arr,
        language=language,
        word_timestamps=False,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        verbose=None,
    )
    return result.get("text", "").strip()


def score_pronunciation(
    wav_bytes: bytes, reference_text: str, language: str = "zh"
) -> tuple[float, str]:
    """Return (phonetic_similarity, unbiased_transcription)."""
    hyp = transcribe_unbiased(wav_bytes, language)
    return similarity(reference_text, hyp), hyp
