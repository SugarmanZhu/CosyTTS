import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


TARGET_PEAK_DBFS = -1.0   # output / reference peak target
EPS = 1e-9


def peak_normalize(x: np.ndarray, target_dbfs: float = TARGET_PEAK_DBFS) -> np.ndarray:
    """Scale array so its absolute peak hits target_dbfs (default -1 dBFS).

    Skips silent/all-zero inputs. Operates on float arrays in [-1, 1].
    """
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak < EPS:
        return x
    target_amp = 10.0 ** (target_dbfs / 20.0)  # -1 dBFS -> ~0.891
    return (x * (target_amp / peak)).astype(np.float32, copy=False)


def _decode_with_ffmpeg(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode arbitrary container (m4a/mp3/aac/...) to float32 PCM via ffmpeg.

    Writes the upload to a temp file so ffmpeg can probe it (stdin probing
    is unreliable for some containers), then asks ffmpeg for s16le 16kHz mono.
    """
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", tmp_path,
                "-f", "s16le", "-ac", "1", "-ar", "16000", "-",
            ],
            capture_output=True, check=True,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm, 16000


def normalize_to_mono_16k(wav_bytes: bytes, out_path) -> tuple[int, float]:
    """Read uploaded audio, resample to 16kHz mono, save as 16-bit PCM WAV.

    Tries soundfile first (fast, native: WAV/FLAC/OGG/AIFF). Falls back to
    ffmpeg for compressed containers (M4A/AAC/MP3/etc.) that libsndfile
    can't decode.

    Returns (sample_rate, duration_seconds).
    """
    try:
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
        if data.shape[1] > 1:
            data = data.mean(axis=1)
        else:
            data = data[:, 0]
        if sr != 16000:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    except (sf.LibsndfileError, RuntimeError):
        data, sr = _decode_with_ffmpeg(wav_bytes)

    data = peak_normalize(data)  # normalize ref so spk embedding sees a sane level

    duration = len(data) / 16000.0
    sf.write(str(out_path), data, 16000, subtype="PCM_16")
    return 16000, duration


def tensor_chunks_to_wav_bytes(
    chunks: list[torch.Tensor], sample_rate: int, normalize: bool = True
) -> bytes:
    """Concatenate float32 tensor chunks ([1, N]) and encode as 16-bit PCM WAV bytes.

    When ``normalize`` is True (default), the entire utterance is peak-normalized
    to TARGET_PEAK_DBFS so quiet voice clones come out at consistent loudness.
    """
    arr = torch.cat(chunks, dim=1).squeeze(0).cpu().numpy()
    if normalize:
        arr = peak_normalize(arr)
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, pcm, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def tensor_to_pcm16_bytes(t: torch.Tensor, gain: float = 1.0) -> bytes:
    """Encode a single tensor chunk as raw int16 PCM bytes (for streaming).

    Per-chunk gain is applied before clipping. For streaming we can't do a true
    full-utterance peak normalize (we don't know the global peak in advance),
    so callers pass a pre-computed gain factor (e.g. derived from the reference
    audio's amplitude or a fixed boost for known-quiet voices).
    """
    arr = t.squeeze(0).cpu().numpy()
    if gain != 1.0:
        arr = arr * gain
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype(np.int16)
    return pcm.tobytes()
