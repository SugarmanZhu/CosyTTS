import os
import sys
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import soundfile as sf
import torch

from app.config import COSYVOICE_REPO, MATCHA_TTS, MODEL_DIR, FP16, LOAD_JIT, LOAD_TRT

# Make CUDA/cuDNN DLLs (bundled with the torch wheel) and the conda FFmpeg
# DLLs findable via Win32 LoadLibrary. Doing this in-process means the service
# works no matter how it's launched (uvicorn, run.ps1, or directly under NSSM
# with a minimal PATH) — onnxruntime-gpu and torchaudio find their DLLs.
_TORCH_LIB = Path(torch.__file__).resolve().parent / "lib"
_LIBRARY_BIN = Path(sys.executable).resolve().parent.parent / "Library" / "bin"
for _d in (_TORCH_LIB, _LIBRARY_BIN):
    if _d.is_dir():
        try:
            os.add_dll_directory(str(_d))
        except (AttributeError, OSError):
            pass
        os.environ["PATH"] = str(_d) + os.pathsep + os.environ.get("PATH", "")

# CosyVoice imports rely on sys.path containing the repo root and Matcha-TTS.
sys.path.insert(0, str(MATCHA_TTS))
sys.path.insert(0, str(COSYVOICE_REPO))

from cosyvoice.cli.cosyvoice import CosyVoice2  # noqa: E402


def load_wav(path: str, target_sr: int) -> torch.Tensor:
    """Load WAV via soundfile + torch resample.

    Replaces CosyVoice's torchaudio-based loader, which on torchaudio 2.9+
    routes through torchcodec and breaks on Windows when FFmpeg DLLs don't
    match torchcodec's bundled core. soundfile reads the file directly.
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    # to mono
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    speech = torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0)
    if sr != target_sr:
        import torchaudio.transforms as T
        speech = T.Resample(orig_freq=sr, new_freq=target_sr)(speech)
    return speech


class CosyVoiceEngine:
    """Loads CosyVoice2 once; reused across requests."""

    def __init__(
        self,
        model_dir: str = str(MODEL_DIR),
        fp16: bool = FP16,
        load_jit: bool = LOAD_JIT,
        load_trt: bool = LOAD_TRT,
    ):
        self.model = CosyVoice2(
            model_dir, load_jit=load_jit, load_trt=load_trt, fp16=fp16
        )
        self.sample_rate = self.model.sample_rate

    def synth(
        self,
        text: str,
        ref_wav_path: str,
        prompt_text: Optional[str] = None,
        mode: str = "zero_shot",
        instruct_text: Optional[str] = None,
        stream: bool = False,
        speed: float = 1.0,
    ) -> Iterator[torch.Tensor]:
        ref = load_wav(ref_wav_path, 16000)

        if mode == "zero_shot":
            if not prompt_text:
                raise ValueError("zero_shot mode requires prompt_text")
            gen = self.model.inference_zero_shot(
                text, prompt_text, ref, stream=stream, speed=speed
            )
        elif mode == "cross_lingual":
            gen = self.model.inference_cross_lingual(
                text, ref, stream=stream, speed=speed
            )
        elif mode == "instruct2":
            if not instruct_text:
                raise ValueError("instruct2 mode requires instruct_text")
            gen = self.model.inference_instruct2(
                text, instruct_text, ref, stream=stream, speed=speed
            )
        else:
            raise ValueError(f"unknown mode: {mode}")

        for chunk in gen:
            yield chunk["tts_speech"]

    def gpu_info(self) -> dict:
        if not torch.cuda.is_available():
            return {"cuda_available": False, "gpu": None, "vram_used_mb": None}
        return {
            "cuda_available": True,
            "gpu": torch.cuda.get_device_name(0),
            "vram_used_mb": round(torch.cuda.memory_allocated(0) / 1024 / 1024, 1),
        }
