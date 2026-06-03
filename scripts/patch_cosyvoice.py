"""Patch CosyVoice's load_wav for Windows + torchaudio >= 2.9.

torchaudio 2.9 routes torchaudio.load() through torchcodec, which fails to
load on Windows when the conda-forge FFmpeg ABI doesn't match torchcodec's
bundled core DLLs. We replace the file load with a direct soundfile read
(no torchcodec dependency), keeping the rest of load_wav intact.

Idempotent: safe to run multiple times.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "CosyVoice" / "cosyvoice" / "utils" / "file_utils.py"

SENTINEL = "# [CosyTTS] soundfile load patch"

ORIGINAL = (
    "    speech, sample_rate = torchaudio.load(wav, backend='soundfile')\n"
    "    speech = speech.mean(dim=0, keepdim=True)\n"
)

REPLACEMENT = (
    f"    {SENTINEL}\n"
    "    import soundfile as _sf\n"
    "    import numpy as _np\n"
    "    if isinstance(wav, (str, bytes)) or hasattr(wav, 'read'):\n"
    "        _data, sample_rate = _sf.read(wav, dtype='float32', always_2d=True)\n"
    "        speech = torch.from_numpy(_np.ascontiguousarray(_data.T))\n"
    "    else:\n"
    "        speech, sample_rate = torchaudio.load(wav, backend='soundfile')\n"
    "    speech = speech.mean(dim=0, keepdim=True)\n"
)


def main():
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found. Clone CosyVoice first "
              f"(git clone --recursive ...).", file=sys.stderr)
        sys.exit(1)

    text = TARGET.read_text(encoding="utf-8")

    if SENTINEL in text:
        print("Already patched. Nothing to do.")
        return

    if ORIGINAL not in text:
        print("WARNING: expected original load_wav body not found. The upstream "
              "code may have changed — patch manually (see README step 7).",
              file=sys.stderr)
        sys.exit(2)

    TARGET.write_text(text.replace(ORIGINAL, REPLACEMENT), encoding="utf-8")
    print(f"Patched {TARGET}")


if __name__ == "__main__":
    main()
