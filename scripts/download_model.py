"""Download CosyVoice2-0.5B model from HuggingFace into pretrained_models/."""
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "pretrained_models" / "CosyVoice2-0.5B"

if __name__ == "__main__":
    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"Downloading CosyVoice2-0.5B to {TARGET} ...")
    snapshot_download(
        repo_id="FunAudioLLM/CosyVoice2-0.5B",
        local_dir=str(TARGET),
    )
    print("Done.")
