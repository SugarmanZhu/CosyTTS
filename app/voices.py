import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.audio import normalize_to_mono_16k
from app.config import VOICES_DIR


SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _slugify(name: str) -> str:
    s = SLUG_RE.sub("-", name.strip()).strip("-")
    return s.lower() or "voice"


class VoiceRegistry:
    """Filesystem-backed voice registry.

    Layout: data/voices/<voice_id>/{ref.wav, meta.json}
    voice_id = slugified name; if collision, append short uuid.
    """

    def __init__(self, root: Path = VOICES_DIR):
        self.root = root

    def _voice_dir(self, voice_id: str) -> Path:
        return self.root / voice_id

    def register(self, name: str, prompt_text: str, audio_bytes: bytes) -> dict:
        voice_id = _slugify(name)
        if self._voice_dir(voice_id).exists():
            voice_id = f"{voice_id}-{uuid.uuid4().hex[:6]}"
        vdir = self._voice_dir(voice_id)
        vdir.mkdir(parents=True, exist_ok=False)

        ref_path = vdir / "ref.wav"
        sr, duration = normalize_to_mono_16k(audio_bytes, ref_path)

        meta = {
            "voice_id": voice_id,
            "name": name,
            "prompt_text": prompt_text,
            "sample_rate": sr,
            "duration_s": duration,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (vdir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return meta

    def get(self, voice_id: str) -> Optional[dict]:
        vdir = self._voice_dir(voice_id)
        meta_path = vdir / "meta.json"
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["ref_path"] = str(vdir / "ref.wav")
        return meta

    def list(self) -> list[dict]:
        out = []
        for vdir in sorted(self.root.iterdir()):
            if not vdir.is_dir():
                continue
            meta_path = vdir / "meta.json"
            if meta_path.exists():
                out.append(json.loads(meta_path.read_text(encoding="utf-8")))
        return out

    def delete(self, voice_id: str) -> bool:
        vdir = self._voice_dir(voice_id)
        if not vdir.exists():
            return False
        for p in vdir.iterdir():
            p.unlink()
        vdir.rmdir()
        return True
