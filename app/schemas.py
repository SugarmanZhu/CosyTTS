from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    # `speaker` is the canonical name; `voice_id` is kept as an alias for
    # backward compatibility. Either is optional — when both are omitted,
    # the server falls back to config.DEFAULT_VOICE.
    speaker: Optional[str] = None
    voice_id: Optional[str] = None
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech rate multiplier")
    mode: Literal["zero_shot", "cross_lingual", "instruct2"] = "zero_shot"
    instruct_text: Optional[str] = None
    stream: bool = False
    # When true, response is JSON {audio_b64, sample_rate, duration, timestamps}
    # instead of audio/wav. Costs ~0.3-1.0 s extra for whisper alignment.
    timestamps: bool = False
    # Convert Traditional Chinese characters to Simplified before synthesis.
    # CosyVoice 2 is trained on Simplified text; raw Traditional input causes
    # stutters and mispronunciation. Default on. Disable for explicit raw mode.
    simplify_chinese: bool = True
    # Pronunciation verification (rejection sampling). When true, the server
    # judges each generated take with Whisper and regenerates (up to
    # max_attempts) if it scores below the configured threshold. Non-stream
    # only. None = use server default (TTS_VERIFY_DEFAULT).
    verify: Optional[bool] = None
    max_attempts: Optional[int] = Field(None, ge=1, le=6)
    # Override the accept threshold for this request (phonetic similarity 0-1).
    # Higher = stricter = more retries. None = server default.
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _coalesce_speaker(self):
        # Treat speaker as the source of truth; fall back to voice_id.
        if not self.speaker and self.voice_id:
            self.speaker = self.voice_id
        return self


class VoiceCreated(BaseModel):
    voice_id: str
    name: str
    sample_rate: int
    duration_s: float


class CharTimestamp(BaseModel):
    char: str
    start: float
    end: float


class VerificationInfo(BaseModel):
    attempts: int
    score: float
    passed: bool
    heard: str  # Whisper's unbiased transcription of the chosen take


class TTSJsonResponse(BaseModel):
    audio_b64: str
    sample_rate: int
    duration_s: float
    speaker: str
    timestamps: list[CharTimestamp]
    verification: Optional[VerificationInfo] = None


class VoiceInfo(BaseModel):
    voice_id: str
    name: str
    prompt_text: str
    duration_s: float
    created_at: str


class HealthResponse(BaseModel):
    status: str
    gpu: Optional[str]
    cuda_available: bool
    vram_used_mb: Optional[float]
    model_loaded: bool
