# CosyTTS — Consumer Guide

A local HTTP TTS service built on **CosyVoice 2**: zero-shot voice cloning, char/word-level timestamps, low-latency streaming, OpenAI-style REST.

> Replace `<HOST>:<PORT>` below with wherever you've deployed the server.
> Default port is `8765`. On a LAN it might be `192.168.x.x:8765`; if you front it with a hostname (Tailscale MagicDNS, mDNS, reverse proxy, etc.) just use that name.

## Minimal call

```http
POST http://<HOST>:<PORT>/tts
Content-Type: application/json

{"text": "下面这首,周杰伦的《简单爱》。"}
```

Response: `Content-Type: audio/wav`, 24 kHz / 16-bit / mono, peak-normalized to -1 dBFS.

If no `speaker` is given, the server uses its configured default voice (`TTS_DEFAULT_VOICE` env var, or first registered voice).

## `/tts` request schema

```jsonc
{
  "text":             "...",        // required, 1-4000 chars
  "speaker":          "demo",       // optional, name of a registered voice
  "speed":            1.0,          // optional, 0.5 - 2.0
  "timestamps":       false,        // optional, see "JSON mode" below
  "stream":           false,        // optional, chunked PCM output
  "simplify_chinese": true,         // optional, Traditional→Simplified before synth
  "verify":           false,        // optional, judge + regenerate mispronounced takes
  "max_attempts":     3,            // optional, retry cap when verify=true (max 6)
  "min_score":        0.90,         // optional, accept threshold 0-1 (higher = stricter)
  "mode":             "zero_shot",  // optional. zero_shot | cross_lingual | instruct2
  "instruct_text":    null          // required for mode=instruct2, e.g. "用悲伤的语调"
}
```

`timestamps` and `stream` are mutually exclusive (server returns 400 if both true).
`verify` only applies to non-streaming requests.

### Pronunciation verification (`verify: true`)

Each generated take is transcribed by Whisper (without being shown the target
text) and compared phonetically to what you asked for. Takes scoring below
`min_score` are regenerated, up to `max_attempts`. The best take is returned.

- **Works well** for Chinese content and gross garbling (dropped/wrong words).
- **Does NOT reliably help** English proper nouns in cloned voices — Whisper
  mis-transcribes accented English names, so its judgement is unreliable there.
- Costs ~1.5 s per attempt (Whisper pass). Off by default.

When `verify` is on, the binary-WAV response carries `X-TTS-Attempts`,
`X-TTS-Score`, `X-TTS-Passed` headers; the JSON response (timestamps mode)
includes a `verification` object:

```json
"verification": {"attempts": 2, "score": 0.97, "passed": true,
                 "heard": "whisper's unbiased transcription"}
```

## Response modes

### 1. Default — binary WAV

Fastest. Pipe into a player, save to disk, hand to an audio library.

### 2. `timestamps: true` — JSON with per-char/word alignment

```json
{
  "audio_b64": "UklGRiIAA...",
  "sample_rate": 24000,
  "duration_s": 2.72,
  "speaker": "demo",
  "timestamps": [
    {"char": "下", "start": 0.000, "end": 0.280},
    {"char": "面", "start": 0.280, "end": 0.560},
    {"char": "这", "start": 0.560, "end": 0.740},
    {"char": "首", "start": 0.740, "end": 0.860},
    {"char": ",", "start": 0.860, "end": 0.980},
    {"char": "周", "start": 1.040, "end": 1.220}
  ]
}
```

`audio_b64` decodes to a full WAV. `timestamps` is ordered to match the input text:

- Chinese characters and CJK punctuation → one entry per char
- English words / numbers → one entry per word (e.g. `"CosyVoice"`, `"5090,"`)
- Original character forms are preserved — i.e. if you sent `萧敬腾`, the response will contain `萧`, not Whisper's homophone substitution `肖`

Timing precision: ~±50 ms, derived from Whisper large-v3 forced alignment.

### 3. `stream: true` — chunked raw PCM

`Content-Type: audio/L16; rate=24000; channels=1`. Raw 16-bit PCM, no WAV header. Consumer either feeds it to a player directly or prepends a WAV header to save.

Typical TTFB: 1-3 s. Steady-state RTF ~0.5 on a modern NVIDIA GPU.

## Voice management

### List

```http
GET http://<HOST>:<PORT>/voices
```

```json
[
  {
    "voice_id": "demo",
    "name": "demo",
    "prompt_text": "Reference transcript here.",
    "duration_s": 5.2,
    "created_at": "2026-06-03T..."
  }
]
```

### Register (multipart/form-data)

```http
POST http://<HOST>:<PORT>/voices
Content-Type: multipart/form-data

audio:        <file>     # WAV / MP3 / M4A / AAC / FLAC — server normalizes to 16 kHz mono
name:         my_voice
prompt_text:  <verbatim transcript of the audio>
```

Returns `{"voice_id": "my_voice", ...}`.

**Reference audio tips**
- 3-10 seconds works best; up to ~15 seconds is fine
- Clean speech, no music / echo / background voices
- `prompt_text` **must** match the audio verbatim — phoneme alignment depends on it
- Server auto-normalizes peak to -1 dBFS so different recording levels don't change output loudness

### Delete

```http
DELETE http://<HOST>:<PORT>/voices/my_voice
→ 204 No Content
```

### Set the default

`TTS_DEFAULT_VOICE=my_voice` env var, or edit `app/config.py`. Falls back to the first registered voice if the configured name doesn't exist.

## Client examples

### curl

```bash
# Minimal
curl -X POST http://<HOST>:<PORT>/tts \
     -H "Content-Type: application/json" \
     -d '{"text":"你好,世界。"}' \
     -o out.wav

# speed + speaker
curl -X POST http://<HOST>:<PORT>/tts \
     -H "Content-Type: application/json" \
     -d '{"text":"快一点说这句话","speaker":"demo","speed":1.3}' \
     -o fast.wav

# Timestamps (JSON)
curl -X POST http://<HOST>:<PORT>/tts \
     -H "Content-Type: application/json" \
     -d '{"text":"测试时间戳","timestamps":true}' \
     -o ts.json
```

### Python

```python
import base64, requests

# Plain WAV
r = requests.post("http://<HOST>:<PORT>/tts",
                  json={"text": "你好,世界。", "speaker": "demo"})
open("out.wav", "wb").write(r.content)

# Timestamps
r = requests.post("http://<HOST>:<PORT>/tts",
                  json={"text": "你好,世界。", "timestamps": True})
data = r.json()
open("out.wav", "wb").write(base64.b64decode(data["audio_b64"]))
for ts in data["timestamps"]:
    print(f"{ts['char']}  {ts['start']:.2f}-{ts['end']:.2f}")

# Streaming → real-time playback
import numpy as np, sounddevice as sd
r = requests.post("http://<HOST>:<PORT>/tts",
                  json={"text": "长文本...", "stream": True}, stream=True)
with sd.OutputStream(samplerate=24000, channels=1, dtype="int16") as out:
    for chunk in r.iter_content(chunk_size=None):
        out.write(np.frombuffer(chunk, dtype=np.int16).reshape(-1, 1))
```

### JavaScript / TypeScript

```ts
// Plain — fetch into Audio element
const res = await fetch("http://<HOST>:<PORT>/tts", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "你好世界" }),
});
new Audio(URL.createObjectURL(await res.blob())).play();

// Timestamps — for subtitles / karaoke
const res = await fetch("http://<HOST>:<PORT>/tts", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "你好世界", timestamps: true }),
});
const { audio_b64, timestamps, sample_rate } = await res.json();
// timestamps: [{char, start, end}, ...] in seconds
```

CORS is wide-open by default (`allow_origins=["*"]`) — fine when the service is on a private network, lock down via `TTS_CORS_ORIGINS` env var for public deployments.

## Other endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | `{status, gpu, cuda_available, vram_used_mb, model_loaded}` |
| `GET /docs` | Swagger UI |
| `GET /openapi.json` | OpenAPI spec |

## Reference performance (RTX 5090)

| Call | Steady-state latency | Notes |
|---|---|---|
| plain / `speed` / `speaker` | ~1.5-2 s | for 4-8 s of audio |
| `stream` short text | TTFB ~2 s | usually 1 chunk |
| `stream` long text | TTFB ~3 s, RTF ~0.56 | sustained playback without underrun |
| `timestamps` (cold) | ~13 s | first call loads Whisper |
| `timestamps` (warm) | ~2.2 s | includes alignment |

Very first synthesis after server boot is ~40 s while CUDA JIT-compiles kernels. Subsequent calls are steady-state.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Connect timeout from another host | Server firewall blocking the port | Allow inbound on the service port |
| Hostname won't resolve | DNS / MagicDNS / mDNS not set up | Use the IP, or set up the name resolver |
| 422 Unprocessable Entity | JSON encoding | `Content-Type: application/json; charset=utf-8`, UTF-8 body for CJK |
| 404 voice_id not found | `speaker` typo | `GET /voices` to see what's available |
| 400 timestamps and stream both true | mutually exclusive | pick one |
| Output sounds quiet | Old voice registered before peak-normalize landed | Re-register: `DELETE /voices/<id>` then `POST /voices` |
| English word read letter-by-letter | All-caps multi-word phrase | Server lowercases automatically; if you see it, your sentence probably had a single all-caps word — that case is intentionally left alone (acronyms) |

See the repo's `README.md` for setup, model download, and the full pipeline architecture.
