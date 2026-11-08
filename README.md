# Arabic Live Transcription and Translation

Live Arabic-to-German transcription and translation for a LiveKit room:

- **Google Colab (GPU)** runs [bot.py](./bot.py) with Silero-VAD speech
  segmentation, selectable ASR backends and a 4-bit Qwen3-8B translator. Final
  Arabic utterances of the backend chosen per participant are translated to
  German using the previous three utterances as speaker-specific context.
  - ASR backends (env `ASR_BACKENDS`): `qwen` (Qwen3-ASR-1.7B, default),
    `cohere` (CohereLabs/cohere-transcribe-arabic-07-2026) or both.
  - XTTS-v2 German speech is optional per browser session and is disabled by
    default. Enable **German speech output (XTTS-v2)** before starting the
    microphone. The TTS model is downloaded and loaded only when selected.
- **Your PC** runs only a small web server with the browser microphone client.
- **LiveKit Cloud** connects both.

## 1. Backend in Colab

1. Open [colab_agent.ipynb](./colab_agent.ipynb) in Google Colab (Runtime: T4 GPU).
2. **Runtime > Run all**. Select your `.env` when the upload dialog appears
   (or add the three values as Colab secrets, or upload `.env` to `/content`
   beforehand; then nothing is asked).
3. Wait for the log line that starts with `READY`. The first start downloads the model.
4. Leave the last cell running.

### ASR backends

The bot loads every backend listed in `ASR_BACKENDS` (comma separated,
default `qwen`). The web app offers a view per loaded backend plus a
side-by-side mode, and each participant chooses which backend's output gets
translated to German.

| Variable | Default | Description |
|---|---|---|
| `ASR_BACKENDS` | `qwen` | Backends to load: `qwen`, `cohere` or `qwen,cohere` |
| `HF_TOKEN` | - | HuggingFace token, **required for `cohere`** (gated repo) |
| `COHERE_MODEL` | `CohereLabs/cohere-transcribe-arabic-07-2026` | Cohere ASR model |
| `COHERE_4BIT` | `0` | Load the Cohere model in 4-bit to save VRAM |
| `COHERE_LANGUAGE` | `ar` | Language hint passed to Cohere ASR |
| `QWEN_MODEL` | `Qwen/Qwen3-ASR-1.7B` | Qwen ASR model |
| `QWEN_LANGUAGE` | `Arabic` | Language hint passed to Qwen ASR |
| `LIVEKIT_ROOM` | `qwen-asr-live` | Room name (must match the web server) |
| `XTTS_SPEAKER` | - | XTTS-v2 speaker name |
| `XTTS_SPEAKER_WAV` | - | Path to a reference voice available in Colab |
| `MAX_UTTERANCE_SECONDS` | `12.0` | Longest utterance (seconds) sent to the ASR, then split |

**Cohere setup (gated repo):** accept the model terms at
https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026 and set a
matching `HF_TOKEN` in `.env`, then start with `ASR_BACKENDS=qwen,cohere`.

**VRAM on a T4 (15.75 GB):** both backends in 16-bit plus the 4-bit
Qwen3-8B translator fit without XTTS (~13 GB). Set `COHERE_4BIT=1` if you
need more headroom.

## 2. Web client on your PC

The app loads `.env` from this directory or the parent directory
(`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, see `.env.example`).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>, optionally enable German speech, and press
**Start microphone**. Choose the ASR view (Qwen3-ASR, Cohere Transcribe
Arabic, or both side by side) and which backend's output should be translated
to German; these choices are remembered per browser. Arabic interim text
appears while speaking; final Arabic and the German translation appear after
a short pause.

The Colab runtime needs a GPU with enough memory for the selected ASR
backend(s), the 4-bit translator and optionally XTTS-v2 (see the VRAM notes
above). XTTS uses a built-in German speaker by default. To use a reference
voice, set `XTTS_SPEAKER_WAV` to an audio file available in the Colab
runtime; check the XTTS-v2 model license before using it commercially.

All bot environment variables are documented in the table above and in
[.env.example](./.env.example).
