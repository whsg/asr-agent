# Arabic Live Transcription and Translation

Live Arabic-to-German transcription and translation for a LiveKit room:

- **Google Colab (GPU)** runs [bot.py](./bot.py) with Silero-VAD speech
  segmentation, Cohere Transcribe Arabic ASR, and a 4-bit Qwen3 German
  translator. The Cohere model repository is gated and requires a Hugging Face
  account, accepted model terms, and `HF_TOKEN`. Qwen3 is public.
  - German speech output is optional per browser session and disabled by
    default (engine chosen by env `TTS_BACKEND`: `phonon` = Gradium Phonon,
    default, or `xtts` = XTTS-v2). Phonon is a lightweight German model that
    speaks in a fixed voice from `gradium/phonon-voices` (default **Henrik**)
    instead of cloning a reference; its repo is gated. Enable **German speech output**
    before starting the microphone - the TTS model is downloaded and loaded
    only when selected.
- **Your PC** runs only a small web server with the browser microphone client.
- **LiveKit Cloud** connects both.

## 1. Backend in Colab

1. Open [colab_agent.ipynb](./colab_agent.ipynb) in Google Colab (Runtime: T4 GPU).
2. **Runtime > Run all**. Select your `.env` when the upload dialog appears
  (or add the LiveKit values and `HF_TOKEN` as Colab secrets, or upload `.env` to `/content`
   beforehand; then nothing is asked).
3. Wait for the log line that starts with `READY`. The first start downloads the model.
4. Leave the last cell running.

### Model configuration

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | - | Hugging Face token; required for Cohere ASR |
| `COHERE_MODEL` | `CohereLabs/cohere-transcribe-arabic-07-2026` | Cohere ASR model |
| `COHERE_4BIT` | `0` | Load the Cohere model in 4-bit to save VRAM |
| `COHERE_LANGUAGE` | `ar` | Language hint passed to Cohere ASR |
| `TRANSLATION_MODEL` | `Qwen/Qwen3-4B-Instruct-2507` | Qwen3 translator model; public, no token needed |
| `TRANSLATION_TARGET_LANG` | `de` | Target language for translation (ISO 639-1 code) |
| `LIVEKIT_ROOM` | `arabic-asr-live` | Room name (must match the web server) |
| `TTS_BACKEND` | `phonon` | Speech engine: `phonon` (Gradium Phonon, default) or `xtts` (XTTS-v2) |
| `XTTS_SPEAKER` | - | XTTS-v2 speaker name (only used when `TTS_BACKEND=xtts`) |
| `XTTS_SPEAKER_WAV` | - | XTTS-v2 reference voice path in Colab (only used when `TTS_BACKEND=xtts`) |
| `PHONON_MODEL_NAME` | `gradium/phonon-40m-de` | Gradium Phonon model (only used when `TTS_BACKEND=phonon`; gated repo, needs `HF_TOKEN`) |
| `PHONON_VOICE` | `henrik` | Speaker from [gradium/phonon-voices](https://huggingface.co/gradium/phonon-voices) by name, raw 16-char voice ID, or `default` (only used when `TTS_BACKEND=phonon`) |
| `PHONON_TEMPERATURE` | `1.0` | Phonon temperature parameter (only used when `TTS_BACKEND=phonon`) |
| `MAX_UTTERANCE_SECONDS` | `20.0` | Longest utterance (seconds) sent to the ASR, then split |
| `TRANSLATION_CONTEXT_TURNS` | `5` | Number of previous source/translation pairs sent as context; the translator also holds back source text whose meaning is incomplete until its continuation arrives |

**Hugging Face setup:** accept the model terms for
[Cohere Transcribe Arabic](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026)
— and, if you use `TTS_BACKEND=phonon`, also for
[gradium/phonon-40m-de](https://huggingface.co/gradium/phonon-40m-de) (and
`gradium/phonon-voices`) — then set an authorized `HF_TOKEN` in `.env` or
Colab Secrets. Colab downloads Cohere weights into its runtime. Qwen3 is
public and needs no token. Phonon weights download the first time
speech output is requested. Note that Phonon needs a transformers build with
`GradiumPhononTTSForConditionalGeneration`
(`pip install -U 'git+https://github.com/huggingface/transformers'`).

**VRAM on a T4 (15.75 GB):** Cohere ASR plus the 4-bit Qwen3
translator run without TTS. The default Phonon engine is lightweight
(40M parameters) and fits easily in the remaining headroom;
XTTS-v2 uses noticeably more RAM/VRAM. Set
`COHERE_4BIT=1` if you need more headroom.

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
**Start microphone**. Cohere's interim Arabic transcript appears while
speaking; the final transcript and German translation appear after a short
pause.

The Colab runtime needs a GPU with enough memory for Cohere ASR, the 4-bit
Qwen3 translator, and optionally the TTS engine (see the VRAM
notes above). The default engine, Gradium Phonon (`TTS_BACKEND=phonon`),
speaks in a fixed voice from the `gradium/phonon-voices` catalog (default
**Henrik**, a German male voice; change with `PHONON_VOICE`) instead of
cloning a reference. Note that Phonon's model/voice repos are CC BY-NC, so
check the license before commercial use. With `TTS_BACKEND=xtts`, XTTS-v2
uses a built-in German speaker by default; to use a reference voice there,
set `XTTS_SPEAKER_WAV` to an audio file available in the Colab runtime.

All bot environment variables are documented in the table above and in
[.env.example](./.env.example).
