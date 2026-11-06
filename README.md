# Qwen3 Arabic Live Transcription
# Arabic-to-German Live Translation

- **Google Colab (GPU)** runs [bot.py](./bot.py) with Qwen3-ASR-1.7B. It joins a
- **Google Colab (GPU)** runs [bot.py](./bot.py) with Qwen3-ASR-1.7B and a
  4-bit Qwen3-8B translator. Final Arabic utterances are translated to German
  with the previous three utterances as speaker-specific context.
- XTTS-v2 German speech is optional per browser session and is disabled by
  default. Enable **Deutsche Sprachausgabe (XTTS-v2)** before starting the
  microphone. The TTS model is downloaded and loaded only when selected.
4. When `BEREIT` appears: at the PC start the web app, optionally enable German
  speech, and turn on the microphone.
- **Your PC** runs only a small web server with the browser microphone client.
- **LiveKit Cloud** connects both.

## 1. Backend in Colab

1. Open [colab_agent.ipynb](./colab_agent.ipynb) in Google Colab (Runtime: T4 GPU).
2. **Runtime > Run all**. Select your `.env` when the upload dialog appears
   (or add the three values as Colab secrets, or upload `.env` to `/content`
   beforehand; then nothing is asked).
3. Wait for the log line `BEREIT`. The first start downloads the model.
4. Leave the last cell running.

## 2. Web client on your PC

The app loads `.env` from this directory or the parent directory
(`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, see `.env.example`).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000> and press **Mikrofon starten**. Partial text
Open <http://127.0.0.1:8000>, optionally enable German speech, and press
**Mikrofon starten**. Arabic interim text appears while speaking; final Arabic
and German text appear after a short pause.

The Colab runtime needs a GPU with enough memory for ASR, the 4-bit translator,
and XTTS-v2. XTTS uses a built-in German speaker by default. To use a reference
voice, set `XTTS_SPEAKER_WAV` to an audio file available in the Colab runtime;
check the XTTS-v2 model license before using it commercially.

Optional environment variables for the bot: `QWEN_MODEL`, `QWEN_LANGUAGE`
(default `Arabic`), `LIVEKIT_ROOM` (must match the web server).
# asr-agent
