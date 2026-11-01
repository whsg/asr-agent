# Qwen3 Arabic Live Transcription

- **Google Colab (GPU)** runs [bot.py](./bot.py) with Qwen3-ASR-1.7B. It joins a
  fixed LiveKit room and sends transcripts back as data messages.
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
(grey) appears about every 1.5 s, final text after a short pause.

Optional environment variables for the bot: `QWEN_MODEL`, `QWEN_LANGUAGE`
(default `Arabic`), `LIVEKIT_ROOM` (must match the web server).
# asr-agent
