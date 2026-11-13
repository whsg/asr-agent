# Arabic Live Transcription and Translation

Live Arabic-to-German speech-to-speech for a LiveKit room:

- **Google Colab (GPU)** runs [bot.py](./bot.py) with Silero-VAD speech
  segmentation, Cohere Transcribe Arabic ASR, a Qwen3 translator and F5-TTS
  speech output. Only the Cohere repository is gated and requires a Hugging
  Face account, accepted model terms, and `HF_TOKEN`.
  - German speech output is optional per browser session and disabled by
    default (engine chosen by env `TTS_BACKEND`: `f5` = F5-TTS, default, or
    `xtts` = XTTS-v2). For a German target, F5-TTS uses the fine-tuned
    `aihpi/F5-TTS-German` model; other targets use the standard F5-TTS base
    model. Enable **German speech output** before starting the microphone.
- **Your PC** runs only a small web server with the browser microphone client.
- **LiveKit Cloud** connects both.

## Pipeline architecture

Every stage is an independent worker with its own queue, so stage N of one
utterance overlaps stage N+1 of the previous one:

```text
mic track -> VAD thread -> ASR -> translation (token stream) -> TTS (clause chunks) -> playout -> LiveKit audio
             per speaker   |________ bounded, priority-scheduled GPU slots _________|            real time
```

What this buys, and why the previous build got slower the longer it ran:

| Property | Why it matters |
|---|---|
| **Playout is its own task** | `AudioSource.capture_frame()` blocks at playback speed. In the old single-loop design that blocking happened *inside* the translation worker, so every turn added its full playback duration to the next turn's latency and the lag grew without bound. |
| **No global GPU lock** | ASR, translation and TTS no longer share one lock held across all three. A bounded `GpuScheduler` admits a few calls at once and serves waiters by priority, so TTS and the final ASR always outrank best-effort live captions. |
| **fp16 translator** | 4-bit NF4 dequantises every weight on every decoded token: 2-4x slower than fp16 on a T4 *and* measurably worse translations. fp16 is now the default, with an automatic 4-bit fallback if it does not fit. |
| **Streaming translation into chunked TTS** | The first finished clause is spoken while the rest is still being generated, instead of waiting for the whole sentence. |
| **VAD on a worker thread** | Silero VAD no longer runs on the event loop, and the interim snapshot is cached instead of re-concatenated (that was quadratic). |
| **Backpressure** | If synthesis is persistently slower than the incoming speech, the oldest unplayed audio is dropped so the agent catches up. |

### Keeping sentences whole

A speaker who pauses for effect mid-sentence used to be split into fragments,
and that one problem caused both of the symptoms you notice:

- The translator received sentence fragments with no context, so the German
  drifted - the LLM has to guess how a half-clause ends.
- Every extra turn pays the TTS engine's **fixed per-call cost**. F5-TTS
  synthesises the reference clip together with the target text on *every* call
  and crops it afterwards, so an N-second reference clip is N seconds of wasted
  generation per chunk. Enough fragments and synthesis falls behind real time,
  the backlog grows, and whole turns get skipped - sentences simply go missing
  from the audio.

Two changes fix this at the source:

- **Adaptive end-of-speech.** A segment that already holds a full clause still
  ends on a short 0.4 s pause, so latency is unchanged. A segment with less
  than `MIN_SENTENCE_SECONDS` of speech needs a full
  `SHORT_SEGMENT_SILENCE_SECONDS` pause instead, so a breath never ends a
  sentence.
- **Growing TTS chunks.** The first chunk stays short for a fast start; later
  chunks grow up to `TTS_MAX_CHUNK_CHARS`, so a long sentence costs a handful
  of TTS calls rather than a dozen.

**Check your reference clip.** It is the single biggest factor in speech-output
throughput. The bot logs its length at startup and warns above 8 s; a 5-6 s
mono clip is the sweet spot. The per-turn report shows `speech` (how much audio
the turn contained) so fragmentation is visible immediately.

### Truncation safeguards

Anything cut off mid-sentence is treated as a bug, so each possible cause is
either removed or made loud:

- The ASR and translator token budgets are derived from the input with a wide
  margin, and the bot **logs a warning naming the variable to raise** whenever
  a request actually hits its budget - a silent cut is no longer possible.
- Backlog relief skips **whole turns**, never individual audio chunks, and
  never the turn currently being spoken; losing one complete sentence is
  recoverable, half a sentence is not.
- `split_speakable` is covered by a property test that streams 200 randomised
  translations through it and asserts the spoken chunks reassemble the input
  word for word.
- F5-TTS never receives a fragment below `TTS_MIN_CHUNK_CHARS`, because it
  clips very short inputs.

Two correctness fixes came along the way: the rolling chat context is now sent
as interleaved source/translation pairs (it used to send every source first and
every translation afterwards, which breaks the few-shot pattern), and the
per-request token budget is derived from the input length instead of always
allowing 256 tokens.

`test/bench_pipeline.py` replays both designs with identical simulated stage
costs (no GPU required):

```text
old: single loop, GPU lock held across translation + TTS + playout
  lag [s]:   4.3    8.1   11.9   15.8   19.6   23.4   27.2   31.1   (growth +26.7 s)
new: streaming pipeline, playout decoupled
  lag [s]:   1.9    1.9    1.9    1.9    1.9    1.9    1.9    1.9   (growth +0.0 s)
```

Every turn is reported on one line, so the behaviour on a real GPU is visible:

```text
turn a1b2-7 | asr 410ms | mt 820ms (first token 170ms) | tts 640ms in 3 chunk(s)
            -> 4.1s audio (rtf 0.16) | first audio 1.12s after speech end | backlog 0.6s
```

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
| `HF_TOKEN` | - | Hugging Face token; required for the gated Cohere ASR repo |
| `COHERE_MODEL` | `CohereLabs/cohere-transcribe-arabic-07-2026` | Cohere ASR model |
| `COHERE_4BIT` | `0` | Load the Cohere model in 4-bit to free VRAM for the fp16 translator |
| `COHERE_LANGUAGE` | `ar` | Language hint passed to Cohere ASR |
| `TRANSLATION_MODEL` | `Qwen/Qwen3-4B-Instruct-2507` | Translator; an 8B variant translates better but decodes slower |
| `TRANSLATION_4BIT` | `0` | Force the slower 4-bit translator; used automatically if fp16 does not fit |
| `TRANSLATION_CONTEXT_TURNS` | `4` | Previous turns sent as chat context; `0` is fastest |
| `LIVEKIT_ROOM` | `arabic-asr-live` | Room name (must match the web server) |
| `TTS_BACKEND` | `f5` | Speech engine: `f5` (F5-TTS) or `xtts` (XTTS-v2) |
| `F5_REF_WAV` | - | **Required for `f5`**: reference voice clip (~5-10 s wav) in the Colab runtime or an http(s) URL |
| `F5_REF_TEXT` | - | **Required for `f5`**: the exact spoken text of the reference clip |
| `F5_NFE_STEPS` | `8` | ODE inference steps, lower is faster |
| `STREAM_TTS` | `1` | Speak each finished clause instead of waiting for the full sentence |
| `INTERIM_CAPTIONS` | `1` | Live Arabic captions while speaking; `0` for the lowest latency |
| `GPU_CONCURRENCY` | `2` | Model calls allowed on the GPU at once |
| `MAX_AUDIO_BACKLOG_SECONDS` | `10` | Unplayed audio after which the agent skips whole queued turns |
| `MAX_GEN_TOKENS` | `448` | Runaway guard only; a warning is logged if a request ever hits it |
| `MAX_UTTERANCE_SECONDS` | `12.0` | Longest utterance (seconds) sent to the ASR, then split |
| `MIN_SENTENCE_SECONDS` | `2.5` | Below this much speech, a pause is treated as mid-sentence |
| `SHORT_SEGMENT_SILENCE_SECONDS` | `1.0` | Pause required to close a segment that is still short |

All variables are documented in [.env.example](./.env.example).

**Hugging Face setup:** accept the model terms for
[Cohere Transcribe Arabic](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026),
then set an authorized `HF_TOKEN` in `.env` or Colab Secrets. Qwen3 is public
and needs no token. Colab downloads the weights into its runtime; F5-TTS
weights are fetched at startup.

**VRAM on a T4 (15.75 GB):** Cohere ASR, the fp16 translator and F5-TTS are
sized to fit together. If the fp16 translator does not fit, the bot logs a
warning and falls back to 4-bit automatically - set `COHERE_4BIT=1` to free
roughly 3 GB and keep the faster fp16 translator instead. The VRAM actually in
use is logged after each model loads.

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
speaking; the final transcript and the German translation appear right after,
and the translation text streams in while it is generated.

F5-TTS (`TTS_BACKEND=f5`, default) is a voice-cloning model: set `F5_REF_WAV`
to a short reference clip (~5-10 s wav of the desired voice, in the Colab
runtime or as an http(s) URL) and `F5_REF_TEXT` to its exact spoken text.
Check the F5-TTS / German fine-tune licenses before using them commercially.
With `TTS_BACKEND=xtts`, XTTS-v2 uses a built-in German speaker by default; to
use a reference voice there, set `XTTS_SPEAKER_WAV` to an audio file available
in the Colab runtime.

## 3. Tests

The pipeline tests replace every GPU dependency with a controllable fake, so
they run on any machine:

```bash
python -m pip install numpy pytest
python -m pytest test
python test/bench_pipeline.py
```
