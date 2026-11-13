# Arabic Live Transcription and Translation

Live Arabic-to-German speech-to-speech for a LiveKit room:

- **A GPU notebook** ([Kaggle](./kaggle_agent.ipynb) or [Colab](./colab_agent.ipynb))
  runs [bot.py](./bot.py) with Silero-VAD speech segmentation, Cohere Transcribe
  Arabic ASR, a translator (TranslateGemma 12B on Kaggle, Qwen3-4B on Colab) and
  Gradium Phonon speech output on the CPU. The Cohere and Phonon German
  repositories are gated and require a Hugging Face account, accepted model
  terms, and `HF_TOKEN`.
  - German speech output is optional per browser session and disabled by
    default (engine chosen by env `TTS_BACKEND`: `phonon`, default, or `xtts`
    = XTTS-v2). Enable **German speech output** before starting the microphone.
- **Your PC** runs only a small web server with the browser microphone client.
- **LiveKit Cloud** connects both.

> **License:** Phonon's German model is **CC BY-NC 4.0** - not for commercial
> use without an agreement with Gradium. XTTS-v2 is non-commercial as well.

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
- Every extra turn pays the TTS engine's **fixed per-call cost**. Enough
  fragments and synthesis falls behind real time, the backlog grows, and whole
  turns get skipped - sentences simply go missing from the audio.

Two changes fix this at the source:

- **Adaptive end-of-speech.** A segment that already holds a full clause still
  ends on a short 0.4 s pause, so latency is unchanged. A segment with less
  than `MIN_SENTENCE_SECONDS` of speech needs a full
  `SHORT_SEGMENT_SILENCE_SECONDS` pause instead, so a breath never ends a
  sentence.
- **Growing TTS chunks.** The first chunk stays short for a fast start; later
  chunks grow up to `TTS_MAX_CHUNK_CHARS`, so a long sentence costs a handful
  of TTS calls rather than a dozen.

The per-turn report shows `speech` (how much audio the turn contained), so
fragmentation is visible immediately.

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
- The TTS never receives a fragment below `TTS_MIN_CHUNK_CHARS`, because very
  short fragments sound clipped.

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

## 1. GPU backend

### Kaggle: Cohere + TranslateGemma 12B + Phonon on two GPUs

Upload [kaggle_agent.ipynb](./kaggle_agent.ipynb) into **one private Kaggle
notebook**. Select **GPU T4 x2** and enable **Internet**. Check your account's
actual hardware availability, quota, and current platform terms; two cards are
not guaranteed merely by having an account.

| Device | Model | Precision |
|---|---|---|
| `cuda:0` | `google/translategemma-12b-it` | NF4 4-bit, bfloat16 compute |
| `cuda:1` | Cohere Arabic ASR | fp16 |
| CPU | Gradium Phonon 40m, German | Phonon's own Rust runtime |

This is **stage placement, not tensor parallelism**: no model is split across
GPUs and there is no CPU weight offload. Each device has its own admission
queue, so the next sentence is transcribed on GPU 1 while the previous one is
translated on GPU 0, and Phonon synthesises on the CPU without waiting for
either GPU.

**Why Phonon.** MOSS-TTS was tried first and removed: a measured Kaggle run had
it at a real-time factor of 4.4-4.9 on a T4 (about 5x slower than playback),
and it never emitted its end token, so every sentence ran to its token budget.
Phonon has 40M parameters and runs faster than real time on a CPU per Gradium's
benchmarks; its German model is gated and **CC BY-NC 4.0**.

1. Accept the Hugging Face access terms for
   [Cohere](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026),
   [TranslateGemma](https://huggingface.co/google/translategemma-12b-it)
   (Gemma terms, not Apache-2.0) and
   [Phonon German](https://huggingface.co/gradium/phonon-40m-de) (automatic).
2. Under **Add-ons > Secrets**, add and enable `HF_TOKEN`, `LIVEKIT_URL`,
   `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET`. Use a read token with access
   to all three gated models. Never publish credentials in cells or datasets.
3. Run all cells. Installation checks two visible GPUs, keeps Kaggle's
   preinstalled, mutually matched PyTorch/torchaudio/torchvision (a pip
   constraints file prevents replacing them), and verifies actual CUDA
   operations in a fresh process. Gated model access is checked before weight
   downloads.
4. The offline test loads all models, generates German speech, and saves
   `smoke-test/german.wav`, `smoke-test/metrics.json`, and `smoke-test.log`
   under `/kaggle/working`. To test ASR on real speech, set `TEST_AUDIO` in
   the smoke-test cell to a private Arabic WAV/FLAC of at most 12 seconds.
   Otherwise it tests translation/TTS with fixed Arabic text and explicitly
   reports that ASR accuracy was not evaluated.
5. After that test succeeds, the last cell starts the live bot in a new
   process. Wait for `READY`, then use the existing local web app with the
   same room (`LIVEKIT_ROOM`, default `arabic-asr-live`). Enable **German
   speech output** before starting the microphone. Kaggle must permit
   outgoing LiveKit/WebRTC connectivity; no notebook HTTP tunnel is used.

TranslateGemma uses the official `AutoProcessor` /
`AutoModelForImageTextToText` API and structured `ar` -> `de` requests.
Qwen's sermon system prompt and rolling chat context are intentionally not
sent to this backend. Translation still streams token-by-token into the same
clause chunker. Inputs exceeding the documented 2048-token input context
raise an error rather than silently truncating Arabic text. On a T4 it
computes in bfloat16: Gemma 3 activations exceed float16's range there, which
produced only special tokens.

**Real-time status: unverified on hardware.** Offline metrics measure
sequential stages after warm-up, not live end-to-end latency. `tts_rtf < 1`
is necessary to synthesize faster than playback, but does not prove that the
whole pipeline keeps up. Listen to the saved audio, then measure live
first-audio delay and backlog over several minutes.

Installation and model-access logs and `/kaggle/working/bot.log` retain
tracebacks after cell output is cleared. Download needed files before the
session is discarded; keep recordings, translations, notebooks and logs
private. Successful local stub tests do not certify GPU compatibility,
translation quality, or real-time performance.

### Colab: single-GPU Qwen profile

1. Open [colab_agent.ipynb](./colab_agent.ipynb) in Google Colab with a
   **T4 GPU** runtime.
2. **Runtime > Run all**. Select your `.env` when the upload dialog appears
  (or add the LiveKit values and `HF_TOKEN` as Colab secrets, or upload `.env` to `/content`
   beforehand; then nothing is asked).
3. Wait for the log line that starts with `READY`. The first start downloads the model.
4. Leave the last cell running.

| Component | Weight memory (GiB, excluding caches/activations) |
|---|---:|
| Cohere ASR, fp16 | 3.85 |
| Qwen3-4B, fp16 | 7.49 |
| Phonon TTS | CPU, no GPU memory |

These figures come from Hugging Face's published safetensors metadata for
[ASR](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026) and
[Qwen3](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507): about 11.3 GiB
on a 15 GiB T4, before inference memory.

`READY` requires successful ASR, TTS, and translation warm-ups; empty or
non-finite synthesized audio aborts startup. The notebook installs only the
selected TTS engine's dependencies and keeps `torchaudio` on the same release
as Colab's installed CUDA `torch`.

If a cell fails, retain `/content/install.log` (installation) and
`/content/bot.log` (full bot output). The last cell raises an error when the bot
exits unexpectedly instead of appearing to complete successfully. Logs survive
cleared cell output, but not deletion/reset of the Colab runtime; download them
before resetting, and redact any credentials before sharing.

### Model configuration

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | - | Hugging Face token; required for the gated Cohere, TranslateGemma and Phonon repos |
| `COHERE_MODEL` | `CohereLabs/cohere-transcribe-arabic-07-2026` | Cohere ASR model |
| `COHERE_4BIT` | `0` | Load the Cohere model in 4-bit to free VRAM for the fp16 translator |
| `COHERE_LANGUAGE` | `ar` | Language hint passed to Cohere ASR |
| `TRANSLATION_MODEL` | `Qwen/Qwen3-4B-Instruct-2507` | Translator; an 8B variant translates better but decodes slower |
| `TRANSLATION_DTYPE` | `auto` | Translator compute dtype; `auto` uses bfloat16 for TranslateGemma on T4, because Gemma 3 overflows float16 (empty output) |
| `TRANSLATION_BACKEND` | `qwen` | `qwen` chat template or `translategemma` structured translation template |
| `ASR_DEVICE` / `TRANSLATION_DEVICE` / `TTS_DEVICE` | `auto` / `auto` / `cpu` for Phonon | `auto`, `cpu`, or `cuda:N`; Kaggle profile uses `cuda:1` / `cuda:0` / `cpu` |
| `TRANSLATION_4BIT` | `0` | Force the slower 4-bit translator; used automatically if fp16 does not fit |
| `TRANSLATION_CONTEXT_TURNS` | `4` | Previous turns sent as chat context; `0` is fastest |
| `LIVEKIT_ROOM` | `arabic-asr-live` | Room name (must match the web server) |
| `TTS_BACKEND` | `phonon` | Speech engine: `phonon` (Gradium Phonon, CPU) or `xtts` (XTTS-v2) |
| `PHONON_MODEL` / `PHONON_VOICE` | `hf://gradium/phonon-40m-de` / model default | Phonon checkpoint and voice |
| `STREAM_TTS` | `1` | Speak each finished clause instead of waiting for the full sentence |
| `INTERIM_CAPTIONS` | `1` | Live Arabic captions while speaking; `0` for the lowest latency |
| `GPU_CONCURRENCY` | `2` | Model calls per device; Kaggle dual-T4 profile uses `1` |
| `MAX_AUDIO_BACKLOG_SECONDS` | `10` | Unplayed audio after which the agent skips whole queued turns |
| `MAX_GEN_TOKENS` | `448` | Runaway guard only; a warning is logged if a request ever hits it |
| `MAX_UTTERANCE_SECONDS` | `12.0` | Longest utterance (seconds) sent to the ASR, then split |
| `MIN_SENTENCE_SECONDS` | `2.5` | Below this much speech, a pause is treated as mid-sentence |
| `SHORT_SEGMENT_SILENCE_SECONDS` | `1.0` | Pause required to close a segment that is still short |

All variables are documented in [.env.example](./.env.example).

**Hugging Face setup:** accept the model terms for
[Cohere Transcribe Arabic](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026)
and [Phonon German](https://huggingface.co/gradium/phonon-40m-de) (plus
TranslateGemma on Kaggle), then set an authorized `HF_TOKEN` in `.env` or the
notebook's secrets. Qwen3 is public. The notebooks download all weights into
their runtime at startup.

The VRAM actually in use is logged after each model loads. A translator
loading OOM triggers a logged 4-bit retry.

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

Phonon (`TTS_BACKEND=phonon`, default) needs no configuration beyond the
accepted model terms - it speaks with the German model's default voice; set
`PHONON_VOICE` to pick another from the
[voice catalog](https://huggingface.co/gradium/phonon-voices). With
`TTS_BACKEND=xtts`, XTTS-v2 uses a built-in German speaker by default.

## Speech engines

`TTS_BACKEND` selects the engine. Each one only has to implement `load()` and
`synthesize(text) -> waveform`, so adding another is a small, local change.

| Engine | License | German | Reference clip | Notes |
|---|---|---|---|---|
| `phonon` (default) | Code MIT/Apache-2.0; **German model CC BY-NC 4.0** (commercial use only with a Gradium agreement) | `gradium/phonon-40m-de` | not needed (voice catalog) | 40M params, runs on the **CPU**, faster than real time per Gradium's benchmarks |
| `xtts` | Coqui CPML (non-commercial) | yes | not needed (built-in speakers) | runs on `TTS_DEVICE`; needs the optional `coqui-tts` package |

Phonon is installed with `pip install ptts`; the German model downloads from
Hugging Face on first use with `HF_TOKEN`. Its quality for sermon register and
Arabic names has to be judged by listening to the smoke-test audio.

## 3. Tests

The pipeline tests replace every GPU dependency with a controllable fake, so
they run on any machine:

```bash
python -m pip install numpy pytest python-dotenv
python -m pytest test
python test/bench_pipeline.py
```
