# Arabic Live Transcription and Translation

Live Arabic-to-German speech-to-speech for a LiveKit room:

- **Google Colab (GPU)** runs [bot.py](./bot.py) with Silero-VAD speech
  segmentation, Cohere Transcribe Arabic ASR, a Qwen3 translator and MOSS-TTS
  speech output. Only the Cohere repository is gated and requires a Hugging
  Face account, accepted model terms, and `HF_TOKEN`.
  - German speech output is optional per browser session and disabled by
    default (engine chosen by env `TTS_BACKEND`: `moss` = MOSS-TTS, default,
    or `xtts` = XTTS-v2). Enable **German speech output** before starting the
    microphone.
- **Your PC** runs only a small web server with the browser microphone client.
- **LiveKit Cloud** connects both.

For **Kaggle with two T4 GPUs and TranslateGemma 12B**, use the separate
[Kaggle notebook](./kaggle_agent.ipynb), described below. The Colab notebook
keeps its existing Qwen default.

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

### Kaggle: Cohere + TranslateGemma 12B + MOSS on two GPUs

Upload [kaggle_agent.ipynb](./kaggle_agent.ipynb) into **one private Kaggle
notebook**. Select **GPU T4 x2** and enable **Internet**. Check your account's
actual hardware availability, quota, and current platform terms; two cards are
not guaranteed merely by having an account.

| Device | Models | Precision |
|---|---|---|
| `cuda:0` | Cohere Arabic ASR + `google/translategemma-12b-it` | ASR fp16; translator NF4 4-bit with fp16 compute |
| `cuda:1` | MOSS Local + MOSS audio codec | Model fp16; codec retains float32 |

This is **stage placement, not tensor parallelism**. No model is split across
GPUs and there is no CPU weight offload. Each device has one admission slot,
so ASR and translation serialize on GPU 0 while TTS on GPU 1 can overlap.
GPU memory is not pooled. GPU 1 still needs about **12.3 GiB for weights alone**;
inference headroom and the 12B translator's quantization overhead must be
validated on the actual session.

1. Accept the Hugging Face terms/access requirements for
   [Cohere](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026)
   and [TranslateGemma](https://huggingface.co/google/translategemma-12b-it).
   TranslateGemma uses Gemma terms, not Apache-2.0.
2. Under **Add-ons > Secrets**, add and enable `HF_TOKEN`, `LIVEKIT_URL`,
   `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET`. Use a read token with access
   to both gated models. Never publish credentials in cells or datasets.
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

**Two `transformers` versions, two processes.** Cohere ASR exists only from
`transformers` 5.4.0 on, but 5.4.0 turned model configs into strict
dataclasses, which breaks MOSS's published remote code (`TypeError:
non-default argument 'sampling_rate' follows default argument`). No single
release loads both. The notebook therefore installs `transformers==5.3.0`
(without dependencies, reusing the shared ones) into
`/kaggle/working/moss-transformers`, and `TTS_WORKER_PYTHONPATH` makes the bot
run MOSS in a child process with that directory first on its `PYTHONPATH`.
The installation cell verifies that the real MOSS configs load there. The
main process keeps 5.4.0. Only text goes to the worker and float32 audio comes
back; the worker also has its own GIL, so MOSS decoding does not compete with
the translator's decode loop. If the worker crashes, the error is logged and
no further speech is produced until the bot restarts.

TranslateGemma uses the official `AutoProcessor` /
`AutoModelForImageTextToText` API and structured `ar` -> `de` requests.
Qwen's sermon system prompt and rolling chat context are intentionally not
sent to this backend. Translation still streams token-by-token into the same
clause chunker. Inputs exceeding the documented 2048-token input context
raise an error rather than silently truncating Arabic text.

**Real-time status: unverified on hardware.** Offline metrics measure
sequential stages after warm-up, not live end-to-end latency. `tts_rtf < 1`
is necessary to synthesize faster than playback, but does not prove that the
whole pipeline keeps up. Listen to the saved audio, then measure live
first-audio delay and backlog over several minutes. Lower `MOSS_VQ_DEPTH`
only as an explicit quality/speed tradeoff; the default remains 32.
The fp16 MOSS path on a T4 also needs a real audio-quality check.

Installation and model-access logs and `/kaggle/working/bot.log` retain
tracebacks after cell output is cleared. Download needed files before the
session is discarded; keep recordings, translations, notebooks and logs
private. Successful local stub tests do not certify GPU compatibility,
translation quality, or real-time performance.

### Colab: existing single-GPU Qwen profile

1. Open [colab_agent.ipynb](./colab_agent.ipynb) in Google Colab. For MOSS,
   select an **A100 40 GB** or another larger GPU (32 GiB+ recommended).
   **The default model combination does not fit a T4.**
2. **Runtime > Run all**. Select your `.env` when the upload dialog appears
  (or add the LiveKit values and `HF_TOKEN` as Colab secrets, or upload `.env` to `/content`
   beforehand; then nothing is asked).
3. Wait for the log line that starts with `READY`. The first start downloads the model.
4. Leave the last cell running.

The startup order is **ASR -> TTS including its audio codec -> translator**.
The codec has its own weights and GPU memory requirement; the TTS parameter
count alone is not the total footprint. Loading TTS first lets the translator's
4-bit loading fallback account for that memory. This does not guarantee that
every model combination fits a T4, or that concurrent inference fits just
because loading succeeds.

The published Local checkpoint currently contains **3.06B parameters**, not
1.7B, and its separate codec contains **1.77B float32 parameters**:

| Component | Weight memory (GiB, excluding caches/activations) |
|---|---:|
| Cohere ASR, fp16/bf16 | 3.85 |
| MOSS Local checkpoint, fp16/bf16 | 5.70 |
| MOSS audio codec, float32 | 6.61 |
| Qwen3-4B, fp16/bf16 | 7.49 |

These are derived from Hugging Face's published safetensors parameter metadata
for [ASR](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026),
[MOSS](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer),
[the codec](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer), and
[Qwen3](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507). Full-precision
weights total about **23.7 GiB**, before inference memory. Startup rejects the
known default combination before loading weights when even its conservative
floor (allowing a 4-bit translator) exceeds the GPU's entire memory.
This check is not a guarantee of inference headroom or real-time performance.

`READY` requires successful ASR, TTS, and translation warm-ups; empty or
non-finite synthesized audio aborts startup. The notebook installs only the
selected TTS engine's dependencies and keeps `torchaudio` on the same release
as Colab's installed CUDA `torch`. MOSS model/codec code is loaded through
`trust_remote_code=True`, without installing the upstream demo package.

If a cell fails, retain `/content/install.log` (installation) and
`/content/bot.log` (full bot output). The last cell raises an error when the bot
exits unexpectedly instead of appearing to complete successfully. Logs survive
cleared cell output, but not deletion/reset of the Colab runtime; download them
before resetting, and redact any credentials before sharing.

### Model configuration

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | - | Hugging Face token; required for the gated Cohere ASR repo |
| `COHERE_MODEL` | `CohereLabs/cohere-transcribe-arabic-07-2026` | Cohere ASR model |
| `COHERE_4BIT` | `0` | Load the Cohere model in 4-bit to free VRAM for the fp16 translator |
| `COHERE_LANGUAGE` | `ar` | Language hint passed to Cohere ASR |
| `TRANSLATION_MODEL` | `Qwen/Qwen3-4B-Instruct-2507` | Translator; an 8B variant translates better but decodes slower |
| `TRANSLATION_DTYPE` | `auto` | Translator compute dtype; `auto` uses bfloat16 for TranslateGemma on T4, because Gemma 3 overflows float16 (empty output) |
| `TRANSLATION_BACKEND` | `qwen` | `qwen` chat template or `translategemma` structured translation template |
| `ASR_DEVICE` / `TRANSLATION_DEVICE` / `TTS_DEVICE` | `auto` | `auto`, `cpu`, or `cuda:N`; Kaggle profile uses `cuda:0` / `cuda:0` / `cuda:1` |
| `TRANSLATION_4BIT` | `0` | Force the slower 4-bit translator; used automatically if fp16 does not fit |
| `TRANSLATION_CONTEXT_TURNS` | `4` | Previous turns sent as chat context; `0` is fastest |
| `LIVEKIT_ROOM` | `arabic-asr-live` | Room name (must match the web server) |
| `TTS_BACKEND` | `moss` | Speech engine: `moss` (MOSS-TTS) or `xtts` (XTTS-v2) |
| `MOSS_MODEL` | `OpenMOSS-Team/MOSS-TTS-Local-Transformer` | Current checkpoint: 3.06B parameters plus a separate 1.77B audio codec; larger GPU required |
| `MOSS_REF_WAV` | - | Optional voice cloning; empty uses the built-in voice |
| `MOSS_VQ_DEPTH` | `32` | RVQ depth (max 32); lower is faster and less detailed |
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
[Cohere Transcribe Arabic](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026),
then set an authorized `HF_TOKEN` in `.env` or Colab Secrets. Qwen3 is public
and needs no token, and MOSS-TTS is Apache-2.0. Colab downloads all weights
into its runtime at startup.

The VRAM actually in use is logged after each model loads. A translator
loading OOM triggers a logged 4-bit retry, but is not a solution to the
default MOSS stack's T4 memory deficit described above.

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

MOSS-TTS (`TTS_BACKEND=moss`, default) needs no configuration - it speaks with
a built-in voice. Set `MOSS_REF_WAV` only if you want to clone a specific
voice. With `TTS_BACKEND=xtts`, XTTS-v2 uses a built-in German speaker by
default; note that XTTS-v2 is published under the Coqui Public Model License,
which is **not** commercially free, while MOSS-TTS is Apache-2.0.

## Speech engines

`TTS_BACKEND` selects the engine. Each one only has to implement `load()` and
`synthesize(text) -> waveform`, so adding another is a small, local change.

| Engine | License | German | Reference clip | Notes |
|---|---|---|---|---|
| `moss` (default) | **Apache-2.0** | `de` in the official language list | **Not needed** (built-in voice) | `MOSS_VQ_DEPTH` is the speed/quality dial |
| `xtts` | Coqui CPML (non-commercial) | yes | not needed (built-in speakers) | fallback engine |

**MOSS-TTS** is the default: its model card lists `de` *and* `ar` among 20
languages, it is Apache-2.0, and - decisively for this pipeline - it speaks
with a built-in voice, so nothing extra is synthesised per call.

What is **not** yet known, and has to be measured on your own GPU: MOSS-TTS is
an autoregressive audio-token model, so its real-time factor on the selected
larger GPU is still unverified. Its weights plus the audio codec compete for
the VRAM that the ASR and translator already use. Both are reported in the log - the
per-turn line shows `tts ... rtf`, and the VRAM in use is logged after each
model loads. Use the larger GPU selected for the MOSS trial.

MOSS-TTS and its codec are loaded with `trust_remote_code=True`; dependencies
are listed in `requirements-agent.txt`. No Git installation of the upstream
demo project is needed. XTTS requires the optional `coqui-tts` package instead.

## 3. Tests

The pipeline tests replace every GPU dependency with a controllable fake, so
they run on any machine:

```bash
python -m pip install numpy pytest python-dotenv
python -m pytest test
python test/bench_pipeline.py
```
