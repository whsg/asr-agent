"""Arabic-to-German live speech-to-speech agent for a LiveKit room.

The agent is a streaming pipeline. Every stage is an independent worker with
its own queue, so stage N of one utterance overlaps stage N+1 of the previous
one and the GPU never idles while the CPU works (or vice versa):

    mic track -> VAD thread -> ASR -> translation (token stream)
              -> TTS (clause chunks) -> playout task -> LiveKit audio track

Results are sent back to the speaker as data messages:

  topic "cohere-transcript":  {"type": "interim" | "final", "id": str, "text": str}
  topic "asr-translation":    {"type": "translation", "id": str, "text": str}

The translation is published incrementally while it is generated, and each
finished clause is spoken immediately instead of waiting for the whole
sentence, which is what keeps the spoken output close to real time.

Runs on a single GPU (tuned for a Colab T4).
"""
from __future__ import annotations

import asyncio
import collections
import contextlib
import heapq
import itertools
import json
import logging
import os
import queue
import re
import secrets
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from livekit import api, rtc
from silero_vad import load_silero_vad

load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO",
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("asr-bot")
logging.getLogger("livekit").setLevel(logging.WARNING)

# Fragmentation is the usual reason a T4 runs out of memory halfway through a
# session even though the totals fit; expandable segments avoid that.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_positive_float(name: str, default: float) -> float:
    """Reads a positive float from the env, falling back to the default."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s instead", name, raw, default)
        return default
    if value <= 0:
        logger.warning("%s must be > 0; using %s instead", name, default)
        return default
    return value


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Reads an integer >= minimum from the env, falling back to the default."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s instead", name, raw, default)
        return default
    if value < minimum:
        logger.warning("%s must be >= %d; using %s instead", name, minimum, default)
        return default
    return value


ROOM_NAME = os.getenv("LIVEKIT_ROOM", "arabic-asr-live")
BOT_IDENTITY = "arabic-asr-bot"
SETTINGS_TOPIC = "asr-settings"
TRANSLATION_TOPIC = "asr-translation"
TRANSCRIPT_TOPIC = "cohere-transcript"

HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None

# --- ASR -------------------------------------------------------------------
COHERE_MODEL_NAME = os.getenv("COHERE_MODEL", "CohereLabs/cohere-transcribe-arabic-07-2026")
COHERE_4BIT = _env_flag("COHERE_4BIT")
COHERE_LANGUAGE = os.getenv("COHERE_LANGUAGE", "ar").strip() or "ar"

# --- Translation -----------------------------------------------------------
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
TRANSLATION_SOURCE_LANG = os.getenv("TRANSLATION_SOURCE_LANG", "ar").strip() or "ar"
TRANSLATION_TARGET_LANG = os.getenv("TRANSLATION_TARGET_LANG", "de").strip() or "de"
# 4-bit NF4 dequantises every weight on every decoded token, which on a T4 is
# 2-4x slower than fp16 *and* costs translation quality. fp16 is therefore the
# default; this flag forces 4-bit, and 4-bit is also used automatically when
# fp16 does not fit into the remaining VRAM.
TRANSLATION_4BIT = _env_flag("TRANSLATION_4BIT")
# System prompt for the translator. The default is tuned for a Friday sermon
# (Jumu'ah khutba); set to an empty value for the plain "translate, no
# explanation" prompt.
TRANSLATION_SYSTEM_PROMPT = os.getenv("TRANSLATION_SYSTEM_PROMPT", "").strip() or (
    "You are a professional simultaneous interpreter rendering a Friday sermon "
    "(Jumu'ah khutba) from {src} to {tgt}. The input arrives in short spoken "
    "fragments and may end mid-sentence: render each one faithfully, keep the "
    "solemn sermon register, keep proper names and Qur'anic phrases in their "
    "standard transliteration, and stay consistent with the earlier turns. "
    "Reply with the {tgt} translation only - no notes, no explanation, no "
    "repetition of the {src} source."
).format(src=TRANSLATION_SOURCE_LANG, tgt=TRANSLATION_TARGET_LANG)
# How many previous (source, translation) pairs to send as chat context so the
# translator stays consistent across turns (0 disables the context). Every turn
# of context is re-prefilled on each request, so this trades latency for
# consistency; 4 is a good balance.
TRANSLATION_CONTEXT_TURNS = _env_int("TRANSLATION_CONTEXT_TURNS", 4, minimum=0)
# Hard ceiling for generated tokens, shared by the ASR and the translator. The
# effective limit per request is derived from the input, so normal turns stop
# long before this; it only exists to stop a degenerate decode.
MAX_GEN_TOKENS = _env_int("MAX_GEN_TOKENS", 448)
# Tokens budgeted per second of audio for the ASR. Speech tops out around five
# words per second and Arabic script costs a few tokens per word, so 25 leaves
# a wide margin over real speech.
ASR_TOKENS_PER_SECOND = _env_int("ASR_TOKENS_PER_SECOND", 25)
# Output-token budget as a multiple of the source length for the translator.
TRANSLATION_TOKEN_RATIO = _env_positive_float("TRANSLATION_TOKEN_RATIO", 3.0)

# --- TTS -------------------------------------------------------------------
TTS_BACKEND = os.getenv("TTS_BACKEND", "f5").strip().lower()
if TTS_BACKEND not in ("f5", "xtts"):
    logger.warning("Invalid TTS_BACKEND=%r; using 'f5' instead", TTS_BACKEND)
    TTS_BACKEND = "f5"
F5_GERMAN_REPO = os.getenv("F5_GERMAN_REPO", "aihpi/F5-TTS-German").strip() or "aihpi/F5-TTS-German"
F5_GERMAN_CKPT = os.getenv("F5_GERMAN_CKPT", "F5TTS_Base/model_420000.safetensors").strip() or "F5TTS_Base/model_420000.safetensors"
F5_BASE_MODEL = os.getenv("F5_BASE_MODEL", "F5TTS_v1_Base").strip() or "F5TTS_v1_Base"
# F5-TTS is a voice-cloning model: it needs a short reference clip (~5-10 s)
# of the desired voice and the spoken text of it (for German, a German clip).
# F5_REF_WAV: file in the Colab runtime (e.g. /content/reference.wav) or http(s) URL.
F5_REF_WAV = os.getenv("F5_REF_WAV", "").strip()
F5_REF_TEXT = os.getenv("F5_REF_TEXT", "").strip()
F5_USE_EMA = _env_flag("F5_USE_EMA", True)
# ODE inference steps (lower is much faster; 8 is a good speed/quality balance,
# 32 is the F5-TTS default and the slowest).
F5_NFE_STEPS = _env_int("F5_NFE_STEPS", 8)
F5_SPEED = _env_positive_float("F5_SPEED", 1.0)
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# Speak each finished clause as soon as the translator has produced it instead
# of waiting for the full sentence. This is the single biggest win for
# time-to-first-audio; disable it if chunk boundaries hurt the prosody.
STREAM_TTS = _env_flag("STREAM_TTS", True)
# Minimum characters before the first / any later clause is handed to the TTS.
# The first chunk is deliberately short so audio starts early; later chunks are
# longer because F5-TTS sounds better with more context.
TTS_FIRST_CHUNK_CHARS = _env_int("TTS_FIRST_CHUNK_CHARS", 40)
TTS_CHUNK_CHARS = _env_int("TTS_CHUNK_CHARS", 110)
# Force a chunk break once the buffer gets this long, even without punctuation.
TTS_MAX_CHUNK_CHARS = _env_int("TTS_MAX_CHUNK_CHARS", 240)
# Never hand the TTS a fragment shorter than this: F5-TTS needs a few words of
# context and clips very short inputs.
TTS_MIN_CHUNK_CHARS = _env_int("TTS_MIN_CHUNK_CHARS", 25)

# --- Audio output ----------------------------------------------------------
AUDIO_SAMPLE_RATE = 24_000
AUDIO_FRAME_SAMPLES = AUDIO_SAMPLE_RATE // 50
# LiveKit buffer in front of the encoder. Small enough that our own backlog
# accounting stays accurate, large enough to survive a scheduling hiccup.
AUDIO_QUEUE_MS = _env_int("AUDIO_QUEUE_MS", 800)
# Silence inserted between two turns so they do not run into each other.
TURN_GAP_SECONDS = 0.12
# If the agent falls behind by more than this many seconds of unplayed audio,
# the oldest pending chunks are dropped. Without this the lag grows forever as
# soon as synthesis is slower than real time.
MAX_AUDIO_BACKLOG_SECONDS = _env_positive_float("MAX_AUDIO_BACKLOG_SECONDS", 10.0)

# --- Scheduling ------------------------------------------------------------
# How many model calls may occupy the GPU at once. >1 lets the Python-side
# pre/post-processing of one stage overlap the CUDA kernels of another, which
# is where most of the pipelining win comes from on a single GPU.
GPU_CONCURRENCY = _env_int("GPU_CONCURRENCY", 2)
# Live Arabic captions while someone is still speaking. They are pure extra
# GPU work on the critical path's budget, so they only ever run when the
# ASR -> translation -> TTS path is completely idle.
INTERIM_CAPTIONS = _env_flag("INTERIM_CAPTIONS", True)

PRIO_SPEECH = 0  # TTS: closest to the listener's ear, an underrun is audible
PRIO_TURN = 1  # final ASR and translation
PRIO_INTERIM = 8  # best-effort live captions
PRIO_CRITICAL_MAX = PRIO_TURN

# --- Segmentation ----------------------------------------------------------
SAMPLE_RATE = 16_000
VAD_CHUNK = 512  # Silero VAD requires exactly 32 ms windows at 16 kHz
CHUNK_SECONDS = VAD_CHUNK / SAMPLE_RATE
SPEECH_START_PROB = 0.5
SPEECH_KEEP_PROB = 0.35
# Trailing silence that ends an utterance once it already contains a full
# clause (env END_SILENCE_SECONDS, default 0.4 s).
END_SILENCE_SECONDS = _env_positive_float("END_SILENCE_SECONDS", 0.4)
END_SILENCE_CHUNKS = round(END_SILENCE_SECONDS / CHUNK_SECONDS)
# A speaker pausing for effect mid-sentence must not end the utterance: below
# MIN_SENTENCE_SECONDS of actual speech, a much longer pause is required before
# the segment is closed. Fragmenting sentences is expensive twice over - the
# translator loses the context it needs, and every extra turn pays the TTS
# engine's fixed per-call cost, which is what pushes synthesis past real time.
MIN_SENTENCE_SECONDS = _env_positive_float("MIN_SENTENCE_SECONDS", 2.5)
SHORT_SEGMENT_SILENCE_SECONDS = _env_positive_float("SHORT_SEGMENT_SILENCE_SECONDS", 1.0)
SHORT_SEGMENT_SILENCE_CHUNKS = round(SHORT_SEGMENT_SILENCE_SECONDS / CHUNK_SECONDS)
PRE_ROLL_CHUNKS = round(0.25 / CHUNK_SECONDS)
TAIL_CHUNKS = round(0.15 / CHUNK_SECONDS)
MIN_SPEECH_CHUNKS = round(0.25 / CHUNK_SECONDS)
# Env MAX_UTTERANCE_SECONDS: longest utterance (seconds) sent to the ASR.
MAX_UTTERANCE_SECONDS = _env_positive_float("MAX_UTTERANCE_SECONDS", 12.0)
MAX_UTTERANCE_CHUNKS = round(MAX_UTTERANCE_SECONDS / CHUNK_SECONDS)
SPLIT_SEARCH_CHUNKS = round(2.0 / CHUNK_SECONDS)
INTERIM_STEP_CHUNKS = round(0.6 / CHUNK_SECONDS)

_utterance_counter = itertools.count()
_boot_id = secrets.token_hex(3)


def new_utterance_id() -> str:
    # Unique across bot restarts so the browser never overwrites old lines.
    return f"{_boot_id}-{next(_utterance_counter)}"


# ---------------------------------------------------------------------------
# Speech segmentation
# ---------------------------------------------------------------------------


class Segmenter:
    """Splits a continuous audio stream into utterances using Silero VAD.

    Runs on a worker thread, never on the event loop. Not thread-safe: every
    speaker owns one instance plus one thread.
    """

    def __init__(self) -> None:
        self.vad = load_silero_vad()
        self.pending = np.empty(0, dtype=np.float32)
        self.pre_roll: collections.deque[np.ndarray] = collections.deque(maxlen=PRE_ROLL_CHUNKS)
        self.chunks: list[np.ndarray] = []
        self.probs: list[float] = []
        self.speaking = False
        self.silent_chunks = 0
        self.speech_chunks = 0
        # Cached concatenation of self.chunks so repeated interim snapshots do
        # not re-copy the whole utterance every time (that was quadratic).
        self._snapshot: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(self.chunks)

    def snapshot(self) -> np.ndarray:
        if self._snapshot is None or self._snapshot.size != self.size * VAD_CHUNK:
            self._snapshot = np.concatenate(self.chunks) if self.chunks else np.empty(0, dtype=np.float32)
        return self._snapshot

    def _append(self, chunk: np.ndarray, prob: float) -> None:
        self.chunks.append(chunk)
        self.probs.append(prob)
        if prob >= SPEECH_KEEP_PROB:
            self.speech_chunks += 1
        self._snapshot = None

    def _silence_needed(self) -> int:
        """Trailing silence (in VAD chunks) required to end the current segment.

        A segment that already holds a full clause ends on a short pause, which
        keeps latency low. A short segment needs a much longer pause, because a
        mid-sentence breath must not be mistaken for the end of a sentence.
        """
        if self.speech_chunks * CHUNK_SECONDS >= MIN_SENTENCE_SECONDS:
            return END_SILENCE_CHUNKS
        return max(END_SILENCE_CHUNKS, SHORT_SEGMENT_SILENCE_CHUNKS)

    @torch.inference_mode()
    def feed(self, pcm: np.ndarray) -> list[np.ndarray | None]:
        """Adds audio and returns finished utterances (None = too short to be speech)."""
        self.pending = np.concatenate((self.pending, pcm)) if self.pending.size else pcm
        finished: list[np.ndarray | None] = []

        offset = 0
        while self.pending.size - offset >= VAD_CHUNK:
            chunk = self.pending[offset:offset + VAD_CHUNK].copy()
            offset += VAD_CHUNK
            prob = float(self.vad(torch.from_numpy(chunk), SAMPLE_RATE))

            if not self.speaking:
                if prob < SPEECH_START_PROB:
                    self.pre_roll.append(chunk)
                    continue
                self.speaking = True
                self.chunks = list(self.pre_roll)
                self.probs = [0.0] * len(self.chunks)
                self.speech_chunks = 0
                self._snapshot = None
                self.pre_roll.clear()
                self.silent_chunks = 0

            self._append(chunk, prob)
            self.silent_chunks = 0 if prob >= SPEECH_KEEP_PROB else self.silent_chunks + 1

            if self.silent_chunks >= self._silence_needed():
                end = self.size - self.silent_chunks + TAIL_CHUNKS
                self.pre_roll.extend(self.chunks[end:])
                finished.append(self._cut(end))
                self.speaking = False
            elif self.size >= MAX_UTTERANCE_CHUNKS:
                # Split long speech at the quietest moment of the last seconds, not mid-word.
                window = self.probs[-SPLIT_SEARCH_CHUNKS:]
                end = self.size - len(window) + int(np.argmin(window)) + 1
                finished.append(self._cut(end))
                self.silent_chunks = 0
        self.pending = self.pending[offset:]
        return finished

    def flush(self) -> np.ndarray | None:
        if not self.speaking:
            return None
        self.speaking = False
        return self._cut(self.size)

    def _cut(self, end: int) -> np.ndarray | None:
        """Returns the first `end` chunks as an utterance and keeps the rest."""
        head_chunks, head_probs = self.chunks[:end], self.probs[:end]
        self.chunks, self.probs = self.chunks[end:], self.probs[end:]
        self.speech_chunks = sum(p >= SPEECH_KEEP_PROB for p in self.probs)
        self._snapshot = None
        if sum(p >= SPEECH_KEEP_PROB for p in head_probs) < MIN_SPEECH_CHUNKS:
            return None
        return np.concatenate(head_chunks)


class SpeakerStream:
    """One participant: a VAD thread plus the state the pipeline needs.

    The LiveKit audio callback only drops raw PCM into a queue; all VAD work
    happens on a dedicated thread so the event loop stays free to run the
    pipeline stages. The thread drains every frame that arrived in the meantime
    in one batch, which keeps up even if the loop was blocked for a moment.
    """

    _STOP = object()

    def __init__(self, identity: str, loop: asyncio.AbstractEventLoop, on_utterance) -> None:
        self.identity = identity
        self.utterance_id = new_utterance_id()
        self._loop = loop
        self._on_utterance = on_utterance
        self._inbox: queue.SimpleQueue = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._segmenter: Segmenter | None = None
        # Published by the VAD thread, read by the interim worker.
        self.speaking = False
        self.pending_chunks = 0
        self.pending_speech_chunks = 0
        self._thread = threading.Thread(target=self._run, name=f"vad-{identity}", daemon=True)
        self._thread.start()

    def push(self, pcm: np.ndarray) -> None:
        self._inbox.put(pcm)

    def close(self) -> None:
        self._inbox.put(self._STOP)

    def snapshot(self) -> np.ndarray | None:
        """Thread-safe copy of the audio received for the current utterance."""
        with self._lock:
            if self._segmenter is None or not self._segmenter.speaking:
                return None
            return self._segmenter.snapshot()

    def next_utterance(self) -> None:
        self.utterance_id = new_utterance_id()

    def _run(self) -> None:
        segmenter = Segmenter()
        with self._lock:
            self._segmenter = segmenter
        try:
            while True:
                item = self._inbox.get()
                if item is self._STOP:
                    break
                batch = [item]
                stopped = False
                while True:
                    try:
                        nxt = self._inbox.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is self._STOP:
                        stopped = True
                        break
                    batch.append(nxt)
                pcm = batch[0] if len(batch) == 1 else np.concatenate(batch)
                with self._lock:
                    finished = segmenter.feed(pcm)
                    self.speaking = segmenter.speaking
                    self.pending_chunks = segmenter.size
                    self.pending_speech_chunks = segmenter.speech_chunks
                for audio in finished:
                    self._emit(audio)
                if stopped:
                    break
            with self._lock:
                tail = segmenter.flush()
                self.speaking = False
            if tail is not None:
                self._emit(tail)
        except Exception:  # noqa: BLE001 - a dead VAD thread must be visible
            logger.exception("VAD thread for %s crashed", self.identity)

    def _emit(self, audio: np.ndarray | None) -> None:
        self._loop.call_soon_threadsafe(self._on_utterance, self, audio, time.monotonic())


# ---------------------------------------------------------------------------
# GPU scheduling
# ---------------------------------------------------------------------------


class GpuScheduler:
    """Priority admission control for the single shared GPU.

    A bounded number of model calls may run at once: CUDA interleaves their
    kernels and the Python-side pre/post-processing of one stage overlaps the
    kernels of another, while the bound keeps activation memory from spiking
    into an OOM. Waiters are served by priority, so a best-effort live caption
    can never delay the ASR -> translation -> TTS path of a finished utterance.
    """

    def __init__(self, concurrency: int) -> None:
        self._slots = max(1, concurrency)
        self._active = 0
        self._waiting: list[tuple[int, int, asyncio.Future]] = []
        self._seq = itertools.count()
        self.critical_inflight = 0

    @property
    def busy(self) -> bool:
        return self.critical_inflight > 0 or bool(self._waiting)

    @contextlib.asynccontextmanager
    async def reserve(self, priority: int):
        await self._acquire(priority)
        critical = priority <= PRIO_CRITICAL_MAX
        if critical:
            self.critical_inflight += 1
        try:
            yield
        finally:
            if critical:
                self.critical_inflight -= 1
            self._release()

    async def run(self, priority: int, func, *args):
        """Runs a blocking model call on a worker thread under a GPU slot."""
        async with self.reserve(priority):
            return await asyncio.to_thread(func, *args)

    async def _acquire(self, priority: int) -> None:
        if self._active < self._slots and not self._waiting:
            self._active += 1
            return
        future = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiting, (priority, next(self._seq), future))
        try:
            await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                # The slot was handed over right before the cancellation
                # arrived; give it back instead of leaking it.
                self._release()
            raise

    def _release(self) -> None:
        while self._waiting:
            _, _, future = heapq.heappop(self._waiting)
            if not future.done():
                future.set_result(None)  # hands the slot over, _active unchanged
                return
        self._active -= 1


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def pick_device_dtype() -> tuple[torch.dtype, str]:
    if not torch.cuda.is_available():
        logger.warning("No GPU found; the models will run on the CPU and be slow.")
        return torch.float32, "cpu"
    major, _ = torch.cuda.get_device_capability(0)
    # A T4 is sm_75 and has no bf16 tensor cores, so fp16 is the fast path there.
    return (torch.bfloat16 if major >= 8 else torch.float16), "cuda:0"


def _dtype_kwargs(dtype) -> dict:
    """`dtype=` on transformers >= 4.56, `torch_dtype=` on older releases."""
    import transformers

    parts = transformers.__version__.split(".")
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 else 0
    if major > 4 or (major == 4 and minor >= 56):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def load_with_sdpa(loader, name: str, **kwargs):
    """Loads a model with SDPA attention, retrying without it if unsupported."""
    try:
        return loader.from_pretrained(name, attn_implementation="sdpa", **kwargs)
    except (ValueError, ImportError) as exc:
        if "attn_implementation" not in str(exc) and "scaled_dot_product" not in str(exc):
            raise
        logger.info("%s has no SDPA kernel; falling back to the eager attention.", name)
        return loader.from_pretrained(name, **kwargs)


def _four_bit_config(dtype):
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )


def _vram_report(stage: str) -> None:
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info(0)
    logger.info("VRAM after %s: %.1f GB used of %.1f GB", stage, (total - free) / 2**30, total / 2**30)


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _device_map(device: str):
    """Explicit single-device placement.

    device_map="auto" silently offloads layers to the CPU when VRAM looks
    tight and then runs an order of magnitude slower; an explicit map either
    fits on the GPU or raises an OOM we can react to.
    """
    return {"": 0} if device != "cpu" else {"": "cpu"}


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------


class AsrEngine:
    """Cohere Transcribe Arabic. Blocking calls, serialised by its own lock."""

    topic = TRANSCRIPT_TOPIC

    def __init__(self, dtype, device: str) -> None:
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        load_kwargs: dict = {"device_map": _device_map(device)}
        if HF_TOKEN:
            load_kwargs["token"] = HF_TOKEN
        if COHERE_4BIT:
            load_kwargs["quantization_config"] = _four_bit_config(dtype)
        else:
            load_kwargs.update(_dtype_kwargs(dtype))
        try:
            self.processor = AutoProcessor.from_pretrained(COHERE_MODEL_NAME, token=HF_TOKEN)
            self.model = load_with_sdpa(
                CohereAsrForConditionalGeneration, COHERE_MODEL_NAME, **load_kwargs
            )
        except Exception as exc:  # noqa: BLE001 - wrapped with actionable guidance below
            raise RuntimeError(
                "Could not load the Cohere model: "
                f"{exc}. The repo is gated: accept the terms on huggingface.co and "
                "set HF_TOKEN in the .env."
            ) from exc
        self.model.eval()
        self.model.generation_config.use_cache = True
        self._lock = threading.Lock()

    def transcribe(self, audio: np.ndarray) -> str:
        seconds = audio.size / SAMPLE_RATE
        # Budget generously: speech tops out around five words per second and
        # Arabic script costs a few tokens per word, so this only ever stops a
        # hallucinating decode - it must never clip a normal transcript.
        budget = min(MAX_GEN_TOKENS, int(seconds * ASR_TOKENS_PER_SECOND) + 48)
        with self._lock:
            inputs = self.processor(
                audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", language=COHERE_LANGUAGE
            )
            inputs = inputs.to(self.model.device, dtype=self.model.dtype)
            with torch.inference_mode():
                start = time.perf_counter()
                outputs = self.model.generate(**inputs, max_new_tokens=budget, do_sample=False)
            ms = (time.perf_counter() - start) * 1000
            decoded = self.processor.decode(outputs, skip_special_tokens=True)
        if isinstance(decoded, list):
            decoded = decoded[0] if decoded else ""
        # Encoder-decoder ASR models have no input_ids (audio features only) and
        # generate() returns decoder tokens only; decoder-only models prepend the
        # prompt, so subtract its length when present.
        input_ids = inputs.get("input_ids")
        prompt_len = input_ids.shape[1] if input_ids is not None else 0
        generated = outputs.shape[1] - prompt_len
        if generated >= budget:
            logger.warning(
                "ASR hit its %d-token budget for %.1f s of audio - the transcript is cut off. "
                "Raise MAX_GEN_TOKENS or ASR_TOKENS_PER_SECOND.",
                budget, seconds,
            )
        logger.debug("ASR %.0f ms for %.1f s audio (%d tokens)", ms, seconds, generated)
        return decoded.strip()


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


class Translator:
    """Qwen3 translator with token streaming.

    Streaming matters more than raw tokens/second here: the pipeline can start
    speaking the first clause while the rest is still being generated.
    """

    def __init__(self, dtype, device: str) -> None:
        from transformers import AutoTokenizer

        load_kwargs: dict = {}
        if HF_TOKEN:
            load_kwargs["token"] = HF_TOKEN
        self.tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL, **load_kwargs)
        self.quantized = TRANSLATION_4BIT
        if TRANSLATION_4BIT:
            self.model = self._load(dtype, device, load_kwargs, four_bit=True)
        else:
            try:
                self.model = self._load(dtype, device, load_kwargs, four_bit=False)
            except Exception as exc:  # noqa: BLE001 - only OOM is recoverable
                if not _is_oom(exc):
                    raise
                logger.warning(
                    "%s does not fit next to the ASR model in fp16/bf16 (%s); falling back to "
                    "4-bit. 4-bit decodes slower and translates slightly worse - use a GPU with "
                    "more VRAM or set COHERE_4BIT=1 to free headroom.",
                    TRANSLATION_MODEL,
                    exc.__class__.__name__,
                )
                torch.cuda.empty_cache()
                self.model = self._load(dtype, device, load_kwargs, four_bit=True)
                self.quantized = True
        self.model.eval()
        self.model.generation_config.use_cache = True
        self._lock = threading.Lock()

    @staticmethod
    def _load(dtype, device: str, load_kwargs: dict, four_bit: bool):
        from transformers import AutoModelForCausalLM

        extra = (
            {"quantization_config": _four_bit_config(dtype)} if four_bit else _dtype_kwargs(dtype)
        )
        return load_with_sdpa(
            AutoModelForCausalLM,
            TRANSLATION_MODEL,
            device_map=_device_map(device),
            **extra,
            **load_kwargs,
        )

    def _build_inputs(self, text: str, context: list[tuple[str, str]]):
        messages: list[dict[str, str]] = []
        if TRANSLATION_SYSTEM_PROMPT:
            messages.append({"role": "system", "content": TRANSLATION_SYSTEM_PROMPT})
        # Interleaved source/translation pairs. Sending all sources first and
        # all translations afterwards (as an earlier version did) destroys the
        # few-shot pattern and measurably degrades the output.
        for previous_source, previous_target in context:
            messages.append({"role": "user", "content": previous_source})
            messages.append({"role": "assistant", "content": previous_target})
        messages.append({"role": "user", "content": text})
        # enable_thinking=False: some Qwen3 instruct models think with a
        # reasoning block by default, which is slow and can eat the whole output
        # budget before reaching the translation. Templates without a thinking
        # branch simply ignore the kwarg.
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

    def stream(self, text: str, context: list[tuple[str, str]], emit) -> None:
        """Generates the translation and calls `emit(piece)` for each new piece."""
        from transformers import TextIteratorStreamer

        with self._lock:
            inputs = self._build_inputs(text, context)
            input_len = inputs["input_ids"].shape[1]
            source_tokens = len(self.tokenizer(text)["input_ids"])
            # German can be noticeably longer than the Arabic source, so budget
            # generously: this cap exists to stop a degenerate repetition loop,
            # not to shorten a legitimate translation.
            budget = max(
                64, min(MAX_GEN_TOKENS, int(source_tokens * TRANSLATION_TOKEN_RATIO) + 48)
            )
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120.0
            )
            error: list[BaseException] = []
            produced: list[int] = []

            def generate() -> None:
                try:
                    with torch.inference_mode():
                        output = self.model.generate(
                            **inputs,
                            max_new_tokens=budget,
                            do_sample=False,
                            repetition_penalty=1.05,
                            streamer=streamer,
                        )
                    produced.append(output.shape[1] - input_len)
                except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
                    error.append(exc)
                    streamer.end()

            worker = threading.Thread(target=generate, name="qwen-generate", daemon=True)
            worker.start()
            try:
                for piece in streamer:
                    if piece:
                        emit(piece)
            finally:
                worker.join()
            if error:
                raise error[0]
            if produced and produced[0] >= budget:
                logger.warning(
                    "Translation hit its %d-token budget for a %d-token source - the output is "
                    "cut off. Raise TRANSLATION_TOKEN_RATIO or MAX_GEN_TOKENS.",
                    budget, source_tokens,
                )

    def translate(self, text: str, context: list[tuple[str, str]] = ()) -> str:
        parts: list[str] = []
        self.stream(text, list(context), parts.append)
        return "".join(parts).strip()


# Strong clause boundaries (sentence end, colon, semicolon, newline) and the
# weak ones used only when a chunk would otherwise grow too long.
_STRONG_BREAK = re.compile(r"[.!?؟…:;\n]")
_WEAK_BREAK = re.compile(r"[,،)\u00bb\"]")


def speakable_target(index: int) -> int:
    """Target length of the chunk with this index within a translation.

    The first chunk is deliberately short so audio starts early; later chunks
    grow quickly because every TTS call has a large fixed cost - F5-TTS
    re-synthesises the reference clip on each call and crops it afterwards, so
    splitting one sentence into many chunks multiplies the GPU work.
    """
    if index <= 0:
        return TTS_FIRST_CHUNK_CHARS
    return min(TTS_MAX_CHUNK_CHARS, TTS_CHUNK_CHARS * index)


def split_speakable(buffer: str, index: int, flush: bool) -> tuple[str, str]:
    """Splits off the next chunk that is worth speaking.

    Returns (chunk, rest); an empty chunk means "keep buffering". The cut is
    the *earliest* clause boundary, so audio starts as soon as a speakable unit
    exists instead of waiting for the longest one that still fits. The target
    length only decides when to start looking: a complete sentence that falls
    somewhat short of it is still a better chunk than a padded one.
    """
    text = buffer.lstrip()
    if not text:
        return "", ""
    if flush:
        return text, ""
    target = speakable_target(index)
    if len(text) < target:
        return "", buffer
    # F5-TTS needs a few words of context to sound natural, so never cut below
    # this even when a sentence ends very early.
    floor = max(TTS_MIN_CHUNK_CHARS, target // 2)

    window = text[: max(target, TTS_MAX_CHUNK_CHARS)]
    cut = min((m.end() for m in _STRONG_BREAK.finditer(window) if m.end() >= floor), default=0)
    if not cut and len(text) >= TTS_MAX_CHUNK_CHARS:
        cut = min((m.end() for m in _WEAK_BREAK.finditer(window) if m.end() >= floor), default=0)
        if not cut:
            cut = window.rfind(" ")
    if cut <= 0:
        return "", buffer
    return text[:cut].strip(), text[cut:]


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or audio.size == 0:
        return audio
    duration = audio.size / source_rate
    target_len = max(1, int(round(duration * target_rate)))
    source_t = np.linspace(0.0, duration, num=audio.size, endpoint=False, dtype=np.float64)
    target_t = np.linspace(0.0, duration, num=target_len, endpoint=False, dtype=np.float64)
    return np.interp(target_t, source_t, audio).astype(np.float32)


def _audio_info(path: str) -> tuple[float, int]:
    """(duration in seconds, channel count) of an audio file; (0.0, 0) if unreadable."""
    try:
        import soundfile

        info = soundfile.info(path)
        return info.frames / float(info.samplerate), int(info.channels)
    except Exception:  # noqa: BLE001 - fall back to the stdlib parser
        pass
    try:
        import wave

        with wave.open(path, "rb") as handle:
            return handle.getnframes() / float(handle.getframerate()), handle.getnchannels()
    except Exception:  # noqa: BLE001 - purely informational
        return 0.0, 0


def _resolve_f5_ref_wav() -> str:
    """Local path of the F5-TTS reference audio; downloads http(s) URLs once."""
    if F5_REF_WAV.startswith(("http://", "https://")):
        dest = Path.home() / ".cache" / "asr-agent" / "f5_ref.wav"
        if not dest.exists():
            logger.info("Downloading F5-TTS reference audio to %s ...", dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(F5_REF_WAV, str(dest))
        return str(dest)
    path = Path(F5_REF_WAV)
    if not path.is_file():
        raise RuntimeError(f"F5 reference file not found: {F5_REF_WAV}")
    return str(path)


class F5Synthesizer:
    """F5-TTS voice cloning (default engine).

    A German target language uses the fine-tune aihpi/F5-TTS-German, any other
    target the standard F5-TTS base model.
    """

    def __init__(self) -> None:
        self.model = None
        self.ref_wav = ""
        self.ref_text = ""
        self.ref_seconds = 0.0
        self.sample_rate = AUDIO_SAMPLE_RATE

    def load(self):
        if self.model is not None:
            return self.model
        from f5_tts.api import F5TTS
        from huggingface_hub import hf_hub_download

        if not F5_REF_WAV:
            raise RuntimeError(
                "TTS backend 'f5' is enabled but F5_REF_WAV is not set. F5-TTS is a "
                "voice-cloning model: set F5_REF_WAV to a short reference clip "
                "(~5-10 s wav of the desired voice) available in the Colab runtime and "
                "F5_REF_TEXT to the spoken text of the clip (or set TTS_BACKEND=xtts)."
            )
        if not F5_REF_TEXT:
            raise RuntimeError(
                "TTS backend 'f5' is enabled but F5_REF_TEXT is not set. The spoken "
                "text of the reference clip is required (without it F5-TTS would load "
                "a ~1.6 GB Whisper model just to transcribe the clip)."
            )

        if TRANSLATION_TARGET_LANG.strip().lower() == "de":
            logger.info("Loading F5-TTS: German fine-tune %s (%s) ...", F5_GERMAN_REPO, F5_GERMAN_CKPT)
            ckpt_file = hf_hub_download(repo_id=F5_GERMAN_REPO, filename=F5_GERMAN_CKPT)
            vocab_file = hf_hub_download(repo_id=F5_GERMAN_REPO, filename="vocab.txt")
            model_name = "F5TTS_Base"
        else:
            logger.info("Loading F5-TTS: standard base model %s ...", F5_BASE_MODEL)
            ckpt_file = ""
            vocab_file = ""
            model_name = F5_BASE_MODEL
        self.model = F5TTS(
            model=model_name,
            ckpt_file=ckpt_file,
            vocab_file=vocab_file,
            use_ema=F5_USE_EMA,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        self.sample_rate = int(getattr(self.model, "target_sample_rate", AUDIO_SAMPLE_RATE))
        self._prepare_reference()
        return self.model

    def _prepare_reference(self) -> None:
        """Trims and caches the reference clip once instead of on every call."""
        self.ref_wav = _resolve_f5_ref_wav()
        self.ref_text = F5_REF_TEXT
        try:
            from f5_tts.infer.utils_infer import preprocess_ref_audio_text

            self.ref_wav, self.ref_text = preprocess_ref_audio_text(
                self.ref_wav, F5_REF_TEXT, show_info=lambda *_: None
            )
        except Exception:  # noqa: BLE001 - optional fast path, F5 redoes it internally
            logger.debug("Could not pre-process the F5 reference clip; F5 will do it per call.")
        self.ref_seconds, channels = _audio_info(self.ref_wav)
        if not self.ref_seconds:
            return
        # F5-TTS synthesises reference + target together on every call and
        # crops the reference afterwards, so the clip's length is pure overhead
        # paid again for every spoken chunk.
        typical_chunk_audio = 3.0
        waste = self.ref_seconds / (self.ref_seconds + typical_chunk_audio)
        logger.info(
            "F5 reference clip: %.1f s, %d channel(s) - roughly %.0f%% of each TTS call is "
            "spent re-synthesising it.",
            self.ref_seconds, channels, waste * 100,
        )
        if self.ref_seconds > 8:
            logger.warning(
                "The F5 reference clip is %.1f s long. Every spoken chunk re-synthesises it and "
                "then throws it away, so this multiplies the cost of the whole speech output. "
                "Cut F5_REF_WAV to a 5-6 s mono clip and set F5_REF_TEXT to exactly what is "
                "spoken in it - that alone can nearly halve the TTS time per chunk.",
                self.ref_seconds,
            )
        if channels > 1:
            logger.warning(
                "The F5 reference clip has %d channels; F5-TTS expects mono and has to downmix "
                "it. Convert it to mono to avoid surprises.", channels,
            )

    def synthesize(self, text: str) -> np.ndarray:
        model = self.load()
        wav, sample_rate, _spectrogram = model.infer(
            ref_file=self.ref_wav,
            ref_text=self.ref_text,
            gen_text=text,
            show_info=lambda *args: None,
            progress=None,
            nfe_step=F5_NFE_STEPS,
            speed=F5_SPEED,
        )
        if wav is None:
            return np.zeros(0, dtype=np.float32)
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        return _resample(audio, int(sample_rate or self.sample_rate), AUDIO_SAMPLE_RATE)


class XttsSynthesizer:
    """XTTS-v2 voice (only used when TTS_BACKEND=xtts)."""

    def __init__(self) -> None:
        self.model = None
        self.speaker = os.getenv("XTTS_SPEAKER", "").strip() or None
        self.speaker_wav = os.getenv("XTTS_SPEAKER_WAV", "").strip() or None
        self.sample_rate = AUDIO_SAMPLE_RATE

    def load(self):
        if self.model is not None:
            return self.model
        if self.speaker_wav and not Path(self.speaker_wav).is_file():
            raise RuntimeError(f"XTTS speaker file not found: {self.speaker_wav}")
        from TTS.api import TTS

        self.model = TTS(XTTS_MODEL_NAME, gpu=torch.cuda.is_available(), progress_bar=False)
        if not self.speaker and not self.speaker_wav:
            speakers = self.model.speakers or []
            if not speakers:
                raise RuntimeError("XTTS has no default voice; configure XTTS_SPEAKER_WAV.")
            self.speaker = speakers[0]
        rate = getattr(getattr(self.model, "synthesizer", None), "output_sample_rate", None)
        self.sample_rate = int(rate or AUDIO_SAMPLE_RATE)
        return self.model

    def synthesize(self, text: str) -> np.ndarray:
        model = self.load()
        options = {"text": text, "language": TRANSLATION_TARGET_LANG}
        if self.speaker_wav:
            options["speaker_wav"] = self.speaker_wav
        elif self.speaker:
            options["speaker"] = self.speaker
        audio = np.asarray(model.tts(**options), dtype=np.float32).reshape(-1)
        return _resample(audio, self.sample_rate, AUDIO_SAMPLE_RATE)


class SpeechSynthesizer:
    """Spoken output for the translation; the engine is picked by TTS_BACKEND."""

    def __init__(self) -> None:
        self.engine = XttsSynthesizer() if TTS_BACKEND == "xtts" else F5Synthesizer()
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            self.engine.load()

    def synthesize(self, text: str) -> np.ndarray:
        with self._lock:
            start = time.perf_counter()
            audio = self.engine.synthesize(text)
            ms = (time.perf_counter() - start) * 1000
        seconds = audio.size / AUDIO_SAMPLE_RATE
        logger.debug("TTS %.0f ms for %.1f s audio (rtf %.2f)", ms, seconds, (ms / 1000) / max(seconds, 1e-6))
        return audio


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """One utterance travelling through the pipeline."""

    identity: str
    utterance_id: str
    audio: np.ndarray | None
    captured_at: float
    source_text: str = ""
    translation: str = ""
    wants_audio: bool = False
    asr_ms: float = 0.0
    translation_ms: float = 0.0
    first_token_ms: float = 0.0
    tts_ms: float = 0.0
    audio_seconds: float = 0.0
    first_audio_at: float | None = None
    chunks: int = 0


@dataclass
class AudioChunk:
    turn: Turn
    samples: np.ndarray
    last: bool = False
    tail_gap: float = field(default=TURN_GAP_SECONDS)


class Pipeline:
    """Stage workers for one LiveKit room.

    Every stage owns a task and a queue, so an utterance can be transcribed
    while the previous one is being translated, spoken, and played back. The
    playout stage is deliberately separate: it runs in real time and must never
    be able to stall a GPU stage, which is what made the old single-loop design
    drift further behind with every turn.
    """

    def __init__(self, asr: AsrEngine, translator: Translator, speech: SpeechSynthesizer,
                 room: rtc.Room, audio_source: rtc.AudioSource, scheduler: GpuScheduler) -> None:
        self.asr = asr
        self.translator = translator
        self.speech = speech
        self.room = room
        self.audio_source = audio_source
        self.gpu = scheduler

        self.asr_queue: asyncio.Queue[Turn] = asyncio.Queue()
        self.translation_queue: asyncio.Queue[Turn] = asyncio.Queue()
        self.tts_queue: asyncio.Queue[tuple[Turn, str, bool]] = asyncio.Queue()
        self.playout_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()

        self.speakers: dict[str, SpeakerStream] = {}
        self.audio_enabled: dict[str, bool] = {}
        self.translation_context: collections.deque[tuple[str, str]] = collections.deque(
            maxlen=TRANSLATION_CONTEXT_TURNS or 1
        )
        self._interim_sizes: dict[str, int] = {}
        self._playout_backlog = 0.0
        self._playing_turn: str | None = None
        self._background: set[asyncio.Task] = set()

    def _spawn(self, coro) -> asyncio.Task:
        """Fire-and-forget send that must not hold up a GPU slot."""
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    # -- stage 0: VAD callback ---------------------------------------------

    def on_utterance(self, speaker: SpeakerStream, audio: np.ndarray | None, captured_at: float) -> None:
        turn = Turn(speaker.identity, speaker.utterance_id, audio, captured_at)
        speaker.next_utterance()
        self._interim_sizes.pop(speaker.identity, None)
        self.asr_queue.put_nowait(turn)

    # -- stage 1: ASR -------------------------------------------------------

    async def run_asr(self) -> None:
        while True:
            turn = await self.asr_queue.get()
            try:
                if turn.audio is None:
                    await self._send(turn.identity, "final", turn.utterance_id, "", TRANSCRIPT_TOPIC)
                    continue
                started = time.monotonic()
                try:
                    text = await self.gpu.run(PRIO_TURN, self.asr.transcribe, turn.audio)
                except Exception:
                    logger.exception("Transcription failed")
                    continue
                turn.asr_ms = (time.monotonic() - started) * 1000
                turn.source_text = (text or "").strip()
                await self._send(turn.identity, "final", turn.utterance_id, turn.source_text, TRANSCRIPT_TOPIC)
                if turn.source_text:
                    turn.wants_audio = self.audio_enabled.get(turn.identity, False)
                    self.translation_queue.put_nowait(turn)
            finally:
                self.asr_queue.task_done()

    # -- stage 1b: best-effort live captions --------------------------------

    async def run_interim(self) -> None:
        """Live Arabic captions, only ever while the critical path is idle.

        Re-transcribing a growing buffer is quadratic work, so this stage polls
        at a low rate and yields the GPU to anything on the ASR -> translation
        -> TTS path.
        """
        if not INTERIM_CAPTIONS:
            return
        while True:
            speaker = self._next_interim()
            if speaker is None or self._critical_busy():
                await asyncio.sleep(0.15)
                continue
            audio = speaker.snapshot()
            if audio is None or audio.size == 0:
                await asyncio.sleep(0.15)
                continue
            utterance_id = speaker.utterance_id
            self._interim_sizes[speaker.identity] = speaker.pending_chunks
            try:
                text = await self.gpu.run(PRIO_INTERIM, self.asr.transcribe, audio)
            except Exception:
                logger.exception("Interim transcription failed")
                await asyncio.sleep(0.5)
                continue
            # Skip if the utterance was finalised meanwhile; its final follows.
            if text and utterance_id == speaker.utterance_id:
                await self._send(speaker.identity, "interim", utterance_id, text, TRANSCRIPT_TOPIC)

    def _critical_busy(self) -> bool:
        return (
            self.gpu.busy
            or self.asr_queue.qsize() > 0
            or self.translation_queue.qsize() > 0
            or self.tts_queue.qsize() > 0
        )

    def _next_interim(self) -> SpeakerStream | None:
        candidates = [
            s for s in self.speakers.values()
            if s.speaking
            and s.pending_chunks - self._interim_sizes.get(s.identity, 0) >= INTERIM_STEP_CHUNKS
            and s.pending_speech_chunks >= MIN_SPEECH_CHUNKS
        ]
        return max(
            candidates,
            key=lambda s: s.pending_chunks - self._interim_sizes.get(s.identity, 0),
            default=None,
        )

    # -- stage 2: translation ----------------------------------------------

    async def run_translation(self) -> None:
        while True:
            turn = await self.translation_queue.get()
            try:
                await self._translate(turn)
            except Exception:
                logger.exception("Translation for %s failed", turn.identity)
            finally:
                self.translation_queue.task_done()

    async def _translate(self, turn: Turn) -> None:
        loop = asyncio.get_running_loop()
        pieces: asyncio.Queue[str | None] = asyncio.Queue()
        context = list(self.translation_context) if TRANSLATION_CONTEXT_TURNS else []
        started = time.monotonic()
        failure: list[BaseException] = []

        def emit(piece: str) -> None:
            loop.call_soon_threadsafe(pieces.put_nowait, piece)

        def generate() -> None:
            try:
                self.translator.stream(turn.source_text, context, emit)
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                failure.append(exc)
            finally:
                loop.call_soon_threadsafe(pieces.put_nowait, None)

        async with self.gpu.reserve(PRIO_TURN):
            worker = asyncio.create_task(asyncio.to_thread(generate))
            buffer = ""
            spoken = ""
            chunk_index = 0
            last_published = 0.0
            partial: asyncio.Task | None = None
            try:
                while True:
                    piece = await pieces.get()
                    if piece is None:
                        break
                    if not turn.first_token_ms:
                        turn.first_token_ms = (time.monotonic() - started) * 1000
                    turn.translation += piece
                    buffer += piece
                    # Live translation text: the browser replaces the line by
                    # id. Sent fire-and-forget so the network never holds the
                    # GPU slot this stage is sitting on.
                    now = time.monotonic()
                    if now - last_published > 0.15:
                        last_published = now
                        partial = self._spawn(self._send(
                            turn.identity, "translation", turn.utterance_id,
                            turn.translation.strip(), TRANSLATION_TOPIC,
                        ))
                    if turn.wants_audio and STREAM_TTS:
                        while True:
                            chunk, buffer = split_speakable(buffer, chunk_index, flush=False)
                            if not chunk:
                                break
                            chunk_index += 1
                            spoken += chunk
                            self.tts_queue.put_nowait((turn, chunk, False))
            finally:
                await worker
        if failure:
            raise failure[0]

        turn.translation = turn.translation.strip()
        turn.translation_ms = (time.monotonic() - started) * 1000
        if partial is not None:
            # The last partial must be on the wire before the final text, or it
            # could overwrite the complete translation in the browser.
            await asyncio.gather(partial, return_exceptions=True)
        await self._send(
            turn.identity, "translation", turn.utterance_id, turn.translation, TRANSLATION_TOPIC
        )
        if turn.translation and TRANSLATION_CONTEXT_TURNS:
            self.translation_context.append((turn.source_text, turn.translation))

        if not turn.wants_audio or not turn.translation:
            self._log_turn(turn)
            return
        rest = buffer.strip() if STREAM_TTS else turn.translation
        if rest:
            self.tts_queue.put_nowait((turn, rest, True))
        elif STREAM_TTS and spoken:
            self.tts_queue.put_nowait((turn, "", True))
        else:
            self._log_turn(turn)

    # -- stage 3: TTS -------------------------------------------------------

    async def run_tts(self) -> None:
        while True:
            turn, text, last = await self.tts_queue.get()
            samples = np.zeros(0, dtype=np.float32)
            try:
                if text:
                    started = time.monotonic()
                    samples = await self.gpu.run(PRIO_SPEECH, self.speech.synthesize, text)
                    turn.tts_ms += (time.monotonic() - started) * 1000
                    turn.chunks += 1
                    turn.audio_seconds += samples.size / AUDIO_SAMPLE_RATE
            except Exception:
                logger.exception("Speech synthesis for %s failed", turn.identity)
                samples = np.zeros(0, dtype=np.float32)
            finally:
                self.tts_queue.task_done()
            # Exactly one chunk per turn carries `last`, and only the playout
            # stage reports the turn, so the timing line is logged once.
            if samples.size or last:
                self._enqueue_playout(AudioChunk(turn, samples, last))

    # -- stage 4: playout ---------------------------------------------------

    def _enqueue_playout(self, chunk: AudioChunk) -> None:
        self._playout_backlog += chunk.samples.size / AUDIO_SAMPLE_RATE
        self.playout_queue.put_nowait(chunk)
        if self._playout_backlog > MAX_AUDIO_BACKLOG_SECONDS:
            self._drop_backlog()

    def _drop_backlog(self) -> None:
        """Drops whole queued turns so the agent can catch up.

        Only reachable when synthesis is persistently slower than the incoming
        speech; without it the spoken output drifts behind forever. Dropping
        happens per turn, never per chunk: losing one complete sentence is
        recoverable, a sentence cut in half mid-word is not. The turn currently
        being spoken and the newest turn are always kept.
        """
        target = MAX_AUDIO_BACKLOG_SECONDS / 2
        pending: list[AudioChunk] = []
        while not self.playout_queue.empty():
            pending.append(self.playout_queue.get_nowait())
            self.playout_queue.task_done()

        groups: list[tuple[Turn, list[AudioChunk]]] = []
        for chunk in pending:
            if groups and groups[-1][0] is chunk.turn:
                groups[-1][1].append(chunk)
            else:
                groups.append((chunk.turn, [chunk]))

        dropped_seconds = 0.0
        dropped_turns: list[str] = []
        kept: list[AudioChunk] = []
        for index, (turn, chunks) in enumerate(groups):
            droppable = (
                index < len(groups) - 1
                and turn.utterance_id != self._playing_turn
                and self._playout_backlog - dropped_seconds > target
            )
            if droppable:
                dropped_seconds += sum(c.samples.size for c in chunks) / AUDIO_SAMPLE_RATE
                dropped_turns.append(turn.utterance_id)
                continue
            kept.extend(chunks)
        for chunk in kept:
            self.playout_queue.put_nowait(chunk)

        if dropped_turns:
            logger.warning(
                "Speech output was %.1f s behind; skipped %d whole turn(s) (%.1f s): %s. "
                "Lower F5_NFE_STEPS or shorten MAX_UTTERANCE_SECONDS if this repeats.",
                self._playout_backlog, len(dropped_turns), dropped_seconds, ", ".join(dropped_turns),
            )
            self._playout_backlog -= dropped_seconds
        elif self._playout_backlog > MAX_AUDIO_BACKLOG_SECONDS:
            logger.warning(
                "Speech output is %.1f s behind but nothing can be skipped without cutting a "
                "sentence in half; the lag will persist until the speaker pauses.",
                self._playout_backlog,
            )

    async def run_playout(self) -> None:
        """Pushes synthesised audio to LiveKit in real time.

        capture_frame() blocks at the speed of playback, which is exactly why
        this must be its own task and must not hold a GPU slot.
        """
        while True:
            chunk = await self.playout_queue.get()
            try:
                self._playing_turn = chunk.turn.utterance_id
                if chunk.samples.size:
                    if chunk.turn.first_audio_at is None:
                        chunk.turn.first_audio_at = time.monotonic()
                    await self._stream_samples(chunk.samples)
                    self._playout_backlog = max(
                        0.0, self._playout_backlog - chunk.samples.size / AUDIO_SAMPLE_RATE
                    )
                if chunk.last:
                    if chunk.tail_gap > 0:
                        await self._stream_samples(
                            np.zeros(int(chunk.tail_gap * AUDIO_SAMPLE_RATE), dtype=np.float32)
                        )
                    self._playing_turn = None
                    self._log_turn(chunk.turn)
            except Exception:
                logger.exception("Audio playout failed")
            finally:
                self.playout_queue.task_done()

    async def _stream_samples(self, waveform: np.ndarray) -> None:
        samples = (np.clip(np.nan_to_num(waveform), -1.0, 1.0) * 32767).astype(np.int16)
        for offset in range(0, samples.size, AUDIO_FRAME_SAMPLES):
            frame_samples = np.ascontiguousarray(samples[offset:offset + AUDIO_FRAME_SAMPLES])
            if frame_samples.size == 0:
                continue
            frame = rtc.AudioFrame(
                data=frame_samples.tobytes(),
                sample_rate=AUDIO_SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=frame_samples.size,
            )
            await self.audio_source.capture_frame(frame)

    # -- reporting ----------------------------------------------------------

    def _log_turn(self, turn: Turn) -> None:
        speech = (turn.audio.size / SAMPLE_RATE) if turn.audio is not None else 0.0
        parts = [
            f"turn {turn.utterance_id}",
            f"speech {speech:.1f}s",
            f"asr {turn.asr_ms:.0f}ms",
            f"mt {turn.translation_ms:.0f}ms (first token {turn.first_token_ms:.0f}ms)",
        ]
        if turn.chunks:
            rtf = (turn.tts_ms / 1000) / max(turn.audio_seconds, 1e-6)
            parts.append(
                f"tts {turn.tts_ms:.0f}ms in {turn.chunks} chunk(s) "
                f"-> {turn.audio_seconds:.1f}s audio (rtf {rtf:.2f})"
            )
        if turn.first_audio_at is not None:
            parts.append(f"first audio {turn.first_audio_at - turn.captured_at:.2f}s after speech end")
        else:
            parts.append(f"text ready {time.monotonic() - turn.captured_at:.2f}s after speech end")
        parts.append(f"backlog {self._playout_backlog:.1f}s")
        logger.info(" | ".join(parts))

    # -- LiveKit plumbing ---------------------------------------------------

    async def _send(self, identity: str, kind: str, utterance_id: str, text: str, topic: str) -> None:
        try:
            await self.room.local_participant.publish_data(
                json.dumps({"type": kind, "id": utterance_id, "text": text}, ensure_ascii=False),
                reliable=True,
                destination_identities=[identity],
                topic=topic,
            )
        except Exception as exc:  # noqa: BLE001 - a dropped caption must not kill the stage
            logger.warning("Sending to %s failed: %s", identity, exc)

    def on_data(self, packet: rtc.DataPacket) -> None:
        if packet.topic != SETTINGS_TOPIC or packet.participant is None:
            return
        try:
            message = json.loads(packet.data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if message.get("type") != "settings":
            return
        enabled = message.get("audio_output")
        if isinstance(enabled, bool):
            self.audio_enabled[packet.participant.identity] = enabled

    async def listen(self, track: rtc.Track, identity: str) -> None:
        loop = asyncio.get_running_loop()
        speaker = SpeakerStream(identity, loop, self.on_utterance)
        self.speakers[track.sid] = speaker
        self._interim_sizes.pop(identity, None)
        stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1)
        logger.info("Transcribing %s", identity)
        try:
            async for event in stream:
                # The event loop only converts and forwards; the VAD itself runs
                # on the speaker's own thread.
                pcm = np.frombuffer(event.frame.data, dtype=np.int16).astype(np.float32) / 32768.0
                speaker.push(pcm)
        finally:
            self.speakers.pop(track.sid, None)
            self._interim_sizes.pop(identity, None)
            # Non-blocking: the VAD thread flushes its tail utterance and exits
            # on its own, so this stays safe inside a cancelled task.
            speaker.close()
            with contextlib.suppress(Exception):
                await stream.aclose()
            logger.info("%s finished", identity)

    def workers(self) -> list:
        return [
            asyncio.create_task(self.run_asr(), name="asr"),
            asyncio.create_task(self.run_interim(), name="interim"),
            asyncio.create_task(self.run_translation(), name="translate"),
            asyncio.create_task(self.run_tts(), name="tts"),
            asyncio.create_task(self.run_playout(), name="playout"),
        ]


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def prepare_models() -> tuple[AsrEngine, Translator, SpeechSynthesizer]:
    dtype, device = pick_device_dtype()
    rng = np.random.default_rng(0)
    warmup_audio = (rng.standard_normal(SAMPLE_RATE) * 0.01).astype(np.float32)

    logger.info("Loading Cohere ASR %s%s", COHERE_MODEL_NAME, " (4-bit)" if COHERE_4BIT else f" ({dtype})")
    asr = AsrEngine(dtype, device)
    asr.transcribe(warmup_audio)
    _vram_report("ASR")

    logger.info("Loading translator %s (%s)", TRANSLATION_MODEL, "4-bit" if TRANSLATION_4BIT else str(dtype))
    translator = Translator(dtype, device)
    try:
        translator.translate("warm-up")
    except Exception:
        logger.exception("Translator warm-up failed (non-fatal)")
    logger.info("Translator ready in %s", "4-bit" if translator.quantized else f"{dtype} (full precision)")
    _vram_report("translator")

    load_silero_vad()  # Cache the VAD weights before the first speaker thread starts.

    speech = SpeechSynthesizer()
    if TTS_BACKEND == "xtts" or (F5_REF_WAV and F5_REF_TEXT):
        logger.info("Preloading %s so the first spoken turn is fast ...", TTS_BACKEND)
        try:
            speech.load()
            speech.synthesize("Guten Morgen.")
        except Exception:
            logger.exception("TTS warm-up failed (non-fatal)")
        _vram_report("TTS")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return asr, translator, speech


def ws_url(url: str) -> str:
    if url.startswith("https://"):
        return "wss://" + url[len("https://"):]
    if url.startswith("http://"):
        return "ws://" + url[len("http://"):]
    return url


def bot_token(key: str, secret: str) -> str:
    return (
        api.AccessToken(key, secret)
        .with_identity(BOT_IDENTITY)
        .with_name("Arabic ASR Bot")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=ROOM_NAME,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .to_jwt()
    )


async def check_credentials(url: str, key: str, secret: str) -> None:
    lkapi = api.LiveKitAPI(url=url, api_key=key, api_secret=secret)
    try:
        await lkapi.room.list_rooms(api.ListRoomsRequest())
    finally:
        await lkapi.aclose()


async def run_room(asr: AsrEngine, translator: Translator, speech: SpeechSynthesizer,
                   url: str, key: str, secret: str) -> None:
    room = rtc.Room()
    audio_source = rtc.AudioSource(
        sample_rate=AUDIO_SAMPLE_RATE, num_channels=1, queue_size_ms=AUDIO_QUEUE_MS
    )
    scheduler = GpuScheduler(GPU_CONCURRENCY)
    pipeline = Pipeline(asr, translator, speech, room, audio_source, scheduler)
    stage_workers = pipeline.workers()
    listeners: dict[str, asyncio.Task] = {}
    disconnected = asyncio.Event()

    def start_listening(track: rtc.Track, participant: rtc.RemoteParticipant) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO or track.sid in listeners:
            return
        task = asyncio.create_task(pipeline.listen(track, participant.identity))
        listeners[track.sid] = task
        task.add_done_callback(lambda _: listeners.pop(track.sid, None))

    def stop_listening(track: rtc.Track) -> None:
        task = listeners.get(track.sid)
        if task:
            task.cancel()

    room.on("track_subscribed", lambda track, _pub, participant: start_listening(track, participant))
    room.on("track_unsubscribed", lambda track, _pub, _participant: stop_listening(track))
    room.on("data_received", pipeline.on_data)
    room.on("disconnected", lambda *_: disconnected.set())

    try:
        await room.connect(ws_url(url), bot_token(key, secret))
        audio_track = rtc.LocalAudioTrack.create_audio_track("german-translation", audio_source)
        await room.local_participant.publish_track(audio_track)
        logger.info(
            "READY: Bot is in room '%s' (streaming pipeline, %d GPU slot(s)). "
            "Now open the web app and start the microphone.",
            ROOM_NAME,
            GPU_CONCURRENCY,
        )
        for participant in room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.track is not None:
                    start_listening(publication.track, participant)
        await disconnected.wait()
    finally:
        tasks = [*listeners.values(), *stage_workers, *pipeline._background]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await room.disconnect()
        await audio_source.aclose()


async def main() -> None:
    # asyncio.to_thread shares one executor; Colab's 2 vCPUs would otherwise
    # cap it at six threads for ASR, translation, TTS and the VAD handoff.
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=8, thread_name_prefix="model")
    )
    url = os.environ.get("LIVEKIT_URL", "").strip()
    key = os.environ.get("LIVEKIT_API_KEY", "").strip()
    secret = os.environ.get("LIVEKIT_API_SECRET", "").strip()
    missing = [n for n, v in (("LIVEKIT_URL", url), ("LIVEKIT_API_KEY", key), ("LIVEKIT_API_SECRET", secret)) if not v]
    if missing:
        raise SystemExit(f"Missing in .env: {', '.join(missing)}")

    logger.info("Checking LiveKit credentials ...")
    try:
        await check_credentials(url, key, secret)
    except Exception as exc:
        raise SystemExit(f"LiveKit rejected the credentials or is not reachable: {exc}")
    logger.info("LiveKit credentials are valid.")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info("GPU: %s (%.1f GB)", props.name, props.total_memory / 2**30)
    logger.info("ASR model: %s", COHERE_MODEL_NAME)
    logger.info("Translation model: %s (%s)", TRANSLATION_MODEL, "4-bit" if TRANSLATION_4BIT else "fp16/bf16")
    if TTS_BACKEND == "f5":
        if not F5_REF_WAV or not F5_REF_TEXT:
            logger.warning(
                "TTS backend is f5 (F5-TTS) but F5_REF_WAV/F5_REF_TEXT are not set - "
                "spoken output will fail until the reference clip is configured "
                "(or set TTS_BACKEND=xtts)."
            )
        else:
            logger.info("TTS backend: f5 (reference voice: %s, ODE steps: %d)", F5_REF_WAV, F5_NFE_STEPS)
    else:
        logger.info("TTS backend: xtts")
    logger.info(
        "Pipeline: %d GPU slot(s), streaming TTS %s, live captions %s",
        GPU_CONCURRENCY,
        "on" if STREAM_TTS else "off",
        "on" if INTERIM_CAPTIONS else "off",
    )
    logger.info("Max utterance length: %.2f s", MAX_UTTERANCE_SECONDS)
    logger.info("End-of-speech silence: %.2f s", END_SILENCE_SECONDS)
    logger.info("Max generated tokens per pass: %d", MAX_GEN_TOKENS)
    if not HF_TOKEN:
        logger.warning("Cohere ASR is enabled but HF_TOKEN is missing - the repo is gated, so loading will fail.")

    asr, translator, speech = await asyncio.to_thread(prepare_models)
    while True:
        try:
            await run_room(asr, translator, speech, url, key, secret)
            logger.warning("Connection ended, retrying in 5 seconds")
        except Exception:
            logger.exception("Connection error, retrying in 5 seconds")
        await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
