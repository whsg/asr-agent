"""Arabic live transcription bot with selectable ASR backends (Qwen3-ASR, Cohere Transcribe Arabic).

Joins a LiveKit room, transcribes the audio of every other participant with each
enabled backend (env ASR_BACKENDS: "qwen", "cohere" or "qwen,cohere") and sends
the text back to the speaker as data messages:

  topic "<backend>-transcript": {"type": "interim" | "final", "id": str, "text": str}

Finished utterances of the backend the speaker picked in the web app (default
Qwen) are additionally translated to German by the translation model chosen via
env TRANSLATION_BACKEND ("translategemma" or "qwen3"); only one is loaded:

  topic "asr-translation": {"type": "translation", "id": str, "text": str}

On request the translation is also spoken: TTS_BACKEND picks the engine
("f5" = F5-TTS, default, or "xtts" = XTTS-v2).

Runs on a GPU (e.g. Colab).
"""
from __future__ import annotations

import asyncio
import collections
import itertools
import json
import logging
import os
import secrets
import sys
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from livekit import api, rtc
from silero_vad import load_silero_vad

load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("qwen-bot")
logging.getLogger("livekit").setLevel(logging.WARNING)

ROOM_NAME = os.getenv("LIVEKIT_ROOM", "qwen-asr-live")
BOT_IDENTITY = "qwen-asr-bot"
SETTINGS_TOPIC = "qwen-settings"
TRANSLATION_TOPIC = "asr-translation"
TOPIC_BY_BACKEND = {"qwen": "qwen-transcript", "cohere": "cohere-transcript"}


def _parse_backends() -> tuple[str, ...]:
    raw = os.getenv("ASR_BACKENDS", "qwen")
    names = [part.strip().lower() for part in raw.replace(";", ",").split(",") if part.strip()]
    valid = [name for name in dict.fromkeys(names) if name in TOPIC_BY_BACKEND]
    return tuple(valid) or ("qwen",)


ASR_BACKENDS = _parse_backends()
MODEL_NAME = os.getenv("QWEN_MODEL", "Qwen/Qwen3-ASR-1.7B")
COHERE_MODEL_NAME = os.getenv("COHERE_MODEL", "CohereLabs/cohere-transcribe-arabic-07-2026")
COHERE_4BIT = os.getenv("COHERE_4BIT", "").strip().lower() in ("1", "true", "yes", "on")
COHERE_LANGUAGE = os.getenv("COHERE_LANGUAGE", "ar").strip() or "ar"
HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None
LANGUAGE = os.getenv("QWEN_LANGUAGE", "Arabic").strip() or None
TRANSLATION_CONTEXT_SIZE = 3
# Translation backend: exactly ONE of these is loaded at startup (never both).
#   "translategemma" -> google/translategemma-12b-it (default)
#   "qwen3"          -> Qwen/Qwen3-8B
TRANSLATION_BACKEND = os.getenv("TRANSLATION_BACKEND", "translategemma").strip().lower()
if TRANSLATION_BACKEND not in ("translategemma", "qwen3"):
    logger.warning("Invalid TRANSLATION_BACKEND=%r; using 'translategemma' instead", TRANSLATION_BACKEND)
    TRANSLATION_BACKEND = "translategemma"
QWEN_TRANSLATION_MODEL = os.getenv("QWEN_TRANSLATION_MODEL", "Qwen/Qwen3-8B")
GEMMA_TRANSLATION_MODEL = os.getenv("GEMMA_TRANSLATION_MODEL", "google/translategemma-12b-it")
TRANSLATION_SOURCE_LANG = os.getenv("TRANSLATION_SOURCE_LANG", "ar").strip() or "ar"
TRANSLATION_TARGET_LANG = os.getenv("TRANSLATION_TARGET_LANG", "de").strip() or "de"
AUDIO_SAMPLE_RATE = 24_000
AUDIO_FRAME_SAMPLES = AUDIO_SAMPLE_RATE // 50


def _env_seconds(name: str, default: float) -> float:
    """Reads a positive number of seconds from the env, falling back to the default."""
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


def _env_int(name: str, default: int) -> int:
    """Reads a positive integer from the env, falling back to the default."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s instead", name, raw, default)
        return default
    if value <= 0:
        logger.warning("%s must be > 0; using %s instead", name, default)
        return default
    return value


# Optional spoken output for the translation; TTS_BACKEND picks the engine:
#   "f5"   (default) F5-TTS - a German target language speaks the German
#           fine-tune aihpi/F5-TTS-German, other target languages speak the
#           standard F5-TTS base model (e.g. English).
#   "xtts" XTTS-v2 (previous engine).
TTS_BACKEND = os.getenv("TTS_BACKEND", "f5").strip().lower()
if TTS_BACKEND not in ("f5", "xtts"):
    logger.warning("Invalid TTS_BACKEND=%r; using 'f5' instead", TTS_BACKEND)
    TTS_BACKEND = "f5"

# F5-TTS settings.
F5_GERMAN_REPO = os.getenv("F5_GERMAN_REPO", "aihpi/F5-TTS-German").strip() or "aihpi/F5-TTS-German"
F5_GERMAN_CKPT = os.getenv("F5_GERMAN_CKPT", "F5TTS_Base/model_420000.safetensors").strip() or "F5TTS_Base/model_420000.safetensors"
F5_BASE_MODEL = os.getenv("F5_BASE_MODEL", "F5TTS_v1_Base").strip() or "F5TTS_v1_Base"
# F5-TTS is a voice-cloning model: it needs a short reference clip (~5-10 s)
# of the desired voice and the spoken text of it (for German, a German clip).
# F5_REF_WAV: file in the Colab runtime (e.g. /content/reference.wav) or http(s) URL.
F5_REF_WAV = os.getenv("F5_REF_WAV", "").strip()
F5_REF_TEXT = os.getenv("F5_REF_TEXT", "").strip()
# Load EMA weights (recommended for the fully fine-tuned checkpoints).
F5_USE_EMA = os.getenv("F5_USE_EMA", "1").strip().lower() in ("1", "true", "yes", "on")
# ODE inference steps (lower is faster, 32 is the F5-TTS default).
F5_NFE_STEPS = _env_int("F5_NFE_STEPS", 32)

# XTTS-v2 (only used when TTS_BACKEND=xtts).
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"


SAMPLE_RATE = 16_000
VAD_CHUNK = 512  # Silero VAD requires exactly 32 ms windows at 16 kHz
CHUNK_SECONDS = VAD_CHUNK / SAMPLE_RATE
SPEECH_START_PROB = 0.5
SPEECH_KEEP_PROB = 0.35
END_SILENCE_CHUNKS = round(0.55 / CHUNK_SECONDS)
PRE_ROLL_CHUNKS = round(0.25 / CHUNK_SECONDS)
TAIL_CHUNKS = round(0.15 / CHUNK_SECONDS)
MIN_SPEECH_CHUNKS = round(0.25 / CHUNK_SECONDS)
# Env MAX_UTTERANCE_SECONDS: longest utterance (seconds) sent to the ASR.
MAX_UTTERANCE_SECONDS = _env_seconds("MAX_UTTERANCE_SECONDS", 12.0)
MAX_UTTERANCE_CHUNKS = round(MAX_UTTERANCE_SECONDS / CHUNK_SECONDS)
SPLIT_SEARCH_CHUNKS = round(2.0 / CHUNK_SECONDS)
INTERIM_STEP_CHUNKS = round(0.4 / CHUNK_SECONDS)

_utterance_counter = itertools.count()
_boot_id = secrets.token_hex(3)


def new_utterance_id() -> str:
    # Unique across bot restarts so the browser never overwrites old lines.
    return f"{_boot_id}-{next(_utterance_counter)}"


class Segmenter:
    """Splits a continuous audio stream into utterances using Silero VAD."""

    def __init__(self) -> None:
        self.vad = load_silero_vad()
        self.pending = np.empty(0, dtype=np.float32)
        self.pre_roll: collections.deque[np.ndarray] = collections.deque(maxlen=PRE_ROLL_CHUNKS)
        self.chunks: list[np.ndarray] = []
        self.probs: list[float] = []
        self.speaking = False
        self.silent_chunks = 0

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def speech_chunks(self) -> int:
        return sum(p >= SPEECH_KEEP_PROB for p in self.probs)

    def snapshot(self) -> np.ndarray:
        return np.concatenate(self.chunks)

    @torch.inference_mode()
    def feed(self, pcm: np.ndarray) -> list[np.ndarray | None]:
        """Adds audio and returns finished utterances (None = too short to be speech)."""
        self.pending = np.concatenate((self.pending, pcm))
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
                self.pre_roll.clear()
                self.silent_chunks = 0

            self.chunks.append(chunk)
            self.probs.append(prob)
            self.silent_chunks = 0 if prob >= SPEECH_KEEP_PROB else self.silent_chunks + 1

            if self.silent_chunks >= END_SILENCE_CHUNKS:
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
        if sum(p >= SPEECH_KEEP_PROB for p in head_probs) < MIN_SPEECH_CHUNKS:
            return None
        return np.concatenate(head_chunks)


def pick_dtype():
    if not torch.cuda.is_available():
        logger.warning("No GPU found; the ASR models will run on the CPU and be slow.")
        return torch.float32, "cpu"
    major, _ = torch.cuda.get_device_capability(0)
    return (torch.bfloat16 if major >= 8 else torch.float16), "cuda:0"


def load_model(dtype, device_map):
    from qwen_asr import Qwen3ASRModel

    return Qwen3ASRModel.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        device_map=device_map,
        max_inference_batch_size=1,
        max_new_tokens=256,
    )


def load_cohere(dtype):
    """Loads Cohere Transcribe Arabic (gated repo, needs transformers>=5.4.0)."""
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration

    load_kwargs: dict = {}
    if HF_TOKEN:
        load_kwargs["token"] = HF_TOKEN
    if COHERE_4BIT:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        load_kwargs["torch_dtype"] = dtype
    try:
        processor = AutoProcessor.from_pretrained(COHERE_MODEL_NAME, **load_kwargs)
        model = CohereAsrForConditionalGeneration.from_pretrained(
            COHERE_MODEL_NAME,
            device_map="auto",
            **load_kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - wrapped with actionable guidance below
        raise RuntimeError(
            "Could not load the Cohere model: "
            f"{exc}. The repo is gated: accept the terms on huggingface.co and "
            "set HF_TOKEN in the .env, or remove 'cohere' from ASR_BACKENDS."
        ) from exc
    model.eval()
    return processor, model


def cohere_transcribe(processor, model, audio: np.ndarray) -> str:
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", language=COHERE_LANGUAGE)
    inputs = inputs.to(model.device, dtype=model.dtype)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=256)
    return processor.decode(outputs, skip_special_tokens=True).strip()


class AsrBackend:
    """One ASR engine; GPU usage is serialized through the shared lock."""

    name: str

    def __init__(self, gpu_lock: asyncio.Lock) -> None:
        self.gpu_lock = gpu_lock

    @property
    def topic(self) -> str:
        return TOPIC_BY_BACKEND[self.name]

    def transcribe_sync(self, audio: np.ndarray) -> str:
        raise NotImplementedError

    async def transcribe(self, audio: np.ndarray) -> str | None:
        try:
            async with self.gpu_lock:
                return await asyncio.to_thread(self.transcribe_sync, audio)
        except Exception:
            logger.exception("Transcription with %s failed", self.name)
            return None


class QwenBackend(AsrBackend):
    name = "qwen"

    def __init__(self, gpu_lock: asyncio.Lock, model) -> None:
        super().__init__(gpu_lock)
        self.model = model

    def transcribe_sync(self, audio: np.ndarray) -> str:
        results = self.model.transcribe(audio=(audio, SAMPLE_RATE), language=LANGUAGE)
        return results[0].text.strip()


class CohereBackend(AsrBackend):
    name = "cohere"

    def __init__(self, gpu_lock: asyncio.Lock, processor, model) -> None:
        super().__init__(gpu_lock)
        self.processor = processor
        self.model = model

    def transcribe_sync(self, audio: np.ndarray) -> str:
        return cohere_transcribe(self.processor, self.model, audio)


def _bits_and_bytes_4bit(dtype):
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )


def load_qwen_translator(dtype, device_map):
    """Loads Qwen3-8B (4-bit); supports conversation context for reference resolution."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device_map == "cpu":
        raise RuntimeError("Qwen3-8B requires a GPU for live translation.")

    load_kwargs = {}
    if HF_TOKEN:
        load_kwargs["token"] = HF_TOKEN
    tokenizer = AutoTokenizer.from_pretrained(QWEN_TRANSLATION_MODEL, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_TRANSLATION_MODEL,
        torch_dtype=dtype,
        quantization_config=_bits_and_bytes_4bit(dtype),
        device_map="auto",
        **load_kwargs,
    )
    model.eval()
    return model, tokenizer


def load_translategemma(dtype, device_map):
    """Loads TranslateGemma 12B (4-bit) for direct text translation (gated repo)."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    load_kwargs = {}
    if HF_TOKEN:
        load_kwargs["token"] = HF_TOKEN
    processor = AutoProcessor.from_pretrained(GEMMA_TRANSLATION_MODEL, **load_kwargs)
    model = AutoModelForImageTextToText.from_pretrained(
        GEMMA_TRANSLATION_MODEL,
        quantization_config=_bits_and_bytes_4bit(dtype),
        device_map="auto",
        **load_kwargs,
    )
    model.eval()
    return model, processor


def _translate_text_qwen(model, tokenizer, text: str, context: Sequence[tuple[str, str]]) -> str:
    context_text = "\n".join(f"Arabic: {source}\nGerman: {target}" for source, target in context)
    context_block = f"\nConversation context so far:\n{context_text}\n" if context_text else ""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional Arabic-to-German translator. "
                "Translate the current Arabic segment completely and faithfully "
                "into natural, spoken German. Use the context only to resolve "
                "references. Do not invent anything, do not summarize, and output "
                "only the German translation."
            ),
        },
        {
            "role": "user",
            "content": f"{context_block}\nCurrent Arabic segment:\n{text}\n\nGerman translation:",
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.05,
        )
    generated = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _translate_text_translategemma(model, processor, text: str, context: Sequence[tuple[str, str]]) -> str:
    # TranslateGemma uses a fixed chat template: one user turn carrying the
    # source/target language codes and the text only. No conversation context.
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": TRANSLATION_SOURCE_LANG,
                    "target_lang_code": TRANSLATION_TARGET_LANG,
                    "text": text,
                }
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    generated = output[0, input_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def translate_text(model, tokenizer, text: str, context: Sequence[tuple[str, str]]) -> str:
    """Dispatches to the loaded translation backend (see TRANSLATION_BACKEND)."""
    if TRANSLATION_BACKEND == "translategemma":
        return _translate_text_translategemma(model, tokenizer, text, context)
    return _translate_text_qwen(model, tokenizer, text, context)


class SpeechSynthesizer:
    """Spoken output for the translation; the engine is picked by TTS_BACKEND.

    "f5" (default): F5-TTS - a German target language via the German
    fine-tune aihpi/F5-TTS-German, other target languages via the standard
    F5-TTS base model. "xtts": XTTS-v2.
    The model loads lazily on the first spoken translation request.
    """

    def __init__(self) -> None:
        self.engine = XttsSynthesizer() if TTS_BACKEND == "xtts" else F5Synthesizer()

    def synthesize(self, text: str) -> np.ndarray:
        return self.engine.synthesize(text)


class XttsSynthesizer:
    """XTTS-v2 voice (previous engine; only used when TTS_BACKEND=xtts)."""

    def __init__(self) -> None:
        self.model = None
        self.speaker = os.getenv("XTTS_SPEAKER", "").strip() or None
        self.speaker_wav = os.getenv("XTTS_SPEAKER_WAV", "").strip() or None

    def synthesize(self, text: str) -> np.ndarray:
        if self.speaker_wav and not Path(self.speaker_wav).is_file():
            raise RuntimeError(f"XTTS speaker file not found: {self.speaker_wav}")
        if self.model is None:
            from TTS.api import TTS

            self.model = TTS(XTTS_MODEL_NAME, gpu=torch.cuda.is_available(), progress_bar=False)
            if not self.speaker and not self.speaker_wav:
                speakers = self.model.speakers or []
                if not speakers:
                    raise RuntimeError("XTTS has no default voice; configure XTTS_SPEAKER_WAV.")
                self.speaker = speakers[0]

        options = {"text": text, "language": "de"}
        if self.speaker_wav:
            options["speaker_wav"] = self.speaker_wav
        elif self.speaker:
            options["speaker"] = self.speaker
        return np.asarray(self.model.tts(**options), dtype=np.float32).reshape(-1)


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
    """F5-TTS voice (default engine).

    The voice is cloned from a reference clip (F5_REF_WAV + F5_REF_TEXT). A
    German target language uses the German fine-tune aihpi/F5-TTS-German, any
    other target language the standard F5-TTS base model.
    """

    def __init__(self) -> None:
        self.model = None
        self.ref_wav = ""

    def _load(self):
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
        self.ref_wav = _resolve_f5_ref_wav()

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
        return self.model

    def synthesize(self, text: str) -> np.ndarray:
        model = self._load()
        wav, _sample_rate, _spectrogram = model.infer(
            ref_file=self.ref_wav,
            ref_text=F5_REF_TEXT,
            gen_text=text,
            show_info=lambda *args: None,
            progress=None,
            nfe_step=F5_NFE_STEPS,
        )
        if wav is None:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(wav, dtype=np.float32).reshape(-1)


def prepare_models():
    dtype, device_map = pick_dtype()
    models: dict[str, object] = {}
    rng = np.random.default_rng(0)
    warmup = (rng.standard_normal(SAMPLE_RATE) * 0.01).astype(np.float32)
    if "qwen" in ASR_BACKENDS:
        logger.info("Loading ASR %s (%s on %s)", MODEL_NAME, dtype, device_map)
        asr_model = load_model(dtype, device_map)
        asr_model.transcribe(audio=(warmup, SAMPLE_RATE), language=LANGUAGE)  # Warm-up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        models["qwen"] = asr_model
    if "cohere" in ASR_BACKENDS:
        logger.info("Loading Cohere ASR %s%s", COHERE_MODEL_NAME, " (4-bit)" if COHERE_4BIT else "")
        models["cohere"] = load_cohere(dtype)
        cohere_transcribe(*models["cohere"])  # Warm-up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if TRANSLATION_BACKEND == "translategemma":
        logger.info("Loading translator %s in 4-bit", GEMMA_TRANSLATION_MODEL)
        translator, tokenizer = load_translategemma(dtype, device_map)
    else:
        logger.info("Loading translator %s in 4-bit", QWEN_TRANSLATION_MODEL)
        translator, tokenizer = load_qwen_translator(dtype, device_map)
    load_silero_vad()
    return models, translator, tokenizer


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


class Speaker:
    def __init__(self, identity: str) -> None:
        self.identity = identity
        self.segmenter = Segmenter()
        self.utterance_id = new_utterance_id()

    def next_utterance(self) -> None:
        self.utterance_id = new_utterance_id()


class _PipelineState:
    """Per-backend job queue and interim progress; VAD segmentation is shared."""

    __slots__ = ("finals", "interim_sizes", "wake")

    def __init__(self) -> None:
        self.finals: collections.deque[tuple[str, str, np.ndarray | None]] = collections.deque()
        self.interim_sizes: dict[str, int] = {}
        self.wake = asyncio.Event()


class Transcriber:
    """Shared VAD plus one worker per ASR backend: finished utterances first, live updates in between."""

    def __init__(self, backends: dict[str, AsrBackend], translator, tokenizer, room: rtc.Room, audio_source: rtc.AudioSource) -> None:
        self.backends = backends
        self.pipelines: dict[str, _PipelineState] = {name: _PipelineState() for name in backends}
        self.translator = translator
        self.tokenizer = tokenizer
        self.room = room
        self.audio_source = audio_source
        self.speech = SpeechSynthesizer()
        self.speakers: dict[str, Speaker] = {}
        self.translation_jobs: asyncio.Queue[tuple[str, str, str]] = asyncio.Queue()
        self.translation_context: collections.defaultdict[str, collections.deque[tuple[str, str]]] = (
            collections.defaultdict(lambda: collections.deque(maxlen=TRANSLATION_CONTEXT_SIZE))
        )
        self.audio_enabled: dict[str, bool] = {}
        self.translation_source: dict[str, str] = {}
        self.default_translation_source = "qwen" if "qwen" in backends else next(iter(backends))
        self.gpu_lock = next(iter(backends.values())).gpu_lock

    async def run_pipeline(self, name: str) -> None:
        state = self.pipelines[name]
        topic = self.backends[name].topic
        while True:
            state.wake.clear()
            if state.finals:
                identity, utterance_id, audio = state.finals.popleft()
                text = await self.backends[name].transcribe(audio) if audio is not None else ""
                await self._send(identity, "final", utterance_id, text or "", topic)
                if text and self._effective_translation_source(identity) == name:
                    await self.translation_jobs.put((identity, utterance_id, text))
                continue

            speaker = self._next_interim(name)
            if speaker is None:
                await state.wake.wait()
                continue
            utterance_id = speaker.utterance_id
            state.interim_sizes[speaker.identity] = speaker.segmenter.size
            text = await self.backends[name].transcribe(speaker.segmenter.snapshot())
            # Skip if the utterance was finalized meanwhile; its final result follows.
            if text and utterance_id == speaker.utterance_id:
                await self._send(speaker.identity, "interim", utterance_id, text, topic)

    async def run_translations(self) -> None:
        while True:
            identity, utterance_id, source_text = await self.translation_jobs.get()
            try:
                context = self.translation_context[identity]
                async with self.gpu_lock:
                    translated = await asyncio.to_thread(
                        translate_text, self.translator, self.tokenizer, source_text, list(context)
                    )
                    if not translated:
                        continue
                    context.append((source_text, translated))
                    await self._send(identity, "translation", utterance_id, translated, TRANSLATION_TOPIC)
                    wants_audio = self.audio_enabled.get(identity, False)
                    waveform = (
                        await asyncio.to_thread(self.speech.synthesize, translated)
                        if wants_audio
                        else None
                    )
                if waveform is not None:
                    await self._publish_audio(waveform)
            except Exception:
                logger.exception("Translation or speech output for %s failed", identity)
            finally:
                self.translation_jobs.task_done()

    async def _publish_audio(self, waveform: np.ndarray) -> None:
        samples = (np.clip(np.nan_to_num(waveform), -1.0, 1.0) * 32767).astype(np.int16)
        for offset in range(0, len(samples), AUDIO_FRAME_SAMPLES):
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

    def on_data(self, packet: rtc.DataPacket) -> None:
        if packet.topic != SETTINGS_TOPIC or packet.participant is None:
            return
        try:
            message = json.loads(packet.data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if message.get("type") != "settings":
            return
        identity = packet.participant.identity
        enabled = message.get("audio_output")
        if isinstance(enabled, bool):
            self.audio_enabled[identity] = enabled
        source = message.get("translation_asr")
        if isinstance(source, str) and source in self.backends:
            self.translation_source[identity] = source
        # Tell the client which backends are actually loaded in this bot.
        capabilities = {
            "type": "capabilities",
            "backends": list(self.backends),
            "default_translation_asr": self.default_translation_source,
        }
        asyncio.create_task(self._publish(identity, capabilities, SETTINGS_TOPIC))

    def _next_interim(self, name: str) -> Speaker | None:
        state = self.pipelines[name]
        candidates = [
            s for s in self.speakers.values()
            if s.segmenter.speaking
            and s.segmenter.size - state.interim_sizes.get(s.identity, 0) >= INTERIM_STEP_CHUNKS
            and s.segmenter.speech_chunks >= MIN_SPEECH_CHUNKS
        ]
        return max(
            candidates,
            key=lambda s: s.segmenter.size - state.interim_sizes.get(s.identity, 0),
            default=None,
        )

    def _effective_translation_source(self, identity: str) -> str:
        return self.translation_source.get(identity) or self.default_translation_source

    async def _publish(self, identity: str, payload: dict, topic: str) -> None:
        try:
            await self.room.local_participant.publish_data(
                json.dumps(payload, ensure_ascii=False),
                reliable=True,
                destination_identities=[identity],
                topic=topic,
            )
        except Exception as exc:
            logger.warning("Sending to %s failed: %s", identity, exc)

    async def _send(self, identity: str, kind: str, utterance_id: str, text: str, topic: str) -> None:
        await self._publish(identity, {"type": kind, "id": utterance_id, "text": text}, topic)

    def _queue_final(self, speaker: Speaker, audio: np.ndarray | None) -> None:
        utterance_id = speaker.utterance_id
        for state in self.pipelines.values():
            state.finals.append((speaker.identity, utterance_id, audio))
            state.wake.set()
        speaker.next_utterance()

    async def listen(self, track: rtc.Track, identity: str) -> None:
        speaker = Speaker(identity)
        self.speakers[track.sid] = speaker
        for state in self.pipelines.values():
            state.interim_sizes.pop(identity, None)
        stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1)
        logger.info("Transcribing %s", identity)
        try:
            async for event in stream:
                pcm = np.frombuffer(event.frame.data, dtype=np.int16).astype(np.float32) / 32768.0
                finished = speaker.segmenter.feed(pcm)
                for audio in finished:
                    self._queue_final(speaker, audio)
                if finished or speaker.segmenter.speaking:
                    for state in self.pipelines.values():
                        state.wake.set()
        finally:
            self.speakers.pop(track.sid, None)
            if speaker.segmenter.speaking:
                self._queue_final(speaker, speaker.segmenter.flush())
            await stream.aclose()
            logger.info("%s finished", identity)


async def run_room(models, translator, tokenizer, url: str, key: str, secret: str) -> None:
    room = rtc.Room()
    audio_source = rtc.AudioSource(sample_rate=AUDIO_SAMPLE_RATE, num_channels=1, queue_size_ms=3000)
    gpu_lock = asyncio.Lock()
    backends: dict[str, AsrBackend] = {}
    if "qwen" in models:
        backends["qwen"] = QwenBackend(gpu_lock, models["qwen"])
    if "cohere" in models:
        processor, cohere_model = models["cohere"]
        backends["cohere"] = CohereBackend(gpu_lock, processor, cohere_model)
    transcriber = Transcriber(backends, translator, tokenizer, room, audio_source)
    pipeline_workers = [asyncio.create_task(transcriber.run_pipeline(name)) for name in backends]
    translation_worker = asyncio.create_task(transcriber.run_translations())
    listeners: dict[str, asyncio.Task] = {}
    disconnected = asyncio.Event()

    def start_listening(track: rtc.Track, participant: rtc.RemoteParticipant) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO or track.sid in listeners:
            return
        task = asyncio.create_task(transcriber.listen(track, participant.identity))
        listeners[track.sid] = task
        task.add_done_callback(lambda _: listeners.pop(track.sid, None))

    def stop_listening(track: rtc.Track) -> None:
        task = listeners.get(track.sid)
        if task:
            task.cancel()

    room.on("track_subscribed", lambda track, _pub, participant: start_listening(track, participant))
    room.on("track_unsubscribed", lambda track, _pub, _participant: stop_listening(track))
    room.on("data_received", transcriber.on_data)
    room.on("disconnected", lambda *_: disconnected.set())

    try:
        await room.connect(ws_url(url), bot_token(key, secret))
        audio_track = rtc.LocalAudioTrack.create_audio_track("german-translation", audio_source)
        await room.local_participant.publish_track(audio_track)
        logger.info(
            "READY: Bot is in room '%s' (ASR: %s). Now open the web app and start the microphone.",
            ROOM_NAME,
            ", ".join(backends),
        )
        for participant in room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.track is not None:
                    start_listening(publication.track, participant)
        await disconnected.wait()
    finally:
        tasks = [*listeners.values(), *pipeline_workers, translation_worker]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await room.disconnect()
        await audio_source.aclose()


async def main() -> None:
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
    logger.info("ASR backends: %s", ", ".join(ASR_BACKENDS))
    logger.info("Translation backend: %s", TRANSLATION_BACKEND)
    if TTS_BACKEND == "f5":
        if not F5_REF_WAV or not F5_REF_TEXT:
            logger.warning(
                "TTS backend is f5 (F5-TTS) but F5_REF_WAV/F5_REF_TEXT are not set - "
                "spoken output will fail until the reference clip is configured "
                "(or set TTS_BACKEND=xtts)."
            )
        else:
            logger.info("TTS backend: f5 (reference voice: %s)", F5_REF_WAV)
    else:
        logger.info("TTS backend: xtts")
    logger.info("Max utterance length: %.2f s", MAX_UTTERANCE_SECONDS)
    if "cohere" in ASR_BACKENDS and not HF_TOKEN:
        logger.warning("Cohere ASR is enabled but HF_TOKEN is missing - the repo is gated, so loading will fail.")
    if TRANSLATION_BACKEND == "translategemma" and not HF_TOKEN:
        logger.warning("TranslateGemma is enabled but HF_TOKEN is missing - the repo is gated, so loading will fail.")

    models, translator, tokenizer = await asyncio.to_thread(prepare_models)
    while True:
        try:
            await run_room(models, translator, tokenizer, url, key, secret)
            logger.warning("Connection ended, reconnecting ...")
        except Exception:
            logger.exception("Connection error, retrying in 5 seconds")
        await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
