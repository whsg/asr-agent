"""Arabic live transcription bot using Cohere Transcribe Arabic.

Joins a LiveKit room, transcribes the audio of every other participant and sends
the text back to the speaker as data messages:

  topic "<backend>-transcript": {"type": "interim" | "final", "id": str, "text": str}

Finished utterances are translated to German with Qwen3:

  topic "asr-translation": {"type": "translation", "id": str, "text": str}

On request the translation is also spoken: TTS_BACKEND picks the engine
("phonon" = Gradium Phonon with a fixed voice from gradium/phonon-voices,
default, "xtts" = XTTS-v2).

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
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from livekit import api, rtc
from silero_vad import load_silero_vad

load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("asr-bot")
logging.getLogger("livekit").setLevel(logging.WARNING)

ROOM_NAME = os.getenv("LIVEKIT_ROOM", "arabic-asr-live")
BOT_IDENTITY = "arabic-asr-bot"
SETTINGS_TOPIC = "asr-settings"
TRANSLATION_TOPIC = "asr-translation"
TOPIC_BY_BACKEND = {"cohere": "cohere-transcript"}
COHERE_MODEL_NAME = os.getenv("COHERE_MODEL", "CohereLabs/cohere-transcribe-arabic-07-2026")
COHERE_4BIT = os.getenv("COHERE_4BIT", "").strip().lower() in ("1", "true", "yes", "on")
COHERE_LANGUAGE = os.getenv("COHERE_LANGUAGE", "ar").strip() or "ar"
HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
TRANSLATION_SOURCE_LANG = os.getenv("TRANSLATION_SOURCE_LANG", "ar").strip() or "ar"
TRANSLATION_TARGET_LANG = os.getenv("TRANSLATION_TARGET_LANG", "de").strip() or "de"
# System prompt for the translator. The default is tuned for a Friday sermon
# (Jumu'ah khutba); set to an empty value for the plain "translate, no
# explanation" prompt.
TRANSLATION_SYSTEM_PROMPT = (
    "You are a professional simultaneous interpreter rendering a Friday sermon "
    "(Jumu'ah khutba) from {src} to {tgt}, streaming. You see the earlier "
    "turns, then a PENDING part (source text you held back before because its "
    "meaning was incomplete) and the current NEW INPUT. Add the new input to "
    "the pending text and judge how much of it now forms coherent, complete "
    "sentences. Never translate word-for-word: produce one compact, fluent "
    "rendering; merge the fragments into clean prose, drop filler words, "
    "greetings and repetitions, and silently fix obvious speech-recognition "
    "errors. Keep the solemn sermon register, keep proper names, translate "
    "Qur'anic phrases, and stay consistent with the earlier turns. Reply in "
    "exactly this format, with no other text:\n"
    "<translate>\nThe {tgt} text to release now, complete sentences only, or "
    "leave empty if nothing is coherent yet.\n</translate>\n<pending>\n"
    "{src} source text still awaiting its continuation, or leave empty.\n"
    "</pending>\n"
    "The pending part must only contain text from what you were given; when "
    "in doubt, release it in <translate>. "
).format(src=TRANSLATION_SOURCE_LANG, tgt=TRANSLATION_TARGET_LANG)
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
#   "phonon"  (default) Gradium Phonon model for German (gradium/phonon-40m-de);
#             the speaker is a fixed voice from gradium/phonon-voices
#             (PHONON_VOICE, default "henrik"), not a cloned reference.
#   "xtts"    XTTS-v2 (previous engine).
TTS_BACKEND = os.getenv("TTS_BACKEND", "phonon").strip().lower()
if TTS_BACKEND not in ("xtts", "phonon"):
    logger.warning("Invalid TTS_BACKEND=%r; using 'phonon' instead", TTS_BACKEND)
    TTS_BACKEND = "phonon"

# Max tokens the ASR / translator may generate per pass. A 20 s utterance
# transcribes to well under a few hundred tokens, so the cap guards against
# runaway generation while keeping normal turns unaffected.
MAX_GEN_TOKENS = _env_int("MAX_GEN_TOKENS", 512)
# How many previous (source, translation) pairs to send as chat context so the
# translator stays consistent across turns (0 disables the context).
TRANSLATION_CONTEXT_TURNS = _env_int("TRANSLATION_CONTEXT_TURNS", 5)

# XTTS-v2 (only used when TTS_BACKEND=xtts).
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# Gradium Phonon model (TTS_BACKEND=phonon, the default).
PHONON_MODEL_NAME = os.getenv("PHONON_MODEL_NAME", "gradium/phonon-40m-de").strip() or "gradium/phonon-40m-de"
# Voice from gradium/phonon-voices by name (case-insensitive), raw 16-char voice
# ID, or "default" for the checkpoint's built-in voice.
PHONON_VOICE = os.getenv("PHONON_VOICE", "henrik").strip() or "henrik"
PHONON_VOICE_IDS = {
    "henrik": "yyS1KYWs6mXoEw7D",
}
# Temperature for phonon inference (lower is more deterministic).
PHONON_TEMPERATURE = float(os.getenv("PHONON_TEMPERATURE", "1.0").strip() or "1.0")

SAMPLE_RATE = 16_000
VAD_CHUNK = 512  # Silero VAD requires exactly 32 ms windows at 16 kHz
CHUNK_SECONDS = VAD_CHUNK / SAMPLE_RATE
SPEECH_START_PROB = 0.5
SPEECH_KEEP_PROB = 0.35
# Trailing silence that ends an utterance (env END_SILENCE_SECONDS, default 1.0 s).
# Lower responds faster; 0.4 s chops up natural sermon pauses into short segments.
END_SILENCE_SECONDS = _env_seconds("END_SILENCE_SECONDS", 1.0)
END_SILENCE_CHUNKS = round(END_SILENCE_SECONDS / CHUNK_SECONDS)
PRE_ROLL_CHUNKS = round(0.25 / CHUNK_SECONDS)
TAIL_CHUNKS = round(0.15 / CHUNK_SECONDS)
MIN_SPEECH_CHUNKS = round(0.25 / CHUNK_SECONDS)
# Env MAX_UTTERANCE_SECONDS: longest utterance (seconds) sent to the ASR.
MAX_UTTERANCE_SECONDS = _env_seconds("MAX_UTTERANCE_SECONDS", 20.0)
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
            "set HF_TOKEN in the .env."
        ) from exc
    model.eval()
    return processor, model


def cohere_transcribe(processor, model, audio: np.ndarray) -> str:
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", language=COHERE_LANGUAGE)
    inputs = inputs.to(model.device, dtype=model.dtype)
    with torch.inference_mode():
        start = time.perf_counter()
        outputs = model.generate(**inputs, max_new_tokens=MAX_GEN_TOKENS)
    ms = (time.perf_counter() - start) * 1000
    decoded = processor.decode(outputs, skip_special_tokens=True)
    if isinstance(decoded, list):
        decoded = decoded[0] if decoded else ""
    # Encoder-decoder ASR models have no input_ids (audio features only) and
    # generate() returns decoder tokens only; decoder-only models prepend the
    # prompt, so subtract its length when present.
    input_ids = inputs.get("input_ids")
    prompt_len = input_ids.shape[1] if input_ids is not None else 0
    logger.info("ASR %.0f ms (%d out tokens)", ms, outputs.shape[1] - prompt_len)
    return decoded.strip()


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


def load_qwen3(dtype, device_map):
    """Loads the 4-bit Qwen3 translator set by TRANSLATION_MODEL.

    Qwen3 is a decoder-only chat model with a built-in chat template, so no
    external Jinja template is needed. The 4B default is a good speed/VRAM fit
    for a T4; set TRANSLATION_MODEL to an 8B variant for better quality.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs = {}
    if HF_TOKEN:
        load_kwargs["token"] = HF_TOKEN
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        TRANSLATION_MODEL,
        quantization_config=_bits_and_bytes_4bit(dtype),
        device_map="auto",
        **load_kwargs,
    )
    model.eval()
    return model, tokenizer


def _build_translator_messages(text: str, context: list[tuple[str, str]], pending: str = "") -> list[dict]:
    if TRANSLATION_SYSTEM_PROMPT:
        messages = [{"role": "system", "content": TRANSLATION_SYSTEM_PROMPT}]
        messages += [
            {"role": "user", "content": previous}
            for previous, _ in context
        ] + [
            {"role": "assistant", "content": target}
            for _, target in context
        ]
    else:
        messages = []
    if pending:
        content = f"PENDING (held over, still missing its continuation): {pending}\nNEW INPUT: {text}"
    else:
        content = text
    messages.append(
        {
            "role": "user",
            "content": (
                f"Translate the following text from {TRANSLATION_SOURCE_LANG} to "
                f"{TRANSLATION_TARGET_LANG}. Return only the translation, with no "
                f"explanation.\n\n{content}"
            ),
        }
    )
    return messages


def _parse_translation_reply(raw: str, original: str) -> tuple[str, str]:
    """Splits the model's <translate>/<pending> reply; any malformation falls back
    to the whole (tag-stripped) reply as the translation so no text is lost."""
    import re

    def _between(open_tag: str, close_tag: str) -> str | None:
        start = raw.find(open_tag)
        end = raw.find(close_tag)
        if start != -1 and end != -1 and end > start:
            return raw[start + len(open_tag) : end].strip()
        return None

    released = _between("<translate>", "</translate>")
    pending = _between("<pending>", "</pending>")
    if released is None:
        # No well-formed <translate> part: treat the whole reply as the
        # translation and drop the pending bookkeeping.
        released = re.sub(r"</?(?:translate|pending)>", "", raw).strip()
        released = re.sub(r"^\s*translate\s*:\s*", "", released, flags=re.IGNORECASE)
        pending = ""
    if not released:
        # Nothing coherent was released: fall back to the new input itself so
        # no source text is ever lost to an empty reply.
        released, pending = original, ""
    return released, pending


def _translate_text_qwen3(
    model, tokenizer, text: str, context: list[tuple[str, str]] = (), pending: str = ""
) -> tuple[str, str]:
    messages = _build_translator_messages(text, context, pending)
    # enable_thinking=False: some Qwen3 instruct models (e.g. Qwen3-8B) think
    # with a reasoning block by default, which is slow and can eat the whole
    # output budget before reaching the translation. We only ever want the
    # direct translation, so force thinking off. Harmless for templates without
    # a thinking branch (e.g. Qwen3-4B-Instruct-2507), which ignore the kwarg.
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        start = time.perf_counter()
        output = model.generate(**inputs, max_new_tokens=MAX_GEN_TOKENS, do_sample=False)
    ms = (time.perf_counter() - start) * 1000
    generated = output[0, input_len:]
    logger.info("Translate %.0f ms (%d out tokens)", ms, generated.shape[0])
    return _parse_translation_reply(
        tokenizer.decode(generated, skip_special_tokens=True).strip(), text
    )


def translate_text(
    model, tokenizer, text: str, context: list[tuple[str, str]] = (), pending: str = ""
) -> tuple[str, str]:
    return _translate_text_qwen3(model, tokenizer, text, context, pending)


class SpeechSynthesizer:
    """Spoken output for the translation; the engine is picked by TTS_BACKEND.

    "phonon" (default): Gradium Phonon model for German with a fixed voice
    from gradium/phonon-voices. "xtts": XTTS-v2. The model loads lazily on
    the first spoken translation request.
    """

    def __init__(self) -> None:
        if TTS_BACKEND == "xtts":
            self.engine = XttsSynthesizer()
        else:
            self.engine = PhononSynthesizer()

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


# Shared Phonon model cache: built once (e.g. by the startup preload) and reused
# by every PhononSynthesizer instance so the heavy load happens only once.
_PHONON_CACHE = None  # (model, processor, voice_latents)


def _phonon_voice_id(name: str) -> str:
    """A German voice name as its gradium/phonon-voices ID (case-insensitive).

    Unknown names are assumed to already be a raw 16-char voice ID and passed
    through unchanged.
    """
    lowered = name.lower()
    for voice_name, voice_id in PHONON_VOICE_IDS.items():
        if voice_name.lower() == lowered:
            return voice_id
    return name


def _phonon_decode_audio(decoded) -> np.ndarray:
    """Flattens PocketTTSProcessor.decode() output to float32 mono samples."""
    audio = decoded["audio"] if isinstance(decoded, dict) else decoded
    if isinstance(audio, (list, tuple)):
        audio = torch.cat(audio) if audio else torch.zeros(0)
    if torch.is_tensor(audio):
        audio = audio.float().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


class PhononSynthesizer:
    """Gradium Phonon German TTS (default engine, TTS_BACKEND=phonon).

    Not a voice-cloning model: the voice is a precomputed 512-d speaker
    embedding from the gradium/phonon-voices catalog (default: Henrik, a
    German male voice; PHONON_VOICE picks another German voice name, a raw
    voice ID, or "default" for the checkpoint's own voice). Needs a
    transformers build with GradiumPhononTTSForConditionalGeneration (git
    main) and a valid HF_TOKEN (the model repo is gated). Emits 24 kHz mono
    speech.
    """

    def __init__(self) -> None:
        self._state = None  # (model, processor, voice_latents)

    def _load(self):
        global _PHONON_CACHE
        if self._state is not None:
            return self._state
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoProcessor, GradiumPhononTTSForConditionalGeneration

        if _PHONON_CACHE is not None:
            self._state = _PHONON_CACHE
            return self._state

        use_cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if use_cuda else torch.float32
        logger.info("Loading Phonon model %s ...", PHONON_MODEL_NAME)
        load_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
        try:
            processor = AutoProcessor.from_pretrained(PHONON_MODEL_NAME, **load_kwargs)
            model = GradiumPhononTTSForConditionalGeneration.from_pretrained(
                PHONON_MODEL_NAME,
                torch_dtype=dtype,
                device_map="cuda:0" if use_cuda else "cpu",
                **load_kwargs,
            )
            model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Could not load Phonon model {PHONON_MODEL_NAME}: {exc}. "
                "It needs a transformers build with "
                "GradiumPhononTTSForConditionalGeneration (pip install -U "
                "'git+https://github.com/huggingface/transformers') and a valid "
                "HF_TOKEN (the repo requires accepting its license on Hugging Face)."
            ) from exc

        if PHONON_VOICE == "default":
            logger.info("Phonon voice: built-in default of %s", PHONON_MODEL_NAME)
            voice_path = hf_hub_download(PHONON_MODEL_NAME, "default-voice.safetensors", **load_kwargs)
        else:
            voice_id = _phonon_voice_id(PHONON_VOICE)
            logger.info("Phonon voice: %s (%s)", PHONON_VOICE, voice_id)
            voice_path = hf_hub_download(
                "gradium/phonon-voices", f"voices/{voice_id}.safetensors", **load_kwargs
            )
        # Speaker embedding latents of shape (num_frames, 512).
        voice = load_file(voice_path)["speaker_wavs"][0].T.float()

        self._state = _PHONON_CACHE = (model, processor, voice)
        return self._state

    def synthesize(self, text: str) -> np.ndarray:
        model, processor, voice = self._load()

        start = time.perf_counter()
        try:
            parts = []
            for chunk in processor.split_text(text):
                # An empty "audio" content item marks where the voice goes.
                conversation = [
                    {
                        "role": "user",
                        "content": [{"type": "audio"}, {"type": "text", "text": chunk}],
                    }
                ]
                inputs = processor.apply_chat_template(
                    conversation,
                    tokenize=True,
                    return_dict=True,
                    add_generation_prompt=True,
                    processor_kwargs={"audio_prompt_latents": [voice]},
                ).to(model.device)
                with torch.inference_mode():
                    outputs = model.generate(**inputs, temperature=PHONON_TEMPERATURE)
                samples = _phonon_decode_audio(
                    processor.decode(outputs.audio_latents, outputs.audio_latents_mask)
                )
                if samples.size:
                    parts.append(samples)
            audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
            ms = (time.perf_counter() - start) * 1000
            logger.info("TTS %.0f ms (%.1f s audio)", ms, audio.size / AUDIO_SAMPLE_RATE)
            return audio
        except Exception:
            logger.exception("Phonon synthesis failed")
            ms = (time.perf_counter() - start) * 1000
            logger.info("TTS %.0f ms (failed)", ms)
            return np.zeros(0, dtype=np.float32)


def prepare_models():
    dtype, device_map = pick_dtype()
    rng = np.random.default_rng(0)
    warmup = (rng.standard_normal(SAMPLE_RATE) * 0.01).astype(np.float32)
    logger.info("Loading Cohere ASR %s%s", COHERE_MODEL_NAME, " (4-bit)" if COHERE_4BIT else "")
    models = {"cohere": load_cohere(dtype)}
    cohere_transcribe(*models["cohere"], warmup)  # Warm-up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Loading Qwen3 %s in 4-bit", TRANSLATION_MODEL)
    translator, tokenizer = load_qwen3(dtype, device_map)
    try:
        translate_text(translator, tokenizer, "warm-up")  # Warm Qwen3's first inference (reply discarded)
    except Exception:
        logger.exception("Qwen3 warm-up failed (non-fatal)")
    load_silero_vad()
    if TTS_BACKEND == "phonon":
        logger.info("Preloading Phonon model so the first spoken turn is fast ...")
        speech = PhononSynthesizer()
        try:
            speech._load()
            speech.synthesize("warm-up")  # Warm Phonon's first inference once
        except Exception:
            logger.exception("Phonon warm-up failed (non-fatal)")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # `processor` here is the Qwen3 tokenizer (run_room tokenizes with it).
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
        # Source text held back by the translator, awaiting its continuation.
        self.pending_source = ""

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

    def __init__(self, backends: dict[str, AsrBackend], translator, processor, room: rtc.Room, audio_source: rtc.AudioSource) -> None:
        self.backends = backends
        self.pipelines: dict[str, _PipelineState] = {name: _PipelineState() for name in backends}
        self.translator = translator
        self.processor = processor
        self.room = room
        self.audio_source = audio_source
        self.speech = SpeechSynthesizer()
        self.speakers: dict[str, Speaker] = {}
        # (identity, utterance_id, text, enqueued_at) jobs for translate + TTS.
        self.translation_jobs: asyncio.Queue[tuple[str, str, str, float]] = asyncio.Queue()
        # >0 while a TTS job is running; live captions then skip so they cannot
        # delay the spoken reply on the shared GPU.
        self._tts_in_flight = 0
        # Recent (source, translation) pairs, sent to the translator as chat
        # context so consecutive fragments form a coherent sermon rendering.
        self.translation_context: collections.deque[tuple[str, str]] = collections.deque(
            maxlen=TRANSLATION_CONTEXT_TURNS
        )
        self.audio_enabled: dict[str, bool] = {}
        self.gpu_lock = next(iter(backends.values())).gpu_lock

    async def run_pipeline(self, name: str) -> None:
        state = self.pipelines[name]
        topic = self.backends[name].topic
        while True:
            state.wake.clear()
            if state.finals:
                identity, utterance_id, audio = state.finals.popleft()
                t0 = time.monotonic()
                text = await self.backends[name].transcribe(audio) if audio is not None else ""
                logger.info(
                    "Final ASR for %s after %.1f s (lock wait + inference)",
                    identity,
                    time.monotonic() - t0,
                )
                await self._send(identity, "final", utterance_id, text or "", topic)
                if text:
                    await self.translation_jobs.put((identity, utterance_id, text, time.monotonic()))
                continue

            speaker = self._next_interim(name)
            if speaker is None:
                await state.wake.wait()
                continue
            if self.translation_jobs.qsize() or self._tts_in_flight:
                # A reply is already queued or being spoken: skip the live
                # caption so it cannot push the answer further down the queue.
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
            identity, utterance_id, source_text, enqueued_at = await self.translation_jobs.get()
            started = time.monotonic()
            waveform = None
            try:
                async with self.gpu_lock:
                    speaker = self._speaker(identity)
                    pending = speaker.pending_source
                    translated, pending = await asyncio.to_thread(
                        translate_text,
                        self.translator,
                        self.processor,
                        source_text,
                        list(self.translation_context),
                        pending,
                    )
                    speaker.pending_source = pending
                    if pending:
                        logger.info("Holding %d chars of %s's source text back", len(pending), identity)
                    if not translated:
                        continue
                    self.translation_context.append((source_text, translated))
                    await self._send(identity, "translation", utterance_id, translated, TRANSLATION_TOPIC)
                    wants_audio = self.audio_enabled.get(identity, False)
                    if wants_audio:
                        self._tts_in_flight += 1
                        try:
                            waveform = await asyncio.to_thread(self.speech.synthesize, translated)
                        finally:
                            self._tts_in_flight -= 1
                if waveform is not None:
                    await self._publish_audio(waveform)
                logger.info(
                    "Turn for %s ready in %.1f s (%.1f s queued before the GPU)",
                    identity,
                    time.monotonic() - started,
                    started - enqueued_at,
                )
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

    def _speaker(self, identity: str) -> Speaker:
        speaker = self.speakers.get(identity)
        if speaker is None:
            speaker = Speaker(identity)
            self.speakers[identity] = speaker
        return speaker

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


async def run_room(models, translator, processor, url: str, key: str, secret: str) -> None:
    room = rtc.Room()
    audio_source = rtc.AudioSource(sample_rate=AUDIO_SAMPLE_RATE, num_channels=1, queue_size_ms=3000)
    gpu_lock = asyncio.Lock()
    cohere_processor, cohere_model = models["cohere"]
    backends = {"cohere": CohereBackend(gpu_lock, cohere_processor, cohere_model)}
    transcriber = Transcriber(backends, translator, processor, room, audio_source)
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
    logger.info("ASR model: %s", COHERE_MODEL_NAME)
    logger.info("Translation model: %s", TRANSLATION_MODEL)
    logger.info("TTS backend: %s%s", TTS_BACKEND, "" if TTS_BACKEND != "phonon" else f" (voice: {PHONON_VOICE})")
    logger.info("Max utterance length: %.2f s", MAX_UTTERANCE_SECONDS)
    logger.info("End-of-speech silence: %.2f s", END_SILENCE_SECONDS)
    logger.info("Max generated tokens per pass: %d", MAX_GEN_TOKENS)
    if not HF_TOKEN:
        logger.warning("Cohere ASR is enabled but HF_TOKEN is missing - the repo is gated, so loading will fail.")

    models, translator, processor = await asyncio.to_thread(prepare_models)
    while True:
        try:
            await run_room(models, translator, processor, url, key, secret)
            logger.warning("Connection ended, reconnecting ...")
        except Exception:
            logger.exception("Connection error, retrying in 5 seconds")
        await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
