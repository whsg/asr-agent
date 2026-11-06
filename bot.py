"""Qwen3-ASR live transcription bot.

Joins a LiveKit room, transcribes the audio of every other participant and sends
the text back to the speaker as data messages on topic "qwen-transcript":
{"type": "interim" | "final", "id": str, "text": str}. Runs on a GPU (e.g. Colab).
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
TOPIC = "qwen-transcript"
SETTINGS_TOPIC = "qwen-settings"
MODEL_NAME = os.getenv("QWEN_MODEL", "Qwen/Qwen3-ASR-1.7B")
TRANSLATION_MODEL_NAME = os.getenv("QWEN_TRANSLATION_MODEL", "Qwen/Qwen3-8B")
LANGUAGE = os.getenv("QWEN_LANGUAGE", "Arabic").strip() or None
TRANSLATION_CONTEXT_SIZE = 3
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
AUDIO_SAMPLE_RATE = 24_000
AUDIO_FRAME_SAMPLES = AUDIO_SAMPLE_RATE // 50

SAMPLE_RATE = 16_000
VAD_CHUNK = 512  # Silero VAD requires exactly 32 ms windows at 16 kHz
CHUNK_SECONDS = VAD_CHUNK / SAMPLE_RATE
SPEECH_START_PROB = 0.5
SPEECH_KEEP_PROB = 0.35
END_SILENCE_CHUNKS = round(0.55 / CHUNK_SECONDS)
PRE_ROLL_CHUNKS = round(0.25 / CHUNK_SECONDS)
TAIL_CHUNKS = round(0.15 / CHUNK_SECONDS)
MIN_SPEECH_CHUNKS = round(0.25 / CHUNK_SECONDS)
MAX_UTTERANCE_CHUNKS = round(12.0 / CHUNK_SECONDS)
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
        logger.warning("Keine GPU gefunden, Qwen3-ASR laeuft auf der CPU und ist langsam.")
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


def load_translator(dtype, device_map):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if device_map == "cpu":
        raise RuntimeError("Qwen3-8B benoetigt fuer die Live-Uebersetzung eine GPU.")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        TRANSLATION_MODEL_NAME,
        torch_dtype=dtype,
        quantization_config=quantization,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def translate_text(model, tokenizer, text: str, context: Sequence[tuple[str, str]]) -> str:
    context_text = "\n".join(f"Arabisch: {source}\nDeutsch: {target}" for source, target in context)
    context_block = f"\nBisheriger Gespraechskontext:\n{context_text}\n" if context_text else ""
    messages = [
        {
            "role": "system",
            "content": (
                "Du bist ein professioneller Arabisch-Deutsch-Uebersetzer. "
                "Uebersetze den aktuellen arabischen Abschnitt vollstaendig und sinngemaess "
                "in natuerliches, gesprochenes Deutsch. Nutze den Kontext nur zur Klaerung "
                "von Bezuengen. Erfinde nichts, fasse nicht zusammen und gib ausschliesslich "
                "die deutsche Uebersetzung aus."
            ),
        },
        {
            "role": "user",
            "content": f"{context_block}\nAktueller arabischer Abschnitt:\n{text}\n\nDeutsche Uebersetzung:",
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


class SpeechSynthesizer:
    """Loads XTTS only when a connected client requests spoken output."""

    def __init__(self) -> None:
        self.model = None
        self.speaker = os.getenv("XTTS_SPEAKER", "").strip() or None
        self.speaker_wav = os.getenv("XTTS_SPEAKER_WAV", "").strip() or None

    def synthesize(self, text: str) -> np.ndarray:
        if self.speaker_wav and not Path(self.speaker_wav).is_file():
            raise RuntimeError(f"XTTS-Sprecherdatei nicht gefunden: {self.speaker_wav}")
        if self.model is None:
            from TTS.api import TTS

            self.model = TTS(XTTS_MODEL_NAME, gpu=torch.cuda.is_available(), progress_bar=False)
            if not self.speaker and not self.speaker_wav:
                speakers = self.model.speakers or []
                if not speakers:
                    raise RuntimeError("XTTS hat keine Standardstimme; XTTS_SPEAKER_WAV konfigurieren.")
                self.speaker = speakers[0]

        options = {"text": text, "language": "de"}
        if self.speaker_wav:
            options["speaker_wav"] = self.speaker_wav
        elif self.speaker:
            options["speaker"] = self.speaker
        return np.asarray(self.model.tts(**options), dtype=np.float32).reshape(-1)


def prepare_models():
    dtype, device_map = pick_dtype()
    logger.info("Lade ASR %s (%s auf %s)", MODEL_NAME, dtype, device_map)
    asr_model = load_model(dtype, device_map)
    rng = np.random.default_rng(0)
    transcribe(asr_model, (rng.standard_normal(SAMPLE_RATE) * 0.01).astype(np.float32))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Lade Uebersetzer %s in 4-Bit", TRANSLATION_MODEL_NAME)
    translator, tokenizer = load_translator(dtype, device_map)
    load_silero_vad()
    return asr_model, translator, tokenizer


def transcribe(model, audio: np.ndarray) -> str:
    results = model.transcribe(audio=(audio, SAMPLE_RATE), language=LANGUAGE)
    return results[0].text.strip()


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
        .with_name("Qwen3 ASR")
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
        self.interim_size = 0

    def next_utterance(self) -> None:
        self.utterance_id = new_utterance_id()
        self.interim_size = 0


class Transcriber:
    """One GPU worker: finished utterances first, live updates whenever the GPU is free."""

    def __init__(self, model, translator, tokenizer, room: rtc.Room, audio_source: rtc.AudioSource) -> None:
        self.model = model
        self.translator = translator
        self.tokenizer = tokenizer
        self.room = room
        self.audio_source = audio_source
        self.speech = SpeechSynthesizer()
        self.speakers: dict[str, Speaker] = {}
        self.finals: collections.deque[tuple[str, str, np.ndarray | None]] = collections.deque()
        self.translation_jobs: asyncio.Queue[tuple[str, str, str]] = asyncio.Queue()
        self.translation_context: collections.defaultdict[str, collections.deque[tuple[str, str]]] = (
            collections.defaultdict(lambda: collections.deque(maxlen=TRANSLATION_CONTEXT_SIZE))
        )
        self.audio_enabled: dict[str, bool] = {}
        self.gpu_lock = asyncio.Lock()
        self.wake = asyncio.Event()

    async def run(self) -> None:
        while True:
            self.wake.clear()
            if self.finals:
                identity, utterance_id, audio = self.finals.popleft()
                text = await self._transcribe(audio) if audio is not None else ""
                await self._send(identity, "final", utterance_id, text or "")
                if text:
                    await self.translation_jobs.put((identity, utterance_id, text))
                continue

            speaker = self._next_interim()
            if speaker is None:
                await self.wake.wait()
                continue
            utterance_id = speaker.utterance_id
            speaker.interim_size = speaker.segmenter.size
            text = await self._transcribe(speaker.segmenter.snapshot())
            # Skip if the utterance was finalized meanwhile; its final result follows.
            if text and utterance_id == speaker.utterance_id:
                await self._send(speaker.identity, "interim", utterance_id, text)

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
                    await self._send(identity, "translation", utterance_id, translated)
                    wants_audio = self.audio_enabled.get(identity, False)
                    waveform = (
                        await asyncio.to_thread(self.speech.synthesize, translated)
                        if wants_audio
                        else None
                    )
                if waveform is not None:
                    await self._publish_audio(waveform)
            except Exception:
                logger.exception("Uebersetzung oder Sprachausgabe fuer %s fehlgeschlagen", identity)
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
        enabled = message.get("audio_output")
        if message.get("type") == "settings" and isinstance(enabled, bool):
            self.audio_enabled[packet.participant.identity] = enabled

    def _next_interim(self) -> Speaker | None:
        candidates = [
            s for s in self.speakers.values()
            if s.segmenter.speaking
            and s.segmenter.size - s.interim_size >= INTERIM_STEP_CHUNKS
            and s.segmenter.speech_chunks >= MIN_SPEECH_CHUNKS
        ]
        return max(candidates, key=lambda s: s.segmenter.size - s.interim_size, default=None)

    async def _transcribe(self, audio: np.ndarray) -> str | None:
        try:
            async with self.gpu_lock:
                return await asyncio.to_thread(transcribe, self.model, audio)
        except Exception:
            logger.exception("Transkription fehlgeschlagen")
            return None

    async def _send(self, identity: str, kind: str, utterance_id: str, text: str) -> None:
        payload = json.dumps({"type": kind, "id": utterance_id, "text": text}, ensure_ascii=False)
        try:
            await self.room.local_participant.publish_data(
                payload, reliable=True, destination_identities=[identity], topic=TOPIC
            )
        except Exception as exc:
            logger.warning("Senden an %s fehlgeschlagen: %s", identity, exc)

    def _queue_final(self, speaker: Speaker, audio: np.ndarray | None) -> None:
        self.finals.append((speaker.identity, speaker.utterance_id, audio))
        speaker.next_utterance()

    async def listen(self, track: rtc.Track, identity: str) -> None:
        speaker = Speaker(identity)
        self.speakers[track.sid] = speaker
        stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1)
        logger.info("Transkribiere %s", identity)
        try:
            async for event in stream:
                pcm = np.frombuffer(event.frame.data, dtype=np.int16).astype(np.float32) / 32768.0
                finished = speaker.segmenter.feed(pcm)
                for audio in finished:
                    self._queue_final(speaker, audio)
                if finished or speaker.segmenter.speaking:
                    self.wake.set()
        finally:
            self.speakers.pop(track.sid, None)
            if speaker.segmenter.speaking:
                self._queue_final(speaker, speaker.segmenter.flush())
                self.wake.set()
            await stream.aclose()
            logger.info("%s beendet", identity)


async def run_room(model, translator, tokenizer, url: str, key: str, secret: str) -> None:
    room = rtc.Room()
    audio_source = rtc.AudioSource(sample_rate=AUDIO_SAMPLE_RATE, num_channels=1, queue_size_ms=3000)
    transcriber = Transcriber(model, translator, tokenizer, room, audio_source)
    worker = asyncio.create_task(transcriber.run())
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
        logger.info("BEREIT: Bot ist im Raum '%s'. Jetzt die Web-App oeffnen und das Mikrofon starten.", ROOM_NAME)
        for participant in room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.track is not None:
                    start_listening(publication.track, participant)
        await disconnected.wait()
    finally:
        tasks = [*listeners.values(), worker, translation_worker]
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
        raise SystemExit(f"In .env fehlt: {', '.join(missing)}")

    logger.info("Pruefe LiveKit-Zugangsdaten ...")
    try:
        await check_credentials(url, key, secret)
    except Exception as exc:
        raise SystemExit(f"LiveKit lehnt die Zugangsdaten ab oder ist nicht erreichbar: {exc}")
    logger.info("LiveKit-Zugangsdaten sind gueltig.")

    model, translator, tokenizer = await asyncio.to_thread(prepare_models)
    while True:
        try:
            await run_room(model, translator, tokenizer, url, key, secret)
            logger.warning("Verbindung beendet, verbinde neu ...")
        except Exception:
            logger.exception("Verbindungsfehler, neuer Versuch in 5 Sekunden")
        await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
