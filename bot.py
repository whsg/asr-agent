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
MODEL_NAME = os.getenv("QWEN_MODEL", "Qwen/Qwen3-ASR-1.7B")
LANGUAGE = os.getenv("QWEN_LANGUAGE", "Arabic").strip() or None

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


def transcribe(model, audio: np.ndarray) -> str:
    results = model.transcribe(audio=(audio, SAMPLE_RATE), language=LANGUAGE)
    return results[0].text.strip()


def prepare_model():
    dtype, device_map = pick_dtype()
    logger.info("Lade %s (%s auf %s) - beim ersten Mal dauert der Download einige Minuten", MODEL_NAME, dtype, device_map)
    model = load_model(dtype, device_map)
    # Warm-up compiles the GPU kernels so the first real utterance is fast; also fails early on a broken setup.
    rng = np.random.default_rng(0)
    for seconds in (1, 4):
        transcribe(model, (rng.standard_normal(seconds * SAMPLE_RATE) * 0.01).astype(np.float32))
    load_silero_vad()
    return model


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
                can_publish=False,
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

    def __init__(self, model, room: rtc.Room) -> None:
        self.model = model
        self.room = room
        self.speakers: dict[str, Speaker] = {}
        self.finals: collections.deque[tuple[str, str, np.ndarray | None]] = collections.deque()
        self.wake = asyncio.Event()

    async def run(self) -> None:
        while True:
            self.wake.clear()
            if self.finals:
                identity, utterance_id, audio = self.finals.popleft()
                text = await self._transcribe(audio) if audio is not None else ""
                await self._send(identity, "final", utterance_id, text or "")
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


async def run_room(model, url: str, key: str, secret: str) -> None:
    room = rtc.Room()
    transcriber = Transcriber(model, room)
    worker = asyncio.create_task(transcriber.run())
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
    room.on("disconnected", lambda *_: disconnected.set())

    await room.connect(ws_url(url), bot_token(key, secret))
    logger.info("BEREIT: Bot ist im Raum '%s'. Jetzt die Web-App oeffnen und das Mikrofon starten.", ROOM_NAME)
    for participant in room.remote_participants.values():
        for publication in participant.track_publications.values():
            if publication.track is not None:
                start_listening(publication.track, participant)
    try:
        await disconnected.wait()
    finally:
        for task in list(listeners.values()):
            task.cancel()
        worker.cancel()
        await room.disconnect()


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

    model = await asyncio.to_thread(prepare_model)
    while True:
        try:
            await run_room(model, url, key, secret)
            logger.warning("Verbindung beendet, verbinde neu ...")
        except Exception:
            logger.exception("Verbindungsfehler, neuer Versuch in 5 Sekunden")
        await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
