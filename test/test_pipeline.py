"""Pipeline behaviour tests for bot.py, run with stubbed models (see conftest).

The point of these tests is the scheduling, not the model quality: they prove
that the stages really overlap, that real-time playout cannot stall the GPU
stages (the regression that made the spoken output drift further behind with
every turn), and that the agent catches up when it falls behind.
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeAsr:
    def __init__(self, seconds: float = 0.10) -> None:
        self.seconds = seconds
        self.calls: list[tuple[float, float]] = []

    def transcribe(self, audio: np.ndarray) -> str:
        start = time.monotonic()
        time.sleep(self.seconds)
        self.calls.append((start, time.monotonic()))
        return "نص عربي"


class FakeTranslator:
    """Emits fixed pieces with a delay so token streaming can be observed."""

    PIECES = [
        "Lob sei Gott, dem Herrn ",
        "der Welten. ",
        "Wir preisen Ihn und ",
        "bitten Ihn um Beistand. ",
        "Friede sei mit euch.",
    ]

    def __init__(self, gap: float = 0.04) -> None:
        self.gap = gap
        self.contexts: list[list[tuple[str, str]]] = []

    def stream(self, text: str, context, emit) -> None:
        self.contexts.append(list(context))
        for piece in self.PIECES:
            time.sleep(self.gap)
            emit(piece)


class FakeSpeech:
    def __init__(self, seconds: float = 0.08, audio_seconds: float = 0.5) -> None:
        self.seconds = seconds
        self.audio_seconds = audio_seconds
        self.texts: list[str] = []
        self.calls: list[tuple[float, float]] = []

    def synthesize(self, text: str) -> np.ndarray:
        start = time.monotonic()
        time.sleep(self.seconds)
        self.calls.append((start, time.monotonic()))
        self.texts.append(text)
        return np.zeros(int(24_000 * self.audio_seconds), dtype=np.float32)


class FakeParticipant:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    async def publish_data(self, payload, *, reliable, destination_identities, topic):
        self.messages.append((topic, payload))


class FakeRoom:
    def __init__(self) -> None:
        self.local_participant = FakeParticipant()


class RealtimeAudioSource:
    """Mimics LiveKit's AudioSource: capture_frame blocks at playback speed."""

    def __init__(self, frame_seconds: float = 0.02) -> None:
        self.frame_seconds = frame_seconds
        self.frames = 0
        self.busy_until = 0.0

    async def capture_frame(self, _frame) -> None:
        self.frames += 1
        await asyncio.sleep(self.frame_seconds)
        self.busy_until = time.monotonic()


def make_pipeline(bot, *, asr=None, translator=None, speech=None, source=None, concurrency=2):
    room = FakeRoom()
    audio_source = source if source is not None else RealtimeAudioSource()
    pipeline = bot.Pipeline(
        asr or FakeAsr(),
        translator or FakeTranslator(),
        speech or FakeSpeech(),
        room,
        audio_source,
        bot.GpuScheduler(concurrency),
    )
    pipeline.audio_enabled["speaker"] = True
    return pipeline, room, audio_source


def utterance(bot, index: int, seconds: float = 1.0):
    return bot.Turn(
        identity="speaker",
        utterance_id=f"t{index}",
        audio=np.zeros(int(bot.SAMPLE_RATE * seconds), dtype=np.float32),
        captured_at=time.monotonic(),
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_first_chunk_is_released_early(bot):
    text = "Lob sei Gott, dem Herrn der Welten. Wir preisen Ihn und bitten Ihn um Beistand."
    chunk, rest = bot.split_speakable(text, 0, flush=False)
    assert chunk == "Lob sei Gott, dem Herrn der Welten."
    assert rest.strip().startswith("Wir preisen")


def test_short_buffer_keeps_buffering(bot):
    chunk, rest = bot.split_speakable("Lob sei Gott,", 0, flush=False)
    assert chunk == ""
    assert rest == "Lob sei Gott,"


def test_flush_returns_everything(bot):
    chunk, rest = bot.split_speakable("  ein Rest ohne Punkt", 1, flush=True)
    assert chunk == "ein Rest ohne Punkt"
    assert rest == ""


def test_long_run_without_punctuation_breaks_on_a_word(bot):
    text = ("wort " * 80).strip()
    chunk, rest = bot.split_speakable(text, 1, flush=False)
    assert chunk
    assert len(chunk) <= bot.TTS_MAX_CHUNK_CHARS
    assert not chunk.endswith("wor")
    assert rest.strip()


# ---------------------------------------------------------------------------
# GPU scheduler
# ---------------------------------------------------------------------------


def test_scheduler_serves_critical_work_before_interim(bot):
    async def scenario():
        scheduler = bot.GpuScheduler(1)
        order: list[str] = []

        async with scheduler.reserve(bot.PRIO_TURN):
            # Both queue up behind the held slot; priority decides who is next.
            interim = asyncio.create_task(hold(scheduler, bot.PRIO_INTERIM, "interim", order))
            await asyncio.sleep(0.02)
            speech = asyncio.create_task(hold(scheduler, bot.PRIO_SPEECH, "speech", order))
            await asyncio.sleep(0.02)
        await asyncio.gather(interim, speech)
        return order

    async def hold(scheduler, priority, label, order):
        async with scheduler.reserve(priority):
            order.append(label)
            await asyncio.sleep(0)

    assert asyncio.run(scenario()) == ["speech", "interim"]


def test_scheduler_releases_slots_after_cancellation(bot):
    async def scenario():
        scheduler = bot.GpuScheduler(1)
        async with scheduler.reserve(bot.PRIO_TURN):
            waiter = asyncio.create_task(enter(scheduler))
            await asyncio.sleep(0.02)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        # The slot must be free again for the next job.
        async with scheduler.reserve(bot.PRIO_TURN):
            return True

    async def enter(scheduler):
        async with scheduler.reserve(bot.PRIO_TURN):
            await asyncio.sleep(1)

    assert asyncio.run(asyncio.wait_for(scenario(), 2))


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def test_segmenter_cuts_on_trailing_silence(bot):
    segmenter = bot.Segmenter()
    rng = np.random.default_rng(1)
    loud = (rng.standard_normal(bot.SAMPLE_RATE * 3) * 0.5).astype(np.float32)
    quiet = np.zeros(bot.SAMPLE_RATE, dtype=np.float32)

    finished = segmenter.feed(loud)
    assert finished == []
    assert segmenter.speaking

    finished = segmenter.feed(quiet)
    assert len(finished) == 1
    assert finished[0] is not None and finished[0].size > 0
    assert not segmenter.speaking


def test_a_mid_sentence_pause_does_not_end_the_utterance(bot):
    """A preacher pausing for effect must not be split into fragments.

    Fragmenting costs twice: the translator loses sentence context, and every
    extra turn pays the TTS engine's fixed per-call cost, which is what pushed
    synthesis past real time and made whole turns get skipped.
    """
    segmenter = bot.Segmenter()
    rng = np.random.default_rng(3)

    def speech(seconds):
        return (rng.standard_normal(int(bot.SAMPLE_RATE * seconds)) * 0.5).astype(np.float32)

    def silence(seconds):
        return np.zeros(int(bot.SAMPLE_RATE * seconds), dtype=np.float32)

    finished = []
    # One sentence spoken as: 1 s speech, short breath, 1 s speech, short breath.
    for part in (speech(1.0), silence(0.5), speech(1.0), silence(0.5)):
        finished += segmenter.feed(part)
    assert finished == [], "a 0.5 s breath must not close a short segment"
    assert segmenter.speaking

    # The real end of the sentence: a long pause.
    finished += segmenter.feed(silence(1.5))
    assert len(finished) == 1, "the sentence should arrive as exactly one utterance"
    assert finished[0] is not None
    # Everything spoken is in that single utterance, pauses included.
    assert finished[0].size / bot.SAMPLE_RATE > 2.0


def test_a_long_segment_still_ends_on_a_short_pause(bot):
    """Latency must not suffer once a full clause has been spoken."""
    segmenter = bot.Segmenter()
    rng = np.random.default_rng(4)
    long_speech = (rng.standard_normal(int(bot.SAMPLE_RATE * 4)) * 0.5).astype(np.float32)

    assert segmenter.feed(long_speech) == []
    finished = segmenter.feed(np.zeros(int(bot.SAMPLE_RATE * 0.5), dtype=np.float32))
    assert len(finished) == 1 and finished[0] is not None


def test_snapshot_is_cached_until_new_audio_arrives(bot):
    segmenter = bot.Segmenter()
    rng = np.random.default_rng(2)
    segmenter.feed((rng.standard_normal(bot.SAMPLE_RATE // 2) * 0.5).astype(np.float32))
    first = segmenter.snapshot()
    assert first is segmenter.snapshot()
    segmenter.feed((rng.standard_normal(bot.VAD_CHUNK * 4) * 0.5).astype(np.float32))
    assert segmenter.snapshot() is not first


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_turns(bot, pipeline, count: int, timeout: float = 20.0):
    """Pushes `count` utterances through the pipeline and waits for the audio."""

    async def scenario():
        workers = pipeline.workers()
        done: list[float] = []
        original = pipeline._log_turn

        def record(turn):
            done.append(time.monotonic())
            original(turn)

        pipeline._log_turn = record
        start = time.monotonic()
        try:
            for index in range(count):
                pipeline.asr_queue.put_nowait(utterance(bot, index))
            deadline = start + timeout
            while len(done) < count and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
        finally:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
        return start, done

    return asyncio.run(scenario())


def test_turns_are_translated_and_spoken_in_order(bot):
    speech = FakeSpeech()
    pipeline, room, _ = make_pipeline(bot, speech=speech)
    _, done = run_turns(bot, pipeline, 2)

    assert len(done) == 2
    assert speech.texts, "no speech was synthesised"
    # Every piece of every turn's translation is spoken exactly once, in order.
    spoken = " ".join(speech.texts).split()
    assert spoken == "".join(FakeTranslator.PIECES).split() * 2
    translations = [p for topic, p in room.local_participant.messages if topic == bot.TRANSLATION_TOPIC]
    assert translations[-1].count("Friede sei mit euch") == 1


def test_streaming_starts_speaking_before_the_translation_is_finished(bot):
    speech = FakeSpeech()
    pipeline, _, _ = make_pipeline(bot, speech=speech)
    run_turns(bot, pipeline, 1)
    # The translator emits five pieces; a non-streaming pipeline would call the
    # TTS exactly once with the whole text.
    assert len(speech.texts) >= 2
    assert speech.texts[0] != "".join(FakeTranslator.PIECES).strip()


def test_realtime_playout_does_not_stall_the_gpu_stages(bot):
    """Regression test for the ever-growing audio lag.

    Playout runs in real time. If it shares the loop with the translation stage
    (or holds a GPU slot), every turn adds its full playback duration to the
    next turn's latency and the agent drifts behind for good.
    """
    translator = FakeTranslator()
    speech = FakeSpeech(audio_seconds=0.6)
    pipeline, _, source = make_pipeline(bot, translator=translator, speech=speech)

    translated_at: list[float] = []
    original = pipeline._translate

    async def record(turn):
        await original(turn)
        translated_at.append(time.monotonic())

    pipeline._translate = record
    start, done = run_turns(bot, pipeline, 3)

    assert len(done) == 3
    compute_time = translated_at[-1] - start
    playout_time = done[-1] - start
    total_audio = speech.audio_seconds * len(speech.texts)

    # All GPU work finishes while the first turns are still being played back.
    assert compute_time < playout_time, (compute_time, playout_time)
    # And the real-time playback, not the GPU, is what sets the total duration.
    assert playout_time >= total_audio * 0.8, (playout_time, total_audio)
    assert compute_time < total_audio * 0.8, (compute_time, total_audio)


def test_pipeline_overlaps_asr_and_speech_of_different_turns(bot):
    asr = FakeAsr(seconds=0.15)
    speech = FakeSpeech(seconds=0.15, audio_seconds=0.2)
    pipeline, _, _ = make_pipeline(bot, asr=asr, speech=speech)
    run_turns(bot, pipeline, 3)

    def overlaps(a, b):
        return a[0] < b[1] and b[0] < a[1]

    assert any(
        overlaps(asr_call, tts_call) for asr_call in asr.calls for tts_call in speech.calls
    ), "ASR and TTS never ran at the same time - the stages are still serialised"


def test_translation_context_is_interleaved_source_and_target(bot):
    translator = FakeTranslator()
    pipeline, _, _ = make_pipeline(bot, translator=translator)
    run_turns(bot, pipeline, 2)

    assert translator.contexts[0] == []
    assert len(translator.contexts[1]) == 1
    source, target = translator.contexts[1][0]
    assert source == "نص عربي"
    assert target == "".join(FakeTranslator.PIECES).strip()


def test_the_final_translation_is_never_overwritten_by_a_partial(bot):
    """The browser replaces a line by id, so the last message must be complete."""

    class SlowParticipant(FakeParticipant):
        async def publish_data(self, payload, *, reliable, destination_identities, topic):
            # Partials are published while the GPU stage streams; a slow link
            # must not let one of them land after the final text.
            await asyncio.sleep(0.05)
            self.messages.append((topic, payload))

    pipeline, room, _ = make_pipeline(bot)
    room.local_participant = SlowParticipant()
    run_turns(bot, pipeline, 1)

    translations = [p for topic, p in room.local_participant.messages if topic == bot.TRANSLATION_TOPIC]
    assert translations, "no translation was published"
    assert "Friede sei mit euch" in translations[-1]


def test_chunking_never_loses_text(bot):
    """Property check: the spoken chunks must reassemble the full translation.

    This is the direct guard against truncated speech output - whatever the
    translator produces has to leave the splitter exactly once, in order.
    """
    import random

    rng = random.Random(7)
    words = ["Lob", "sei", "Gott", "dem", "Herrn", "der", "Welten", "wahrlich",
             "Barmherzigen", "und", "Beistand", "Gesandten", "gepriesen"]
    enders = [". ", "! ", "? ", "; ", ", ", " ", ": "]

    for _ in range(200):
        text = ""
        for _ in range(rng.randint(1, 40)):
            text += rng.choice(words) + rng.choice(enders)
        # Stream it piece by piece exactly as the translator would.
        buffer = ""
        chunks: list[str] = []
        index = 0
        position = 0
        while position < len(text):
            step = rng.randint(2, 9)
            buffer += text[position:position + step]
            position += step
            while True:
                chunk, buffer = bot.split_speakable(buffer, index, flush=False)
                if not chunk:
                    break
                index += 1
                chunks.append(chunk)
        rest, _ = bot.split_speakable(buffer, index, flush=True)
        if rest:
            chunks.append(rest)
        assert " ".join(chunks).split() == text.split(), text


def test_chunks_grow_so_a_turn_needs_few_tts_calls(bot):
    """Every TTS call re-synthesises the reference clip, so calls are costly.

    The first chunk stays short for a fast start, but later chunks must grow,
    otherwise a long sentence fans out into many calls and synthesis falls
    behind real time - which is what made whole turns get skipped.
    """
    sentence = "Lob sei Gott, dem Herrn der Welten, dem Barmherzigen und dem Gnaedigen. "
    text = sentence * 8  # ~580 characters
    chunks: list[str] = []
    buffer = text
    index = 0
    while True:
        chunk, buffer = bot.split_speakable(buffer, index, flush=False)
        if not chunk:
            break
        index += 1
        chunks.append(chunk)
    rest, _ = bot.split_speakable(buffer, index, flush=True)
    if rest:
        chunks.append(rest)

    assert " ".join(chunks).split() == text.split()
    assert len(chunks) <= 5, f"{len(chunks)} TTS calls for {len(text)} characters is too many"
    assert len(chunks[0]) <= bot.TTS_MAX_CHUNK_CHARS // 2, "the first chunk must stay short"
    assert len(chunks[2]) > len(chunks[0]), "later chunks must grow"


@pytest.mark.parametrize("audio_case", ["valid", "missing", "empty", "nonfinite"])
def test_moss_engine_calls_the_documented_api(bot, monkeypatch, audio_case):
    """MOSS-TTS is wired through transformers' AutoModel/AutoProcessor.

    The real weights cannot run here, so this fakes exactly the API the model
    card documents and checks we drive it correctly: a built-in voice (no
    reference), the hierarchical-sampling knobs, and a token budget derived
    from the text rather than a fixed 4096.
    """
    import sys
    import types
    from contextlib import contextmanager

    calls = {}
    inference_active = []

    @contextmanager
    def inference_mode():
        inference_active.append(True)
        try:
            yield
        finally:
            inference_active.pop()

    monkeypatch.setattr(bot.torch, "inference_mode", inference_mode)

    class FakeMessage:
        def __init__(self, seconds):
            # 24 kHz mono waveform, as audio_codes_list[0] would be.
            self.audio_codes_list = [FakeTensor(int(32_000 * seconds))]

    class FakeTensor:
        def __init__(self, size):
            self._array = np.zeros(0 if audio_case == "empty" else size, dtype=np.float32)
            if audio_case == "nonfinite":
                self._array[0] = np.nan

        def detach(self):
            return self

        def to(self, _dtype):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._array

    class FakeProcessor:
        model_config = types.SimpleNamespace(sampling_rate=32_000)
        audio_tokenizer = types.SimpleNamespace(
            to=lambda _device: types.SimpleNamespace(eval=lambda: "tokenizer"))

        def build_user_message(self, text, reference=None):
            calls["reference"] = reference
            calls["text"] = text
            return {"text": text}

        def __call__(self, conversations, mode):
            calls["mode"] = mode
            return {"input_ids": FakeIds(), "attention_mask": FakeIds()}

        def decode(self, _outputs):
            assert inference_active, "codec decoding must not build an autograd graph"
            return [] if audio_case == "missing" else [FakeMessage(2.0)]

    class FakeIds:
        def to(self, _device):
            return self

    class FakeModel:
        device = "cpu"

        def __init__(self):
            self.generation_config = types.SimpleNamespace(cache_implementation=None)

        def eval(self):
            return self

        def to(self, _device):
            return self

        def generate(self, **kwargs):
            calls.update(kwargs)
            return "outputs"

    fake = types.SimpleNamespace(
        __version__="5.4.0",
        AutoProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeProcessor()),
        AutoModel=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeModel()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)

    engine = bot.MossSynthesizer()
    text = "Lob sei Gott, dem Herrn der Welten, dem Barmherzigen und dem Gnaedigen."
    if audio_case != "valid":
        with pytest.raises(RuntimeError, match="audio"):
            engine.synthesize(text)
        return
    audio = engine.synthesize(text)

    assert calls["reference"] is None, "no reference clip when MOSS_REF_WAV is unset"
    assert calls["mode"] == "generation"
    assert calls["n_vq_for_inference"] == bot.MOSS_VQ_DEPTH
    assert calls["audio_top_k"] == bot.MOSS_TOP_K
    # The budget must scale with the text, not be a fixed 4096.
    assert 96 <= calls["max_new_tokens"] < 1000
    assert calls["max_new_tokens"] == engine._token_budget(text)
    # 2 s at the model's 32 kHz, resampled to the pipeline's 24 kHz.
    assert abs(audio.size / bot.AUDIO_SAMPLE_RATE - 2.0) < 0.01


def test_moss_does_not_hide_generation_type_errors(bot, monkeypatch):
    """An internal TypeError must propagate, not trigger another GPU decode."""
    import pytest
    import sys
    import types

    attempts = []

    class FakeProcessor:
        model_config = types.SimpleNamespace(sampling_rate=24_000)
        audio_tokenizer = types.SimpleNamespace(
            to=lambda _device: types.SimpleNamespace(eval=lambda: "tokenizer"))

        def build_user_message(self, text, reference=None):
            return {"text": text}

        def __call__(self, conversations, mode):
            return {"input_ids": _Ids(), "attention_mask": _Ids()}

        def decode(self, _outputs):
            return []

    class _Ids:
        def to(self, _device):
            return self

    class FakeModel:
        device = "cpu"

        def __init__(self):
            self.generation_config = types.SimpleNamespace(cache_implementation=None)

        def eval(self):
            return self

        def to(self, _d):
            return self

        def generate(self, **kwargs):
            attempts.append(kwargs)
            if "n_vq_for_inference" in kwargs:
                raise TypeError("unexpected keyword argument 'n_vq_for_inference'")
            return "outputs"

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        __version__="5.4.0",
        AutoProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeProcessor()),
        AutoModel=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeModel()),
    ))

    engine = bot.MossSynthesizer()
    with pytest.raises(TypeError, match="n_vq_for_inference"):
        engine.synthesize("Guten Morgen.")
    assert len(attempts) == 1, "do not retry arbitrary TypeErrors with different settings"


def test_a_single_turn_is_never_cut_in_half(bot, monkeypatch):
    """Losing a whole sentence is recoverable; half a sentence is not."""
    pipeline, _, _ = make_pipeline(bot)
    monkeypatch.setattr(bot, "MAX_AUDIO_BACKLOG_SECONDS", 2.0)
    turn = utterance(bot, 0)
    one_second = np.zeros(bot.AUDIO_SAMPLE_RATE, dtype=np.float32)

    for _ in range(4):
        pipeline._enqueue_playout(bot.AudioChunk(turn, one_second, last=False))
    pipeline._enqueue_playout(bot.AudioChunk(turn, one_second, last=True))

    remaining = []
    while not pipeline.playout_queue.empty():
        remaining.append(pipeline.playout_queue.get_nowait())
    assert len(remaining) == 5, "the only queued turn must survive intact"
    assert remaining[-1].last


def test_whole_turns_are_skipped_when_the_agent_falls_behind(bot, monkeypatch):
    pipeline, _, _ = make_pipeline(bot)
    monkeypatch.setattr(bot, "MAX_AUDIO_BACKLOG_SECONDS", 4.0)
    one_second = np.zeros(bot.AUDIO_SAMPLE_RATE, dtype=np.float32)
    turns = [utterance(bot, index) for index in range(4)]
    pipeline._playing_turn = turns[0].utterance_id

    for turn in turns:
        pipeline._enqueue_playout(bot.AudioChunk(turn, one_second, last=False))
        pipeline._enqueue_playout(bot.AudioChunk(turn, one_second, last=True))

    remaining = []
    while not pipeline.playout_queue.empty():
        remaining.append(pipeline.playout_queue.get_nowait())
    survivors = {chunk.turn.utterance_id for chunk in remaining}

    assert pipeline._playout_backlog <= 4.0
    # The turn being spoken and the newest turn are always kept ...
    assert turns[0].utterance_id in survivors
    assert turns[-1].utterance_id in survivors
    # ... and whatever was skipped was skipped completely.
    for turn in turns:
        chunks = [c for c in remaining if c.turn.utterance_id == turn.utterance_id]
        assert len(chunks) in (0, 2), "a turn was cut in half"
