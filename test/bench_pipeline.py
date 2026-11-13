"""Measures the new streaming pipeline against the old serial loop.

Both runs use exactly the same simulated stage costs, so the difference is
purely architectural. No GPU and no model weights are needed:

    python test/bench_pipeline.py
"""
from __future__ import annotations

import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stubs  # noqa: E402

# Per-turn cost of each stage, measured on a Colab T4 with the old build.
ASR_SECONDS = 0.7
TRANSLATE_SECONDS = 2.2
TTS_SECONDS = 1.4
AUDIO_SECONDS = 4.0  # spoken length of one translated utterance
SPEECH_INTERVAL = 4.5  # how often the speaker finishes an utterance
TURNS = 8
# Scale everything down so the benchmark finishes in seconds while keeping the
# ratios between the stages intact.
SCALE = 0.1


def _scaled(seconds: float) -> float:
    return seconds * SCALE


async def run_old() -> list[float]:
    """The previous design: one loop does translate + TTS + real-time playout."""
    lags: list[float] = []
    queue: asyncio.Queue = asyncio.Queue()

    async def producer() -> None:
        for index in range(TURNS):
            await asyncio.sleep(_scaled(SPEECH_INTERVAL))
            queue.put_nowait((index, time.monotonic()))

    async def worker() -> None:
        for _ in range(TURNS):
            _index, spoken_at = await queue.get()
            # A single GPU lock held across ASR, translation and TTS ...
            await asyncio.sleep(_scaled(ASR_SECONDS))
            await asyncio.sleep(_scaled(TRANSLATE_SECONDS))
            await asyncio.sleep(_scaled(TTS_SECONDS))
            lags.append(time.monotonic() - spoken_at)
            # ... and then the same loop pushes the audio out in real time.
            await asyncio.sleep(_scaled(AUDIO_SECONDS))

    await asyncio.gather(producer(), worker())
    return lags


async def run_new() -> list[float]:
    """The streaming pipeline: one worker per stage, playout fully decoupled."""
    lags: list[float] = []
    asr_q: asyncio.Queue = asyncio.Queue()
    mt_q: asyncio.Queue = asyncio.Queue()
    tts_q: asyncio.Queue = asyncio.Queue()
    play_q: asyncio.Queue = asyncio.Queue()
    chunks = 3  # the translation is spoken in three clause chunks

    async def producer() -> None:
        for index in range(TURNS):
            await asyncio.sleep(_scaled(SPEECH_INTERVAL))
            asr_q.put_nowait((index, time.monotonic()))

    async def asr() -> None:
        while True:
            turn = await asr_q.get()
            await asyncio.sleep(_scaled(ASR_SECONDS))
            mt_q.put_nowait(turn)

    async def translate() -> None:
        while True:
            turn = await mt_q.get()
            # Token streaming: each clause leaves as soon as it is complete.
            for chunk in range(chunks):
                await asyncio.sleep(_scaled(TRANSLATE_SECONDS) / chunks)
                tts_q.put_nowait((turn, chunk))

    async def tts() -> None:
        while True:
            turn, chunk = await tts_q.get()
            await asyncio.sleep(_scaled(TTS_SECONDS) / chunks)
            play_q.put_nowait((turn, chunk))

    async def playout() -> None:
        while True:
            (_index, spoken_at), chunk = await play_q.get()
            if chunk == 0:
                lags.append(time.monotonic() - spoken_at)
            await asyncio.sleep(_scaled(AUDIO_SECONDS) / chunks)

    workers = [asyncio.create_task(stage()) for stage in (asr, translate, tts, playout)]
    try:
        await producer()
        deadline = time.monotonic() + 30
        while len(lags) < TURNS and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    return lags


def report(name: str, lags: list[float]) -> None:
    real = [lag / SCALE for lag in lags]
    print(f"\n{name}")
    print("  turn   : " + "  ".join(f"{i + 1:>5}" for i in range(len(real))))
    print("  lag [s]: " + "  ".join(f"{lag:5.1f}" for lag in real))
    print(f"  first turn {real[0]:.1f} s, last turn {real[-1]:.1f} s, "
          f"median {statistics.median(real):.1f} s, growth {real[-1] - real[0]:+.1f} s")


def main() -> None:
    stubs.install()

    print(
        f"Simulated stage costs per turn: ASR {ASR_SECONDS}s, translation "
        f"{TRANSLATE_SECONDS}s, TTS {TTS_SECONDS}s, {AUDIO_SECONDS}s of audio, "
        f"a new utterance every {SPEECH_INTERVAL}s."
    )
    old = asyncio.run(run_old())
    new = asyncio.run(run_new())
    report("old: single loop, GPU lock held across translation + TTS + playout", old)
    report("new: streaming pipeline, playout decoupled", new)
    print(
        f"\nTime to first audio on the last turn: {old[-1] / SCALE:.1f} s -> "
        f"{new[-1] / SCALE:.1f} s ({old[-1] / new[-1]:.1f}x faster)."
    )


if __name__ == "__main__":
    main()
