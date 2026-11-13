"""Importable stubs for the GPU/LiveKit dependencies used by bot.py.

The real models are not importable outside Colab, so the tests and the
benchmark replace them with controllable fakes and exercise the scheduling and
streaming logic of `bot.py` instead.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Deterministic configuration so a developer's local .env cannot shift results.
TEST_ENV = {
    "ASR_DEVICE": "auto",
    "TRANSLATION_DEVICE": "auto",
    "TTS_DEVICE": "auto",
    "TRANSLATION_BACKEND": "qwen",
    "STREAM_TTS": "1",
    "INTERIM_CAPTIONS": "0",
    "TTS_FIRST_CHUNK_CHARS": "40",
    "TTS_CHUNK_CHARS": "110",
    "TTS_MAX_CHUNK_CHARS": "240",
    "GPU_CONCURRENCY": "2",
    "TRANSLATION_CONTEXT_TURNS": "4",
    "MAX_AUDIO_BACKLOG_SECONDS": "10",
}


def install() -> None:
    """Puts every stub in place; safe to call more than once."""
    _install_torch_stub()
    _install_livekit_stub()
    _install_misc_stubs()
    for key, value in TEST_ENV.items():
        os.environ[key] = value


def import_bot():
    install()
    sys.path.insert(0, str(PROJECT_ROOT))
    import bot

    return bot


class _InferenceMode:
    def __call__(self, func=None):
        if func is None:
            return self
        return func

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _install_torch_stub() -> None:
    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"
    torch.inference_mode = _InferenceMode()
    torch.from_numpy = lambda array: array

    cuda = types.SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
        get_device_capability=lambda _index=0: (7, 5),
        get_device_properties=lambda _index=0: types.SimpleNamespace(name="stub", total_memory=0),
        empty_cache=lambda: None,
        mem_get_info=lambda _index=0: (0, 0),
        OutOfMemoryError=type("OutOfMemoryError", (RuntimeError,), {}),
    )
    torch.cuda = cuda
    torch.OutOfMemoryError = cuda.OutOfMemoryError
    sys.modules["torch"] = torch


def _install_livekit_stub() -> None:
    livekit = types.ModuleType("livekit")
    api = types.ModuleType("livekit.api")
    rtc = types.ModuleType("livekit.rtc")

    class AccessToken:
        def __init__(self, *_args):
            pass

        def with_identity(self, *_a):
            return self

        def with_name(self, *_a):
            return self

        def with_ttl(self, *_a):
            return self

        def with_grants(self, *_a):
            return self

        def to_jwt(self):
            return "token"

    api.AccessToken = AccessToken
    api.VideoGrants = lambda **kwargs: kwargs
    api.ListRoomsRequest = lambda: None
    api.LiveKitAPI = object

    class AudioFrame:
        def __init__(self, data, sample_rate, num_channels, samples_per_channel):
            self.data = data
            self.sample_rate = sample_rate
            self.num_channels = num_channels
            self.samples_per_channel = samples_per_channel

    rtc.AudioFrame = AudioFrame
    rtc.Room = object
    rtc.Track = object
    rtc.AudioSource = object
    rtc.AudioStream = object
    rtc.LocalAudioTrack = object
    rtc.RemoteParticipant = object
    rtc.DataPacket = object
    rtc.TrackKind = types.SimpleNamespace(KIND_AUDIO="audio")

    livekit.api = api
    livekit.rtc = rtc
    sys.modules["livekit"] = livekit
    sys.modules["livekit.api"] = api
    sys.modules["livekit.rtc"] = rtc


def _install_misc_stubs() -> None:
    silero = types.ModuleType("silero_vad")

    def load_silero_vad():
        # Speech whenever the window is loud enough; deterministic for tests.
        return lambda chunk, _rate: float(np.abs(np.asarray(chunk)).mean() > 0.05)

    silero.load_silero_vad = load_silero_vad
    sys.modules["silero_vad"] = silero

    # Always stub dotenv so a developer's local .env cannot change the test
    # configuration.
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *_a, **_k: None
    sys.modules["dotenv"] = dotenv
