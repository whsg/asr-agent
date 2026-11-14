"""Phonon (Gradium) TTS backend, driven through a fake `ptts` module."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest


class FakeTTS:
    """Mirrors the documented API: TTS(model, lang=...), .synth(text, voice=...), .sample_rate."""

    instances = []

    def __init__(self, model=None, *, lang):
        self.model = model
        self.lang = lang
        self.sample_rate = 48_000
        self.voices = ["Anna", "Jonas"]
        self.calls = []
        self.output = None
        FakeTTS.instances.append(self)

    def synth(self, text, voice=None):
        self.calls.append((text, voice))
        if self.output is not None:
            return self.output
        return np.full(48_000, 0.1, dtype=np.float32)  # 1 s at 48 kHz


@pytest.fixture
def phonon(bot, monkeypatch):
    FakeTTS.instances.clear()
    monkeypatch.setitem(sys.modules, "ptts", SimpleNamespace(TTS=FakeTTS))
    monkeypatch.setattr(bot, "PHONON_MODEL", "hf://gradium/phonon-40m-de")
    monkeypatch.setattr(bot, "PHONON_VOICE", "")
    return bot


def test_phonon_loads_german_model_and_resamples(phonon):
    engine = phonon.PhononSynthesizer()
    audio = engine.synthesize("Guten Morgen.")
    tts = FakeTTS.instances[0]
    assert tts.model == "hf://gradium/phonon-40m-de"
    assert tts.lang == phonon.TRANSLATION_TARGET_LANG == "de"
    assert tts.calls == [("Guten Morgen.", None)], "empty PHONON_VOICE must use the model default"
    # 1 s of 48 kHz model audio becomes 1 s at the pipeline's 24 kHz.
    assert audio.dtype == np.float32
    assert abs(audio.size / phonon.AUDIO_SAMPLE_RATE - 1.0) < 0.01
    engine.synthesize("Noch ein Satz.")
    assert len(FakeTTS.instances) == 1, "the model is loaded once and reused"


def test_phonon_passes_selected_voice(phonon, monkeypatch):
    monkeypatch.setattr(phonon, "PHONON_VOICE", "Jonas")
    engine = phonon.PhononSynthesizer()
    engine.synthesize("Guten Morgen.")
    assert FakeTTS.instances[0].calls == [("Guten Morgen.", "Jonas")]


@pytest.mark.parametrize("output", [np.zeros(0, dtype=np.float32), np.array([np.nan], dtype=np.float32)])
def test_phonon_rejects_empty_or_non_finite_audio(phonon, output):
    engine = phonon.PhononSynthesizer()
    engine.load()
    FakeTTS.instances[0].output = output
    with pytest.raises(RuntimeError, match="empty or non-finite"):
        engine.synthesize("Guten Morgen.")


def test_phonon_threads_are_set_before_the_model_loads(phonon, monkeypatch):
    order = []

    def set_num_threads(count):
        order.append(("threads", count, len(FakeTTS.instances)))

    monkeypatch.setitem(sys.modules, "ptts", SimpleNamespace(TTS=FakeTTS, set_num_threads=set_num_threads))
    monkeypatch.setattr(phonon, "PHONON_THREADS", 2)
    phonon.PhononSynthesizer().load()
    assert order == [("threads", 2, 0)], "the global pool must be sized before TTS() is created"


def test_phonon_threads_default_leaves_the_library_alone(phonon, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "ptts", SimpleNamespace(TTS=FakeTTS, set_num_threads=calls.append))
    monkeypatch.setattr(phonon, "PHONON_THREADS", 0)
    phonon.PhononSynthesizer().load()
    assert calls == []


def test_phonon_load_error_explains_gated_model(phonon, monkeypatch):
    def refuse(*args, **kwargs):
        raise PermissionError("401 Client Error: gated repo")

    monkeypatch.setitem(sys.modules, "ptts", SimpleNamespace(TTS=refuse))
    with pytest.raises(RuntimeError, match="gated: accept the terms"):
        phonon.PhononSynthesizer().load()


def test_phonon_is_selectable_and_runs_on_cpu_scheduler(phonon, monkeypatch):
    monkeypatch.setattr(phonon, "TTS_BACKEND", "phonon")
    assert isinstance(phonon.SpeechSynthesizer().engine, phonon.PhononSynthesizer)

    monkeypatch.setattr(phonon.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(phonon.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(phonon, "ASR_DEVICE", "cuda:0")
    monkeypatch.setattr(phonon, "TRANSLATION_DEVICE", "cuda:0")
    monkeypatch.setattr(phonon, "TTS_DEVICE", "cpu")
    asr, translation, speech = phonon.stage_schedulers()
    # CPU synthesis must not wait for the GPU 0 slot held by ASR/translation.
    assert asr is translation and speech is not translation


def test_explicit_cpu_stage_does_not_warn(bot, caplog):
    with caplog.at_level("WARNING", logger="asr-bot"):
        assert bot.pick_device_dtype("cpu") == (bot.torch.float32, "cpu")
    assert not caplog.records, "a deliberately CPU-placed stage (Phonon) is not a problem"
