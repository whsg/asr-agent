"""The TTS worker process: real subprocess, real pipes, fake TTS engine."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

TEST_DIR = Path(__file__).resolve().parent

WORKER = """
import os, sys, types
sys.path.insert(0, {test_dir!r})
import numpy as np
import stubs

sys.modules["transformers"] = types.SimpleNamespace(__version__="5.3.0")
bot = stubs.import_bot()
mode = os.environ["FAKE_TTS_MODE"]


class Engine:
    def load(self):
        # Libraries print to stdout; that must not corrupt the protocol.
        print("noise from a library on stdout")
        assert os.environ["PYTHONPATH"].split(os.pathsep)[0] == os.environ["EXPECTED_SITE"]
        if mode == "load-error":
            raise ValueError("model files are broken")

    def synthesize(self, text):
        if mode == "crash":
            os._exit(3)
        if text == "fail":
            raise ValueError("bad chunk")
        print("more noise")
        return np.arange(len(text), dtype=np.float32) / 10


bot.SpeechSynthesizer._ENGINES[bot.TTS_BACKEND] = Engine
sys.exit(bot.run_tts_worker())
"""


@pytest.fixture
def client(bot, tmp_path, monkeypatch):
    script = tmp_path / "worker.py"
    script.write_text(WORKER.format(test_dir=str(TEST_DIR)))
    site = tmp_path / "isolated-site"
    monkeypatch.setattr(bot, "TTS_WORKER_PYTHONPATH", str(site))
    monkeypatch.setenv("EXPECTED_SITE", str(site))
    monkeypatch.setenv("FAKE_TTS_MODE", "ok")
    worker = bot.TtsWorkerClient()
    worker.command = [sys.executable, "-u", str(script)]
    yield worker
    worker.close()


def test_worker_round_trips_audio_and_survives_a_failed_chunk(client):
    audio = client.synthesize("Hallo")
    np.testing.assert_allclose(audio, np.arange(5, dtype=np.float32) / 10)
    assert audio.dtype == np.float32

    with pytest.raises(RuntimeError, match="bad chunk"):
        client.synthesize("fail")
    # One failed chunk must not take the worker down.
    assert client.synthesize("Guten Morgen.").size == len("Guten Morgen.")

    process = client.process
    client.close()
    assert process.returncode == 0, "the worker must exit cleanly so its GPU memory is freed"
    assert client.process is None


def test_worker_load_error_is_reported(client, monkeypatch):
    monkeypatch.setenv("FAKE_TTS_MODE", "load-error")
    with pytest.raises(RuntimeError, match="model files are broken"):
        client.load()


def test_worker_crash_is_reported_not_hung(client, monkeypatch):
    monkeypatch.setenv("FAKE_TTS_MODE", "crash")
    with pytest.raises(RuntimeError, match=r"exited \(code 3\)"):
        client.synthesize("Hallo")


def test_speech_synthesizer_uses_worker_only_when_configured(bot, monkeypatch):
    monkeypatch.setattr(bot, "TTS_WORKER_PYTHONPATH", "")
    assert isinstance(bot.SpeechSynthesizer().engine, bot.MossSynthesizer)
    monkeypatch.setattr(bot, "TTS_WORKER_PYTHONPATH", os.fspath(TEST_DIR))
    assert isinstance(bot.SpeechSynthesizer().engine, bot.TtsWorkerClient)
