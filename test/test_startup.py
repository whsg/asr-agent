"""Startup regressions; no GPU or model downloads required."""
import ast
import importlib
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent.parent


def notebook_source(index):
    notebook = json.loads((ROOT / "colab_agent.ipynb").read_text())
    return "".join(notebook["cells"][index]["source"])


def fake_startup(bot, monkeypatch, *, translation_error=None, speech_error=None, audio=None):
    calls = []

    def translate(text):
        calls.append("translate")
        if translation_error:
            raise translation_error
        return "Guten Morgen."

    def synthesize(text):
        calls.append("synthesize")
        if speech_error:
            raise speech_error
        return np.ones(2400, dtype=np.float32) if audio is None else audio

    monkeypatch.setattr(bot, "AsrEngine", lambda *args: SimpleNamespace(
        transcribe=lambda audio: calls.append("asr")))
    monkeypatch.setattr(bot, "Translator", lambda *args: (
        calls.append("load-translator") or SimpleNamespace(translate=translate, quantized=False)))
    monkeypatch.setattr(bot, "SpeechSynthesizer", lambda: SimpleNamespace(
        load=lambda: calls.append("load-speech"), synthesize=synthesize))
    monkeypatch.setattr(bot, "load_silero_vad", lambda: None)
    return calls


def test_speech_reserves_memory_before_translator_fallback(bot, monkeypatch):
    calls = fake_startup(bot, monkeypatch)
    bot.prepare_models()
    assert calls.index("synthesize") < calls.index("load-translator")


@pytest.mark.parametrize("stage", ["translation", "speech"])
def test_failed_warmup_prevents_readiness(bot, monkeypatch, stage):
    error = RuntimeError(f"{stage} warmup failed")
    fake_startup(bot, monkeypatch, **{f"{stage}_error": error})
    with pytest.raises(RuntimeError, match=f"{stage} warmup failed"):
        bot.prepare_models()


@pytest.mark.parametrize("audio", [np.zeros(0), np.array([np.nan]), np.array([np.inf])])
def test_invalid_warmup_audio_prevents_readiness(bot, monkeypatch, audio):
    fake_startup(bot, monkeypatch, audio=audio)
    with pytest.raises(RuntimeError, match="audio"):
        bot.prepare_models()


def test_empty_translation_prevents_readiness(bot, monkeypatch):
    fake_startup(bot, monkeypatch)
    monkeypatch.setattr(bot, "Translator", lambda *args: SimpleNamespace(
        translate=lambda text: " ", quantized=False))
    with pytest.raises(RuntimeError, match="empty translation"):
        bot.prepare_models()


def test_notebook_embeds_current_agent():
    assert notebook_source(4) == "%%writefile /content/bot.py\n" + (ROOT / "bot.py").read_text()


def test_notebook_cells_compile():
    for index in (2, 3, 5, 6):
        compile(notebook_source(index), f"notebook-cell-{index}", "exec")
    compile(notebook_source(4).split("\n", 1)[1], "embedded-bot", "exec")


@pytest.mark.parametrize("gib, allowed", [(15, False), (24, True), (40, True)])
def test_default_moss_weight_floor(bot, monkeypatch, gib, allowed):
    monkeypatch.setattr(bot, "pick_device_dtype", lambda requested="auto": (bot.torch.float16, "cuda:0"))
    monkeypatch.setattr(bot, "TTS_BACKEND", "moss")
    monkeypatch.setattr(bot, "MOSS_MODEL", "OpenMOSS-Team/MOSS-TTS-Local-Transformer")
    monkeypatch.setattr(bot, "MOSS_DTYPE", "auto")
    monkeypatch.setattr(bot, "COHERE_MODEL_NAME", "CohereLabs/cohere-transcribe-arabic-07-2026")
    monkeypatch.setattr(bot, "COHERE_4BIT", False)
    monkeypatch.setattr(bot, "TRANSLATION_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
    monkeypatch.setattr(bot.torch.cuda, "get_device_properties",
                        lambda index: SimpleNamespace(total_memory=gib * 2**30))
    if allowed:
        bot._check_moss_memory(bot.torch.float16, "cuda:0")
    else:
        with pytest.raises(RuntimeError, match="cannot fit a T4"):
            bot._check_moss_memory(bot.torch.float16, "cuda:0")


def test_missing_optional_secrets_preserve_required_values(tmp_path, monkeypatch):
    import types

    class SecretNotFoundError(Exception):
        pass

    class NotebookAccessError(Exception):
        pass

    secrets = {
        "LIVEKIT_URL": "wss://example.invalid", "LIVEKIT_API_KEY": "test-key",
        "LIVEKIT_API_SECRET": "test-secret", "HF_TOKEN": "test-token",
    }

    def get(key):
        if key in secrets:
            return secrets[key]
        raise SecretNotFoundError(key)

    colab = types.ModuleType("google.colab")
    colab.userdata = SimpleNamespace(
        get=get, SecretNotFoundError=SecretNotFoundError, NotebookAccessError=NotebookAccessError)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.userdata", colab.userdata)
    parsed = ast.parse(notebook_source(2))
    statements = [node for node in parsed.body if (
        isinstance(node, ast.FunctionDef) and node.name == "find_existing"
    )]
    namespace = {
        "importlib": importlib,
        "Path": lambda _: tmp_path, "TARGET": tmp_path / ".env",
        "REQUIRED": tuple(secrets), "OPTIONAL": ("MOSS_DTYPE", "TTS_BACKEND"),
    }
    exec(compile(ast.Module(body=statements, type_ignores=[]), "secrets-cell", "exec"), namespace)
    assert namespace["find_existing"]() == secrets

    def unexpected_failure(key):
        raise RuntimeError("secret service unavailable")

    colab.userdata.get = unexpected_failure
    with pytest.raises(RuntimeError, match="secret service unavailable"):
        namespace["find_existing"]()


def test_launch_failure_is_logged_and_raised(tmp_path):
    class FakeProcess:
        stdout = io.StringIO("Traceback: model loading failed\n")

        def poll(self):
            return 1

        def wait(self, timeout):
            return 1

    captured = {}

    def popen(*args, **kwargs):
        captured.update(kwargs)
        return FakeProcess()

    namespace = {
        "os": SimpleNamespace(path=SimpleNamespace(exists=lambda _: True),
                              environ={"HF_TOKEN": "stale-token"}),
        "subprocess": SimpleNamespace(
            Popen=popen, DEVNULL=-3, PIPE=-1, STDOUT=-2, TimeoutExpired=subprocess.TimeoutExpired),
        "sys": SimpleNamespace(executable="python"),
        "Path": lambda _: tmp_path / "bot.log",
        "dotenv_values": lambda *args, **kwargs: {"HF_TOKEN": "selected-test-token"},
    }
    parsed = ast.parse(notebook_source(6))
    parsed.body = [node for node in parsed.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        exec(compile(parsed, "launch-cell", "exec"), namespace)
    assert "model loading failed" in (tmp_path / "bot.log").read_text()
    assert "Bot exit code: 1" in (tmp_path / "bot.log").read_text()
    assert captured["env"]["HF_TOKEN"] == "selected-test-token"


@pytest.mark.parametrize("backend", ["moss", "xtts"])
def test_installation_selects_only_needed_backend_and_matches_torch(tmp_path, backend):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    namespace = {
        "importlib": SimpleNamespace(metadata=SimpleNamespace(version=lambda _: "2.9.1+cu128")),
        "subprocess": SimpleNamespace(run=run, STDOUT=-2),
        "sys": SimpleNamespace(executable="python"),
        "Path": lambda _: tmp_path / "install.log",
        "dotenv_values": lambda *args, **kwargs: {"TTS_BACKEND": backend},
    }
    parsed = ast.parse(notebook_source(3))
    parsed.body = [node for node in parsed.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(parsed, "install-cell", "exec"), namespace)
    packages = commands[1]
    assert "torch==2.9.1" in packages
    assert "torchaudio==2.9.1" in packages
    assert not any("git+" in package for package in packages)
    assert ("coqui-tts>=0.27" in packages) == (backend == "xtts")
    assert ("from TTS.api import TTS" in commands[2][-1]) == (backend == "xtts")
    assert "PreTrainedAudioTokenizerBase" in commands[2][-1]


def test_dotenv_serialization_roundtrip():
    # A separate interpreter avoids the pipeline suite's intentionally stubbed dotenv.
    code = """
import io
import tempfile
from pathlib import Path
from dotenv import dotenv_values, set_key
value = "test # quote's ${HOME}"
with tempfile.TemporaryDirectory() as directory:
    target = Path(directory) / ".env"
    target.touch()
    set_key(target, "VALUE", value, quote_mode="always")
    assert dotenv_values(target, interpolate=False)["VALUE"] == value
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
