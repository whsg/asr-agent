"""TRANSLATION_ENGINE=llamacpp: a real child process speaking llama-server's HTTP/SSE API."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Mirrors the documented llama-server contract: GET /health, POST /completion
# with server-sent events and a final event carrying stop/stop_type/timings.
FAKE_SERVER = r'''#!{python}
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
record = os.environ["FAKE_LLAMA_RECORD"]
mode = os.environ.get("FAKE_LLAMA_MODE", "ok")
with open(record + ".argv", "w") as handle:
    json.dump({"argv": args, "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}, handle)
if mode == "crash":
    print("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 12.6 GiB: out of memory", flush=True)
    sys.exit(1)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path != "/health":
            self.send_error(404)
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/detokenize":
            text = "<unused7>ilk" if mode == "mismatch" else "صباح الخير. Guten Morgen."
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"content": text if body["tokens"] == [1, 2, 3] else "?"}).encode())
            return
        with open(record + ".request", "w") as handle:
            json.dump({"path": self.path, "body": body}, handle)
        if mode == "http500":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error": {"code": 500, "message": "context overflow"}}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        events = [{"content": "Guten ", "stop": False}, {"content": "Morgen.", "stop": False}]
        if mode != "truncated":
            events.append({"content": "", "stop": True, "tokens_predicted": 3,
                           "stop_type": "limit" if mode == "limit" else "eos",
                           "timings": {"prompt_n": 4, "prompt_ms": 20.0, "predicted_n": 3,
                                       "predicted_per_second": 18.5}})
        for event in events:
            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            self.wfile.flush()


HTTPServer(("127.0.0.1", int(args[args.index("--port") + 1])), Handler).serve_forever()
'''

PROMPT_IDS = [2, 105, 2364, 107, 9, 106, 107, 105, 4368, 107]


class Tokenizer:
    bos_token = "<bos>"

    def __call__(self, text, **kwargs):
        if kwargs.get("return_tensors") != "pt":
            return {"input_ids": [1, 2, 3]}
        ids = np.array([PROMPT_IDS])
        return {"input_ids": ids, "attention_mask": np.ones_like(ids)}


class Processor:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.messages = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append((messages, kwargs))
        # The processor's tokenizing path cannot handle the template's own
        # format, so the agent must render text-only and tokenize separately.
        assert kwargs.get("tokenize") is False
        return "<bos>rendered prompt"


@pytest.fixture
def llama(bot, monkeypatch, tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_text(FAKE_SERVER.replace("{python}", sys.executable))
    binary.chmod(0o755)
    model = tmp_path / "translategemma-12b-it.Q8_0.gguf"
    model.write_bytes(b"GGUF")
    processor = Processor()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor.tokenizer),
    ))
    monkeypatch.setattr(bot, "TRANSLATION_BACKEND", "translategemma")
    monkeypatch.setattr(bot, "LLAMA_SERVER_BIN", str(binary))
    monkeypatch.setattr(bot, "LLAMA_GGUF_FILE", str(model))
    monkeypatch.setattr(bot, "LLAMA_SERVER_LOG", str(tmp_path / "llama-server.log"))
    monkeypatch.setattr(bot, "LLAMA_STARTUP_SECONDS", 20.0)
    monkeypatch.setattr(bot.LlamaServer, "PID_FILE", tmp_path / "llama.pid")
    monkeypatch.setenv("FAKE_LLAMA_RECORD", str(tmp_path / "record"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
    yield SimpleNamespace(bot=bot, tmp=tmp_path, processor=processor,
                          record=lambda kind: json.loads((tmp_path / f"record.{kind}").read_text()))
    for server in list(bot.LlamaServer.running):
        server.stop()


def start_translator(llama, mode="ok", monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("FAKE_LLAMA_MODE", mode)
    server = llama.bot.LlamaServer("cuda:1")
    server.start()
    return server, llama.bot.LlamaCppTranslator(server)


def test_server_runs_only_on_its_gpu_fully_offloaded_and_cpu_light(llama, monkeypatch):
    server, _ = start_translator(llama, monkeypatch=monkeypatch)
    launched = llama.record("argv")
    argv = launched["argv"]
    # cuda:1 of this process is physical GPU 7: the server must see only that GPU.
    assert launched["cuda"] == "7"
    option = lambda name: argv[argv.index(name) + 1]  # noqa: E731
    assert option("-m") == llama.bot.LLAMA_GGUF_FILE
    assert option("--host") == "127.0.0.1"
    assert option("-ngl") == "999" and option("--fit") == "off", "no silent CPU offload"
    assert option("-c") == str(llama.bot.LLAMA_CTX)
    assert option("-np") == "1"
    assert option("-t") == option("-tb") == "1" and option("--poll") == "0", "leave cores to Phonon"
    assert int(llama.bot.LlamaServer.PID_FILE.read_text()) == server.process.pid


def test_translation_uses_official_token_ids_greedy_and_streams(llama, monkeypatch):
    _, translator = start_translator(llama, monkeypatch=monkeypatch)
    pieces = []
    translator.stream("صباح الخير", [("alt", "alt")], pieces.append)
    assert pieces == ["Guten ", "Morgen."]
    request = llama.record("request")
    assert request["path"] == "/completion"
    body = request["body"]
    # Token ids from the official template: llama-server adds no second BOS
    # and applies no template of its own.
    assert body["prompt"] == PROMPT_IDS
    assert body["temperature"] == 0.0 and body["stream"] is True
    assert body["n_predict"] == llama.bot.translation_budget(3)
    messages, kwargs = llama.processor.messages[-1]
    assert messages == [
        {"role": "user", "content": [{
            "type": "text", "source_lang_code": "ar", "target_lang_code": "de", "text": "alt"}]},
        {"role": "assistant", "content": "alt"},
        {"role": "user", "content": [{
            "type": "text", "source_lang_code": "ar", "target_lang_code": "de",
            "text": "صباح الخير"}]},
    ]
    assert kwargs["add_generation_prompt"] is True
    assert translator.last_hit_budget is False
    assert translator.last_timings["predicted_per_second"] == 18.5
    assert translator.quantized is False and "Q8_0" in translator.dtype
    assert translator.translate("صباح الخير") == "Guten Morgen."


def test_budget_limit_is_reported(llama, monkeypatch, caplog):
    _, translator = start_translator(llama, "limit", monkeypatch)
    with caplog.at_level("WARNING", logger="asr-bot"):
        translator.translate("صباح الخير")
    assert translator.last_hit_budget is True
    assert "budget" in caplog.text


@pytest.mark.parametrize("mode, message", [
    ("http500", "HTTP 500.*context overflow"),
    ("truncated", "without a final event"),
])
def test_server_errors_are_raised_not_swallowed(llama, monkeypatch, mode, message):
    _, translator = start_translator(llama, mode, monkeypatch)
    with pytest.raises(RuntimeError, match=message):
        translator.translate("صباح الخير")


def test_gguf_with_a_different_vocabulary_is_rejected_at_startup(llama, monkeypatch):
    with pytest.raises(RuntimeError, match="does not match the tokenizer"):
        start_translator(llama, "mismatch", monkeypatch)


def test_llamacpp_with_qwen_requires_its_own_gguf():
    import subprocess

    root = Path(__file__).resolve().parent.parent
    code = (
        "import os, sys; sys.path.insert(0, %r); import stubs; stubs.install(); "
        "os.environ.update(TRANSLATION_ENGINE='llamacpp', TRANSLATION_BACKEND='qwen'); "
        "os.environ.pop('LLAMA_GGUF_FILE', None); sys.path.insert(0, %r); import bot"
    ) % (str(root / "test"), str(root))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode != 0
    assert "needs LLAMA_GGUF_REPO/LLAMA_GGUF_FILE" in result.stderr


@pytest.mark.parametrize("engine, asr, translation, expected", [
    ("llamacpp", "cuda:1", "cuda:0", [1]),      # worker threads default to GPU 0
    ("llamacpp", "cuda:0", "cuda:0", []),       # same GPU: nothing to protect
    ("transformers", "cuda:1", "cuda:0", []),   # unchanged in-process behaviour
])
def test_worker_threads_are_pinned_to_the_asr_gpu(bot, monkeypatch, engine, asr, translation, expected):
    calls = []
    monkeypatch.setattr(bot.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bot.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(bot.torch.cuda, "set_device", calls.append, raising=False)
    monkeypatch.setattr(bot, "TRANSLATION_ENGINE", engine)
    monkeypatch.setattr(bot, "ASR_DEVICE", asr)
    monkeypatch.setattr(bot, "TRANSLATION_DEVICE", translation)
    bot.pin_cuda_device()
    assert calls == expected


def test_crash_while_loading_shows_the_server_log(llama, monkeypatch):
    monkeypatch.setenv("FAKE_LLAMA_MODE", "crash")
    server = llama.bot.LlamaServer("cuda:0")
    server.start()
    with pytest.raises(RuntimeError, match="exited with code 1(.|\n)*out of memory"):
        server.wait_ready()


def test_missing_binary_names_the_setting(llama, monkeypatch):
    monkeypatch.setattr(llama.bot, "LLAMA_SERVER_BIN", str(llama.tmp / "nowhere" / "llama-server"))
    with pytest.raises(RuntimeError, match="LLAMA_SERVER_BIN"):
        llama.bot.LlamaServer("cuda:0").start()


def test_stop_frees_the_gpu_and_a_leftover_server_is_killed(llama, monkeypatch):
    first, _ = start_translator(llama, monkeypatch=monkeypatch)
    leftover = first.process.pid
    # A run that was killed hard leaves its server behind; the next start stops it.
    llama.bot.LlamaServer.running.discard(first)
    first.process = None
    second = llama.bot.LlamaServer("cuda:0")
    started = time.monotonic()
    second.start()
    assert time.monotonic() - started < 5
    assert not llama.bot.LlamaServer._alive(leftover)
    with pytest.raises(ChildProcessError):  # killed and already reaped or reapable
        while os.waitpid(leftover, 0):
            pass
    process = second.process
    second.stop()
    assert process.poll() is not None
    assert not llama.bot.LlamaServer.PID_FILE.exists()
    assert second not in llama.bot.LlamaServer.running


def test_prepare_models_starts_server_before_loading_asr(bot, monkeypatch):
    calls = []

    class Server:
        model_path = "/models/translategemma-12b-it.Q8_0.gguf"

        def __init__(self, device):
            calls.append(("server", device))

        def start(self):
            calls.append("start-server")

    monkeypatch.setattr(bot, "TRANSLATION_ENGINE", "llamacpp")
    monkeypatch.setattr(bot, "LlamaServer", Server)
    monkeypatch.setattr(bot, "LlamaCppTranslator", lambda server: calls.append("translator") or SimpleNamespace(
        translate=lambda text: "Guten Morgen.", quantized=False, dtype="llama.cpp Q8_0"))
    monkeypatch.setattr(bot, "Translator", lambda *args: pytest.fail("transformers engine must not load"))
    monkeypatch.setattr(bot, "AsrEngine", lambda *args: SimpleNamespace(
        transcribe=lambda audio: calls.append("asr")))
    monkeypatch.setattr(bot, "SpeechSynthesizer", lambda: SimpleNamespace(
        load=lambda: None, synthesize=lambda text: np.ones(2400, dtype=np.float32)))
    monkeypatch.setattr(bot, "load_silero_vad", lambda: None)
    bot.prepare_models()
    assert calls.index("start-server") < calls.index("asr") < calls.index("translator")


def test_physical_gpu_mapping(bot, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert bot._physical_gpu("cuda:1") == "1"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert bot._physical_gpu("cuda:0") == "2"
    with pytest.raises(RuntimeError, match="not in CUDA_VISIBLE_DEVICES"):
        bot._physical_gpu("cuda:2")
