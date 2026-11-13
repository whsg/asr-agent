"""Kaggle profile, official translation contract and dual-GPU routing tests."""
import ast
import asyncio
import io
import json
import queue
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def kaggle_cells():
    return json.loads((ROOT / "kaggle_agent.ipynb").read_text())["cells"]


def dual_gpu(bot, monkeypatch):
    monkeypatch.setattr(bot.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bot.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(bot.torch.cuda, "get_device_properties", lambda index: SimpleNamespace(
        name=f"T4-{index}", total_memory=15 * 2**30))
    monkeypatch.setattr(bot, "ASR_DEVICE", "cuda:0")
    monkeypatch.setattr(bot, "TRANSLATION_DEVICE", "cuda:0")
    monkeypatch.setattr(bot, "TTS_DEVICE", "cuda:1")
    monkeypatch.setattr(bot, "GPU_CONCURRENCY", 1)


def run_kaggle_install_cell(tmp_path, *, urlopen=None, pip_output="ok\n", pip_code=0, commands=None, envs=None):
    commands = [] if commands is None else commands
    envs = {} if envs is None else envs

    def run(command, **kwargs):
        commands.append(command)
        envs[id(command)] = kwargs.get("env") or {}
        is_pip = command[1:3] == ["-m", "pip"]
        kwargs["stdout"].write(pip_output if is_pip else "ok\n")
        return SimpleNamespace(returncode=pip_code if is_pip else 0)

    versions = {"torch": "2.6.0+cu124", "torchaudio": "2.6.0+cu124", "torchvision": "0.21.0+cu124"}
    tree = ast.parse("".join(kaggle_cells()[1]["source"]))
    tree.body = [node for node in tree.body
                 if not isinstance(node, (ast.Import, ast.ImportFrom))
                 and not (isinstance(node, ast.Assign) and any(
                     isinstance(t, ast.Name) and t.id == "WORK" for t in node.targets))
                 and not isinstance(node, ast.If)]
    namespace = {
        "WORK": tmp_path, "os": SimpleNamespace(environ={}), "print": lambda *a, **k: None,
        "sys": SimpleNamespace(executable="python"),
        "subprocess": SimpleNamespace(run=run, STDOUT=-2),
        "metadata": SimpleNamespace(version=versions.__getitem__, PackageNotFoundError=KeyError),
        "urllib": SimpleNamespace(request=SimpleNamespace(
            urlopen=urlopen or (lambda *a, **k: SimpleNamespace(close=lambda: None)))),
    }
    (tmp_path / "gpu-preflight.log").write_text("ok\n")
    exec(compile(tree, "kaggle-install", "exec"), namespace)
    return commands, versions


def test_kaggle_install_keeps_preinstalled_torch_stack(tmp_path):
    commands, versions = run_kaggle_install_cell(tmp_path)
    pip_calls = [c for c in commands if c[1:3] == ["-m", "pip"]]
    assert not any("download.pytorch.org" in part for call in pip_calls for part in call)
    assert not any(part.startswith(("torch==", "torchaudio==", "torchvision=="))
                   for call in pip_calls for part in call), "torch must never be reinstalled"
    agent_call = next(c for c in pip_calls if "-c" in c)
    constraint_file = agent_call[agent_call.index("-c") + 1]
    assert "transformers==5.4.0" in agent_call
    assert Path(constraint_file).read_text() == "".join(
        f"{name}=={version}\n" for name, version in versions.items())


def test_kaggle_install_isolates_moss_transformers(tmp_path):
    envs = {}
    commands, _ = run_kaggle_install_cell(tmp_path, envs=envs)
    moss_call = next(c for c in commands if "--target" in c)
    assert "transformers==5.3.0" in moss_call
    assert "--no-deps" in moss_call, "5.3.0 must reuse the shared deps, not replace them"
    site = moss_call[moss_call.index("--target") + 1]
    assert site == str(tmp_path / "moss-transformers")

    check = next(c for c in commands if c[1] == "-c" and "MOSS-Audio-Tokenizer" in c[2])
    assert "transformers.__version__ == '5.3.0'" in check[2]
    assert "AutoConfig.from_pretrained('OpenMOSS-Team/MOSS-Audio-Tokenizer'" in check[2]
    assert envs[id(check)]["PYTHONPATH"] == site, "the MOSS check must see the isolated transformers"
    main_check = next(c for c in commands if c[1] == "-c" and "CohereAsrForConditionalGeneration" in c[2])
    assert "PYTHONPATH" not in envs[id(main_check)], "the main process must keep transformers 5.4.0"

    profile = "".join(kaggle_cells()[2]["source"])
    assert "'TTS_WORKER_PYTHONPATH': '/kaggle/working/moss-transformers'" in profile


def test_kaggle_install_reports_missing_internet_before_pip(tmp_path):
    def offline(*args, **kwargs):
        raise OSError("[Errno -3] Temporary failure in name resolution")

    commands = []
    with pytest.raises(RuntimeError, match="NO Internet") as error:
        run_kaggle_install_cell(tmp_path, urlopen=offline, commands=commands)
    assert "Internet: On" in str(error.value)
    assert commands, "the GPU check must still have run"
    assert not any(c[1:3] == ["-m", "pip"] for c in commands)


def test_kaggle_install_http_status_counts_as_online(tmp_path):
    class HttpStatus(OSError):
        code = 401

    def gated(url, **kwargs):
        if "huggingface" in url:
            raise HttpStatus("401")
        return SimpleNamespace(close=lambda: None)

    commands, _ = run_kaggle_install_cell(tmp_path, urlopen=gated)
    assert any(c[1:3] == ["-m", "pip"] for c in commands)


def test_kaggle_install_failure_puts_cause_in_traceback(tmp_path):
    output = "Collecting x\nERROR: No matching distribution found for livekit>=1.0\n"
    with pytest.raises(RuntimeError) as error:
        run_kaggle_install_cell(tmp_path, pip_output=output, pip_code=1)
    assert "No matching distribution found for livekit" in str(error.value)


def test_kaggle_notebook_embeds_agent_and_compiles():
    for cell in kaggle_cells():
        if cell["cell_type"] != "code":
            continue
        code = "".join(cell["source"])
        if code.startswith("%%writefile"):
            header, code = code.split("\n", 1)
            assert header == "%%writefile /kaggle/working/bot.py"
            assert code == (ROOT / "bot.py").read_text()
        compile(code, "kaggle-cell", "exec")


def test_kaggle_profile_has_two_devices_and_quantized_gemma():
    tree = ast.parse("".join(kaggle_cells()[2]["source"]))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "PROFILE"
                              for target in node.targets))
    profile = ast.literal_eval(assignment.value)
    assert profile["ASR_DEVICE"] == profile["TRANSLATION_DEVICE"] == "cuda:0"
    assert profile["TTS_DEVICE"] == "cuda:1"
    assert profile["TRANSLATION_MODEL"] == "google/translategemma-12b-it"
    assert profile["TRANSLATION_BACKEND"] == "translategemma"
    assert profile["TRANSLATION_4BIT"] == "1"
    assert profile["TRANSLATION_CONTEXT_TURNS"] == "0"
    assert profile["GPU_CONCURRENCY"] == "1"
    assert profile["TTS_BACKEND"] == "moss"


def test_explicit_device_placement_and_dtype(bot, monkeypatch):
    dual_gpu(bot, monkeypatch)
    monkeypatch.setattr(bot.torch.cuda, "get_device_capability",
                        lambda index: (7, 5) if index == 0 else (8, 0))
    assert bot.pick_device_dtype("cuda:0") == (bot.torch.float16, "cuda:0")
    assert bot.pick_device_dtype("cuda:1") == (bot.torch.bfloat16, "cuda:1")
    assert bot._device_map("cuda:1") == {"": 1}
    assert bot._device_map("cpu") == {"": "cpu"}
    with pytest.raises(RuntimeError, match="does not expose"):
        bot.pick_device_dtype("cuda:2")
    with pytest.raises(ValueError, match="Invalid device"):
        bot.pick_device_dtype("gpu1")


def test_moss_floor_is_per_device_not_total_across_gpus(bot, monkeypatch):
    dual_gpu(bot, monkeypatch)
    monkeypatch.setattr(bot, "MOSS_DTYPE", "float16")
    monkeypatch.setattr(bot, "TTS_BACKEND", "moss")
    monkeypatch.setattr(bot, "TRANSLATION_MODEL", "google/translategemma-12b-it")
    bot._check_moss_memory(bot.torch.float16, "cuda:0")
    monkeypatch.setattr(bot, "TTS_DEVICE", "cuda:0")
    with pytest.raises(RuntimeError, match="cannot fit a T4"):
        bot._check_moss_memory(bot.torch.float16, "cuda:0")


def test_device_schedulers_allow_tts_during_translation(bot, monkeypatch):
    dual_gpu(bot, monkeypatch)

    async def scenario():
        asr, translation, speech = bot.stage_schedulers()
        assert asr is translation
        assert speech is not translation
        async with translation.reserve(bot.PRIO_TURN):
            async with speech.reserve(bot.PRIO_SPEECH):
                assert translation.busy and speech.busy
            waiter = asyncio.create_task(asr._acquire(bot.PRIO_TURN))
            await asyncio.sleep(0)
            assert not waiter.done(), "ASR and translation must serialize on GPU 0"
        await waiter
        asr._release()

    asyncio.run(scenario())


def test_single_device_keeps_shared_scheduler(bot, monkeypatch):
    for key in ("ASR_DEVICE", "TRANSLATION_DEVICE", "TTS_DEVICE"):
        monkeypatch.setattr(bot, key, "cpu")
    asr, translation, speech = bot.stage_schedulers()
    assert asr is translation is speech


def test_vram_report_uses_selected_gpu(bot, monkeypatch, caplog):
    dual_gpu(bot, monkeypatch)
    calls = []
    monkeypatch.setattr(bot.torch.cuda, "mem_get_info", lambda index: (
        calls.append(index) or (2 * 2**30, 15 * 2**30)))
    with caplog.at_level("INFO", logger="asr-bot"):
        bot._vram_report("TTS", "cuda:1")
    assert calls == [1]
    assert "cuda:1" in caplog.text


def test_moss_model_and_codec_both_load_on_gpu_one(bot, monkeypatch):
    dual_gpu(bot, monkeypatch)
    calls = {}

    class Codec:
        def to(self, device):
            calls["codec_device"] = device
            return self

        def eval(self):
            return self

    def load_model(model_id, **kwargs):
        calls["model_kwargs"] = kwargs
        calls["model"] = SimpleNamespace(
            eval=lambda: None, generation_config=SimpleNamespace(cache_implementation=None))
        return calls["model"]

    def load_processor(model_id, **kwargs):
        # MOSS's real processor forwards unknown kwargs into ProcessorMixin,
        # which rejects them ("Unexpected keyword argument token").
        unexpected = set(kwargs) - {"trust_remote_code"}
        if unexpected:
            raise TypeError(f"Unexpected keyword argument {', '.join(sorted(unexpected))}.")
        return SimpleNamespace(audio_tokenizer=Codec(), model_config=SimpleNamespace(sampling_rate=24_000))

    monkeypatch.setattr(bot.torch, "backends", SimpleNamespace(
        cuda=SimpleNamespace(enable_cudnn_sdp=lambda enabled: None)), raising=False)
    monkeypatch.setattr(bot, "MOSS_DTYPE", "auto")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        __version__="5.4.0",
        AutoProcessor=SimpleNamespace(from_pretrained=load_processor),
        AutoModel=SimpleNamespace(from_pretrained=load_model),
    ))
    engine = bot.MossSynthesizer()
    engine.load()
    assert calls["codec_device"] == "cuda:1"
    assert calls["model_kwargs"]["device_map"] == {"": 1}
    assert calls["model_kwargs"]["dtype"] == bot.torch.float16
    assert engine.sample_rate == 24_000
    # Without this, transformers 5.3.0 builds DynamicCache(config=...) and
    # MOSS fails with "'MossTTSDelayConfig' object has no attribute 'num_hidden_layers'".
    assert calls["model"].generation_config.cache_implementation == "dynamic_full"


@pytest.mark.parametrize("prompt_length", [30, 2049])
def test_translategemma_official_api_and_token_stream(bot, monkeypatch, prompt_length):
    calls = {}

    class Inputs(dict):
        def to(self, device):
            calls["inputs_device"] = device
            return self

    class Tokenizer:
        def __call__(self, text):
            return {"input_ids": [1, 2, 3]}

    class Processor:
        tokenizer = Tokenizer()

        def apply_chat_template(self, messages, **kwargs):
            calls["messages"] = messages
            calls["template_kwargs"] = kwargs
            return Inputs(input_ids=SimpleNamespace(shape=(1, prompt_length)))

    class Streamer:
        def __init__(self, tokenizer, **kwargs):
            self.items = queue.Queue()

        def end(self):
            self.items.put(None)

        def __iter__(self):
            while True:
                item = self.items.get(timeout=2)
                if item is None:
                    return
                yield item

    class Model:
        device = "cuda:0"
        generation_config = SimpleNamespace(use_cache=False)

        def eval(self):
            return self

        def generate(self, **kwargs):
            calls["generation"] = kwargs
            for piece in ("Guten ", "Morgen."):
                kwargs["streamer"].items.put(piece)
            kwargs["streamer"].end()
            return SimpleNamespace(shape=(1, 34))

    def load_model(model_id, **kwargs):
        calls["model_id"] = model_id
        calls["load_kwargs"] = kwargs
        return Model()

    def reject_qwen(*args, **kwargs):
        raise AssertionError("TranslateGemma must not use the causal-only Qwen loader")

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        __version__="5.4.0", AutoTokenizer=SimpleNamespace(from_pretrained=reject_qwen),
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Processor()),
        AutoModelForImageTextToText=SimpleNamespace(from_pretrained=load_model),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=reject_qwen),
        BitsAndBytesConfig=lambda **kwargs: kwargs, TextIteratorStreamer=Streamer,
    ))
    monkeypatch.setattr(bot, "TRANSLATION_BACKEND", "translategemma")
    monkeypatch.setattr(bot, "TRANSLATION_MODEL", "google/translategemma-12b-it")
    monkeypatch.setattr(bot, "TRANSLATION_4BIT", True)
    engine = bot.Translator(bot.torch.bfloat16, "cuda:0")
    pieces = []
    if prompt_length > 2048:
        with pytest.raises(ValueError, match="not truncated"):
            engine.stream("صباح الخير", [], pieces.append)
        assert "generation" not in calls
        assert not pieces
        return
    engine.stream("صباح الخير", [("old source", "old target")], pieces.append)
    assert "".join(pieces) == "Guten Morgen."
    assert calls["model_id"] == "google/translategemma-12b-it"
    assert calls["load_kwargs"]["device_map"] == {"": 0}
    # Gemma 3 overflows float16, so both the stored activations and the 4-bit
    # matmuls must run in the dtype chosen by translation_dtype().
    assert calls["load_kwargs"]["dtype"] == bot.torch.bfloat16
    assert calls["load_kwargs"]["quantization_config"]["load_in_4bit"]
    assert calls["load_kwargs"]["quantization_config"]["bnb_4bit_compute_dtype"] == bot.torch.bfloat16
    assert engine.dtype == bot.torch.bfloat16
    assert engine.last_hit_budget is False
    assert calls["inputs_device"] == "cuda:0"
    assert calls["messages"] == [{
        "role": "user", "content": [{
            "type": "text", "source_lang_code": "ar", "target_lang_code": "de",
            "text": "صباح الخير",
        }],
    }]
    assert "enable_thinking" not in calls["template_kwargs"]
    assert calls["generation"]["do_sample"] is False
    assert engine.quantized


@pytest.mark.parametrize("backend, setting, bf16_ok, expected", [
    ("translategemma", "auto", True, "bfloat16"),   # T4: Gemma 3 overflows float16
    ("translategemma", "auto", False, "float32"),   # no usable bf16: safe fallback
    ("qwen", "auto", True, "float16"),              # Qwen works in float16; keep it fast
    ("translategemma", "float16", True, "float16"),  # explicit setting wins
    ("qwen", "float32", True, "float32"),
])
def test_translation_dtype_avoids_gemma_float16_overflow(bot, monkeypatch, backend, setting, bf16_ok, expected):
    monkeypatch.setattr(bot, "TRANSLATION_BACKEND", backend)
    monkeypatch.setattr(bot, "TRANSLATION_DTYPE", setting)
    monkeypatch.setattr(bot, "_bf16_works", lambda device: bf16_ok)
    assert bot.translation_dtype(bot.torch.float16, "cuda:0") == getattr(bot.torch, expected)


def test_translation_dtype_keeps_native_bf16_and_cpu_defaults(bot, monkeypatch):
    monkeypatch.setattr(bot, "TRANSLATION_BACKEND", "translategemma")
    monkeypatch.setattr(bot, "TRANSLATION_DTYPE", "auto")
    monkeypatch.setattr(bot, "_bf16_works", lambda device: pytest.fail("no probe needed"))
    assert bot.translation_dtype(bot.torch.bfloat16, "cuda:0") == bot.torch.bfloat16  # A100 etc.
    assert bot.translation_dtype(bot.torch.float32, "cpu") == bot.torch.float32


def test_translation_dtype_rejects_unknown_setting(bot, monkeypatch):
    monkeypatch.setattr(bot, "TRANSLATION_DTYPE", "fp8")
    with pytest.raises(ValueError, match="TRANSLATION_DTYPE"):
        bot.translation_dtype(bot.torch.float16, "cuda:0")


def test_special_token_only_warmup_names_numeric_overflow(bot, monkeypatch):
    """The Kaggle failure: 64 tokens generated, all special, empty text."""
    monkeypatch.setattr(bot, "AsrEngine", lambda *args: SimpleNamespace(transcribe=lambda audio: ""))
    monkeypatch.setattr(bot, "SpeechSynthesizer", lambda: SimpleNamespace(
        load=lambda: None, synthesize=lambda text: np.ones(2400, dtype=np.float32)))
    monkeypatch.setattr(bot, "load_silero_vad", lambda: None)
    monkeypatch.setattr(bot, "Translator", lambda dtype, device: SimpleNamespace(
        translate=lambda text: "", quantized=True, dtype=dtype, last_hit_budget=True))
    with pytest.raises(RuntimeError, match="NaN logits.*TRANSLATION_DTYPE=float32"):
        bot.prepare_models()


def test_qwen_template_keeps_interleaved_context(bot):
    calls = {}

    class Inputs(dict):
        def to(self, device):
            return self

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls["messages"] = messages
            calls["kwargs"] = kwargs
            return Inputs()

    engine = bot.Translator.__new__(bot.Translator)
    engine.backend = "qwen"
    engine.tokenizer = Tokenizer()
    engine.model = SimpleNamespace(device="cpu")
    engine._build_inputs("new source", [("old source", "old target")])
    assert calls["messages"][-3:] == [
        {"role": "user", "content": "old source"},
        {"role": "assistant", "content": "old target"},
        {"role": "user", "content": "new source"},
    ]
    assert calls["kwargs"]["enable_thinking"] is False


def test_pipeline_uses_separate_tts_admission(bot):
    from test_pipeline import FakeAsr, FakeTranslator, FakeSpeech, FakeRoom, RealtimeAudioSource

    async def scenario():
        speech = FakeSpeech(seconds=0.01)
        scheduler = bot.GpuScheduler(1)
        tts_scheduler = bot.GpuScheduler(1)
        pipeline = bot.Pipeline(FakeAsr(), FakeTranslator(), speech, FakeRoom(),
                                RealtimeAudioSource(), scheduler, scheduler, tts_scheduler)
        turn = bot.Turn("speaker", "test-turn", None, 0)
        pipeline.tts_queue.put_nowait((turn, "Guten Morgen.", True))
        async with scheduler.reserve(bot.PRIO_TURN):
            worker = asyncio.create_task(pipeline.run_tts())
            try:
                await asyncio.wait_for(pipeline.tts_queue.join(), timeout=1)
                assert speech.texts == ["Guten Morgen."]
            finally:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(scenario())


def test_smoke_test_saves_real_outputs_and_honest_metrics(bot, monkeypatch, tmp_path):
    writes = {}
    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(
        write=lambda path, audio, rate, **kwargs: writes.update(
            path=path, audio=audio, rate=rate),
    ))
    translator = SimpleNamespace(
        quantized=True, stream=lambda text, context, emit: emit("Guten Morgen."))
    speech = SimpleNamespace(synthesize=lambda text: np.ones(24_000, dtype=np.float32))
    bot.run_smoke_test(None, translator, speech, str(tmp_path))
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert writes["path"] == tmp_path / "german.wav"
    assert writes["rate"] == bot.AUDIO_SAMPLE_RATE
    assert metrics["german_text"] == "Guten Morgen."
    assert metrics["output_audio_seconds"] == 1
    assert metrics["asr_tested_on_supplied_audio"] is False
    assert metrics["asr_seconds"] is None
    assert "NOT live" in metrics["timing_scope"]


@pytest.mark.parametrize("input_seconds, valid", [(1.0, True), (13.0, False)])
def test_smoke_test_uses_supplied_arabic_audio(bot, monkeypatch, tmp_path, input_seconds, valid):
    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(
        read=lambda *args, **kwargs: (np.zeros((int(16_000 * input_seconds), 2)), 16_000),
        write=lambda *args, **kwargs: None,
    ))
    calls = []

    def transcribe(audio):
        calls.append(audio.shape)
        return "نص عربي"

    asr = SimpleNamespace(transcribe=transcribe)
    translator = SimpleNamespace(
        quantized=True, stream=lambda text, context, emit: emit("Guten Morgen."))
    speech = SimpleNamespace(synthesize=lambda text: np.ones(24_000, dtype=np.float32))
    if not valid:
        with pytest.raises(ValueError, match="between 0 and"):
            bot.run_smoke_test(asr, translator, speech, str(tmp_path), "private.wav")
        assert not calls
        return
    bot.run_smoke_test(asr, translator, speech, str(tmp_path), "private.wav")
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["asr_tested_on_supplied_audio"] is True
    assert metrics["input_audio_seconds"] == input_seconds
    assert metrics["asr_seconds"] is not None
    assert metrics["source_text"] == "نص عربي"
    assert calls == [(16_000,)]


def test_kaggle_runner_puts_child_traceback_into_error(tmp_path):
    output = (
        "INFO Loading MOSS-TTS ...\n"
        "Traceback (most recent call last):\n"
        '  File "bot.py", line 1, in <module>\n'
        "torch.OutOfMemoryError: CUDA out of memory on cuda:1\n"
    )

    class Process:
        stdout = io.StringIO(output)

        def poll(self):
            return 1

        def wait(self, timeout):
            return 1

    tree = ast.parse("".join(kaggle_cells()[2]["source"]))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "run_logged"]
    namespace = {
        "WORK": tmp_path, "env": {}, "print": lambda *a, **k: None,
        "sys": SimpleNamespace(executable="python"),
        "subprocess": SimpleNamespace(
            Popen=lambda *args, **kwargs: Process(), DEVNULL=-3, PIPE=-1, STDOUT=-2),
    }
    exec(compile(tree, "kaggle-runner", "exec"), namespace)
    with pytest.raises(RuntimeError) as error:
        namespace["run_logged"](["bot.py", "--smoke-test", "out"], "smoke-test.log")
    message = str(error.value)
    assert "CUDA out of memory on cuda:1" in message
    assert message.index("Traceback (most recent") < message.index("CUDA out of memory")
    assert "Loading MOSS" not in message.split("--- Cause ---")[1]


@pytest.mark.parametrize("exit_code, live, should_raise", [(0, False, False), (1, False, True), (0, True, True)])
def test_kaggle_runner_preserves_logs_and_checks_exit(tmp_path, exit_code, live, should_raise):
    class Process:
        stdout = io.StringIO("model output\n")

        def poll(self):
            return exit_code

        def wait(self, timeout):
            return exit_code

    tree = ast.parse("".join(kaggle_cells()[2]["source"]))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "run_logged"]
    namespace = {
        "WORK": tmp_path, "env": {},
        "sys": SimpleNamespace(executable="python"),
        "subprocess": SimpleNamespace(
            Popen=lambda *args, **kwargs: Process(), DEVNULL=-3, PIPE=-1, STDOUT=-2),
    }
    exec(compile(tree, "kaggle-runner", "exec"), namespace)
    if should_raise:
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            namespace["run_logged"](["bot.py"], "bot.log", live=live)
    else:
        assert namespace["run_logged"](["bot.py"], "bot.log", live=live) == 0
    assert "model output" in (tmp_path / "bot.log").read_text()
    assert f"Exit code: {exit_code}" in (tmp_path / "bot.log").read_text()
