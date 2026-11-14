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


def run_kaggle_install_cell(tmp_path, *, urlopen=None, pip_output="ok\n", pip_code=0, commands=None, envs=None,
                            has_nvcc=True, build_llama=True, streamed=None, printed=None):
    commands = [] if commands is None else commands
    envs = {} if envs is None else envs
    streamed = [] if streamed is None else streamed
    printed = [] if printed is None else printed

    def finish(command):
        if command[:2] == ["cmake", "--build"]:
            server = tmp_path / "llama.cpp" / "build" / "bin" / "llama-server"
            server.parent.mkdir(parents=True, exist_ok=True)
            server.write_text("binary")

    def run(command, **kwargs):
        commands.append(command)
        envs[id(command)] = kwargs.get("env") or {}
        is_pip = command[1:3] == ["-m", "pip"]
        kwargs["stdout"].write(pip_output if is_pip else "ok\n")
        finish(command)
        return SimpleNamespace(returncode=pip_code if is_pip else 0)

    class FakePopen:
        """The streaming variant: output is consumed line by line, not redirected."""

        def __init__(self, command, **kwargs):
            commands.append(command)
            streamed.append(command)
            envs[id(command)] = kwargs.get("env") or {}
            finish(command)
            self.stdout = iter(["[  7%] Building CUDA object ggml-cuda/mmq.cu.o\n", "ok\n"])

        def wait(self):
            return 0

    versions = {"torch": "2.6.0+cu124", "torchaudio": "2.6.0+cu124", "torchvision": "0.21.0+cu124"}
    tree = ast.parse("".join(kaggle_cells()[1]["source"]))
    tree.body = [node for node in tree.body
                 if not isinstance(node, (ast.Import, ast.ImportFrom))
                 and not (isinstance(node, ast.Assign) and any(
                     isinstance(t, ast.Name) and t.id in ("WORK", "LLAMA_DIR", "BUILD_LLAMA_CPP")
                     for t in node.targets))
                 and not isinstance(node, ast.If)]
    namespace = {
        "WORK": tmp_path, "os": SimpleNamespace(environ={}, cpu_count=lambda: 4),
        "print": lambda *a, **k: printed.append(" ".join(str(part) for part in a)),
        "sys": SimpleNamespace(executable="python"), "Path": Path,
        "time": SimpleNamespace(monotonic=lambda: 1e6),
        "LLAMA_DIR": tmp_path / "llama.cpp", "BUILD_LLAMA_CPP": build_llama,
        "shutil": SimpleNamespace(which=lambda name: str(nvcc) if has_nvcc else None,
                                  rmtree=lambda *a, **k: None),
        "subprocess": SimpleNamespace(run=run, Popen=FakePopen, STDOUT=-2, PIPE=-1),
        "metadata": SimpleNamespace(version=versions.__getitem__, PackageNotFoundError=KeyError),
        "urllib": SimpleNamespace(request=SimpleNamespace(
            urlopen=urlopen or (lambda *a, **k: SimpleNamespace(close=lambda: None)))),
    }
    (tmp_path / "gpu-preflight.log").write_text("ok\n")
    nvcc = tmp_path / "nvcc"
    nvcc.write_text("")
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


def test_kaggle_install_builds_llama_server_for_t4_only(tmp_path):
    commands, _ = run_kaggle_install_cell(tmp_path)
    clone = next(c for c in commands if c[:2] == ["git", "clone"])
    assert clone[clone.index("--branch") + 1] == "b12361", "pinned llama.cpp release"
    configure = next(c for c in commands if c[0] == "cmake" and "-S" in c)
    assert "-DGGML_CUDA=ON" in configure
    assert "-DCMAKE_CUDA_ARCHITECTURES=75" in configure, "native T4 code, no driver JIT"
    assert f"-DCMAKE_CUDA_COMPILER={tmp_path / 'nvcc'}" in configure
    # Kaggle ships no libcuda.so link stub, so CMake never defines the
    # CUDA::cuda_driver target llama.cpp links for its VMM pool.
    assert "-DGGML_CUDA_NO_VMM=ON" in configure
    assert "-DLLAMA_USE_PREBUILT_UI=OFF" in configure, "no web UI download during the build"
    build = next(c for c in commands if c[:2] == ["cmake", "--build"])
    assert build[build.index("--target") + 1] == "llama-server"
    server = str(tmp_path / "llama.cpp" / "build" / "bin" / "llama-server")
    assert [server, "--version"] in commands
    assert server == kaggle_literal("PROFILE")["LLAMA_SERVER_BIN"].replace(
        "/tmp/llama.cpp", str(tmp_path / "llama.cpp"))


def test_kaggle_build_prints_progress_so_it_cannot_look_hung(tmp_path):
    """Half an hour of silence is indistinguishable from a crashed build."""
    streamed, printed = [], []
    run_kaggle_install_cell(tmp_path, streamed=streamed, printed=printed)
    assert [command[:2] for command in streamed] == [["cmake", "--build"]], \
        "only the long compile needs live output"
    assert any("20-40 minutes" in line for line in printed), "the wait must be announced"


def test_kaggle_install_error_report_keeps_the_cause_under_the_error_line(tmp_path):
    """CMake names the missing target *below* "CMake Error", so it must survive.

    A filter that keeps only lines containing "error" turns the one message that
    says what is wrong into a line number and nothing else.
    """
    log = (
        "-- Configuring incomplete\n"
        'CMake Error at ggml/src/ggml-cuda/CMakeLists.txt:181 (target_link_libraries):\n'
        '  Target "ggml-cuda" links to:\n'
        "\n"
        "    CUDA::cuda_driver\n"
        "\n"
        "  but the target was not found.\n"
    )
    with pytest.raises(RuntimeError) as error:
        run_kaggle_install_cell(tmp_path, pip_output=log, pip_code=1)
    assert "CUDA::cuda_driver" in str(error.value)
    assert "but the target was not found" in str(error.value)


def test_kaggle_install_refuses_a_saved_version_run(tmp_path):
    """"Save & Run All" has no microphone, so it can only burn GPU quota.

    The guard is a module-level `if`, which run_kaggle_install_cell strips, so
    this checks the source instead of executing it.
    """
    install = "".join(kaggle_cells()[1]["source"])
    assert "KAGGLE_KERNEL_RUN_TYPE" in install
    assert "Quick Save" in install
    # The GPU check has to name every realistic cause, not just the setting.
    assert "weekly GPU quota" in install and "Accelerator" in install


def test_kaggle_install_reuses_build_and_can_skip_it(tmp_path):
    first, _ = run_kaggle_install_cell(tmp_path)
    assert any(c[:2] == ["cmake", "--build"] for c in first)
    again, _ = run_kaggle_install_cell(tmp_path)
    assert not any(c[0] in ("git", "cmake") for c in again), "same session: reuse the binary"
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    skipped, _ = run_kaggle_install_cell(fallback, build_llama=False)
    assert not any(c[0] in ("git", "cmake") for c in skipped)


def test_kaggle_install_without_nvcc_names_the_fallback(tmp_path):
    if Path("/usr/local/cuda/bin/nvcc").exists():
        pytest.skip("this machine has a real CUDA compiler")
    with pytest.raises(RuntimeError, match="nvcc is missing(.|\n)*TRANSLATION_ENGINE"):
        run_kaggle_install_cell(tmp_path, has_nvcc=False)


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


def kaggle_literal(name):
    tree = ast.parse("".join(kaggle_cells()[2]["source"]))
    node = next(node for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
    return ast.literal_eval(node.value)


def test_kaggle_profile_splits_gpus_and_runs_phonon_on_cpu():
    profile = kaggle_literal("PROFILE")
    # GPU 0 belongs to the 8-bit translator alone, GPU 1 to the ASR.
    assert profile["ASR_DEVICE"] == "cuda:1" and profile["TRANSLATION_DEVICE"] == "cuda:0"
    assert profile["TRANSLATION_MODEL"] == "google/translategemma-12b-it"
    assert profile["TRANSLATION_BACKEND"] == "translategemma"
    assert profile["TRANSLATION_ENGINE"] == "llamacpp"
    assert profile["LLAMA_GGUF_REPO"] == "mradermacher/translategemma-12b-it-GGUF"
    assert profile["LLAMA_GGUF_FILE"] == "translategemma-12b-it.Q8_0.gguf"
    assert profile["LLAMA_THREADS"] == "1"
    assert profile["TRANSLATION_4BIT"] == "1", "kept for the transformers fallback"
    # Gemma now gets discourse context too, but only two turns: they share its
    # 2048-token input window with the source and are re-prefilled every turn.
    assert profile["TRANSLATION_CONTEXT_TURNS"] == "2"
    # A backlog is absorbed by speaking faster before any turn is dropped. The
    # measured expansion sits at ~1.0x, so there is no slack and the ramp has
    # to start almost at once.
    assert profile["MAX_PLAYOUT_SPEED"] == "1.35"
    assert profile["SPEEDUP_BACKLOG_SECONDS"] == "1"
    assert profile["GPU_CONCURRENCY"] == "1"
    assert profile["TTS_BACKEND"] == "phonon" and profile["TTS_DEVICE"] == "cpu"
    assert profile["PHONON_MODEL"] == "hf://gradium/phonon-40m-de"
    assert profile["PHONON_THREADS"] == "2" and profile["TORCH_CPU_THREADS"] == "1"
    install = "".join(kaggle_cells()[1]["source"])
    assert "'ptts'" in install and "import ptts" in install
    notebook = (ROOT / "kaggle_agent.ipynb").read_text().lower()
    assert "moss" not in notebook, "MOSS was removed"


def test_kaggle_access_check_covers_all_gated_models():
    """The gated models must fail fast, before tens of GB are downloaded."""
    tree = ast.parse("".join(kaggle_cells()[2]["source"]))
    start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", None) == "ACCESS_REPOS" for t in n.targets))
    calls = []
    namespace = {"PROFILE": kaggle_literal("PROFILE"),
                 "run_logged": lambda arguments, log_name, **kwargs: calls.append(arguments)}
    for node in tree.body[start:]:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "access", "exec"), namespace)
        if calls:
            break
    assert calls[0][2:] == ["CohereLabs/cohere-transcribe-arabic-07-2026",
                            "google/translategemma-12b-it", "gradium/phonon-40m-de"]
    compile(calls[0][1], "access-check", "exec")


def test_phonon_default_puts_tts_on_its_own_cpu_scheduler():
    """Without TTS_DEVICE, Phonon must not queue behind the GPU 0 slot."""
    import subprocess

    code = (
        "import os, sys; sys.path.insert(0, %r); import stubs; stubs.install(); "
        "os.environ.pop('TTS_DEVICE', None); os.environ.pop('TTS_BACKEND', None); "
        "sys.path.insert(0, %r); import bot; print(bot.TTS_BACKEND, bot.TTS_DEVICE)"
    ) % (str(ROOT / "test"), str(ROOT))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-2:] == ["phonon", "cpu"]


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


@pytest.mark.parametrize("prompt_length", [30, 2049])
def test_translategemma_official_api_and_token_stream(bot, monkeypatch, prompt_length):
    calls = {}

    class Inputs(dict):
        def to(self, device):
            calls["inputs_device"] = device
            return self

    class Tokenizer:
        bos_token = "<bos>"

        def __call__(self, text, **kwargs):
            if kwargs.get("return_tensors") != "pt":
                return {"input_ids": [1, 2, 3]}
            calls["tokenizer_kwargs"] = kwargs
            return Inputs(input_ids=SimpleNamespace(shape=(1, prompt_length)))

    class Processor:
        tokenizer = Tokenizer()

        def apply_chat_template(self, messages, **kwargs):
            calls["messages"] = messages
            calls["template_kwargs"] = kwargs
            return "<bos>rendered prompt"

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
    # The previous turn travels as a user/assistant pair, because the template
    # has no system role: without it every VAD fragment loses its discourse.
    # Its assistant role takes a plain string, not a content list.
    assert calls["messages"] == [
        {"role": "user", "content": [{
            "type": "text", "source_lang_code": "ar", "target_lang_code": "de",
            "text": "old source",
        }]},
        {"role": "assistant", "content": "old target"},
        {"role": "user", "content": [{
            "type": "text", "source_lang_code": "ar", "target_lang_code": "de",
            "text": "صباح الخير",
        }]},
    ]
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


class GemmaTemplate:
    """Enforces the two contracts the real stack imposes, which disagree.

    Verified against google/translategemma-12b-it's chat_template.jinja and
    transformers' ProcessorMixin.apply_chat_template:

    - the template requires the assistant role to carry a plain string, a user
      content list with exactly one item, and strictly alternating roles
      starting with the user;
    - the processor's `tokenize=True` path scans every message for images and
      indexes each content entry, which a plain string cannot survive.

    So only rendering text-only satisfies both.
    """

    def __init__(self, tokens_per_message: int = 100) -> None:
        self.tokens_per_message = tokens_per_message
        self.rendered: list[list[dict]] = []

    def apply_chat_template(self, messages, **kwargs):
        if kwargs.get("tokenize", True):
            for message in messages:
                for content in message["content"]:
                    content["type"]  # a string raises TypeError here, as in transformers
        if messages[0]["role"] != "user":
            raise ValueError("Conversations must start with a user prompt.")
        for index, message in enumerate(messages):
            if (message["role"] == "user") != (index % 2 == 0):
                raise ValueError("Conversation roles must alternate user/assistant/...")
            if message["role"] == "assistant":
                if not isinstance(message["content"], str):
                    raise ValueError("Assistant role must provide content as a string")
            elif not isinstance(message["content"], list) or len(message["content"]) != 1:
                raise ValueError("User role must provide `content` as an iterable with one item.")
        self.rendered.append(messages)
        return "<bos>" + "|".join(str(message["role"]) for message in messages)


class GemmaTokenizer:
    bos_token = "<bos>"

    def __init__(self, template: GemmaTemplate) -> None:
        self.template = template
        self.calls: list[dict] = []

    def __call__(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        length = self.template.tokens_per_message * (text.count("|") + 1)
        return {"input_ids": SimpleNamespace(shape=(1, length))}


def gemma_owner(bot, template=None):
    template = template if template is not None else GemmaTemplate()
    owner = bot.Translator.__new__(bot.Translator)
    owner.backend = "translategemma"
    owner.processor = template
    owner.tokenizer = GemmaTokenizer(template)
    return owner


def test_gemma_prompt_is_rendered_text_only_then_tokenized(bot):
    """The processor's tokenizing path cannot handle the template's own format.

    It indexes every content entry looking for images; the template demands a
    plain string on the assistant role. Rendering text-only is the only way to
    satisfy both, and the template already emits the BOS token.
    """
    owner = gemma_owner(bot)
    bot._gemma_inputs(owner, "now", [("src", "tgt")])

    assert owner.processor.rendered, "the official template was not used"
    assert owner.tokenizer.calls[-1]["add_special_tokens"] is False, "BOS would be doubled"
    assert owner.tokenizer.calls[-1]["text"].startswith("<bos>")


def test_gemma_context_is_trimmed_to_its_input_window(bot):
    """Gemma's input is 2048 tokens; the oldest turn is the cheapest to lose."""
    template = GemmaTemplate(tokens_per_message=500)
    context = [("src1", "tgt1"), ("src2", "tgt2"), ("src3", "tgt3")]
    bot._gemma_inputs(gemma_owner(bot, template), "now", context)

    assert [len(attempt) for attempt in template.rendered] == [7, 5, 3], "oldest turns dropped first"
    assert template.rendered[-1][0]["content"][0]["text"] == "src3", "the newest context survives"
    assert template.rendered[-1][-1]["content"][0]["text"] == "now"


def test_gemma_falls_back_to_context_free_if_the_template_refuses(bot, monkeypatch, caplog):
    """A template that rejects earlier turns must not kill every translation."""
    monkeypatch.setattr(bot, "_gemma_context_supported", True)
    monkeypatch.setattr(bot, "_gemma_context_error", "")

    class Refusing(GemmaTemplate):
        def apply_chat_template(self, messages, **kwargs):
            if len(messages) > 1:
                raise ValueError("only a single user turn is supported")
            return super().apply_chat_template(messages, **kwargs)

    template = Refusing()
    owner = gemma_owner(bot, template)
    with caplog.at_level("WARNING", logger="asr-bot"):
        bot._gemma_inputs(owner, "now", [("src", "tgt")])
    # The cause has to travel with the warning; "see the log above" cost a run.
    assert "only a single user turn is supported" in caplog.text
    assert bot._gemma_context_error.endswith("only a single user turn is supported")
    assert len(template.rendered[-1]) == 1
    # The probe runs once, not on every turn.
    bot._gemma_inputs(owner, "later", [("src", "tgt")])
    assert [len(attempt) for attempt in template.rendered] == [1, 1]


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
        quantized=True, stream=lambda text, context, emit: emit("Guten Morgen."),
        translate=lambda text, context=(): "Guten Morgen.")
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


def test_smoke_test_covers_the_rolling_context_prompt(bot, monkeypatch, tmp_path):
    """The context is its own prompt path, and only the real model can judge it.

    The live agent reaches it from the second turn on, so a smoke test that
    only ever translates with an empty history cannot catch a template that
    rejects earlier turns - which is exactly how that bug reached a live run.
    """
    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(
        write=lambda *args, **kwargs: None))
    contexts = []
    translator = SimpleNamespace(
        quantized=True, stream=lambda text, context, emit: emit("Guten Morgen."),
        translate=lambda text, context=(): (contexts.append(list(context)), "Mit Kontext.")[1])
    speech = SimpleNamespace(synthesize=lambda text: np.ones(24_000, dtype=np.float32))

    bot.run_smoke_test(None, translator, speech, str(tmp_path))

    assert contexts == [[("الحمد لله رب العالمين. نسأل الله أن يرزقنا العلم النافع والعمل الصالح.",
                          "Guten Morgen.")]]
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["german_text_with_context"] == "Mit Kontext."


def test_smoke_test_fails_if_gemma_silently_dropped_the_context(bot, monkeypatch, tmp_path):
    """Degrading quietly is right in a live room, but wrong in a pre-flight check."""
    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(
        write=lambda *args, **kwargs: None))
    monkeypatch.setattr(bot, "TRANSLATION_BACKEND", "translategemma")
    monkeypatch.setattr(bot, "_gemma_context_supported", False)
    translator = SimpleNamespace(
        quantized=True, stream=lambda text, context, emit: emit("Guten Morgen."),
        translate=lambda text, context=(): "Ohne Kontext.")
    speech = SimpleNamespace(synthesize=lambda text: np.ones(24_000, dtype=np.float32))

    with pytest.raises(RuntimeError, match="rejected the rolling context"):
        bot.run_smoke_test(None, translator, speech, str(tmp_path))


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
        quantized=True, stream=lambda text, context, emit: emit("Guten Morgen."),
        translate=lambda text, context=(): "Guten Morgen.")
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
        "INFO Loading Phonon ...\n"
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
    assert "Loading Phonon" not in message.split("--- Cause ---")[1]


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
