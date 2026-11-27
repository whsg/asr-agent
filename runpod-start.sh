#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Optional quick-test credentials: uncomment and replace every placeholder.
# Keep real values local to the Pod; do not commit them to the branch.
export LIVEKIT_URL="wss://sts-spxqczbe.livekit.cloud"
export LIVEKIT_API_KEY="APID2mWegf8UgVL"
export LIVEKIT_API_SECRET="4u8tFhhSZeFO9nOXrtnAT3km3O0zfO265rWPZM6FIftB"
export HF_TOKEN="hf_YOUR_HUGGINGFACE_TOKEN"

if [[ -z "${LIVEKIT_URL:-}" || -z "${LIVEKIT_API_KEY:-}" || -z "${LIVEKIT_API_SECRET:-}" || -z "${HF_TOKEN:-}" ]] \
    && [[ ! -f "$PROJECT_ROOT/.env" ]]; then
    echo "Set the required credentials in the Pod environment or provide a .env file in the project root." >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python was not found: $PYTHON_BIN" >&2
    exit 2
fi
if ! "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
    echo "Use the Runpod Base Ubuntu 24.04 image with Python 3.12." >&2
    exit 2
fi

if [[ "$EUID" -eq 0 ]]; then
    APT=(apt-get)
elif command -v sudo >/dev/null 2>&1; then
    APT=(sudo apt-get)
else
    echo "System packages must be installed as root (or with sudo)." >&2
    exit 2
fi

"${APT[@]}" update
DEBIAN_FRONTEND=noninteractive "${APT[@]}" install -y --no-install-recommends \
    build-essential cmake curl ffmpeg git libsndfile1 python3.12-venv unzip

VENV_DIR="${RUNPOD_VENV_DIR:-$PROJECT_ROOT/.venv}"
export PIP_NO_CACHE_DIR=1

mkdir -p "$VENV_DIR"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
VENV_PYTHON="$VENV_DIR/bin/python"
export PATH="$VENV_DIR/bin:$PATH"

"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
"$VENV_PYTHON" -m pip install python-dotenv

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    dotenv_exports="$("$VENV_PYTHON" - "$PROJECT_ROOT/.env" <<'PY'
import os
import re
import shlex
import sys
from dotenv import dotenv_values

for key, value in dotenv_values(sys.argv[1]).items():
    if value is not None and key not in os.environ:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise SystemExit(f"Invalid environment variable name in .env: {key}")
        print(f"export {key}={shlex.quote(value)}")
PY
)"
    # Pod environment values take precedence; dotenv values are shell-quoted.
    eval "$dotenv_exports"
fi

if [[ -z "${LIVEKIT_URL:-}" || -z "${LIVEKIT_API_KEY:-}" || -z "${LIVEKIT_API_SECRET:-}" || -z "${HF_TOKEN:-}" ]]; then
    echo "Missing required values: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, and HF_TOKEN." >&2
    exit 2
fi

for name in LIVEKIT_URL LIVEKIT_API_KEY LIVEKIT_API_SECRET HF_TOKEN; do
    if [[ "${!name}" == *YOUR_* || "${!name}" == *PLACEHOLDER* ]]; then
        echo "Replace the placeholder value for $name before starting." >&2
        exit 2
    fi
done

if ! "$VENV_PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    "$VENV_PYTHON" -m pip install --upgrade --force-reinstall \
        --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
fi

if ! "$VENV_PYTHON" -c 'import torch; print(f"PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}"); raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    echo "CUDA is unavailable. Use a GPU Pod with the Runpod Base CUDA 12.8 image." >&2
    exit 1
fi

"$VENV_PYTHON" -m pip install -r requirements-agent.txt -r requirements.txt
"$VENV_PYTHON" -m pip check

HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/workspace/.cache/torch_extensions}"
export HF_HOME TORCH_EXTENSIONS_DIR
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export LIVEKIT_ROOM="${LIVEKIT_ROOM:-arabic-asr-live}"
export BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
export BACKEND_PORT="${BACKEND_PORT:-8000}"
export TRANSLATION_ENGINE="${TRANSLATION_ENGINE:-llamacpp}"
export TRANSLATION_BACKEND="${TRANSLATION_BACKEND:-translategemma}"
export TRANSLATION_MODEL="${TRANSLATION_MODEL:-google/translategemma-12b-it}"
export LLAMA_GGUF_REPO="${LLAMA_GGUF_REPO:-mradermacher/translategemma-12b-it-GGUF}"
export LLAMA_GGUF_FILE="${LLAMA_GGUF_FILE:-translategemma-12b-it.Q8_0.gguf}"
export LLAMA_CTX="${LLAMA_CTX:-3072}"
export LLAMA_THREADS="${LLAMA_THREADS:-1}"
export LLAMA_SERVER_LOG="${LLAMA_SERVER_LOG:-/tmp/llama-server.log}"
export ASR_DEVICE="${ASR_DEVICE:-cuda:0}"
export TRANSLATION_DEVICE="${TRANSLATION_DEVICE:-cuda:0}"
export TTS_DEVICE="${TTS_DEVICE:-cpu}"
export TTS_BACKEND="${TTS_BACKEND:-phonon}"
export GPU_CONCURRENCY="${GPU_CONCURRENCY:-1}"
export TORCH_CPU_THREADS="${TORCH_CPU_THREADS:-1}"
export PHONON_THREADS="${PHONON_THREADS:-2}"
export COHERE_4BIT="${COHERE_4BIT:-0}"

if [[ ! "$BACKEND_PORT" =~ ^[0-9]{1,5}$ ]] || (( 10#$BACKEND_PORT < 1 || 10#$BACKEND_PORT > 65535 )); then
    echo "BACKEND_PORT must be an integer from 1 to 65535." >&2
    exit 2
fi

mkdir -p "$HF_HOME" "$TORCH_EXTENSIONS_DIR"

echo "Starting the bot and FastAPI backend on ${BACKEND_HOST}:${BACKEND_PORT}."
echo "WARNING: /api/session is unauthenticated and issues LiveKit participant tokens."

bash "$PROJECT_ROOT/runpod-entrypoint.sh" &
BOT_PID=$!
"$VENV_PYTHON" -m uvicorn backend.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" &
API_PID=$!

cleanup() {
    for pid in "$BOT_PID" "$API_PID"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait "$BOT_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

set +e
wait -n "$BOT_PID" "$API_PID"
status=$?
set -e

if [[ "$status" -eq 0 ]]; then
    echo "A service exited unexpectedly." >&2
    status=1
fi
exit "$status"
