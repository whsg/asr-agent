#!/bin/bash
set -euo pipefail

if [[ "${TRANSLATION_ENGINE:-transformers}" != "llamacpp" ]]; then
    exec python bot.py "$@"
fi

LLAMA_TAG="${LLAMA_TAG:-b12361}"
LLAMA_DIR="${LLAMA_DIR:-/tmp/llama.cpp}"
LLAMA_SERVER="${LLAMA_DIR}/build/bin/llama-server"

find_precompiled_server() {
    local candidate

    if [[ -n "${LLAMA_SERVER_BIN:-}" ]] && [[ "${LLAMA_SERVER_BIN}" != "llama-server" ]] && [[ -x "${LLAMA_SERVER_BIN}" ]]; then
        printf '%s\n' "${LLAMA_SERVER_BIN}"
        return 0
    fi

    if command -v llama-server >/dev/null 2>&1; then
        command -v llama-server
        return 0
    fi

    for candidate in \
        "/workspace/bin/llama-server" \
        "/opt/venv/bin/llama-server" \
        "/usr/local/bin/llama-server" \
        "/app/bin/llama-server" \
        "${LLAMA_SERVER}"
    do
        if [[ -x "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    return 1
}

if LLAMA_SERVER="$(find_precompiled_server)"; then
    export LLAMA_SERVER_BIN="${LLAMA_SERVER}"
    echo "Using precompiled llama-server: ${LLAMA_SERVER_BIN}"
    exec python bot.py "$@"
fi

LLAMA_SERVER_DOWNLOAD_URL="${LLAMA_SERVER_DOWNLOAD_URL:-https://github.com/ggml-org/llama.cpp/releases/download/b12361/llama-b12361-bin-ubuntu-cuda-12.8-x64.tar.gz}"

if [[ -n "${LLAMA_SERVER_DOWNLOAD_URL:-}" ]]; then
    echo "Downloading precompiled llama-server..."
    DOWNLOAD_DIR="/tmp/llama-server-download"
    rm -rf "${DOWNLOAD_DIR}"
    mkdir -p "${DOWNLOAD_DIR}"
    DOWNLOAD_FILE="${DOWNLOAD_DIR}/llama-server-package"

    curl -fL --retry 3 "${LLAMA_SERVER_DOWNLOAD_URL}" -o "${DOWNLOAD_FILE}"

    case "${LLAMA_SERVER_DOWNLOAD_URL}" in
        *.zip)
            unzip -q "${DOWNLOAD_FILE}" -d "${DOWNLOAD_DIR}"
            ;;
        *.tar.gz|*.tgz)
            tar -xzf "${DOWNLOAD_FILE}" -C "${DOWNLOAD_DIR}"
            ;;
        *.tar)
            tar -xf "${DOWNLOAD_FILE}" -C "${DOWNLOAD_DIR}"
            ;;
        *)
            chmod +x "${DOWNLOAD_FILE}"
            export LLAMA_SERVER_BIN="${DOWNLOAD_FILE}"
            echo "Using downloaded llama-server: ${LLAMA_SERVER_BIN}"
            exec python bot.py "$@"
            ;;
    esac

    LLAMA_SERVER="$(find "${DOWNLOAD_DIR}" -type f -name 'llama-server' -perm -u+x | head -n 1)"
    if [[ -n "${LLAMA_SERVER}" ]]; then
        export LLAMA_SERVER_BIN="${LLAMA_SERVER}"
        echo "Using downloaded llama-server: ${LLAMA_SERVER_BIN}"
        exec python bot.py "$@"
    fi

    echo "The downloaded archive did not contain an executable llama-server; falling back to source build." >&2
fi

if [[ ! -x "${LLAMA_SERVER}" ]]; then
    echo "Building llama-server for this Runpod GPU..."
    if [[ -z "$(command -v nvcc || true)" ]]; then
        echo "nvcc is missing. Use a CUDA-devel container or set LLAMA_SERVER_BIN." >&2
        exit 1
    fi

    rm -rf "${LLAMA_DIR}"
    git clone --depth 1 --branch "${LLAMA_TAG}" \
        https://github.com/ggml-org/llama.cpp "${LLAMA_DIR}"

    cmake -S "${LLAMA_DIR}" -B "${LLAMA_DIR}/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=native \
        -DGGML_CUDA_FA=OFF \
        -DGGML_CUDA_NO_VMM=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_OPENSSL=OFF \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_USE_PREBUILT_UI=OFF

    cmake --build "${LLAMA_DIR}/build" --target llama-server -j
fi

export LLAMA_SERVER_BIN="${LLAMA_SERVER}"
echo "Using llama-server: ${LLAMA_SERVER_BIN}"
exec python bot.py "$@"
