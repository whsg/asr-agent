FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/huggingface \
    HF_HUB_CACHE=/tmp/huggingface/hub \
    TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    TMPDIR=/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        git \
        python3 \
        python3-pip \
        python3-venv \
        unzip \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app
COPY requirements-agent.txt ./
RUN python3 -m pip install -r requirements-agent.txt

COPY bot.py ./
COPY runpod-entrypoint.sh ./
RUN chmod +x runpod-entrypoint.sh

ENV TRANSLATION_ENGINE=llamacpp \
    TRANSLATION_BACKEND=translategemma \
    TRANSLATION_MODEL=google/translategemma-27b-it \
    LLAMA_GGUF_REPO=mradermacher/translategemma-27b-it-GGUF \
    LLAMA_GGUF_FILE=translategemma-27b-it.Q8_0.gguf \
    LLAMA_CTX=3072 \
    LLAMA_THREADS=1 \
    LLAMA_SERVER_LOG=/tmp/llama-server.log \
    ASR_DEVICE=cuda:0 \
    TRANSLATION_DEVICE=cuda:0 \
    TTS_DEVICE=cpu \
    TTS_BACKEND=phonon \
    GPU_CONCURRENCY=1 \
    TORCH_CPU_THREADS=1 \
    PHONON_THREADS=2 \
    COHERE_4BIT=0

ENTRYPOINT ["./runpod-entrypoint.sh"]
