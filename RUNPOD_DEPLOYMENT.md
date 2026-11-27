# Deploy this app on Runpod

This app runs continuously as a **Runpod Pod**, not as a Serverless endpoint.
Choose either the Runpod Base setup below (no image build; starts both the bot
and web backend) or the Docker image path that follows.

## Runpod Base (no Docker image)

1. Deploy a GPU Pod using Runpod Base for **Ubuntu 24.04 with CUDA 12.8.1**.
   Clone or copy this repository to `/workspace/asr-agent` and use persistent
   workspace storage if available. Upload the source and `.env` separately;
   do not upload a local `.venv` or model cache.
2. Add these values in the Pod's environment settings. Store credentials as
   secrets; do not put them in the Dockerfile or commit them:
   - `LIVEKIT_URL`
   - `LIVEKIT_API_KEY`
   - `LIVEKIT_API_SECRET`
   - `HF_TOKEN`
   - Optionally, `LIVEKIT_ROOM` (defaults to `arabic-asr-live`)

   Before first startup, accept Hugging Face access for
   [Cohere Transcribe Arabic](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026)
   and [Gradium Phonon German](https://huggingface.co/gradium/phonon-40m-de)
   using the account that owns `HF_TOKEN`.

   Alternatively, fill in the blank, git-ignored `.env` file in your local
   repository root (`notepad .env` on Windows), then upload it to
   `/workspace/asr-agent/.env`. Pod environment variables override values from
   `.env`. Transfer it using the Pod's SSH details, for example:

   ```powershell
   scp -P PORT -i "C:\path\to\ssh-key" .env USER@HOST:/workspace/asr-agent/.env
   ```

   Replace `PORT`, the key path, `USER`, and `HOST` with the SSH details shown
   for your Pod. After transfer, run `chmod 600 /workspace/asr-agent/.env` in
   the Pod terminal. Do not upload `.env` to GitHub or share it in chat.
3. Expose port `8000` as HTTP in the Pod settings.
4. In the Pod terminal, run:

   ```bash
   cd /workspace/asr-agent
   bash runpod-start.sh
   ```

On the first run, the script installs system and Python dependencies, checks
that PyTorch can use the GPU, then starts the bot and FastAPI backend together.
The bot downloads its models on first startup. Hugging Face model files are
stored in `/workspace/.cache/huggingface`. The script uses the bot's code
defaults (Qwen3 4B with Transformers); Dockerfile
environment defaults are not applied to a Runpod Base Pod.

Open the Pod's HTTP proxy URL for the web client; `/api/health` checks the web
backend. The backend is intentionally public in this setup, but `/api/session`
has no authentication and issues 30-minute LiveKit participant tokens to
anyone who can reach it. Do not expose it to untrusted users without adding
access control and rate limiting.

## Docker image path

### Prerequisites

You need:

- a Runpod account;
- Docker installed locally;
- a Docker Hub, GitHub Container Registry, or other container-registry account;
- a LiveKit Cloud project;
- a Hugging Face account and token;
- access granted to the required gated models.

Accept the Hugging Face access terms for:

- [Cohere Transcribe Arabic](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026)
- [TranslateGemma](https://huggingface.co/google/translategemma-12b-it)
- [Gradium Phonon German](https://huggingface.co/gradium/phonon-40m-de)

Create a read-capable Hugging Face token at
<https://huggingface.co/settings/tokens>.

## 1. Build the Docker image

Run this command from the repository root:

```powershell
$Image = "YOUR_REGISTRY/asr-agent:runpod"
docker build --platform linux/amd64 -t $Image .
```

Replace `YOUR_REGISTRY` with your Docker Hub username or registry namespace.

Do not rely on a locally built Windows image. Runpod requires the Linux AMD64
image specified by `--platform linux/amd64`.

## 2. Push the image

Log in and push the same image:

```powershell
docker login
docker push $Image
```

If the registry is private, configure the registry credentials in the Runpod
Pod template or deployment form before launching the Pod.

## 3. Create the Runpod Pod

1. Open the Runpod console.
2. Select **Pods**.
3. Click **Deploy** or **New Pod**.
4. Choose a custom template or set a custom container image.
5. Enter the image:

   ```text
   docker.io/<YOUR_REGISTRY>/asr-agent:runpod
   ```

   Replace `<YOUR_REGISTRY>` with your Docker Hub username or registry
   namespace.

6. Select a GPU with at least **48 GB of VRAM**.
7. Configure enough container and volume storage for downloaded model weights,
   temporary llama.cpp files, and logs.
8. Do not expose any HTTP or TCP port. The bot uses outbound LiveKit
   connections and does not need an exposed public port.
9. Do not enable Serverless. This deployment must remain a Pod.

### GPU selection

The default image runs both Cohere ASR and TranslateGemma 12B on one GPU.

- A 48 GB GPU is the documented starting point.
- ASR and translation share `cuda:0` in this default profile.
- TTS runs on the CPU.
- Use a Secure Cloud GPU for the most predictable network and GPU behavior.

GPU availability and model memory can change. If the model or llama.cpp server
fails to load, use a larger GPU or switch to the transformers fallback profile.

## 4. Add environment variables

In the Runpod Pod deployment form, open **Environment Variables** or
**Template Variables**, then add each variable below.

Replace all placeholder values. Do not add quotes around values.

### Required values

| Variable | Example | Purpose |
|---|---|---|
| `HF_TOKEN` | `hf_xxxxxxxxx` | Accesses gated Hugging Face models |
| `LIVEKIT_URL` | `wss://your-project.livekit.cloud` | LiveKit project WebSocket URL |
| `LIVEKIT_API_KEY` | `APIxxxxxxxx` | LiveKit API key |
| `LIVEKIT_API_SECRET` | `xxxxxxxxxxxxxxxx` | LiveKit API secret |
| `LIVEKIT_ROOM` | `arabic-asr-live` | Room joined by the bot |

Add the required values in Runpod using:

```text
HF_TOKEN=hf_xxxxxxxxx
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=APIxxxxxxxx
LIVEKIT_API_SECRET=xxxxxxxxxxxxxxxx
LIVEKIT_ROOM=arabic-asr-live
```

Use the exact same `LIVEKIT_URL`, `LIVEKIT_API_KEY`,
`LIVEKIT_API_SECRET`, and `LIVEKIT_ROOM` values in the local `.env` used by
the web client.

### Recommended model values

The Docker image already sets these defaults. Add them explicitly if you are
creating a custom Runpod template or overriding an earlier template:

```text
TRANSLATION_ENGINE=llamacpp
TRANSLATION_BACKEND=translategemma
TRANSLATION_MODEL=google/translategemma-12b-it
LLAMA_GGUF_REPO=mradermacher/translategemma-12b-it-GGUF
LLAMA_GGUF_FILE=translategemma-12b-it.Q8_0.gguf
LLAMA_CTX=3072
LLAMA_THREADS=1
ASR_DEVICE=cuda:0
TRANSLATION_DEVICE=cuda:0
TTS_DEVICE=cpu
TTS_BACKEND=phonon
GPU_CONCURRENCY=1
TORCH_CPU_THREADS=1
PHONON_THREADS=2
COHERE_4BIT=0
```

If you use an `environment_variables` YAML or API structure instead of the
Runpod console, its format is equivalent to:

```yaml
HF_TOKEN: hf_xxxxxxxxx
LIVEKIT_URL: wss://your-project.livekit.cloud
LIVEKIT_API_KEY: APIxxxxxxxx
LIVEKIT_API_SECRET: xxxxxxxxxxxxxxxx
LIVEKIT_ROOM: arabic-asr-live
TRANSLATION_ENGINE: llamacpp
TRANSLATION_BACKEND: translategemma
TRANSLATION_MODEL: google/translategemma-12b-it
ASR_DEVICE: cuda:0
TRANSLATION_DEVICE: cuda:0
TTS_DEVICE: cpu
TTS_BACKEND: phonon
GPU_CONCURRENCY: "1"
```

### Optional overrides

```text
LOG_LEVEL=INFO
TRANSLATION_CONTEXT_TURNS=4
INTERIM_CAPTIONS=1
STREAM_TTS=1
MAX_UTTERANCE_SECONDS=12
MAX_GEN_TOKENS=448
```

Keep `HF_TOKEN`, LiveKit credentials, and any other secrets out of the
Dockerfile and source repository. Set them only through Runpod environment
variables or secrets.

## 5. Configure the container command

The Docker image has an entrypoint that starts the bot. Leave the container
command and entrypoint at their image defaults.

The entrypoint:

1. looks for a llama.cpp server executable;
2. downloads a precompiled CUDA llama.cpp package when available;
3. falls back to building llama.cpp for the detected GPU; and
4. starts `bot.py`.

Do not set a shell command such as `bash`, `sleep infinity`, or
`python backend/main.py` unless you intentionally want to override bot startup.

## 6. Deploy and verify startup

Deploy the Pod, wait for it to enter the `Running` state, and open its
**Logs**.

The startup sequence normally reports:

- the detected GPU;
- ASR model loading;
- TTS warm-up;
- translator loading;
- LiveKit credential validation;
- `READY: Bot is in room ...`

The final readiness line is:

```text
READY: Bot is in room 'arabic-asr-live' ...
```

If it does not appear:

### Missing Hugging Face access

Check the log for model access or 401/403 errors. Verify that:

- `HF_TOKEN` is correct;
- the token belongs to an account that accepted each model's terms;
- the token has access to the gated repositories.

### Missing LiveKit credentials

Verify `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, and
`LIVEKIT_ROOM`.

The URL must use `wss://` for a LiveKit Cloud project:

```text
wss://your-project.livekit.cloud
```

### Out of memory

Try:

```text
COHERE_4BIT=1
TRANSLATION_4BIT=1
GPU_CONCURRENCY=1
```

This is slower and lower quality for the translation model. The llama.cpp
profile explicitly refuses CPU offload instead of silently slowing down.

Alternatively, use the transformers fallback:

```text
TRANSLATION_ENGINE=transformers
```

Remove or omit `LLAMA_GGUF_REPO` and `LLAMA_GGUF_FILE` when testing the
transformers fallback.

### llama.cpp startup failure

Check for a CUDA architecture or binary compatibility error. The entrypoint
can fall back to a source build, which takes several minutes on the first
startup. The container's `/tmp/llama.cpp` build is ephemeral and may need to
be rebuilt after the Pod restarts or a new container disk is created.

## 7. Start the web client

On your PC, create the matching `.env` file:

```text
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=APIxxxxxxxx
LIVEKIT_API_SECRET=xxxxxxxxxxxxxxxx
LIVEKIT_ROOM=arabic-asr-live
```

Install and run it:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>, enable German speech if desired, and press
**Start microphone**.

The LiveKit bot and browser client must use the same project and room.

## Important model-cache behavior

The current Docker image does not pre-download model weights or GGUF files.
Hugging Face and llama.cpp downloads happen at Pod startup:

- `HF_HOME=/tmp/huggingface`
- `HF_HUB_CACHE=/tmp/huggingface/hub`
- llama.cpp files are downloaded or built under `/tmp`

This means:

- cold starts may take several minutes;
- restarting the container can repeat large downloads;
- attaching only `/workspace` does not preserve `/tmp` model caches;
- a network volume must be configured explicitly and mounted by the application
  to provide persistent caching.
