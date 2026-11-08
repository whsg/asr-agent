import os
import secrets
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from livekit import api
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
if not ENV_FILE.exists():
    ENV_FILE = PROJECT_ROOT.parent / ".env"
load_dotenv(ENV_FILE)

ROOM_NAME = os.getenv("LIVEKIT_ROOM", "qwen-asr-live")


class SessionResponse(BaseModel):
    server_url: str
    room_name: str
    participant_identity: str
    token: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    required = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    yield


app = FastAPI(title="Qwen3 Arabic Live Transcription and Translation", lifespan=lifespan)


def to_ws_url(url: str) -> str:
    url = url.strip()
    if url.startswith("https://"):
        return "wss://" + url[len("https://"):]
    if url.startswith("http://"):
        return "ws://" + url[len("http://"):]
    return url


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/session", response_model=SessionResponse)
async def create_session(response: Response) -> SessionResponse:
    response.headers["Cache-Control"] = "no-store"
    identity = f"browser-{secrets.token_urlsafe(6)}"
    token = (
        api.AccessToken(os.environ["LIVEKIT_API_KEY"].strip(), os.environ["LIVEKIT_API_SECRET"].strip())
        .with_identity(identity)
        .with_ttl(timedelta(minutes=30))
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=ROOM_NAME,
                can_publish=True,
                can_subscribe=True,
                    can_publish_data=True,
            )
        )
        .to_jwt()
    )
    return SessionResponse(
        server_url=to_ws_url(os.environ["LIVEKIT_URL"]),
        room_name=ROOM_NAME,
        participant_identity=identity,
        token=token,
    )


app.mount("/", StaticFiles(directory=PROJECT_ROOT / "frontend", html=True), name="frontend")
