from typing import Dict

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config
from application.agent_session import AgentSession

app = FastAPI(title="Realtime Voice Agent")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws/agent")
async def agent_socket(websocket: WebSocket) -> None:
    if not config.GOOGLE_CLOUD_PROJECT:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "GOOGLE_CLOUD_PROJECT not set"})
        await websocket.close()
        return
    await AgentSession().run(websocket)
