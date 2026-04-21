import asyncio
import base64
import json
import logging
import os
import queue
import statistics
import threading
import time
from collections import deque

from typing import Any, Deque, Dict, Iterable, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.api_core import exceptions as api_exceptions
from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech as cloud_speech_types

from agent import AgentPipeline


PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("SPEECH_LOCATION", "eu")
MODEL = os.getenv("SPEECH_MODEL", "chirp_3")
LANGUAGE_CODE = os.getenv("SPEECH_LANGUAGE", "es-ES")


def _api_endpoint(location: str) -> str:
    if location == "global":
        return "speech.googleapis.com"
    return f"{location}-speech.googleapis.com"


class MetricsStore:
    """Track transcript counters and latency samples for the demo."""

    def __init__(self) -> None:
        self._latencies: Deque[int] = deque(maxlen=300)
        self._transcript_count = 0
        self._lock = threading.Lock()

    def add_latency(self, latency_ms: int) -> None:
        """Record a single latency value."""
        with self._lock:
            self._latencies.append(latency_ms)
            self._transcript_count += 1

    @staticmethod
    def _percentile(values: list[int], pct: float) -> int:
        """Return percentile using nearest-rank style index."""
        if not values:
            return 0
        index = max(0, min(len(values) - 1, round((pct / 100) * (len(values) - 1))))
        return values[index]

    def snapshot(self) -> Dict[str, Any]:
        """Return a metrics snapshot for API/UI consumers."""
        with self._lock:
            values = sorted(self._latencies)
            recent = list(self._latencies)
            if not values:
                return {
                    "transcript_count": self._transcript_count,
                    "avg_ms": 0,
                    "p50_ms": 0,
                    "p95_ms": 0,
                    "min_ms": 0,
                    "max_ms": 0,
                    "recent_ms": recent,
                }

            return {
                "transcript_count": self._transcript_count,
                "avg_ms": round(statistics.mean(values)),
                "p50_ms": self._percentile(values, 50),
                "p95_ms": self._percentile(values, 95),
                "min_ms": values[0],
                "max_ms": values[-1],
                "recent_ms": recent,
            }


class MonitorHub:
    """Manage dashboard monitor sockets and broadcast demo events."""

    def __init__(self) -> None:
        self._sockets: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a monitor websocket."""
        await websocket.accept()
        async with self._lock:
            self._sockets.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a monitor websocket if present."""
        async with self._lock:
            self._sockets.discard(websocket)

    async def broadcast(self, payload: Dict[str, Any]) -> None:
        """Send an event to all active monitor sockets."""
        async with self._lock:
            targets = list(self._sockets)

        dead: list[WebSocket] = []
        for socket in targets:
            try:
                await socket.send_json(payload)
            except Exception:
                dead.append(socket)

        if dead:
            async with self._lock:
                for socket in dead:
                    self._sockets.discard(socket)


class SpeechSession:
    """Bridge one client websocket to Google streaming recognition."""

    def __init__(self, metrics: MetricsStore, monitors: MonitorHub) -> None:
        self._metrics = metrics
        self._monitors = monitors
        self._audio_queue: "queue.Queue[Optional[tuple[bytes, int, int]]]" = queue.Queue()
        self._event_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._stream_start_ms = 0
        self._last_network_ms = 0
        self._last_queue_ms = 0

    def _build_requests(
        self,
    ) -> Iterable[cloud_speech_types.StreamingRecognizeRequest]:
        """Yield config request followed by queued audio requests."""
        recognizer = f"projects/{PROJECT_ID}/locations/{LOCATION}/recognizers/_"
        recognition_config = cloud_speech_types.RecognitionConfig(
            explicit_decoding_config=cloud_speech_types.ExplicitDecodingConfig(
                encoding=cloud_speech_types.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                audio_channel_count=1,
            ),
            language_codes=[LANGUAGE_CODE],
            model=MODEL,
        )

        streaming_config = cloud_speech_types.StreamingRecognitionConfig(
            config=recognition_config,
        )
        if MODEL in ("latest_short", "latest_long"):
            streaming_config = cloud_speech_types.StreamingRecognitionConfig(
                config=recognition_config,
                streaming_features=cloud_speech_types.StreamingRecognitionFeatures(
                    interim_results=True,
                ),
            )

        yield cloud_speech_types.StreamingRecognizeRequest(
            recognizer=recognizer,
            streaming_config=streaming_config,
        )

        first = True
        while not self._stop_event.is_set():
            item = self._audio_queue.get()
            if item is None:
                break
            audio_chunk, sent_at_ms, received_at_ms = item
            consumed_at_ms = int(time.time() * 1000)
            self._last_network_ms = received_at_ms - sent_at_ms
            self._last_queue_ms = consumed_at_ms - received_at_ms
            if first:
                self._stream_start_ms = sent_at_ms
                first = False
            yield cloud_speech_types.StreamingRecognizeRequest(audio=audio_chunk)

    def _worker(self) -> None:
        """Run blocking Speech v2 stream and push transcript events."""
        try:
            if not PROJECT_ID:
                raise RuntimeError("Set GOOGLE_CLOUD_PROJECT before running server.")

            client = SpeechClient(
                client_options=ClientOptions(
                    api_endpoint=_api_endpoint(LOCATION)
                )
            )
            responses = client.streaming_recognize(requests=self._build_requests())

            for response in responses:
                results = [r for r in response.results if r.alternatives]
                if not results:
                    continue

                # latest_short splits long audio into segments; join them all.
                # is_final only when the last (active) segment is done.
                transcript = "".join(r.alternatives[0].transcript for r in results)
                last_result = results[-1]
                is_final = bool(last_result.is_final)

                now_ms = int(time.time() * 1000)
                if self._stream_start_ms and last_result.result_end_offset:
                    end_offset_ms = int(last_result.result_end_offset.total_seconds() * 1000)
                    stt_ms = max(0, now_ms - (self._stream_start_ms + end_offset_ms))
                else:
                    stt_ms = 0
                self._metrics.add_latency(stt_ms)

                tag = "FINAL" if is_final else "PARTIAL"
                logger.info(
                    "[%s] %r | net=%dms queue=%dms stt=%dms",
                    tag, transcript,
                    self._last_network_ms, self._last_queue_ms, stt_ms,
                )

                self._event_queue.put(
                    {
                        "type": "transcript",
                        "transcript": transcript,
                        "is_final": is_final,
                        "latency_ms": stt_ms,
                        "latency_breakdown": {
                            "network_ms": self._last_network_ms,
                            "queue_ms": self._last_queue_ms,
                            "stt_ms": stt_ms,
                        },
                        "server_time_ms": now_ms,
                        "metrics": self._metrics.snapshot(),
                    }
                )
        except api_exceptions.Cancelled:
            logger.debug("Stream cancelled (normal closure)")
        except Exception as exc:
            self._event_queue.put(
                {
                    "type": "error",
                    "message": str(exc),
                    "server_time_ms": int(time.time() * 1000),
                }
            )
        finally:
            self._event_queue.put({"type": "worker_done"})

    async def run(self, websocket: WebSocket) -> None:
        """Handle websocket receive/send loops for one transcription session."""
        await websocket.accept()
        worker_thread = threading.Thread(target=self._worker, daemon=True)
        worker_thread.start()

        forward_task = asyncio.create_task(self._forward_events(websocket))
        try:
            while True:
                raw_message = await websocket.receive_text()
                payload = json.loads(raw_message)
                message_type = payload.get("type")

                if message_type == "audio":
                    audio_b64 = payload.get("audio_b64", "")
                    sent_at_ms = int(payload.get("sent_at_ms", int(time.time() * 1000)))
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                    except Exception:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "Invalid base64 audio payload.",
                                "server_time_ms": int(time.time() * 1000),
                            }
                        )
                        continue

                    received_at_ms = int(time.time() * 1000)
                    self._audio_queue.put((audio_bytes, sent_at_ms, received_at_ms))
                elif message_type == "stop":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            self._stop_event.set()
            self._audio_queue.put(None)
            await asyncio.to_thread(worker_thread.join, 2.0)
            await forward_task

    async def _forward_events(self, websocket: WebSocket) -> None:
        """Forward worker events to the client and dashboard monitors."""
        while True:
            event = await asyncio.to_thread(self._event_queue.get)
            if event.get("type") == "worker_done":
                break

            try:
                await websocket.send_json(event)
            except Exception:
                pass  # client already disconnected
            await self._monitors.broadcast(event)


LLM_DEBOUNCE_MS = int(os.getenv("LLM_DEBOUNCE_MS", "200"))


class AgentSession:
    """Full voice agent: STT → Gemini → TTS per WebSocket connection."""

    def __init__(self) -> None:
        self._audio_queue: "queue.Queue[Optional[tuple[bytes, int, int]]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._stream_start_ms = 0
        self._last_network_ms = 0
        self._last_queue_ms = 0

        self._conversation: list[dict] = []
        self._current_utterance = ""

        self._pipeline_task: Optional[asyncio.Task] = None
        self._debounce_task: Optional[asyncio.Task] = None
        self._send_queue: "asyncio.Queue[Optional[dict]]" = asyncio.Queue()
        self._is_playing = False

    def _build_requests(self) -> Iterable[cloud_speech_types.StreamingRecognizeRequest]:
        recognizer = f"projects/{PROJECT_ID}/locations/{LOCATION}/recognizers/_"
        recognition_config = cloud_speech_types.RecognitionConfig(
            explicit_decoding_config=cloud_speech_types.ExplicitDecodingConfig(
                encoding=cloud_speech_types.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                audio_channel_count=1,
            ),
            language_codes=[LANGUAGE_CODE],
            model=MODEL,
        )
        streaming_config = cloud_speech_types.StreamingRecognitionConfig(
            config=recognition_config,
        )
        if MODEL in ("latest_short", "latest_long"):
            streaming_config = cloud_speech_types.StreamingRecognitionConfig(
                config=recognition_config,
                streaming_features=cloud_speech_types.StreamingRecognitionFeatures(
                    interim_results=True,
                ),
            )
        yield cloud_speech_types.StreamingRecognizeRequest(
            recognizer=recognizer,
            streaming_config=streaming_config,
        )
        first = True
        while not self._stop_event.is_set():
            item = self._audio_queue.get()
            if item is None:
                break
            audio_chunk, sent_at_ms, received_at_ms = item
            consumed_at_ms = int(time.time() * 1000)
            self._last_network_ms = received_at_ms - sent_at_ms
            self._last_queue_ms = consumed_at_ms - received_at_ms
            if first:
                self._stream_start_ms = sent_at_ms
                first = False
            yield cloud_speech_types.StreamingRecognizeRequest(audio=audio_chunk)

    async def _cancel_pipeline(self) -> None:
        if self._pipeline_task and not self._pipeline_task.done():
            self._pipeline_task.cancel()
            try:
                await self._pipeline_task
            except asyncio.CancelledError:
                pass
        self._pipeline_task = None

    async def _cancel_debounce(self) -> None:
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
            try:
                await self._debounce_task
            except asyncio.CancelledError:
                pass
        self._debounce_task = None

    async def _start_pipeline(self, text: str, stt_ms: int) -> None:
        await self._cancel_pipeline()
        await self._send_queue.put({"type": "agent_start"})
        self._is_playing = True

        async def on_audio(audio_bytes: bytes) -> None:
            await self._send_queue.put({
                "type": "audio_chunk",
                "audio_b64": base64.b64encode(audio_bytes).decode(),
                "sample_rate": 16000,
            })

        async def _run() -> None:
            try:
                pipeline = AgentPipeline(
                    conversation=list(self._conversation),
                    user_text=text,
                    on_audio=on_audio,
                )
                latency = await pipeline.run()
                latency["stt_ms"] = stt_ms
                await self._send_queue.put({"type": "agent_done", "latency": latency})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("[LLM/TTS] %s", exc)
                await self._send_queue.put({"type": "agent_error", "message": str(exc)})
            finally:
                self._is_playing = False

        self._pipeline_task = asyncio.create_task(_run())

    def _worker(self, loop: asyncio.AbstractEventLoop) -> None:
        try:
            if not PROJECT_ID:
                raise RuntimeError("Set GOOGLE_CLOUD_PROJECT before running server.")
            client = SpeechClient(
                client_options=ClientOptions(api_endpoint=_api_endpoint(LOCATION))
            )
            responses = client.streaming_recognize(requests=self._build_requests())
            for response in responses:
                results = [r for r in response.results if r.alternatives]
                if not results:
                    continue
                transcript = "".join(r.alternatives[0].transcript for r in results)
                last_result = results[-1]
                is_final = bool(last_result.is_final)
                now_ms = int(time.time() * 1000)
                if self._stream_start_ms and last_result.result_end_offset:
                    end_offset_ms = int(last_result.result_end_offset.total_seconds() * 1000)
                    stt_ms = max(0, now_ms - (self._stream_start_ms + end_offset_ms))
                else:
                    stt_ms = 0

                tag = "FINAL" if is_final else "PARTIAL"
                logger.info("[%s] %r | stt=%dms", tag, transcript, stt_ms)

                asyncio.run_coroutine_threadsafe(
                    self._on_stt_event(transcript, is_final, stt_ms), loop
                )
        except api_exceptions.Cancelled:
            logger.debug("Agent STT stream cancelled")
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                self._send_queue.put({"type": "agent_error", "message": str(exc)}),
                loop,
            )
        finally:
            asyncio.run_coroutine_threadsafe(
                self._send_queue.put(None), loop
            )

    async def _on_stt_event(self, transcript: str, is_final: bool, stt_ms: int) -> None:
        if not is_final:
            if self._is_playing:
                return
            self._current_utterance = transcript
            await self._cancel_debounce()

            async def _debounce() -> None:
                await asyncio.sleep(LLM_DEBOUNCE_MS / 1000)
                await self._start_pipeline(transcript, stt_ms)

            self._debounce_task = asyncio.create_task(_debounce())
        else:
            await self._cancel_debounce()
            await self._cancel_pipeline()
            self._current_utterance = ""
            self._conversation.append({"role": "user", "content": transcript})
            await self._start_pipeline(transcript, stt_ms)

    async def _on_interrupt(self) -> None:
        await self._cancel_debounce()
        await self._cancel_pipeline()
        self._is_playing = False

    async def run(self, websocket: WebSocket) -> None:
        await websocket.accept()
        loop = asyncio.get_event_loop()
        worker_thread = threading.Thread(
            target=self._worker, args=(loop,), daemon=True
        )
        worker_thread.start()

        send_task = asyncio.create_task(self._send_loop(websocket))
        try:
            while True:
                raw = await websocket.receive_text()
                payload = json.loads(raw)
                msg_type = payload.get("type")

                if msg_type == "audio":
                    audio_b64 = payload.get("audio_b64", "")
                    sent_at_ms = int(payload.get("sent_at_ms", int(time.time() * 1000)))
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                    except Exception:
                        await websocket.send_json({"type": "error", "message": "Invalid base64"})
                        continue
                    received_at_ms = int(time.time() * 1000)
                    self._audio_queue.put((audio_bytes, sent_at_ms, received_at_ms))

                elif msg_type == "interrupt":
                    await self._on_interrupt()

                elif msg_type == "stop":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            self._stop_event.set()
            self._audio_queue.put(None)
            await self._cancel_pipeline()
            await self._cancel_debounce()
            await asyncio.to_thread(worker_thread.join, 2.0)
            send_task.cancel()

    async def _send_loop(self, websocket: WebSocket) -> None:
        while True:
            item = await self._send_queue.get()
            if item is None:
                break
            try:
                await websocket.send_json(item)
            except Exception:
                pass


app = FastAPI(title="Realtime Speech Demo")
app.mount("/static", StaticFiles(directory="static"), name="static")
metrics_store = MetricsStore()
monitor_hub = MonitorHub()


@app.get("/")
async def dashboard() -> FileResponse:
    """Serve the realtime dashboard page."""
    return FileResponse("static/index.html")


@app.get("/health")
async def health() -> Dict[str, str]:
    """Basic health endpoint for local checks."""
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Dict[str, Any]:
    """Expose current latency and transcript metrics."""
    return metrics_store.snapshot()


@app.websocket("/ws/monitor")
async def monitor_socket(websocket: WebSocket) -> None:
    """Open a monitor socket for dashboard live updates."""
    await monitor_hub.connect(websocket)
    await websocket.send_json(
        {
            "type": "metrics",
            "metrics": metrics_store.snapshot(),
            "server_time_ms": int(time.time() * 1000),
        }
    )

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await monitor_hub.disconnect(websocket)


@app.websocket("/ws/transcribe")
async def transcribe_socket(websocket: WebSocket) -> None:
    """Open a transcription session socket."""
    session = SpeechSession(metrics_store, monitor_hub)
    await session.run(websocket)


@app.websocket("/ws/agent")
async def agent_socket(websocket: WebSocket) -> None:
    """Open a voice agent session socket."""
    if not PROJECT_ID:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "GOOGLE_CLOUD_PROJECT not set"})
        await websocket.close()
        return
    session = AgentSession()
    await session.run(websocket)
