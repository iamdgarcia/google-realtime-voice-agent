import asyncio
import base64
import json
import logging
import os
import queue
import statistics
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
from collections import deque

from typing import Any, Deque, Dict, Iterable, Optional

def _configure_logging() -> None:
    """Configure console and rotating-file logging for server events."""
    log_dir = Path(os.getenv("SERVER_LOG_DIR", "logs"))
    log_filename = os.getenv("SERVER_LOG_FILE", "server.log")
    max_bytes = int(os.getenv("SERVER_LOG_MAX_BYTES", str(5 * 1024 * 1024)))
    backup_count = int(os.getenv("SERVER_LOG_BACKUP_COUNT", "5"))
    log_level = os.getenv("SERVER_LOG_LEVEL", "INFO").upper()

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_filename

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(
                filename=log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            ),
        ],
        force=True,
    )


_configure_logging()
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
        self._active_pipeline: Optional[AgentPipeline] = None
        self._debounce_task: Optional[asyncio.Task] = None
        self._send_queue: "asyncio.Queue[dict]" = asyncio.Queue()
        self._is_playing = False
        self._pipeline_gen = 0

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
        old_gen = self._pipeline_gen
        self._pipeline_gen += 1
        if self._active_pipeline is not None:
            self._active_pipeline.cancel()
            self._active_pipeline = None
        if self._pipeline_task and not self._pipeline_task.done():
            logger.info("[PIPELINE] cancelling task gen=%d → gen=%d", old_gen, self._pipeline_gen)
            self._pipeline_task.cancel()
            try:
                await self._pipeline_task
            except asyncio.CancelledError:
                pass
        else:
            logger.debug("[PIPELINE] cancel: no active task gen=%d → gen=%d", old_gen, self._pipeline_gen)
        self._pipeline_task = None

    async def _cancel_debounce(self) -> None:
        if self._debounce_task and not self._debounce_task.done():
            logger.info("[DEBOUNCE] cancelling pending debounce")
            self._debounce_task.cancel()
            try:
                await self._debounce_task
            except asyncio.CancelledError:
                pass
        self._debounce_task = None

    async def _start_pipeline(self, text: str, stt_ms: int) -> None:
        await self._cancel_pipeline()
        gen = self._pipeline_gen
        logger.info(
            "[PIPELINE] start gen=%d is_playing=%s text=%r", gen, self._is_playing, text
        )
        if not self._is_playing:
            logger.info("[PIPELINE] sending agent_start to client")
            await self._send_queue.put({"type": "agent_start"})
            self._is_playing = True
        else:
            logger.info("[PIPELINE] agent_start skipped — already playing")

        async def on_audio(audio_bytes: bytes) -> None:
            if gen != self._pipeline_gen:
                logger.debug(
                    "[PIPELINE] on_audio DISCARDED gen=%d current_gen=%d bytes=%d",
                    gen, self._pipeline_gen, len(audio_bytes),
                )
                return  # superseded — discard audio from stale pipeline
            logger.info(
                "[PIPELINE] on_audio queuing chunk gen=%d bytes=%d", gen, len(audio_bytes)
            )
            await self._send_queue.put({
                "type": "audio_chunk",
                "audio_b64": base64.b64encode(audio_bytes).decode(),
                "sample_rate": 16000,
            })

        async def _run() -> None:
            logger.info("[PIPELINE] _run started gen=%d", gen)
            try:
                pipeline = AgentPipeline(
                    conversation=list(self._conversation),
                    user_text=text,
                    on_audio=on_audio,
                )
                self._active_pipeline = pipeline
                latency = await pipeline.run()
                self._active_pipeline = None
                if gen != self._pipeline_gen:
                    logger.info(
                        "[PIPELINE] _run superseded gen=%d current_gen=%d — discarding",
                        gen, self._pipeline_gen,
                    )
                    return
                latency["stt_ms"] = stt_ms
                logger.info(
                    "[PIPELINE] done gen=%d stt=%dms llm=%dms tts=%dms total=%dms response=%r",
                    gen, stt_ms,
                    latency.get("llm_first_token_ms", 0),
                    latency.get("tts_first_audio_ms", 0),
                    latency.get("total_ms", 0),
                    latency.get("response_text", "")[:80],
                )
                await self._send_queue.put({"type": "agent_done", "latency": latency})
            except asyncio.CancelledError:
                logger.info("[PIPELINE] _run cancelled gen=%d", gen)
                raise
            except Exception as exc:
                logger.error("[PIPELINE] _run error gen=%d: %s", gen, exc, exc_info=True)
                if gen == self._pipeline_gen:
                    await self._send_queue.put({"type": "agent_error", "message": str(exc)})
            finally:
                if gen == self._pipeline_gen:
                    logger.info("[PIPELINE] _run finally: clearing is_playing gen=%d", gen)
                    self._is_playing = False
                else:
                    logger.info(
                        "[PIPELINE] _run finally: skipping is_playing reset gen=%d current_gen=%d",
                        gen, self._pipeline_gen,
                    )

        self._pipeline_task = asyncio.create_task(_run())

    def _worker(self, loop: asyncio.AbstractEventLoop) -> None:
        stream_index = 0
        while not self._stop_event.is_set():
            stream_index += 1
            logger.info("[WORKER] opening STT stream #%d", stream_index)
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

                logger.info("[WORKER] STT stream #%d closed naturally", stream_index)

            except api_exceptions.Cancelled:
                logger.info("[WORKER] STT stream #%d cancelled by server — will retry", stream_index)
                # Don't break here; fall through to the _stop_event check and
                # queue swap so a new stream opens unless we're truly shutting down.
            except api_exceptions.Aborted:
                # 409: Google closed the stream because no audio arrived for a
                # while (expected after barge-in or a long silence). Swap the
                # queue to unblock the stale generator and open a fresh stream
                # unless the session is already shutting down.
                if self._stop_event.is_set():
                    logger.debug("[WORKER] STT stream #%d aborted (session stopping)", stream_index)
                    break
                logger.warning("[WORKER] STT stream #%d timed out — reopening", stream_index)
                old_queue = self._audio_queue
                self._audio_queue = queue.Queue()
                old_queue.put(None)
                continue
            except Exception as exc:
                logger.error("[WORKER] STT stream #%d error: %s", stream_index, exc, exc_info=True)
                asyncio.run_coroutine_threadsafe(
                    self._send_queue.put({"type": "agent_error", "message": str(exc)}),
                    loop,
                )
                break

            if self._stop_event.is_set():
                break

            # The STT stream closed naturally (Chirp 3 closes after each FINAL).
            # Swap the audio queue so the old _build_requests generator — which
            # may still be blocked in queue.get() inside a gRPC thread — exits
            # cleanly when it receives None, while the new stream gets a fresh
            # queue with no stale chunks.
            old_queue = self._audio_queue
            self._audio_queue = queue.Queue()
            old_queue.put(None)  # unblock any generator stuck on the old queue

        logger.info("[WORKER] STT worker exited after %d stream(s)", stream_index)

    async def _on_stt_event(self, transcript: str, is_final: bool, stt_ms: int) -> None:
        if self._stop_event.is_set():
            logger.debug("[STT] event ignored — session stopped")
            return
        logger.info(
            "[STT] event is_final=%s is_playing=%s gen=%d transcript=%r",
            is_final, self._is_playing, self._pipeline_gen, transcript,
        )
        if not is_final:
            if self._is_playing:
                logger.debug("[STT] partial ignored — agent is playing")
                return
            self._current_utterance = transcript
            await self._cancel_debounce()

            async def _debounce() -> None:
                logger.info("[DEBOUNCE] sleeping %dms before pipeline", LLM_DEBOUNCE_MS)
                await asyncio.sleep(LLM_DEBOUNCE_MS / 1000)
                logger.info("[DEBOUNCE] firing pipeline for partial %r", transcript)
                await self._start_pipeline(transcript, stt_ms)

            self._debounce_task = asyncio.create_task(_debounce())
            logger.debug("[DEBOUNCE] scheduled for %r", transcript)
        else:
            logger.info("[STT] FINAL — cancelling debounce, starting fresh pipeline")
            await self._cancel_debounce()
            self._current_utterance = ""
            self._conversation.append({"role": "user", "content": transcript})
            await self._start_pipeline(transcript, stt_ms)

    async def _on_interrupt(self) -> None:
        logger.info("[BARGE-IN] interrupt received — cancelling pipeline gen=%d", self._pipeline_gen)
        was_playing = self._is_playing
        await self._cancel_debounce()
        await self._cancel_pipeline()
        self._is_playing = False
        if was_playing:
            await self._send_queue.put({"type": "agent_cancelled"})

    async def run(self, websocket: WebSocket) -> None:
        logger.info("[AGENT] connection accepted")
        await websocket.accept()
        loop = asyncio.get_event_loop()
        worker_thread = threading.Thread(
            target=self._worker, args=(loop,), daemon=True
        )
        worker_thread.start()

        send_task = asyncio.create_task(self._send_loop(websocket))
        audio_chunks_received = 0
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
                        logger.warning("[AGENT] invalid base64 in audio message")
                        await websocket.send_json({"type": "error", "message": "Invalid base64"})
                        continue
                    received_at_ms = int(time.time() * 1000)
                    self._audio_queue.put((audio_bytes, sent_at_ms, received_at_ms))
                    audio_chunks_received += 1
                    if audio_chunks_received % 50 == 0:
                        logger.debug("[AGENT] audio chunks received from client: %d", audio_chunks_received)

                elif msg_type == "interrupt":
                    logger.info("[AGENT] interrupt message from client")
                    await self._on_interrupt()

                elif msg_type == "stop":
                    logger.info("[AGENT] stop message from client")
                    break
        except WebSocketDisconnect:
            logger.info("[AGENT] client disconnected (WebSocketDisconnect)")
        except Exception as exc:
            logger.error("[AGENT] unexpected error in receive loop: %s", exc, exc_info=True)
        finally:
            logger.info(
                "[AGENT] cleaning up — audio_chunks_received=%d gen=%d",
                audio_chunks_received, self._pipeline_gen,
            )
            self._stop_event.set()
            self._audio_queue.put(None)
            await self._cancel_pipeline()
            await self._cancel_debounce()
            await asyncio.to_thread(worker_thread.join, 2.0)
            send_task.cancel()

    async def _send_loop(self, websocket: WebSocket) -> None:
        logger.info("[SEND] send_loop started")
        chunks_sent = 0
        try:
            while True:
                item = await self._send_queue.get()
                msg_type = item.get("type")
                try:
                    await websocket.send_json(item)
                    if msg_type == "audio_chunk":
                        chunks_sent += 1
                        audio_len = len(item.get("audio_b64", "")) * 3 // 4
                        logger.info("[SEND] audio_chunk #%d sent bytes~%d", chunks_sent, audio_len)
                    else:
                        logger.info("[SEND] sent msg type=%r", msg_type)
                except Exception as exc:
                    logger.error("[SEND] failed to send type=%r: %s", msg_type, exc)
        except asyncio.CancelledError:
            logger.info("[SEND] send_loop cancelled (chunks_sent=%d)", chunks_sent)


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
