import asyncio
import base64
import json
import logging
import threading
import time
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect
from google.api_core import exceptions as api_exceptions

import config
from adapters.gcp_stt import SttStream, create_client
from adapters.gemini import stream_response, warmup
from adapters.gcp_tts import synthesize
from core.pipeline import AgentPipeline

logger = logging.getLogger(__name__)


class AgentSession:
    """Full voice pipeline: STT → Gemini → TTS with barge-in per WebSocket connection."""

    def __init__(self) -> None:
        self._stt = SttStream()
        self._conversation: list[dict] = []
        self._current_utterance = ""
        self._pipeline_task: Optional[asyncio.Task] = None
        self._active_pipeline: Optional[AgentPipeline] = None
        self._debounce_task: Optional[asyncio.Task] = None
        self._pending_final: str = ""
        self._send_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._is_playing = False
        self._pipeline_gen = 0
        self._audio_sent_gen: int = -1
        self._last_completed_utterance: str = ""

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
        logger.info("[PIPELINE] start gen=%d is_playing=%s text=%r", gen, self._is_playing, text)
        if not self._is_playing:
            await self._send_queue.put({"type": "agent_start"})
            self._is_playing = True
        else:
            logger.info("[PIPELINE] agent_start skipped — already playing")

        async def on_audio(audio_bytes: bytes) -> None:
            if gen != self._pipeline_gen:
                logger.debug(
                    "[PIPELINE] on_audio DISCARDED gen=%d current=%d bytes=%d",
                    gen, self._pipeline_gen, len(audio_bytes),
                )
                return  # superseded — discard audio from stale pipeline
            self._audio_sent_gen = gen  # TTS has started; future FINALs are likely echo
            logger.info("[PIPELINE] on_audio gen=%d bytes=%d", gen, len(audio_bytes))
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
                    llm_stream_fn=stream_response,
                    tts_fn=synthesize,
                )
                self._active_pipeline = pipeline
                latency = await pipeline.run()
                self._active_pipeline = None
                if gen != self._pipeline_gen:
                    logger.info("[PIPELINE] _run superseded gen=%d — discarding", gen)
                    return
                latency["stt_ms"] = stt_ms
                response_text = latency.get("response_text", "")
                logger.info(
                    "[PIPELINE] done gen=%d stt=%dms llm=%dms tts=%dms total=%dms response=%r",
                    gen, stt_ms,
                    latency.get("llm_first_token_ms", 0),
                    latency.get("tts_first_audio_ms", 0),
                    latency.get("total_ms", 0),
                    response_text[:80],
                )
                self._last_completed_utterance = text
                self._conversation.append({"role": "user", "content": text})
                if response_text:
                    self._conversation.append({"role": "model", "content": response_text})
                logger.info("[PIPELINE] conversation now %d turns", len(self._conversation))
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
                        "[PIPELINE] _run finally: skipping is_playing reset gen=%d current=%d",
                        gen, self._pipeline_gen,
                    )

        self._pipeline_task = asyncio.create_task(_run())

    def _worker(self, loop: asyncio.AbstractEventLoop) -> None:
        stream_index = 0
        while not self._stt.stopped:
            stream_index += 1
            logger.info("[WORKER] opening STT stream #%d", stream_index)
            try:
                if not config.GOOGLE_CLOUD_PROJECT:
                    raise RuntimeError("GOOGLE_CLOUD_PROJECT is not set.")
                client = create_client()
                for response in client.streaming_recognize(requests=self._stt.requests()):
                    results = [r for r in response.results if r.alternatives]
                    if not results:
                        continue
                    transcript = "".join(r.alternatives[0].transcript for r in results)
                    last_result = results[-1]
                    is_final = bool(last_result.is_final)
                    now_ms = int(time.time() * 1000)
                    if self._stt.stream_start_ms and last_result.result_end_offset:
                        end_offset_ms = int(
                            last_result.result_end_offset.total_seconds() * 1000
                        )
                        stt_ms = max(0, now_ms - (self._stt.stream_start_ms + end_offset_ms))
                    else:
                        stt_ms = 0
                    tag = "FINAL" if is_final else "PARTIAL"
                    logger.info("[%s] %r | stt=%dms", tag, transcript, stt_ms)
                    asyncio.run_coroutine_threadsafe(
                        self._on_stt_event(transcript, is_final, stt_ms), loop
                    )
                logger.info("[WORKER] STT stream #%d closed naturally", stream_index)

            except api_exceptions.Cancelled:
                # Chirp 3 closes after each FINAL; reopen unless session is stopping
                logger.info("[WORKER] STT stream #%d cancelled — will retry", stream_index)
            except api_exceptions.Aborted:
                # 409: Google closed the stream after silence or barge-in idle
                if self._stt.stopped:
                    logger.debug("[WORKER] STT stream #%d aborted (session stopping)", stream_index)
                    break
                logger.warning("[WORKER] STT stream #%d timed out — reopening", stream_index)
                self._stt.swap_queue()
                continue
            except Exception as exc:
                logger.error("[WORKER] STT stream #%d error: %s", stream_index, exc, exc_info=True)
                asyncio.run_coroutine_threadsafe(
                    self._send_queue.put({"type": "agent_error", "message": str(exc)}),
                    loop,
                )
                break

            if self._stt.stopped:
                break

            # Chirp 3 closes after each FINAL; swap queue so the stale generator
            # exits cleanly and the next stream reads from a fresh queue
            self._stt.swap_queue()

        logger.info("[WORKER] exited after %d stream(s)", stream_index)

    async def _on_stt_event(self, transcript: str, is_final: bool, stt_ms: int) -> None:
        if self._stt.stopped:
            return
        logger.info(
            "[STT] event is_final=%s is_playing=%s gen=%d transcript=%r",
            is_final, self._is_playing, self._pipeline_gen, transcript,
        )
        if not is_final:
            if self._is_playing:
                return
            if transcript == self._last_completed_utterance:
                logger.info("[STT] PARTIAL ignored — echo of last response transcript=%r", transcript)
                return
            effective = (self._pending_final + " " + transcript).strip() if self._pending_final else transcript
            self._pending_final = ""
            self._current_utterance = effective
            await self._send_queue.put({"type": "transcript", "transcript": effective, "is_final": False})
            await self._cancel_debounce()

            word_count = len(effective.split())
            if word_count < config.LLM_DEBOUNCE_MIN_WORDS:
                logger.info("[DEBOUNCE] skipping — too short (%d word(s)): %r", word_count, effective)
            else:
                async def _debounce() -> None:
                    logger.info("[DEBOUNCE] sleeping %dms", config.LLM_DEBOUNCE_MS)
                    await asyncio.sleep(config.LLM_DEBOUNCE_MS / 1000)
                    logger.info("[DEBOUNCE] firing pipeline for %r", effective)
                    await self._start_pipeline(effective, stt_ms)

                self._debounce_task = asyncio.create_task(_debounce())
        else:
            if self._is_playing:
                if self._audio_sent_gen == self._pipeline_gen:
                    logger.info("[STT] FINAL ignored — echo during playback transcript=%r", transcript)
                    return
                if transcript == self._last_completed_utterance:
                    logger.info("[STT] FINAL ignored — echo restart same as last response transcript=%r", transcript)
                    await self._cancel_debounce()
                    return
                # Pipeline started from a partial but TTS hasn't sent audio yet;
                # FINAL carries the complete utterance — restart with it
                logger.info(
                    "[STT] FINAL before TTS — restarting pipeline text=%r (was %r)",
                    transcript, self._current_utterance,
                )
                await self._send_queue.put({"type": "transcript", "transcript": transcript, "is_final": True})
                await self._cancel_debounce()
                self._current_utterance = ""
                self._pending_final = ""
                await self._start_pipeline(transcript, stt_ms)
                return
            if transcript == self._last_completed_utterance:
                logger.info("[STT] FINAL ignored — already completed by debounce transcript=%r", transcript)
                return
            logger.info("[STT] FINAL — grace window %dms for %r", config.LLM_FINAL_GRACE_MS, transcript)
            await self._send_queue.put({"type": "transcript", "transcript": transcript, "is_final": True})
            await self._cancel_debounce()
            self._current_utterance = ""
            self._pending_final = transcript

            async def _grace() -> None:
                await asyncio.sleep(config.LLM_FINAL_GRACE_MS / 1000)
                self._pending_final = ""
                logger.info("[DEBOUNCE] FINAL grace expired — firing for %r", transcript)
                await self._start_pipeline(transcript, stt_ms)

            self._debounce_task = asyncio.create_task(_grace())

    async def _on_interrupt(self) -> None:
        logger.info("[BARGE-IN] interrupt received gen=%d", self._pipeline_gen)
        was_playing = self._is_playing
        await self._cancel_debounce()
        await self._cancel_pipeline()
        self._is_playing = False
        if was_playing:
            await self._send_queue.put({"type": "agent_interrupted"})

    async def run(self, websocket: WebSocket) -> None:
        logger.info("[AGENT] connection accepted")
        await websocket.accept()
        loop = asyncio.get_event_loop()
        asyncio.create_task(asyncio.to_thread(warmup))
        worker = threading.Thread(target=self._worker, args=(loop,), daemon=True)
        worker.start()
        send_task = asyncio.create_task(self._send_loop(websocket))
        audio_chunks_received = 0
        try:
            while True:
                payload = json.loads(await websocket.receive_text())
                msg_type = payload.get("type")
                if msg_type == "audio":
                    try:
                        audio_bytes = base64.b64decode(payload.get("audio_b64", ""))
                    except Exception:
                        logger.warning("[AGENT] invalid base64 in audio message")
                        await websocket.send_json({"type": "error", "message": "Invalid base64"})
                        continue
                    sent_at_ms = int(payload.get("sent_at_ms", int(time.time() * 1000)))
                    self._stt.push(audio_bytes, sent_at_ms)
                    audio_chunks_received += 1
                    if audio_chunks_received % 50 == 0:
                        logger.debug("[AGENT] audio chunks received: %d", audio_chunks_received)
                elif msg_type == "interrupt":
                    logger.info("[AGENT] interrupt from client")
                    await self._on_interrupt()
                elif msg_type == "stop":
                    logger.info("[AGENT] stop from client")
                    break
        except WebSocketDisconnect:
            logger.info("[AGENT] client disconnected")
        except Exception as exc:
            logger.error("[AGENT] receive loop error: %s", exc, exc_info=True)
        finally:
            logger.info(
                "[AGENT] cleanup — chunks=%d gen=%d",
                audio_chunks_received, self._pipeline_gen,
            )
            self._stt.stop()
            self._stt.close()
            await self._cancel_pipeline()
            await self._cancel_debounce()
            await asyncio.to_thread(worker.join, 2.0)
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
                        logger.info("[SEND] audio_chunk #%d bytes~%d", chunks_sent, audio_len)
                    else:
                        logger.info("[SEND] msg type=%r", msg_type)
                except Exception as exc:
                    logger.error("[SEND] failed type=%r: %s", msg_type, exc)
        except asyncio.CancelledError:
            logger.info("[SEND] send_loop cancelled (chunks_sent=%d)", chunks_sent)
