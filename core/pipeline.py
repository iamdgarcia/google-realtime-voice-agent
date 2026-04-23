import asyncio
import logging
import threading
import time
from typing import Awaitable, Callable, Optional

import config
from core.chunker import SentenceChunker

logger = logging.getLogger(__name__)

LlmStreamFn = Callable[[list[dict], str, Callable[[str], None], threading.Event], None]
TtsFn = Callable[[str], bytes]


class AgentPipeline:
    """STT transcript → Gemini stream → TTS synthesis, sentence by sentence."""

    def __init__(
        self,
        conversation: list[dict],
        user_text: str,
        on_audio: Callable[[bytes], Awaitable[None]],
        llm_stream_fn: LlmStreamFn,
        tts_fn: TtsFn,
    ) -> None:
        self._conversation = conversation
        self._user_text = user_text
        self._on_audio = on_audio
        self._llm_stream_fn = llm_stream_fn
        self._tts_fn = tts_fn
        self._t_start = time.monotonic()
        self._llm_first_token_ms: int = 0
        self._tts_first_audio_ms: int = 0
        self._cancel_flag = threading.Event()

    def cancel(self) -> None:
        self._cancel_flag.set()

    async def run(self) -> dict:
        """Execute the cascade pipeline; return latency breakdown."""
        loop = asyncio.get_running_loop()
        chunker = SentenceChunker()
        audio_queue: asyncio.Queue[Optional[tuple[str, bytes]]] = asyncio.Queue()
        response_parts: list[str] = []
        first_token = True

        def _stream() -> None:
            nonlocal first_token
            token_count = 0
            sentence_count = 0

            def on_token(token: str) -> None:
                nonlocal first_token, token_count, sentence_count
                if self._cancel_flag.is_set():
                    return
                token_count += 1
                if first_token:
                    self._llm_first_token_ms = int(
                        (time.monotonic() - self._t_start) * 1000
                    )
                    logger.info("[LLM] first token at %dms", self._llm_first_token_ms)
                    first_token = False
                for sentence in chunker.push(token):
                    if self._cancel_flag.is_set():
                        return
                    sentence_count += 1
                    logger.info("[TTS] synthesizing sentence #%d: %r", sentence_count, sentence)
                    audio = self._tts_fn(sentence)
                    logger.info("[TTS] sentence #%d bytes=%d", sentence_count, len(audio))
                    asyncio.run_coroutine_threadsafe(
                        audio_queue.put((sentence, audio)), loop
                    )

            try:
                if not self._cancel_flag.is_set():
                    logger.info(
                        "[LLM] streaming model=%s user=%r",
                        config.GEMINI_MODEL, self._user_text[:60],
                    )
                    self._llm_stream_fn(
                        self._conversation, self._user_text, on_token, self._cancel_flag
                    )
                remainder = chunker.flush()
                if remainder and not self._cancel_flag.is_set():
                    sentence_count += 1
                    audio = self._tts_fn(remainder)
                    asyncio.run_coroutine_threadsafe(
                        audio_queue.put((remainder, audio)), loop
                    )
                logger.info(
                    "[LLM] complete tokens=%d sentences=%d", token_count, sentence_count
                )
            except Exception as exc:
                logger.error("[LLM] stream error: %s", exc, exc_info=True)
                raise
            finally:
                asyncio.run_coroutine_threadsafe(audio_queue.put(None), loop)

        executor_task = loop.run_in_executor(None, _stream)
        first_audio = True
        chunk_count = 0
        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        audio_queue.get(), timeout=config.LLM_TIMEOUT_S
                    )
                except asyncio.TimeoutError:
                    raise RuntimeError(f"LLM timed out after {config.LLM_TIMEOUT_S:.0f}s")
                if item is None:
                    logger.info("[PIPELINE] done — %d chunks forwarded", chunk_count)
                    break
                sentence, audio = item
                chunk_count += 1
                if first_audio:
                    self._tts_first_audio_ms = int(
                        (time.monotonic() - self._t_start) * 1000
                    )
                    logger.info(
                        "[PIPELINE] first audio at %dms bytes=%d",
                        self._tts_first_audio_ms, len(audio),
                    )
                    first_audio = False
                response_parts.append(sentence)
                await self._on_audio(audio)
        finally:
            try:
                await executor_task
            except Exception:
                pass  # already logged inside _stream

        total_ms = int((time.monotonic() - self._t_start) * 1000)
        logger.info(
            "[TURN] llm_first=%dms tts_first=%dms total=%dms",
            self._llm_first_token_ms, self._tts_first_audio_ms, total_ms,
        )
        return {
            "llm_first_token_ms": self._llm_first_token_ms,
            "tts_first_audio_ms": self._tts_first_audio_ms,
            "total_ms": total_ms,
            "response_text": " ".join(response_parts),
        }
