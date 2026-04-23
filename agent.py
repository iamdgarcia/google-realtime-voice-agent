"""Compatibility layer for legacy imports and tests.

This module preserves the original `agent` import path while delegating the
actual implementation to the refactored `core` and `adapters` packages.
"""

from __future__ import annotations

import base64
import logging
import threading
from typing import Awaitable, Callable

from google import genai
from google.cloud import texttospeech

import config
from core.chunker import SentenceChunker
from core.pipeline import AgentPipeline as _CoreAgentPipeline

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Eres un asistente conversacional breve y amable. "
    "Responde siempre en español. "
    "Mantén las respuestas cortas (1-3 oraciones)."
)


def _stream_llm(
    conversation: list[dict],
    user_text: str,
    on_token: Callable[[str], None],
    cancel_flag: threading.Event,
) -> None:
    """Stream tokens from Gemini using the legacy module-level client API."""
    if cancel_flag.is_set():
        return

    client = genai.Client()
    contents = list(conversation)
    contents.append({"role": "user", "content": user_text})

    stream = client.models.generate_content_stream(
        model=config.GEMINI_MODEL,
        contents=contents,
        config={"system_instruction": _SYSTEM_PROMPT, "thinking_budget": 0},
    )

    for chunk in stream:
        if cancel_flag.is_set():
            logger.info("[LLM] cancelled mid-stream")
            return
        on_token(getattr(chunk, "text", "") or "")


def _synthesize(text: str) -> bytes:
    """Synthesize LINEAR16 audio and strip the WAV header if present."""
    language_code = "-".join(config.TTS_VOICE.split("-")[:2])
    client = texttospeech.TextToSpeechClient()
    response = client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=text),
        voice=texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=config.TTS_VOICE,
        ),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
        ),
    )
    audio = response.audio_content
    return audio[44:] if audio[:4] == b"RIFF" else audio


class AgentPipeline:
    """Legacy wrapper around the refactored core agent pipeline."""

    def __init__(
        self,
        conversation: list[dict],
        user_text: str,
        on_audio: Callable[[bytes], Awaitable[None]],
    ) -> None:
        self._conversation = conversation
        self._user_text = user_text
        self._on_audio = on_audio

    async def run(self) -> dict:
        """Execute the current pipeline with built-in Gemini and TTS hooks."""
        pipeline = _CoreAgentPipeline(
            conversation=self._conversation,
            user_text=self._user_text,
            on_audio=self._on_audio,
            llm_stream_fn=_stream_llm,
            tts_fn=_synthesize,
        )
        return await pipeline.run()
