import asyncio
import logging
import os
import re
import time
from typing import Awaitable, Callable, Optional

from google import genai
from google.genai import types
from google.cloud import texttospeech

logger = logging.getLogger(__name__)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
TTS_VOICE = os.getenv("TTS_VOICE", "es-ES-Standard-A")

_SYSTEM_PROMPT = (
    "Eres un asistente conversacional breve y amable. "
    "Responde siempre en español. "
    "Mantén las respuestas cortas (1-3 oraciones)."
)


_SENTENCE_END = re.compile(r'[.!?](?=\s|$)')


class SentenceChunker:
    """Accumulate LLM tokens and emit complete sentences."""

    def __init__(self) -> None:
        self._buf = ""

    def push(self, token: str) -> list[str]:
        """Add token to buffer; return list of complete sentences found."""
        self._buf += token
        sentences: list[str] = []
        while True:
            match = _SENTENCE_END.search(self._buf)
            if not match:
                break
            end = match.end()
            sentences.append(self._buf[:end].strip())
            self._buf = self._buf[end:].lstrip()
        return sentences

    def flush(self) -> str:
        """Return and clear any remaining buffered text."""
        remainder = self._buf.strip()
        self._buf = ""
        return remainder

    def reset(self) -> None:
        """Clear buffer without returning content."""
        self._buf = ""


class AgentPipeline:
    """Stream Gemini tokens → chunk into sentences → synthesize each via TTS."""

    def __init__(
        self,
        conversation: list[dict],
        user_text: str,
        on_audio: Callable[[bytes], Awaitable[None]],
    ) -> None:
        self._conversation = conversation
        self._user_text = user_text
        self._on_audio = on_audio
        self._t_start = time.monotonic()
        self._llm_first_token_ms: int = 0
        self._tts_first_audio_ms: int = 0

    def _build_contents(self) -> list[types.Content]:
        contents = []
        for turn in self._conversation:
            contents.append(types.Content(
                role=turn["role"],
                parts=[types.Part(text=turn["content"])],
            ))
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=self._user_text)],
        ))
        return contents

    def _synthesize(self, text: str) -> bytes:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=TTS_VOICE[:5],
            name=TTS_VOICE,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
        )
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )
        return response.audio_content

    async def run(self) -> dict:
        """Execute LLM→TTS pipeline; return latency dict."""
        loop = asyncio.get_event_loop()
        chunker = SentenceChunker()
        full_response = ""
        first_token = True

        def _stream_llm() -> list[tuple[str, bytes]]:
            nonlocal first_token
            sentences_audio: list[tuple[str, bytes]] = []
            client = genai.Client()
            config = types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
            )
            stream = client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=self._build_contents(),
                config=config,
            )
            for chunk in stream:
                token = chunk.text or ""
                if first_token:
                    self._llm_first_token_ms = int(
                        (time.monotonic() - self._t_start) * 1000
                    )
                    first_token = False
                sentences = chunker.push(token)
                for sentence in sentences:
                    audio = self._synthesize(sentence)
                    sentences_audio.append((sentence, audio))
            remainder = chunker.flush()
            if remainder:
                audio = self._synthesize(remainder)
                sentences_audio.append((remainder, audio))
            return sentences_audio

        sentences_audio = await loop.run_in_executor(None, _stream_llm)

        for i, (sentence, audio) in enumerate(sentences_audio):
            if i == 0:
                self._tts_first_audio_ms = int(
                    (time.monotonic() - self._t_start) * 1000
                )
            full_response += (" " if i > 0 else "") + sentence
            await self._on_audio(audio)

        total_ms = int((time.monotonic() - self._t_start) * 1000)
        logger.info(
            "[TURN] llm_first=%dms tts_first=%dms total=%dms",
            self._llm_first_token_ms,
            self._tts_first_audio_ms,
            total_ms,
        )
        return {
            "llm_first_token_ms": self._llm_first_token_ms,
            "tts_first_audio_ms": self._tts_first_audio_ms,
            "total_ms": total_ms,
            "response_text": full_response,
        }
