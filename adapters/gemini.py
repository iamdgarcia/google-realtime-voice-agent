import logging
import threading
from typing import Callable

from google import genai
from google.genai import types

import config

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Eres un asistente conversacional breve y amable. "
    "Responde siempre en español. "
    "Mantén las respuestas cortas (1-3 oraciones)."
)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=config.GOOGLE_CLOUD_PROJECT,
            location=config.GEMINI_LOCATION,
        )
    return _client


def warmup() -> None:
    """Fire a minimal request to pre-establish the Vertex AI TLS connection."""
    try:
        client = _get_client()
        stream = client.models.generate_content_stream(
            model=config.GEMINI_MODEL,
            contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
            config=types.GenerateContentConfig(
                system_instruction="Reply with one word.",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                max_output_tokens=5,
            ),
        )
        for _ in stream:
            break
        logger.info("[LLM] warmup complete")
    except Exception as exc:
        logger.warning("[LLM] warmup failed: %s", exc)


def stream_response(
    conversation: list[dict],
    user_text: str,
    on_token: Callable[[str], None],
    cancel_flag: threading.Event,
) -> None:
    if cancel_flag.is_set():
        return
    client = _get_client()
    if cancel_flag.is_set():
        return
    contents = [
        types.Content(role=turn["role"], parts=[types.Part(text=turn["content"])])
        for turn in conversation
    ]
    contents.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
    stream = client.models.generate_content_stream(
        model=config.GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    for chunk in stream:
        if cancel_flag.is_set():
            logger.info("[LLM] cancelled mid-stream")
            return
        on_token(chunk.text or "")
