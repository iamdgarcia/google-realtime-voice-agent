# Voice Agent Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real-time voice agent that pipes Google STT → Gemini 2.0 Flash (streaming) → Google TTS back to the client, with barge-in support and per-stage latency benchmarking.

**Architecture:** `AgentPipeline` (new `agent.py`) owns LLM streaming, sentence chunking, and TTS synthesis. `AgentSession` (replaces `SpeechSession` in `server.py`) handles STT events, debounce, pipeline cancellation, and conversation history. `client.py` gains a PyAudio output stream and RMS-based barge-in detection.

**Tech Stack:** FastAPI + WebSockets, Google Cloud Speech v2 (`chirp_3`/`latest_short`), `google-generativeai` (Gemini 2.0 Flash), `google-cloud-texttospeech`, `asyncio.Task` cancellation, PyAudio.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `requirements.txt` | Modify | Add `google-generativeai`, `google-cloud-texttospeech`, `pytest-asyncio` |
| `agent.py` | **Create** | `SentenceChunker`, `AgentPipeline` |
| `tests/test_agent.py` | **Create** | Unit tests for `SentenceChunker` and `AgentPipeline` |
| `server.py` | Modify | Add `AgentSession`, new `/ws/agent` endpoint; keep existing `/ws/transcribe` intact |
| `client.py` | Modify | PyAudio output stream, barge-in RMS monitor, `interrupt` message |

---

### Task 1: Add dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Update requirements.txt**

Replace the file contents with:

```
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
google-cloud-speech>=2.26.0
google-generativeai>=0.8.0
google-cloud-texttospeech>=2.16.0
websockets>=12.0
pyaudio>=0.2.14
pytest>=8.0
pytest-asyncio>=0.23
```

- [ ] **Step 2: Install**

```bash
cd /home/developer/proyectos/gcp_audio
source venv/bin/activate
pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 3: Verify imports**

```bash
python -c "import google.generativeai; import google.cloud.texttospeech; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add google-generativeai and texttospeech dependencies"
```

---

### Task 2: SentenceChunker

**Files:**
- Create: `agent.py`
- Create: `tests/__init__.py`
- Create: `tests/test_agent.py`

- [ ] **Step 1: Create the failing test**

Create `tests/__init__.py` (empty) and `tests/test_agent.py`:

```python
import pytest
from agent import SentenceChunker


def test_no_sentence_boundary():
    chunker = SentenceChunker()
    assert chunker.push("Hello world") == []


def test_single_sentence_period():
    chunker = SentenceChunker()
    assert chunker.push("Hello world. ") == ["Hello world."]


def test_single_sentence_question():
    chunker = SentenceChunker()
    assert chunker.push("How are you? ") == ["How are you?"]


def test_single_sentence_exclamation():
    chunker = SentenceChunker()
    assert chunker.push("Great! ") == ["Great!"]


def test_multiple_sentences_in_one_push():
    chunker = SentenceChunker()
    result = chunker.push("Hello world. How are you? ")
    assert result == ["Hello world.", "How are you?"]


def test_sentence_split_across_pushes():
    chunker = SentenceChunker()
    assert chunker.push("Hello ") == []
    assert chunker.push("world. ") == ["Hello world."]


def test_flush_returns_remainder():
    chunker = SentenceChunker()
    chunker.push("Hello world")
    assert chunker.flush() == "Hello world"


def test_flush_empty():
    chunker = SentenceChunker()
    assert chunker.flush() == ""


def test_flush_after_sentence_clears_buffer():
    chunker = SentenceChunker()
    chunker.push("Hello. Bye")
    chunker.flush()
    assert chunker.flush() == ""


def test_reset_clears_state():
    chunker = SentenceChunker()
    chunker.push("Hello world")
    chunker.reset()
    assert chunker.flush() == ""
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /home/developer/proyectos/gcp_audio && source venv/bin/activate
pytest tests/test_agent.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'agent'`

- [ ] **Step 3: Create `agent.py` with SentenceChunker**

Create `/home/developer/proyectos/gcp_audio/agent.py`:

```python
import re
from typing import Optional


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
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_agent.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/__init__.py tests/test_agent.py
git commit -m "feat: add SentenceChunker with TDD"
```

---

### Task 3: AgentPipeline

**Files:**
- Modify: `agent.py` — add `AgentPipeline` class
- Modify: `tests/test_agent.py` — add pipeline tests

- [ ] **Step 1: Write failing tests for AgentPipeline**

Append to `tests/test_agent.py`:

```python
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_agent_pipeline_calls_on_audio_with_bytes():
    """Pipeline should invoke on_audio callback with bytes from TTS."""
    audio_chunks = []

    async def collect(chunk: bytes) -> None:
        audio_chunks.append(chunk)

    fake_token_stream = ["Hello", " there", "."]
    fake_audio = b"\x00\x01" * 100

    with patch("agent.genai") as mock_genai, \
         patch("agent.texttospeech") as mock_tts:

        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        fake_response = MagicMock()
        fake_response.__iter__ = MagicMock(
            return_value=iter([MagicMock(text=t) for t in fake_token_stream])
        )
        mock_model.generate_content.return_value = fake_response

        mock_tts_client = MagicMock()
        mock_tts.TextToSpeechClient.return_value = mock_tts_client
        mock_tts_response = MagicMock()
        mock_tts_response.audio_content = fake_audio
        mock_tts_client.synthesize_speech.return_value = mock_tts_response

        mock_tts.SynthesisInput = MagicMock(side_effect=lambda text: MagicMock())
        mock_tts.VoiceSelectionParams = MagicMock(return_value=MagicMock())
        mock_tts.AudioConfig = MagicMock(return_value=MagicMock())
        mock_tts.AudioEncoding = MagicMock()
        mock_tts.AudioEncoding.LINEAR16 = 1

        from agent import AgentPipeline
        pipeline = AgentPipeline(
            conversation=[],
            user_text="Hello",
            on_audio=collect,
        )
        await pipeline.run()

    assert len(audio_chunks) > 0
    assert audio_chunks[0] == fake_audio


@pytest.mark.asyncio
async def test_agent_pipeline_returns_latency_dict():
    """Pipeline.run() should return a dict with timing keys."""
    fake_audio = b"\x00\x01"

    with patch("agent.genai") as mock_genai, \
         patch("agent.texttospeech") as mock_tts:

        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        fake_response = MagicMock()
        fake_response.__iter__ = MagicMock(
            return_value=iter([MagicMock(text="Fine. ")])
        )
        mock_model.generate_content.return_value = fake_response

        mock_tts_client = MagicMock()
        mock_tts.TextToSpeechClient.return_value = mock_tts_client
        mock_tts_response = MagicMock()
        mock_tts_response.audio_content = fake_audio
        mock_tts_client.synthesize_speech.return_value = mock_tts_response
        mock_tts.SynthesisInput = MagicMock(side_effect=lambda text: MagicMock())
        mock_tts.VoiceSelectionParams = MagicMock(return_value=MagicMock())
        mock_tts.AudioConfig = MagicMock(return_value=MagicMock())
        mock_tts.AudioEncoding = MagicMock()
        mock_tts.AudioEncoding.LINEAR16 = 1

        from agent import AgentPipeline
        pipeline = AgentPipeline(
            conversation=[],
            user_text="Fine",
            on_audio=AsyncMock(),
        )
        result = await pipeline.run()

    assert "llm_first_token_ms" in result
    assert "tts_first_audio_ms" in result
    assert "total_ms" in result
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_agent.py::test_agent_pipeline_calls_on_audio_with_bytes -v
```

Expected: `ImportError` or `AttributeError` (AgentPipeline not defined).

- [ ] **Step 3: Implement AgentPipeline in agent.py**

Append to `agent.py` (after `SentenceChunker`):

```python
import asyncio
import logging
import os
import time
from typing import Awaitable, Callable

import google.generativeai as genai
from google.cloud import texttospeech

logger = logging.getLogger(__name__)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
TTS_VOICE = os.getenv("TTS_VOICE", "es-ES-Standard-A")

_SYSTEM_PROMPT = (
    "Eres un asistente conversacional breve y amable. "
    "Responde siempre en español. "
    "Mantén las respuestas cortas (1-3 oraciones)."
)


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

    def _build_contents(self) -> list[dict]:
        contents = [{"role": "user", "parts": [_SYSTEM_PROMPT]}]
        for turn in self._conversation:
            contents.append({"role": turn["role"], "parts": [turn["content"]]})
        contents.append({"role": "user", "parts": [self._user_text]})
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
        model = genai.GenerativeModel(GEMINI_MODEL)
        chunker = SentenceChunker()
        full_response = ""
        first_token = True
        first_audio = True

        def _stream_llm() -> list[str]:
            """Blocking call; returns list of (sentence, audio_bytes) pairs."""
            nonlocal first_token
            sentences_audio: list[tuple[str, bytes]] = []
            response = model.generate_content(
                self._build_contents(),
                stream=True,
            )
            for chunk in response:
                token = chunk.text
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
                full_response += sentence
            else:
                full_response += " " + sentence
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_agent.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: add AgentPipeline with Gemini streaming and Google TTS"
```

---

### Task 4: AgentSession in server.py

**Files:**
- Modify: `server.py` — add `AgentSession` class and `/ws/agent` endpoint

- [ ] **Step 1: Add imports to server.py**

At the top of `server.py`, after the existing imports, add:

```python
from agent import AgentPipeline
```

- [ ] **Step 2: Add AgentSession class**

Add this class to `server.py` after the `SpeechSession` class (before the `app = FastAPI(...)` line):

```python
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
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

    # ------------------------------------------------------------------ STT

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

    # ------------------------------------------------------------------ pipeline helpers

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
        t_pipeline_start = time.monotonic()

        async def on_audio(audio_bytes: bytes) -> None:
            import base64 as _b64
            await self._send_queue.put({
                "type": "audio_chunk",
                "audio_b64": _b64.b64encode(audio_bytes).decode(),
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

    # ------------------------------------------------------------------ STT worker

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

    # ------------------------------------------------------------------ event handlers

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

    # ------------------------------------------------------------------ main run loop

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
```

- [ ] **Step 3: Add the /ws/agent endpoint**

After the existing `/ws/transcribe` endpoint in `server.py`, add:

```python
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
```

- [ ] **Step 4: Smoke test server starts**

```bash
cd /home/developer/proyectos/gcp_audio && source venv/bin/activate
python -c "import server; print('server imports OK')"
```

Expected: `server imports OK`

- [ ] **Step 5: Commit**

```bash
git add server.py
git commit -m "feat: add AgentSession and /ws/agent endpoint"
```

---

### Task 5: client.py — audio output and barge-in

**Files:**
- Modify: `client.py` — add `--agent` flag, output stream, barge-in

- [ ] **Step 1: Replace client.py**

Replace the full contents of `client.py` with:

```python
import argparse
import asyncio
import base64
import json
import struct
import time
from typing import Any

import websockets

CHUNK_SIZE = 1024
RATE = 16000
CHANNELS = 1
BARGE_IN_THRESHOLD = int(__import__('os').getenv("BARGE_IN_THRESHOLD", "300"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime STT / Voice Agent client")
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8000/ws/transcribe",
        help="Server websocket endpoint",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Connect to /ws/agent instead of /ws/transcribe",
    )
    return parser.parse_args()


def open_microphone() -> Any:
    try:
        import pyaudio
    except ImportError as exc:
        raise RuntimeError(
            "PyAudio is required. Install with `pip install pyaudio`."
        ) from exc
    audio = pyaudio.PyAudio()
    mic = audio.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )
    return audio, mic


def open_speaker(audio: Any) -> Any:
    import pyaudio
    return audio.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=RATE,
        output=True,
        frames_per_buffer=CHUNK_SIZE,
    )


def rms(chunk: bytes) -> float:
    if len(chunk) < 2:
        return 0.0
    samples = struct.unpack(f"<{len(chunk)//2}h", chunk)
    mean_sq = sum(s * s for s in samples) / len(samples)
    return mean_sq ** 0.5


async def run_client(url: str, agent_mode: bool) -> None:
    import pyaudio
    audio, mic_stream = open_microphone()
    speaker_stream = open_speaker(audio) if agent_mode else None

    is_playing = asyncio.Event()

    async with websockets.connect(url, max_size=2**22) as socket:
        print(f"Connected to {url}")
        print("Speak into your microphone. Press Ctrl+C to stop.")

        async def send_audio() -> None:
            sequence = 0
            while True:
                chunk = await asyncio.to_thread(mic_stream.read, CHUNK_SIZE, False)
                if agent_mode and is_playing.is_set():
                    if rms(chunk) > BARGE_IN_THRESHOLD:
                        is_playing.clear()
                        if speaker_stream:
                            speaker_stream.stop_stream()
                        await socket.send(json.dumps({"type": "interrupt"}))
                        print("[BARGE-IN]")
                payload = {
                    "type": "audio",
                    "seq": sequence,
                    "sent_at_ms": int(time.time() * 1000),
                    "audio_b64": base64.b64encode(chunk).decode(),
                }
                await socket.send(json.dumps(payload))
                sequence += 1

        async def receive_events() -> None:
            async for raw in socket:
                event = json.loads(raw)
                t = event.get("type")

                if t == "transcript":
                    state = "FINAL" if event.get("is_final") else "PARTIAL"
                    print(f"[{state}] {event.get('transcript', '')} ({event.get('latency_ms', 0)} ms)")

                elif t == "agent_start":
                    is_playing.set()
                    if speaker_stream:
                        try:
                            speaker_stream.start_stream()
                        except Exception:
                            pass
                    print("[AGENT] speaking...")

                elif t == "audio_chunk" and speaker_stream:
                    if is_playing.is_set():
                        audio_bytes = base64.b64decode(event.get("audio_b64", ""))
                        await asyncio.to_thread(speaker_stream.write, audio_bytes)

                elif t == "agent_done":
                    is_playing.clear()
                    lat = event.get("latency", {})
                    print(
                        f"[DONE] stt={lat.get('stt_ms')}ms "
                        f"llm={lat.get('llm_first_token_ms')}ms "
                        f"tts={lat.get('tts_first_audio_ms')}ms "
                        f"total={lat.get('total_ms')}ms"
                    )

                elif t == "agent_error":
                    print(f"[AGENT ERROR] {event.get('message', '')}")

                elif t == "error":
                    print(f"[ERROR] {event.get('message', 'unknown error')}")

        sender = asyncio.create_task(send_audio())
        receiver = asyncio.create_task(receive_events())

        done, pending = await asyncio.wait(
            [sender, receiver],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in done:
            exc = task.exception()
            if exc is not None:
                raise exc

    mic_stream.stop_stream()
    mic_stream.close()
    if speaker_stream:
        speaker_stream.stop_stream()
        speaker_stream.close()
    audio.terminate()


def main() -> None:
    args = parse_args()
    url = args.url
    if args.agent and "/ws/transcribe" in url:
        url = url.replace("/ws/transcribe", "/ws/agent")
    try:
        asyncio.run(run_client(url, args.agent))
    except KeyboardInterrupt:
        print("Stopping client...")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify imports**

```bash
cd /home/developer/proyectos/gcp_audio && source venv/bin/activate
python -c "import client; print('client imports OK')"
```

Expected: `client imports OK`

- [ ] **Step 3: Commit**

```bash
git add client.py
git commit -m "feat: add audio output and barge-in to client"
```

---

### Task 6: End-to-end verification

**Files:** none — manual test only

- [ ] **Step 1: Start the server**

```bash
cd /home/developer/proyectos/gcp_audio && source venv/bin/activate
GOOGLE_CLOUD_PROJECT=<your-project-id> uvicorn server:app --reload
```

Expected: `Uvicorn running on http://127.0.0.1:8000`

- [ ] **Step 2: Start agent client**

In a second terminal:

```bash
cd /home/developer/proyectos/gcp_audio && source venv/bin/activate
GOOGLE_CLOUD_PROJECT=<your-project-id> python client.py --agent
```

Expected:
```
Connected to ws://127.0.0.1:8000/ws/agent
Speak into your microphone. Press Ctrl+C to stop.
```

- [ ] **Step 3: Speak a sentence and verify latency log**

Say something like "¿Cómo estás?" and wait for the agent to respond.

Expected server log line:
```
HH:MM:SS INFO [TURN] llm_first=NNNms tts_first=NNNms total=NNNms
```

Expected client output:
```
[FINAL] ¿Cómo estás? (NNNms)
[AGENT] speaking...
[DONE] stt=NNNms llm=NNNms tts=NNNms total=NNNms
```

- [ ] **Step 4: Test barge-in**

While the agent is speaking, speak loudly. Expected:
```
[BARGE-IN]
```

Agent audio stops. New utterance is processed.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "chore: voice agent pipeline complete - end-to-end verified"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| STT → LLM → TTS pipeline | Task 3 (AgentPipeline) |
| Sentence chunking before LLM finishes | Task 2 (SentenceChunker) + Task 3 |
| Streaming Gemini response | Task 3 (`stream=True`) |
| Per-sentence TTS synthesis | Task 3 (`_synthesize` per sentence) |
| Conversation history in memory | Task 4 (AgentSession._conversation) |
| Debounce 200ms on PARTIAL | Task 4 (_debounce_task) |
| FINAL commits to history | Task 4 (_on_stt_event is_final branch) |
| Cancel pipeline on interrupt | Task 4 (_on_interrupt) |
| `agent_start` / `audio_chunk` / `agent_done` messages | Task 4 (_start_pipeline) |
| Latency instrumentation: llm_first, tts_first, total | Task 3 (AgentPipeline.run) |
| `stt_ms` included in agent_done | Task 4 (passed to latency dict) |
| GOOGLE_CLOUD_PROJECT guard | Task 4 (/ws/agent endpoint) |
| Client audio output | Task 5 (open_speaker + audio_chunk handler) |
| Client barge-in with RMS | Task 5 (send_audio barge-in block) |
| `--agent` CLI flag | Task 5 (parse_args) |
| Error handling: Gemini fail → agent_error | Task 4 (_run except block) |
| Error handling: client disconnect → cancel | Task 4 (finally block) |

No gaps found.
