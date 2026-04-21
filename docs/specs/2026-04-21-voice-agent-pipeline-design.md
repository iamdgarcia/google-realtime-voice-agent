# Voice Agent Pipeline — Design Spec

**Date:** 2026-04-21
**Status:** Draft
**Project:** gcp_audio

---

## Overview

A real-time voice agent that listens to the user, generates responses via Gemini Flash, and speaks them back via Google TTS. The system is designed to feel conversational — not turn-based — through aggressive streaming, early LLM invocation on partials, and client-side barge-in.

Latency benchmarking is a first-class goal: every pipeline stage is instrumented so bottlenecks can be identified and compared.

---

## Goals

- Sub-1s perceived latency from end of user speech to first audio output
- Interruptible: user can speak over the agent and the agent stops immediately
- Benchmarkable: per-stage latency logged on every turn (`stt_ms`, `llm_first_token_ms`, `tts_first_audio_ms`, `total_ms`)
- Single Google stack: Speech v2 (`latest_short`), Gemini 2.0 Flash, Cloud TTS
- MVP scope: conversation history in memory, no persistence

---

## Architecture

```
client.py                        server.py                        Google APIs
─────────────────────────────────────────────────────────────────────────────
mic audio (chunks) ─────────────→ AgentSession
                                      │
                                      ├─ STT stream (Speech v2 latest_short)
                                      │   ├─ PARTIAL → debounce 200ms → LLM call
                                      │   └─ FINAL  → commit to history
                                      │
                                      ├─ AgentPipeline (cancellable)
                                      │   ├─ Gemini 2.0 Flash (streaming)  ──→ Gemini API
                                      │   │   └─ tokens → sentence chunks
                                      │   └─ Google TTS (per sentence)     ──→ TTS API
                                      │       └─ audio bytes
                                      │
                                      ├─ audio_chunk frames ←──────────────────
                                      │
speaker ←── PyAudio output ──────────
     │
     ├─ RMS monitor while playing
     └─ barge-in → {"type":"interrupt"} ──→ cancel pipeline
```

---

## Components

### `agent.py` — AgentPipeline

Owns LLM streaming, sentence chunking, TTS synthesis, and the rolling prompt state.

**Rolling prompt state:**
```python
{
    "conversation": [
        {"role": "user", "content": "..."},
        {"role": "model", "content": "..."},
    ],
    "current_user_utterance": "I want to boo..."  # updated on each PARTIAL
}
```

**Sentence chunking:** Accumulate Gemini tokens until a sentence boundary (`[.!?]` followed by whitespace or end-of-stream) → call `SynthesizeSpeech` → emit audio bytes. Each sentence is synthesized independently so audio starts playing before the LLM finishes.

**Cancellation:** Each pipeline run is an `asyncio.Task`. Cancellation is cooperative — Gemini stream and pending TTS calls are abandoned immediately when `.cancel()` is called.

**Latency instrumentation:** Records timestamps at each stage boundary and emits them with `agent_done`.

**System prompt:** Kept short (< 100 tokens). Flash latency is sensitive to prompt length.

---

### `server.py` — AgentSession

Replaces `SpeechSession`. Manages the full per-connection lifecycle.

**On `PARTIAL` event (LISTENING state only — ignored during PLAYING):**
1. Update `current_user_utterance` in rolling state
2. Cancel any in-flight `AgentPipeline` task
3. Start debounce timer (200ms)
4. If timer fires without a new PARTIAL → start new `AgentPipeline` call (sends `agent_start`)

**On `FINAL` event:**
1. Cancel debounce timer
2. Cancel any in-flight pipeline
3. Commit utterance to conversation history
4. Start `AgentPipeline` with finalized text (sends `agent_start`)

**On `{"type":"interrupt"}` from client:**
1. Cancel in-flight pipeline immediately
2. Reset to listening state (STT continues uninterrupted)

**Conversation history** stored as a list of `{"role", "content"}` dicts, in memory, scoped to the WebSocket session lifetime.

---

### `client.py` — updates

**Audio output:** Open a second PyAudio stream for output (LINEAR16, 16000Hz). Play `audio_chunk` frames in order as they arrive.

**Barge-in detection:** While `is_playing` flag is True, read mic chunks concurrently. If `rms(chunk) > BARGE_IN_THRESHOLD` (default: 300):
1. Stop audio output immediately
2. Send `{"type": "interrupt"}` to server
3. Clear `is_playing` flag
4. Resume STT streaming (already running)

**State machine:**
```
LISTENING → (PARTIAL received) → LISTENING (update utterance)
LISTENING → (agent_start) → PLAYING
PLAYING   → (barge-in detected) → LISTENING
PLAYING   → (agent_done) → LISTENING
```

---

## WebSocket Protocol

### Existing (unchanged)
```json
{"type": "audio", "seq": 12, "sent_at_ms": 1234, "audio_b64": "..."}
{"type": "transcript", "transcript": "...", "is_final": true, "latency_ms": 400}
{"type": "error", "message": "..."}
```

### New — Client → Server
```json
{"type": "interrupt"}
```

### New — Server → Client
```json
{"type": "agent_start"}
{"type": "audio_chunk", "audio_b64": "...", "sample_rate": 16000}
{"type": "agent_done", "latency": {
    "stt_ms": 400,
    "llm_first_token_ms": 380,
    "tts_first_audio_ms": 210,
    "total_ms": 990
}}
{"type": "agent_error", "message": "..."}
```

---

## Latency Instrumentation

Every turn logs a structured line to the server console:

```
14:22:01 INFO [TURN] stt=412ms llm_first=340ms tts_first=215ms total=967ms
```

`total_ms` is measured from the FINAL transcript arriving to the first audio chunk being sent to the client.

Per-stage definitions:
- `stt_ms`: already measured (end-of-speech offset to FINAL event)
- `llm_first_token_ms`: FINAL event → first Gemini token received
- `tts_first_audio_ms`: first complete sentence → first audio bytes from TTS
- `total_ms`: FINAL event → first `audio_chunk` sent to client

---

## Error Handling

| Scenario | Behavior |
|---|---|
| Gemini call fails | Send `agent_error`, keep session alive, history unchanged |
| TTS synthesis fails | Send `agent_error`, discard remaining audio for that turn |
| Barge-in during TTS | Cancel pipeline silently, restart listening |
| New PARTIAL cancels LLM | Silent cancel, no error event |
| Client disconnects mid-response | Cancel all tasks, clean up audio queue |
| `GOOGLE_CLOUD_PROJECT` not set | Reject session with explicit error on connect |

All errors tagged with stage: `[LLM]`, `[TTS]`, `[STT]`.

---

## Configuration

All via environment variables:

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | — | Required |
| `SPEECH_LOCATION` | `us-central1` | STT region |
| `SPEECH_MODEL` | `latest_short` | STT model |
| `SPEECH_LANGUAGE` | `es-ES` | Language code |
| `GEMINI_MODEL` | `gemini-2.0-flash` | LLM model |
| `TTS_VOICE` | `es-ES-Standard-A` | TTS voice name |
| `BARGE_IN_THRESHOLD` | `300` | RMS threshold for barge-in |
| `LLM_DEBOUNCE_MS` | `200` | Delay before LLM call on partial |

---

## Out of Scope (MVP)

- Persistent conversation history (across sessions)
- WebRTC VAD (Silero or py-webrtcvad) — stretch goal for improved silence detection
- Direct audio input to Gemini 3 Flash Preview — benchmarking mode, deferred
- Dashboard UI updates for agent state
- Echo cancellation (assumes earphones)
- Multi-user sessions
