# Design

## Architecture

- `client.py`: Captures local microphone PCM audio and sends chunks to server WebSocket.
- `server.py`: Hosts FastAPI app, streams audio to Google Speech v2, computes latency metrics, and broadcasts transcript events.
- `static/index.html`: Live dashboard with podcast icon, transcript feed, and latency chart.

## Data Flow

1. Client connects to `ws://localhost:8000/ws/transcribe`.
2. Client sends JSON frames containing `audio_b64`, `seq`, and `sent_at_ms`.
3. Server enqueues decoded audio bytes into a streaming queue.
4. Speech worker thread reads queue and calls `SpeechClient.streaming_recognize`.
5. Server translates responses into events with transcript and latency.
6. Events are pushed to:
   - the originating client WebSocket
   - all monitor WebSockets (`/ws/monitor`) used by dashboard.
7. Dashboard renders live activity, transcript entries, and latency trend line.

## Interfaces

### WebSocket `/ws/transcribe`

Incoming message:

```json
{
  "type": "audio",
  "seq": 12,
  "sent_at_ms": 1713660000123,
  "audio_b64": "..."
}
```

Outgoing message:

```json
{
  "type": "transcript",
  "transcript": "hello world",
  "is_final": false,
  "latency_ms": 245,
  "server_time_ms": 1713660000456
}
```

### WebSocket `/ws/monitor`

Outgoing messages mirror transcript events and include live metrics snapshots.

### HTTP `GET /metrics`

Returns aggregate counters and latency summary.

## Error Handling Matrix

- Missing `GOOGLE_CLOUD_PROJECT`: reject session start with explicit error.
- Mic access failure in client: show local error and stop capture.
- Invalid WebSocket payload: send structured error frame and continue.
- Google streaming failure: send error frame to client and monitors.
- Monitor disconnect: remove subscriber safely.

## Testing Strategy

- Syntax/compile check for Python files.
- Import-time smoke run for server module.
- Manual demo test:
  1. Start server.
  2. Start client.
  3. Open dashboard.
  4. Speak and observe transcript + latency updates.
