# Requirements

## Scope

Build a fully working real-time transcription demo with a client, a server, and a frontend view that visualizes live audio flow and transcription speed.

## EARS Requirements

- WHEN the user starts a demo session, THE SYSTEM SHALL stream microphone audio from a client to the server over WebSocket.
- WHEN the server receives audio chunks, THE SYSTEM SHALL forward those chunks to Google Speech-to-Text v2 streaming recognition.
- WHEN Google returns recognition results, THE SYSTEM SHALL emit transcript events to the connected client and dashboard monitors in near real time.
- WHEN transcript events are emitted, THE SYSTEM SHALL include latency metrics that represent end-to-end response speed.
- WHEN the dashboard is open, THE SYSTEM SHALL display a live status indicator (podcast/microphone style) showing active streaming state.
- WHEN latency metrics are available, THE SYSTEM SHALL render an updating latency chart and summary statistics (avg, p50, p95).
- IF required cloud configuration is missing, THEN THE SYSTEM SHALL return actionable error messages.
- IF microphone capture fails, THEN THE SYSTEM SHALL report microphone setup errors without crashing the whole service.
- WHERE no active session exists, THE SYSTEM SHALL keep the dashboard in an idle state.

## Constraints

- Python 3.10+ compatible implementation.
- Google Speech-to-Text v2 regional endpoint must match resource location.
- Use minimal dependencies and a single-process local demo setup.
