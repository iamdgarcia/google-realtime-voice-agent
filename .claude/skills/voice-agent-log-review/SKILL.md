---
name: voice-agent-log-review
description: Use when reviewing gcp_audio server logs to identify and fix client message synchronisation issues in the STT→LLM→TTS pipeline.
---

# Voice Agent Log Review

## Overview

Reviews `logs/server.log` for sync breakages between server-internal state and the messages visible to the client. The invariant is simple: every `agent_start` must be matched by exactly one of `agent_done`, `agent_interrupted`, or `agent_error`.

## Key Concepts

**Generation counter (`gen`)** — incremented every time a pipeline is cancelled. Lets you trace which pipeline produced each log line and spot stale callbacks.

**`_is_playing`** — server-side boolean. `True` means a pipeline has claimed the audio channel. When it flips to `False` without a matching terminal message to the client, the client is left hanging.

**Client-visible messages**

| Message | Meaning |
|---|---|
| `agent_start` | Agent begins speaking; client should mute mic |
| `audio_chunk` | PCM payload |
| `agent_done` | Pipeline finished cleanly |
| `agent_interrupted` | Barge-in cancelled the pipeline |
| `agent_error` | Fatal pipeline or STT error |

## Review Checklist

1. **Find every `agent_start`** — grep `sent msg type='agent_start'`
2. **Find terminal messages** — grep `agent_done\|agent_interrupted\|agent_error`
3. **Count them** — `agent_start` count must equal terminal count
4. **Check gen jumps** — each `cancelling task gen=N → gen=M` should have M = N+1; a jump of 2 means double-cancel
5. **Check `_is_playing` resets** — look for `clearing is_playing` vs `skipping is_playing reset`; a skip without a subsequent matching terminal message is a bug
6. **STT errors after interrupt** — `409 Stream timed out` after a barge-in is expected (client stopped sending audio); treat as retriable, not fatal

## Expected Happy-Path Flow

```
agent_start
  [SEND] audio_chunk × N
agent_done
```

## Barge-in Flow (correct)

```
agent_start
  [SEND] audio_chunk × N      ← partial playback
[BARGE-IN] interrupt received
agent_interrupted              ← client knows agent stopped
```

## Common Bugs and Fixes

| Symptom in log | Root cause | Fix |
|---|---|---|
| `agent_start` with no terminal message | `_on_interrupt` doesn't notify client | Put `agent_interrupted` in `_send_queue` when `was_playing` |
| `409 Stream timed out` → `agent_error` | STT timeout after interrupt treated as fatal | Catch `api_exceptions.Aborted` separately; swap queue and reopen stream |
| `gen` jumps by 2 | Double `_cancel_pipeline` call | Call cancel only once (inside `_start_pipeline`) |
| LLM hangs silently after N turns | Conversation history malformed — all user turns, no model turns, plus duplicate current user message | Save user+model turns only on successful pipeline completion; don't append in `_on_stt_event` |
| Agent responds to its own speech | FINAL event has no `is_playing` guard — STT picks up TTS output (echo) | Ignore FINAL when `is_playing=True`; return early |
| Pipeline waits indefinitely | No timeout on `audio_queue.get()` in `AgentPipeline.run()` | `asyncio.wait_for(audio_queue.get(), timeout=LLM_TIMEOUT_S)` |

## Conversation State Invariants

Check with `grep "conversation now" logs/server.log`:

- Turns should grow by 2 per completed exchange (user + model)
- No completed exchange → 0 new turns (cancelled/interrupted pipelines don't write history)
- All-user conversation (no model turns) → Gemini will hang or produce wrong output

## Echo Detection

If the PARTIAL/FINAL transcripts repeat phrases the agent just said, the mic is picking up speaker output. Signs:

- PARTIAL arrives < 500ms after `agent_done`
- FINAL text matches the last `response=` in `[PIPELINE] done`

Fix: guard FINAL with `is_playing` check, or add client-side AEC.

## How to Apply

1. `grep -n "agent_start\|agent_done\|agent_interrupted\|agent_cancelled\|agent_error" logs/server.log`
2. Pair each `agent_start` with a terminal. Flag orphans.
3. `grep "conversation now" logs/server.log` — verify turns grow by 2 each exchange.
4. `grep "FINAL.*is_playing=True" logs/server.log` — these are potential echo events being wrongly processed.
5. For `409 Aborted` STT errors: catch `api_exceptions.Aborted` before the generic `except Exception`, swap the audio queue, and continue the worker loop instead of breaking.
