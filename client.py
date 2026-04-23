import argparse
import asyncio
import base64
import json
import struct
import threading
import time
from typing import Any

import websockets

from config import BARGE_IN_THRESHOLD

CHUNK_SIZE = 1024
RATE = 16000
CHANNELS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime Voice Agent CLI client")
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8000/ws/agent",
        help="Server websocket endpoint",
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


async def run_client(url: str) -> None:
    import pyaudio
    audio, mic_stream = open_microphone()
    speaker_stream = open_speaker(audio)
    speaker_lock = threading.Lock()
    is_playing = asyncio.Event()

    async with websockets.connect(url, max_size=2**22) as socket:
        print(f"Connected to {url}")
        print("Speak into your microphone. Press Ctrl+C to stop.")

        async def send_audio() -> None:
            sequence = 0
            while True:
                chunk = await asyncio.to_thread(mic_stream.read, CHUNK_SIZE, False)
                if is_playing.is_set():
                    if rms(chunk) > BARGE_IN_THRESHOLD:
                        is_playing.clear()
                        def _stop() -> None:
                            with speaker_lock:
                                speaker_stream.stop_stream()
                        asyncio.create_task(asyncio.to_thread(_stop))
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
                    try:
                        speaker_stream.start_stream()
                    except Exception:
                        pass
                    print("[AGENT] speaking...")

                elif t == "audio_chunk":
                    if is_playing.is_set():
                        audio_bytes = base64.b64decode(event.get("audio_b64", ""))
                        piece_size = CHUNK_SIZE * 2 * CHANNELS
                        try:
                            for i in range(0, len(audio_bytes), piece_size):
                                if not is_playing.is_set():
                                    break
                                piece = audio_bytes[i:i + piece_size]
                                def _write(p: bytes = piece) -> None:
                                    with speaker_lock:
                                        if is_playing.is_set():
                                            speaker_stream.write(p)
                                await asyncio.to_thread(_write)
                        except OSError:
                            pass

                elif t == "agent_done":
                    is_playing.clear()
                    lat = event.get("latency", {})
                    print(
                        f"[DONE] stt={lat.get('stt_ms')}ms "
                        f"llm={lat.get('llm_first_token_ms')}ms "
                        f"tts={lat.get('tts_first_audio_ms')}ms "
                        f"total={lat.get('total_ms')}ms"
                    )

                elif t == "agent_interrupted":
                    is_playing.clear()
                    print("[AGENT] interrupted")

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
    try:
        asyncio.run(run_client(args.url))
    except KeyboardInterrupt:
        print("Stopping client...")


if __name__ == "__main__":
    main()
