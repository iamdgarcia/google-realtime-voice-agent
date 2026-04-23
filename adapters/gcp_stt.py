import queue
import threading
import time
from typing import Iterable, Optional

from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech as cloud_speech_types

import config


def _api_endpoint(location: str) -> str:
    if location == "global":
        return "speech.googleapis.com"
    return f"{location}-speech.googleapis.com"


def create_client() -> SpeechClient:
    return SpeechClient(
        client_options=ClientOptions(api_endpoint=_api_endpoint(config.SPEECH_LOCATION))
    )


class SttStream:
    """Audio queue + STT request generator for one streaming recognition session."""

    def __init__(self) -> None:
        self._queue: queue.Queue[Optional[tuple[bytes, int, int]]] = queue.Queue()
        self._stop = threading.Event()
        self.stream_start_ms = 0
        self.last_network_ms = 0
        self.last_queue_ms = 0

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def stop(self) -> None:
        self._stop.set()

    def push(self, audio_bytes: bytes, sent_at_ms: int) -> None:
        received_ms = int(time.time() * 1000)
        self._queue.put((audio_bytes, sent_at_ms, received_ms))

    def close(self) -> None:
        self._queue.put(None)

    def swap_queue(self) -> None:
        """Replace the audio queue and unblock the stale generator.

        Chirp 3 closes the gRPC stream after each FINAL result. The old
        _build_requests generator may be blocked in queue.get(); sending None
        lets it exit cleanly while the new stream reads from a fresh queue.
        """
        old = self._queue
        self._queue = queue.Queue()
        old.put(None)

    def requests(self) -> Iterable[cloud_speech_types.StreamingRecognizeRequest]:
        recognizer = (
            f"projects/{config.GOOGLE_CLOUD_PROJECT}"
            f"/locations/{config.SPEECH_LOCATION}/recognizers/_"
        )
        recognition_config = cloud_speech_types.RecognitionConfig(
            explicit_decoding_config=cloud_speech_types.ExplicitDecodingConfig(
                encoding=cloud_speech_types.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                audio_channel_count=1,
            ),
            language_codes=[config.SPEECH_LANGUAGE],
            model=config.SPEECH_MODEL,
        )
        streaming_config = cloud_speech_types.StreamingRecognitionConfig(
            config=recognition_config,
        )
        if config.SPEECH_MODEL in ("latest_short", "latest_long"):
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
        while not self._stop.is_set():
            item = self._queue.get()
            if item is None:
                break
            audio_chunk, sent_at_ms, received_ms = item
            consumed_ms = int(time.time() * 1000)
            self.last_network_ms = received_ms - sent_at_ms
            self.last_queue_ms = consumed_ms - received_ms
            if first:
                self.stream_start_ms = sent_at_ms
                first = False
            yield cloud_speech_types.StreamingRecognizeRequest(audio=audio_chunk)
