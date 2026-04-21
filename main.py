import os
from itertools import chain
from typing import Generator, Iterable

from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech as cloud_speech_types

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = "us-central1"
MODEL = "chirp"

CHUNK_SIZE = 1024
SAMPLE_RATE = 16000
CHANNELS = 1


def microphone_audio_stream() -> Generator[bytes, None, None]:
    """Yield raw PCM chunks captured from the default microphone.

    Yields:
        bytes: A single chunk of microphone audio in LINEAR16 format.
    """
    try:
        import pyaudio
    except ImportError as exc:
        raise RuntimeError(
            "PyAudio is required for microphone streaming. "
            "Install it with `pip install pyaudio`."
        ) from exc

    audio_interface = pyaudio.PyAudio()
    try:
        stream = audio_interface.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )
    except Exception as exc:
        audio_interface.terminate()
        raise RuntimeError(
            "Unable to open default microphone input device. "
            "Check your OS audio input settings and permissions."
        ) from exc

    try:
        while True:
            try:
                yield stream.read(CHUNK_SIZE, exception_on_overflow=False)
            except Exception as exc:
                raise RuntimeError(
                    "Microphone read failed while streaming audio."
                ) from exc
    finally:
        stream.stop_stream()
        stream.close()
        audio_interface.terminate()


def build_streaming_requests(
    recognizer: str,
    audio_chunks: Iterable[bytes],
) -> Generator[cloud_speech_types.StreamingRecognizeRequest, None, None]:
    """Build and yield streaming recognition requests.

    Args:
        recognizer (str): Fully qualified recognizer resource path.
        audio_chunks (Iterable[bytes]): PCM audio chunks.

    Yields:
        cloud_speech_types.StreamingRecognizeRequest: Config then audio requests.
    """
    recognition_config = cloud_speech_types.RecognitionConfig(
        explicit_decoding_config=cloud_speech_types.ExplicitDecodingConfig(
            encoding=cloud_speech_types.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            audio_channel_count=CHANNELS,
        ),
        language_codes=["en-US"],
        model=MODEL,
    )
    streaming_config = cloud_speech_types.StreamingRecognitionConfig(
        config=recognition_config
    )

    yield cloud_speech_types.StreamingRecognizeRequest(
        recognizer=recognizer,
        streaming_config=streaming_config,
    )

    for chunk in audio_chunks:
        yield cloud_speech_types.StreamingRecognizeRequest(audio=chunk)


def transcribe_streaming_v2() -> list[cloud_speech_types.StreamingRecognizeResponse]:
    """Transcribe live microphone audio using Speech-to-Text streaming v2.

    Returns:
        list[cloud_speech_types.StreamingRecognizeResponse]: Recognition responses.
    """
    if not PROJECT_ID:
        raise RuntimeError("Set GOOGLE_CLOUD_PROJECT before starting the script.")

    client = SpeechClient(
        client_options=ClientOptions(
            api_endpoint=f"{LOCATION}-speech.googleapis.com"
        )
    )
    recognizer = (
        f"projects/{PROJECT_ID}/locations/{LOCATION}/recognizers/_"
    )
    audio_chunks = microphone_audio_stream()
    try:
        first_chunk = next(audio_chunks)
    except Exception as exc:
        raise RuntimeError(
            "Microphone initialization failed before request streaming started."
        ) from exc

    requests = build_streaming_requests(
        recognizer,
        chain([first_chunk], audio_chunks),
    )

    responses_iterator = client.streaming_recognize(requests=requests)
    responses: list[cloud_speech_types.StreamingRecognizeResponse] = []
    for response in responses_iterator:
        responses.append(response)
        for result in response.results:
            if result.alternatives:
                transcript = result.alternatives[0].transcript
                print(f"Transcript: {transcript}")

    return responses


if __name__ == "__main__":
    transcribe_streaming_v2()
