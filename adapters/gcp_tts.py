from google.cloud import texttospeech

import config


def synthesize(text: str) -> bytes:
    language_code = "-".join(config.TTS_VOICE.split("-")[:2])
    client = texttospeech.TextToSpeechClient()
    response = client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=text),
        voice=texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=config.TTS_VOICE,
        ),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
        ),
    )
    audio = response.audio_content
    # TTS returns RIFF/WAV; strip 44-byte header to stream raw PCM to the client
    return audio[44:] if audio[:4] == b"RIFF" else audio
