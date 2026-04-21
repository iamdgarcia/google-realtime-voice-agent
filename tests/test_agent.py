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


import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_tts_mock(mock_tts, fake_audio: bytes) -> None:
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


def _make_genai_mock(mock_genai, tokens: list[str]) -> None:
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    fake_stream = MagicMock()
    fake_stream.__iter__ = MagicMock(
        return_value=iter([MagicMock(text=t) for t in tokens])
    )
    mock_client.models.generate_content_stream.return_value = fake_stream


@pytest.mark.asyncio
async def test_agent_pipeline_calls_on_audio_with_bytes():
    """Pipeline should invoke on_audio callback with bytes from TTS."""
    audio_chunks = []

    async def collect(chunk: bytes) -> None:
        audio_chunks.append(chunk)

    fake_audio = b"\x00\x01" * 100

    with patch("agent.genai") as mock_genai, \
         patch("agent.texttospeech") as mock_tts:

        _make_genai_mock(mock_genai, ["Hello", " there", "."])
        _make_tts_mock(mock_tts, fake_audio)

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
    with patch("agent.genai") as mock_genai, \
         patch("agent.texttospeech") as mock_tts:

        _make_genai_mock(mock_genai, ["Fine. "])
        _make_tts_mock(mock_tts, b"\x00\x01")

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
