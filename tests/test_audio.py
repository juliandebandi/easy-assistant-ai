"""Tests for POST /messages/audio: transcription happens before the shared turn logic in
routes/messages.py's `_handle_turn`, so these tests focus on the audio-specific edges
(transcription wiring, size limit, blank-transcript rejection, unsupported format) rather than
re-testing HITL/persistence behavior already covered by test_messages.py and test_actions.py —
`_handle_turn` is the same code path either way.
"""

from fastapi import FastAPI
from httpx import AsyncClient
from openai import AsyncOpenAI

from backend.infra.transcription.whisper_adapter import WhisperAdapter
from tests.conftest import FakeTranscriptionPort


async def test_audio_message_transcribes_and_replies(
    client: AsyncClient, transcription_port: FakeTranscriptionPort
) -> None:
    transcription_port.transcript = "what's on my calendar today?"

    response = await client.post(
        "/messages/audio",
        files={"audio": ("note.ogg", b"fake-audio-bytes", "audio/ogg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"]
    assert body["reply"] == "first reply"  # the fake chat model's canned first response
    assert transcription_port.calls == [{"audio": b"fake-audio-bytes", "mime_type": "audio/ogg"}]


async def test_audio_message_continues_an_existing_text_thread(
    client: AsyncClient, app: FastAPI, transcription_port: FakeTranscriptionPort
) -> None:
    first = await client.post("/messages", json={"text": "hello"})
    thread_id = first.json()["thread_id"]

    transcription_port.transcript = "follow up by voice"
    second = await client.post(
        "/messages/audio",
        data={"thread_id": thread_id},
        files={"audio": ("note.ogg", b"more-audio-bytes", "audio/ogg")},
    )

    assert second.status_code == 200
    assert second.json()["thread_id"] == thread_id
    assert second.json()["reply"] == "second reply"

    # The transcript, not raw bytes, is what actually reached the graph as a human turn —
    # proves the audio route feeds _handle_turn the same way the text route does.
    state = await app.state.graph.aget_state({"configurable": {"thread_id": thread_id}})
    human_turns = [m.content for m in state.values["messages"] if m.type == "human"]
    assert human_turns == ["hello", "follow up by voice"]


async def test_audio_message_rejects_oversized_file(client: AsyncClient) -> None:
    oversized = b"x" * (25 * 1024 * 1024 + 1)

    response = await client.post(
        "/messages/audio",
        files={"audio": ("note.ogg", oversized, "audio/ogg")},
    )

    assert response.status_code == 413


async def test_audio_message_rejects_blank_transcription(
    client: AsyncClient, transcription_port: FakeTranscriptionPort
) -> None:
    transcription_port.transcript = "   "

    response = await client.post(
        "/messages/audio",
        files={"audio": ("silence.ogg", b"fake-audio-bytes", "audio/ogg")},
    )

    assert response.status_code == 422


class _RejectingTranscriptionPort:
    """Implements `TranscriptionPort` structurally, always rejecting — stands in for
    WhisperAdapter's own unsupported-mime-type behavior (see test below) to check the route
    maps that `ValueError` to a 400, not a raw 500."""

    async def transcribe(self, audio: bytes, *, mime_type: str) -> str:
        raise ValueError(f"unsupported audio mime type: {mime_type!r}")


async def test_audio_message_maps_unsupported_format_to_400(
    client: AsyncClient, app: FastAPI
) -> None:
    app.state.transcription_port = _RejectingTranscriptionPort()

    response = await client.post(
        "/messages/audio",
        files={"audio": ("note.xyz", b"fake-audio-bytes", "application/x-unknown")},
    )

    assert response.status_code == 400


async def test_whisper_adapter_rejects_unsupported_mime_type() -> None:
    # No network call ever happens here — the mime-type lookup raises before the client is
    # touched, so a placeholder API key is enough to construct it.
    adapter = WhisperAdapter(AsyncOpenAI(api_key="test-key"))

    try:
        await adapter.transcribe(b"fake-audio-bytes", mime_type="application/x-unknown")
    except ValueError as exc:
        assert "application/x-unknown" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unsupported mime type")
