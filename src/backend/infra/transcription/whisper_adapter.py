"""Whisper adapter: implements `application.ports.TranscriptionPort` against OpenAI's audio
transcription API.

Kept as its own adapter behind `TranscriptionPort`, rather than a call folded into
agent/llm.py's chat model, specifically so transcription stays independent of whichever
provider `llm_model` is configured for chat (see config.py's `openai_api_key` comment) — this
is what lets routes/messages.py's "transcription happens before this endpoint is ever reached"
design note hold regardless of which LLM provider is answering the conversation.
"""

from openai import AsyncOpenAI

from backend.config import Settings

# Whisper infers audio format from the filename extension it's given, not from a declared
# content-type — this maps the mime types real callers actually send (WhatsApp voice notes are
# audio/ogg; a direct API upload might be audio/mpeg or audio/wav) to an extension Whisper
# recognizes. Explicit on purpose rather than guessing from the mime type's subtype: a wrong
# extension makes Whisper misread, or outright reject, an otherwise valid file.
_EXTENSION_BY_MIME = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "mp4",
    "audio/m4a": "m4a",
    "audio/x-m4a": "m4a",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/oga": "oga",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
}


class WhisperAdapter:
    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    async def transcribe(self, audio: bytes, *, mime_type: str) -> str:
        extension = _EXTENSION_BY_MIME.get(mime_type)
        if extension is None:
            raise ValueError(f"unsupported audio mime type: {mime_type!r}")
        # A `(filename, content)` tuple, not a raw BytesIO — the SDK's `file` param reads the
        # format hint from the filename it's given in the tuple, so this is the whole reason
        # the extension lookup above exists.
        transcription = await self._client.audio.transcriptions.create(
            model="whisper-1", file=(f"audio.{extension}", audio)
        )
        return transcription.text


def build_whisper_adapter(settings: Settings) -> WhisperAdapter:
    client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    return WhisperAdapter(client)
