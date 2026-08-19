"""Ports: framework-agnostic interfaces the agent's tools depend on.

No imports from `langchain`, `notion_client`, or any other vendor SDK in this file, on
purpose — that's what lets `agent/tools.py` depend on `EmailPort`/`NotionPort` without ever
knowing whether it's talking to a mock, the real Gmail API, or the real Notion API. Concrete
adapters live under `infra/` and are wired to a port only in `interfaces/api/app.py`, the
composition root. `Protocol` (not `ABC`) means adapters satisfy these structurally, without
inheriting from anything defined here.
"""

from typing import Protocol


class EmailPort(Protocol):
    async def send(self, to: str, subject: str, body: str) -> None: ...


class NotionPort(Protocol):
    async def create_page(self, *, title: str, content: str) -> str:
        """Create a page under whatever parent the adapter is configured with.

        Returns the created page's URL, so a tool can relay it back to the user.
        """
        ...

    async def search_pages(self, query: str) -> list[dict[str, str]]:
        """Search pages by title/content. Returns cheap summaries — `{title, id, url}` per
        result — never full page content, so a tool can list candidates without paying for
        every match's content up front."""
        ...

    async def get_page_content(self, page_id: str) -> str:
        """Fetch one specific page's full text content, flattened to plain text."""
        ...


class TranscriptionPort(Protocol):
    async def transcribe(self, audio: bytes, *, mime_type: str) -> str:
        """Transcribe spoken audio to plain text, in whatever language it's spoken in.

        No translation step — the caller's chat model is already multilingual and handles
        response language as part of ordinary conversation, so this only ever converts
        modality (audio -> text), never language.
        """
        ...
