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
