"""Notion adapter: implements `application.ports.NotionPort` against the real Notion API.

Assumes `settings.notion_parent_page_id` names a Notion *page*, not a database. A page parent
always has a property literally named "title"; a database parent's title property can be
named anything, which would mean introspecting that database's schema first to find it. Out
of scope for the one page-creation action this phase needs — swapping to a database parent
later is a localized change to this file, not an architecture change.
"""

from typing import Any, cast

from notion_client import AsyncClient

from backend.config import Settings


class NotionAdapter:
    def __init__(self, client: AsyncClient, parent_page_id: str) -> None:
        self._client = client
        self._parent_page_id = parent_page_id

    async def create_page(self, *, title: str, content: str) -> str:
        page = cast(
            dict[str, Any],
            await self._client.pages.create(
                parent={"type": "page_id", "page_id": self._parent_page_id},
                properties={"title": {"title": [{"text": {"content": title}}]}},
                children=[
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"type": "text", "text": {"content": content}}]
                        },
                    }
                ],
            ),
        )
        return cast(str, page["url"])


def build_notion_adapter(settings: Settings) -> NotionAdapter:
    client = AsyncClient(auth=settings.notion_api_key.get_secret_value())
    return NotionAdapter(client, settings.notion_parent_page_id)
