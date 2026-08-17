"""Notion adapter: implements `application.ports.NotionPort` against the real Notion API.

`create_page` assumes `settings.notion_parent_page_id` names a Notion *page*, not a database.
A page parent always has a property literally named "title"; a database parent's title
property can be named anything, which would mean introspecting that database's schema first
to find it. Out of scope for the one page-creation action this phase needs — swapping to a
database parent later is a localized change to this file, not an architecture change.

`search_pages`/`get_page_content` don't have that limitation — `_extract_title` finds
whichever property Notion itself marks `type: "title"` (every page has exactly one, regardless
of its key name), so search results read correctly whether a match is a plain page or a
database row.
"""

from typing import Any, cast

from notion_client import AsyncClient

from backend.config import Settings


def _extract_title(properties: dict[str, Any]) -> str:
    for prop in properties.values():
        if prop.get("type") == "title":
            return "".join(t.get("plain_text", "") for t in prop.get("title", []))
    return "(untitled)"


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

    async def search_pages(self, query: str) -> list[dict[str, str]]:
        response = cast(
            dict[str, Any],
            await self._client.search(query=query, filter={"property": "object", "value": "page"}),
        )
        return [
            {"title": _extract_title(page["properties"]), "id": page["id"], "url": page["url"]}
            for page in response["results"]
        ]

    async def get_page_content(self, page_id: str) -> str:
        # Notion stores content as a tree of blocks, not a flat string. Only block types that
        # carry a `rich_text` array under their own type key are flattened here (paragraphs,
        # headings, list items, to-dos, quotes, callouts, ...) — tables, images, embeds, and a
        # few other structural block types are silently skipped rather than rendered, an
        # acceptable simplification for "give the model something to read," not a full export.
        lines: list[str] = []
        cursor: str | None = None
        while True:
            kwargs: dict[str, Any] = {"block_id": page_id}
            if cursor is not None:
                kwargs["start_cursor"] = cursor
            response = cast(dict[str, Any], await self._client.blocks.children.list(**kwargs))
            for block in response["results"]:
                block_type = block["type"]
                rich_text = block.get(block_type, {}).get("rich_text", [])
                text = "".join(t.get("plain_text", "") for t in rich_text)
                if text:
                    lines.append(text)
                if block.get("has_children"):
                    lines.append(await self.get_page_content(block["id"]))
            if not response.get("has_more"):
                break
            cursor = response.get("next_cursor")
        return "\n".join(lines)


def build_notion_adapter(settings: Settings) -> NotionAdapter:
    client = AsyncClient(auth=settings.notion_api_key.get_secret_value())
    return NotionAdapter(client, settings.notion_parent_page_id)
