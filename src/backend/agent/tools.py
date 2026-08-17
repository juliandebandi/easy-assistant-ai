"""LangChain tool wrappers around the application ports.

`build_tools()` is a factory of closures, deliberately mirroring `build_graph()`'s own shape
(see graph.py) — a plain function taking port instances and returning bound tools, so tests
substitute fakes the same way `create_app()` already substitutes a fake `chat_model` /
`checkpointer`. Each closure below depends only on the `Protocol` types from
`application.ports`, never a concrete adapter — that's what makes swapping an adapter (e.g. a
future real `GmailAdapter` for today's `MockEmailAdapter`) a one-line change in app.py, with
zero change to this file.
"""

from collections.abc import Sequence

from langchain_core.tools import BaseTool, tool
from langgraph.types import interrupt

from backend.application.ports import EmailPort, NotionPort


def build_tools(*, notion_port: NotionPort, email_port: EmailPort) -> Sequence[BaseTool]:
    @tool
    async def create_notion_page(title: str, content: str) -> str:
        """Create a new Notion page with the given title and content under the configured
        parent page."""
        url = await notion_port.create_page(title=title, content=content)
        return f"Notion page created: {url}"

    @tool
    async def search_notion_pages(query: str) -> str:
        """Search Notion for pages matching a query. Returns titles and ids only, not full
        content — call get_notion_page with a specific id once you know which page you need."""
        results = await notion_port.search_pages(query)
        if not results:
            return "No matching Notion pages found."
        return "\n".join(f"- {r['title']} (id: {r['id']}, url: {r['url']})" for r in results)

    @tool
    async def get_notion_page(page_id: str) -> str:
        """Fetch the full text content of one specific Notion page by id. Use
        search_notion_pages first to find the right id — don't guess one."""
        content = await notion_port.get_page_content(page_id)
        return content or "(this page has no text content)"

    @tool
    def start_new_conversation() -> str:
        """Start a completely fresh conversation, discarding everything discussed on this
        thread so far. Only call this when the user clearly asks to start over or forget the
        current conversation — not for an ordinary change of topic within the same chat.
        """
        # No side effect here on purpose — a tool mid-run can't safely delete its own thread's
        # checkpoint history out from under itself. routes/messages.py detects this tool by
        # name in the finished result and performs the actual reset *after* this turn's reply
        # is built, so the user still sees this acknowledgment before the thread goes empty.
        return "Sure — let's start fresh! What would you like to talk about?"

    @tool
    async def send_email(to: str, subject: str, body: str) -> str:
        """Send an email on the user's behalf. Requires human confirmation before sending —
        call this as soon as you have the recipient, subject, and body rather than waiting;
        the confirmation step happens automatically."""
        # `interrupt()` pauses the whole graph run here and re-raises a GraphInterrupt that
        # ToolNode explicitly lets bubble up rather than treating as a tool error — the
        # officially supported way to gate a tool's side effect on human approval. Everything
        # before this line may re-run on resume (interrupt()'s documented semantics: the node
        # restarts from the top); the actual send below only ever runs once, after approval.
        decision = interrupt(
            {"tool": "send_email", "args": {"to": to, "subject": subject, "body": body}}
        )
        if not decision.get("approved", False):
            return "Email not sent — the user declined to confirm."
        await email_port.send(to, subject, body)
        return f"Email sent to {to}."

    return [
        create_notion_page,
        search_notion_pages,
        get_notion_page,
        start_new_conversation,
        send_email,
    ]
