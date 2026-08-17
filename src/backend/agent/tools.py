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

    return [create_notion_page, send_email]
