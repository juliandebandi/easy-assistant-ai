"""Tests for the tool-calling loop and its human-in-the-loop gate.

The Notion test overrides the shared `chat_model` fixture (see conftest.py) with one scripted
to emit a tool call first — pytest resolves the module-local fixture over conftest's for every
test in this file, and `app`/`client` (which depend on `chat_model`) pick up the override
automatically. The `send_email` tests build their own app/client per test instead, since each
one needs a differently-scripted model across *two* HTTP calls (the initial tool call, then the
model's turn after the tool resumes) — a single module-level fixture can't vary per test.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from backend.agent.tools import build_tools
from backend.infra.email.mock_adapter import MockEmailAdapter
from backend.interfaces.api.app import create_app
from tests.conftest import FakeNotionPort, FakeTranscriptionPort, ToolCallingFakeChatModel


@pytest.fixture
def chat_model() -> ToolCallingFakeChatModel:
    return ToolCallingFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "create_notion_page",
                            "args": {
                                "title": "Team Sync Notes",
                                "content": "Notes from the meeting.",
                            },
                            "id": "call_1",
                        }
                    ],
                ),
                AIMessage(content="Created the page for you!"),
            ]
        )
    )


async def test_notion_tool_call_creates_a_page(
    client: AsyncClient, notion_port: FakeNotionPort
) -> None:
    response = await client.post(
        "/messages", json={"text": "create a notion page with today's notes"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Created the page for you!"
    assert notion_port.created == [
        {"title": "Team Sync Notes", "content": "Notes from the meeting."}
    ]


async def test_notion_search_then_get_page_chain(
    notion_port: FakeNotionPort, email_port: MockEmailAdapter
) -> None:
    """One HTTP call, two tool-call round trips: the chat<->tools loop keeps going until the
    model stops requesting tools, so a search followed by a targeted fetch happens within a
    single /messages request — the client never sees the intermediate search step."""
    notion_port.search_results = [
        {"title": "Q3 Planning", "id": "page-1", "url": "https://notion.so/page-1"},
    ]
    notion_port.page_content = {"page-1": "Q3 goals: ship the RAG phase."}

    chat_model = ToolCallingFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "search_notion_pages", "args": {"query": "Q3"}, "id": "call_1"}
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_notion_page",
                            "args": {"page_id": "page-1"},
                            "id": "call_2",
                        }
                    ],
                ),
                AIMessage(content="Q3 goals: ship the RAG phase."),
            ]
        )
    )
    async with _running_client(chat_model, notion_port, email_port) as (client, _app):
        response = await client.post("/messages", json={"text": "what are the Q3 goals?"})

    assert response.status_code == 200
    assert response.json()["reply"] == "Q3 goals: ship the RAG phase."


async def test_notion_search_with_no_matches(notion_port: FakeNotionPort) -> None:
    # Exercises the tool function directly rather than a full /messages round trip — no
    # model/graph involvement needed to check this one string-formatting edge case.
    tools = build_tools(notion_port=notion_port, email_port=MockEmailAdapter())
    search_tool = next(t for t in tools if t.name == "search_notion_pages")
    result = await search_tool.ainvoke({"query": "nothing matches this"})
    assert result == "No matching Notion pages found."


_SEND_EMAIL_CALL = {
    "name": "send_email",
    "args": {"to": "a@example.com", "subject": "Hi", "body": "Body text."},
    "id": "call_1",
}
_SEND_EMAIL_ARGS = _SEND_EMAIL_CALL["args"]


@asynccontextmanager
async def _running_client(
    chat_model: ToolCallingFakeChatModel,
    notion_port: FakeNotionPort,
    email_port: MockEmailAdapter,
) -> AsyncGenerator[tuple[AsyncClient, FastAPI]]:
    """Same wiring as conftest.py's `app`/`client` fixtures, as a helper instead of fixtures —
    these tests need a fresh `chat_model` scripted per test (two calls, two different scripted
    replies), which a single shared fixture can't express."""
    application = create_app(
        chat_model=chat_model,
        checkpointer=InMemorySaver(),
        notion_port=notion_port,
        email_port=email_port,
        transcription_port=FakeTranscriptionPort(),
    )
    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, application


def _confirmation_call(decision: str, reply_if_unclear: str | None = None) -> AIMessage:
    """Scripts the model's response to a `classify_reply()` call (see
    agent/confirmation.py) — `with_structured_output()` works by binding the Pydantic schema
    as a tool and parsing a matching tool call back out, so the fake needs to return one named
    after the schema class, not just plain text."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "ConfirmationDecision",
                "args": {"decision": decision, "reply_if_unclear": reply_if_unclear},
                "id": "confirm_1",
            }
        ],
    )


async def test_send_email_pauses_for_confirmation(
    notion_port: FakeNotionPort, email_port: MockEmailAdapter
) -> None:
    chat_model = ToolCallingFakeChatModel(
        messages=iter([AIMessage(content="", tool_calls=[_SEND_EMAIL_CALL])])
    )
    async with _running_client(chat_model, notion_port, email_port) as (client, _app):
        response = await client.post("/messages", json={"text": "email a@example.com"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] is None
    assert body["pending_confirmation"] == {"tool": "send_email", "args": _SEND_EMAIL_ARGS}
    # The gate's entire point: nothing was sent before confirmation.
    assert email_port.sent == []


async def test_send_email_approved_in_plain_conversation_sends(
    notion_port: FakeNotionPort, email_port: MockEmailAdapter
) -> None:
    chat_model = ToolCallingFakeChatModel(
        messages=iter(
            [
                AIMessage(content="", tool_calls=[_SEND_EMAIL_CALL]),
                _confirmation_call("approve"),
                AIMessage(content="Sent it!"),
            ]
        )
    )
    async with _running_client(chat_model, notion_port, email_port) as (client, _app):
        first = await client.post("/messages", json={"text": "email a@example.com"})
        thread_id = first.json()["thread_id"]

        # No special field — just a normal reply, like any other conversation turn.
        second = await client.post(
            "/messages", json={"thread_id": thread_id, "text": "yes, go ahead"}
        )

    assert second.status_code == 200
    body = second.json()
    assert body["reply"] == "Sent it!"
    assert body["pending_confirmation"] is None
    assert email_port.sent == [_SEND_EMAIL_ARGS]


async def test_send_email_declined_in_plain_conversation_does_not_send(
    notion_port: FakeNotionPort, email_port: MockEmailAdapter
) -> None:
    chat_model = ToolCallingFakeChatModel(
        messages=iter(
            [
                AIMessage(content="", tool_calls=[_SEND_EMAIL_CALL]),
                _confirmation_call("decline"),
                AIMessage(content="Okay, I won't send it."),
            ]
        )
    )
    async with _running_client(chat_model, notion_port, email_port) as (client, _app):
        first = await client.post("/messages", json={"text": "email a@example.com"})
        thread_id = first.json()["thread_id"]

        second = await client.post(
            "/messages", json={"thread_id": thread_id, "text": "no, don't send that"}
        )

    assert second.status_code == 200
    assert second.json()["reply"] == "Okay, I won't send it."
    assert email_port.sent == []


async def test_unclear_reply_answers_without_touching_the_paused_thread(
    notion_port: FakeNotionPort, email_port: MockEmailAdapter
) -> None:
    """A clarifying question shouldn't consume the pending confirmation — the thread must
    still be paused (and the email still unsent) afterward, and a later clear "yes" must still
    actually resume it. This is the behavior the aupdate_state() finding in the plan ruled out
    a different implementation for."""
    chat_model = ToolCallingFakeChatModel(
        messages=iter(
            [
                AIMessage(content="", tool_calls=[_SEND_EMAIL_CALL]),
                _confirmation_call("unclear", reply_if_unclear="It's going to a@example.com."),
                _confirmation_call("approve"),
                AIMessage(content="Sent it!"),
            ]
        )
    )
    async with _running_client(chat_model, notion_port, email_port) as (client, _app):
        first = await client.post("/messages", json={"text": "email a@example.com"})
        thread_id = first.json()["thread_id"]

        clarify = await client.post(
            "/messages", json={"thread_id": thread_id, "text": "wait, who's this going to?"}
        )
        assert clarify.status_code == 200
        clarify_body = clarify.json()
        assert clarify_body["reply"] == "It's going to a@example.com."
        # Still pending, same action — the graph was never touched by the unclear reply.
        assert clarify_body["pending_confirmation"] == {
            "tool": "send_email",
            "args": _SEND_EMAIL_ARGS,
        }
        assert email_port.sent == []

        confirm = await client.post(
            "/messages", json={"thread_id": thread_id, "text": "ok yes send it"}
        )

    assert confirm.status_code == 200
    assert confirm.json()["reply"] == "Sent it!"
    assert email_port.sent == [_SEND_EMAIL_ARGS]


async def test_start_new_conversation_tool_resets_thread(
    notion_port: FakeNotionPort, email_port: MockEmailAdapter
) -> None:
    chat_model = ToolCallingFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "start_new_conversation", "args": {}, "id": "call_1"}],
                ),
                AIMessage(content="Sure — let's start fresh! What would you like to talk about?"),
                AIMessage(content="Hi again!"),
            ]
        )
    )
    async with _running_client(chat_model, notion_port, email_port) as (client, app):
        first = await client.post("/messages", json={"text": "forget everything, start over"})
        thread_id = first.json()["thread_id"]
        assert (
            first.json()["reply"] == "Sure — let's start fresh! What would you like to talk about?"
        )

        second = await client.post("/messages", json={"thread_id": thread_id, "text": "hi again"})
        assert second.status_code == 200

        state = await app.state.graph.aget_state({"configurable": {"thread_id": thread_id}})

    human_turns = [m.content for m in state.values["messages"] if m.type == "human"]
    # Only "hi again" — the original "forget everything, start over" turn is gone, wiped by
    # the reset that ran after the first turn's reply was already built.
    assert human_turns == ["hi again"]


async def test_delete_messages_endpoint_resets_thread(client: AsyncClient, app: FastAPI) -> None:
    first = await client.post("/messages", json={"text": "hello"})
    thread_id = first.json()["thread_id"]

    delete_response = await client.delete(f"/messages/{thread_id}")
    assert delete_response.status_code == 204

    state = await app.state.graph.aget_state({"configurable": {"thread_id": thread_id}})
    assert state.values == {}
