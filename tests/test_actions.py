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

from backend.infra.email.mock_adapter import MockEmailAdapter
from backend.interfaces.api.app import create_app
from tests.conftest import FakeNotionPort, ToolCallingFakeChatModel


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
    )
    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, application


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


async def test_send_email_confirm_true_sends(
    notion_port: FakeNotionPort, email_port: MockEmailAdapter
) -> None:
    chat_model = ToolCallingFakeChatModel(
        messages=iter(
            [
                AIMessage(content="", tool_calls=[_SEND_EMAIL_CALL]),
                AIMessage(content="Sent it!"),
            ]
        )
    )
    async with _running_client(chat_model, notion_port, email_port) as (client, _app):
        first = await client.post("/messages", json={"text": "email a@example.com"})
        thread_id = first.json()["thread_id"]

        second = await client.post("/messages", json={"thread_id": thread_id, "confirm": True})

    assert second.status_code == 200
    body = second.json()
    assert body["reply"] == "Sent it!"
    assert body["pending_confirmation"] is None
    assert email_port.sent == [_SEND_EMAIL_ARGS]


async def test_send_email_confirm_false_does_not_send(
    notion_port: FakeNotionPort, email_port: MockEmailAdapter
) -> None:
    chat_model = ToolCallingFakeChatModel(
        messages=iter(
            [
                AIMessage(content="", tool_calls=[_SEND_EMAIL_CALL]),
                AIMessage(content="Okay, I won't send it."),
            ]
        )
    )
    async with _running_client(chat_model, notion_port, email_port) as (client, _app):
        first = await client.post("/messages", json={"text": "email a@example.com"})
        thread_id = first.json()["thread_id"]

        second = await client.post("/messages", json={"thread_id": thread_id, "confirm": False})

    assert second.status_code == 200
    assert second.json()["reply"] == "Okay, I won't send it."
    assert email_port.sent == []


async def test_confirm_on_a_thread_with_nothing_pending_is_rejected(
    client: AsyncClient,
) -> None:
    response = await client.post(
        "/messages", json={"thread_id": "brand-new-thread", "confirm": True}
    )
    assert response.status_code == 409


async def test_plain_message_on_a_paused_thread_is_rejected(
    notion_port: FakeNotionPort, email_port: MockEmailAdapter
) -> None:
    chat_model = ToolCallingFakeChatModel(
        messages=iter([AIMessage(content="", tool_calls=[_SEND_EMAIL_CALL])])
    )
    async with _running_client(chat_model, notion_port, email_port) as (client, _app):
        first = await client.post("/messages", json={"text": "email a@example.com"})
        thread_id = first.json()["thread_id"]

        second = await client.post(
            "/messages", json={"thread_id": thread_id, "text": "hello again"}
        )

    assert second.status_code == 409
