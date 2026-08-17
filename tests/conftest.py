"""Shared pytest fixtures."""

from collections.abc import AsyncGenerator, Callable, Sequence
from typing import Any, cast

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from backend.infra.email.mock_adapter import MockEmailAdapter
from backend.interfaces.api.app import create_app


class ToolCallingFakeChatModel(GenericFakeChatModel):
    """`GenericFakeChatModel` already accepts full `AIMessage` objects (including
    `tool_calls`) or plain strings from its `messages` iterator — that's why it's used here
    instead of `FakeListChatModel`, which only ever returns plain-text replies. The one thing
    it's still missing for this codebase's purposes: `build_graph()` unconditionally calls
    `model.bind_tools(tools)` now, and `BaseChatModel.bind_tools()` raises `NotImplementedError`
    by default — neither fake class overrides it. Overriding it here to just return `self`
    (ignoring the tool schemas), cast to the return type `bind_tools` promises, is enough: tests
    script the exact `AIMessage`, tool calls included, that the model should "decide" to return
    next, rather than relying on the fake to actually reason about which tool to call.
    """

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return cast("Runnable[LanguageModelInput, AIMessage]", self)


@pytest.fixture
def chat_model() -> ToolCallingFakeChatModel:
    """A scripted model that cycles through these responses in call order, completely
    ignoring what it's actually shown. Enough to test endpoint wiring and conversation
    persistence (see test_messages.py) without a real API key, network call, or the
    nondeterminism a real model would introduce — it says nothing about real model quality,
    which isn't what these tests are for.
    """
    return ToolCallingFakeChatModel(messages=iter(["first reply", "second reply", "third reply"]))


class FakeNotionPort:
    """Implements `NotionPort` (structurally — no inheritance needed) with no network call."""

    def __init__(self) -> None:
        self.created: list[dict[str, str]] = []

    async def create_page(self, *, title: str, content: str) -> str:
        self.created.append({"title": title, "content": content})
        return f"https://notion.so/fake-page-{len(self.created)}"


@pytest.fixture
def notion_port() -> FakeNotionPort:
    return FakeNotionPort()


@pytest.fixture
def email_port() -> MockEmailAdapter:
    """The real adapter, not a fake — it's already side-effect-free (no network call) with an
    inspectable `.sent` list, so HITL tests can assert on it directly. See its docstring for
    why it's the real production adapter for this phase, not just a test double."""
    return MockEmailAdapter()


@pytest.fixture
def checkpointer() -> BaseCheckpointSaver[str]:
    """An in-process, in-memory checkpointer instead of the real `AsyncPostgresSaver`.

    Real Postgres-backed persistence was verified directly against the running docker-compose
    stack (a live /messages call, then a follow-up on the same thread_id, through a real Gemini
    call) rather than through this test suite — see app.py's create_app() docstring for why:
    AsyncPostgresSaver's connection pool hangs when repeatedly opened/closed across the fresh
    event loop pytest-asyncio creates per test, on Windows specifically. InMemorySaver has no
    connection pool and no event-loop sensitivity, and it's still a real `BaseCheckpointSaver`
    exercising the exact same `compile(checkpointer=...)` path — good enough to test what these
    tests actually care about: does state persist correctly across two calls sharing a
    thread_id.
    """
    return InMemorySaver()


@pytest.fixture
async def app(
    chat_model: ToolCallingFakeChatModel,
    checkpointer: BaseCheckpointSaver[str],
    notion_port: FakeNotionPort,
    email_port: MockEmailAdapter,
) -> AsyncGenerator[FastAPI]:
    """A fully started app instance, lifespan included.

    `ASGITransport` alone never triggers FastAPI's startup/shutdown lifespan events. Without
    explicitly entering `lifespan_context` here, `app.state.graph` (set during startup — see
    app.py's lifespan) simply wouldn't exist yet, and every /messages request would fail with
    an AttributeError before the actual behavior under test ever ran.
    """
    application = create_app(
        chat_model=chat_model,
        checkpointer=checkpointer,
        notion_port=notion_port,
        email_port=email_port,
    )
    async with application.router.lifespan_context(application):
        yield application


@pytest.fixture
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient]:
    """An HTTP client wired directly to the ASGI app, no network socket involved.

    `ASGITransport` calls the app's ASGI callable in-process instead of routing requests
    through a real TCP port the way `TestClient`-over-`uvicorn` or hitting a running container
    would. That keeps the test suite from needing a server actually listening anywhere, and
    from paying socket setup/teardown cost per test.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
