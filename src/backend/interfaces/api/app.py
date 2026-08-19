"""FastAPI application factory.

A `create_app()` function instead of a module-level `app = FastAPI()` for one concrete reason:
tests need to build an app instance and swap dependencies (e.g. `get_settings`, `get_db_session`)
*before* anything registers against a real database. A module-level `app` is constructed the
moment the module is imported, which is too early to inject test doubles cleanly. The factory
also keeps import-time side effects (route registration, middleware) explicit and out of module
scope, which matters once this app has more than one entrypoint (uvicorn locally, `fastapi run`
in the container, and eventually a test client).

Optional parameters accumulate here for exactly the same reason the factory pattern exists in
the first place — letting tests substitute real dependencies with cheap, deterministic doubles:

- `chat_model`: tests pass a `FakeListChatModel` instead of a real Gemini client, so `/messages`
  can be tested (thread persistence, request/response contract) without a network call, an API
  key, or nondeterministic model output.
- `checkpointer`: tests pass an `InMemorySaver` instead of the real `AsyncPostgresSaver`. This
  isn't cutting a corner — real Postgres-backed persistence was verified directly (a live
  `/messages` call, then a follow-up on the same thread_id, through the actual running
  docker-compose stack and a real Gemini call) rather than through this test suite, and
  `AsyncPostgresSaver`'s connection pool turned out to hang when repeatedly opened and closed
  across the fresh event loop pytest-asyncio creates per test on Windows — a real driver/OS
  interaction, not a bug in this code. `InMemorySaver` has no connection pool and no event-loop
  sensitivity, so it sidesteps the problem entirely while still exercising the exact same
  code path (a real `BaseCheckpointSaver`, same `compile(checkpointer=...)` call) for what
  these tests actually care about: does state correctly persist across two calls sharing a
  thread_id.
- `notion_port`: tests pass a fake `NotionPort` instead of the real `NotionAdapter`, so the
  tool-calling path can be tested without a Notion integration token or a real workspace to
  write into. Same DI seam, same reason.
- `email_port`: no real adapter exists yet to skip (see infra/email/mock_adapter.py) — but the
  seam is here regardless, so `send_email`'s human-in-the-loop gate (see agent/tools.py and
  routes/messages.py) can be tested deterministically against an adapter with an inspectable
  `.sent` list, the same way `checkpointer` lets state persistence be asserted on directly.
- `transcription_port`: tests pass a fake `TranscriptionPort` instead of the real
  `WhisperAdapter`, so `POST /messages/audio` can be tested without an OpenAI API key or a real
  Whisper call. Same DI seam, same reason.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from backend.agent.graph import build_graph
from backend.agent.llm import get_chat_model
from backend.agent.tools import build_tools
from backend.application.ports import EmailPort, NotionPort, TranscriptionPort
from backend.config import get_settings
from backend.infra.db.checkpointer import get_checkpointer_dsn
from backend.infra.email.mock_adapter import MockEmailAdapter
from backend.infra.notion.adapter import build_notion_adapter
from backend.infra.observability.langfuse import configure_langfuse
from backend.infra.transcription.whisper_adapter import build_whisper_adapter
from backend.interfaces.api.routes import health, messages


def create_app(
    *,
    chat_model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver[str] | None = None,
    notion_port: NotionPort | None = None,
    email_port: EmailPort | None = None,
    transcription_port: TranscriptionPort | None = None,
) -> FastAPI:
    settings = get_settings()
    # Without this, every `logger.info()`/`.debug()` call anywhere in this app's own code is
    # silently dropped — Python's root logger defaults to WARNING with no handler attached.
    # Found live: MockEmailAdapter's send log never appeared in `docker logs` despite the send
    # demonstrably happening. Uvicorn's own access logs were unaffected because uvicorn
    # configures its own logger separately; this call is what the rest of `backend.*` needed.
    logging.basicConfig(level=settings.log_level)

    @asynccontextmanager
    async def resolve_checkpointer() -> AsyncGenerator[BaseCheckpointSaver[str]]:
        # A provided checkpointer (tests) is used as-is — it has no pool/connection lifecycle
        # for this context manager to own. Otherwise, open the real Postgres-backed one for
        # the app's entire process lifetime (see infra/db/checkpointer.py) — `async with`
        # guarantees it's cleanly closed on shutdown rather than leaking connections when the
        # container stops.
        if checkpointer is not None:
            yield checkpointer
        else:
            dsn = get_checkpointer_dsn(settings)
            async with AsyncPostgresSaver.from_conn_string(dsn) as pg_checkpointer:
                yield pg_checkpointer

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        # Resolved here, inside the lifespan, rather than in create_app()'s own body: this
        # closure only runs when the ASGI server actually starts, so get_chat_model() and
        # build_notion_adapter() (which need real API keys) are never called at all when a
        # test supplies its own doubles — create_app(chat_model=fake, notion_port=fake) never
        # touches the real settings' secrets. email_port has no "real" counterpart to skip
        # yet (see infra/email/mock_adapter.py) — MockEmailAdapter is what production runs too.
        model = chat_model if chat_model is not None else get_chat_model(settings)
        resolved_notion_port = (
            notion_port if notion_port is not None else build_notion_adapter(settings)
        )
        resolved_email_port = email_port if email_port is not None else MockEmailAdapter()
        resolved_transcription_port = (
            transcription_port
            if transcription_port is not None
            else build_whisper_adapter(settings)
        )
        tools = build_tools(notion_port=resolved_notion_port, email_port=resolved_email_port)

        async with resolve_checkpointer() as cp:
            # The graph is compiled once, not per-request, and stored on app.state for routes
            # to reach via `request.app.state.graph`. Compiling is what binds the topology
            # (graph.py) to this specific checkpointer instance; doing it per-request would
            # mean every request pays graph-construction cost and, worse, would be pointless
            # busywork since the topology never changes between requests.
            app.state.graph = build_graph(model, tools).compile(checkpointer=cp)
            # The *unbound* model, separately from the graph — `model.bind_tools(tools)`
            # happens inside build_graph()'s own closure (see graph.py) and isn't reachable
            # from outside it. routes/messages.py needs a plain model instance too, for
            # classifying a plain-text reply to a paused confirmation (see
            # agent/confirmation.py) — reusing this one rather than constructing a second
            # model avoids a second provider/API-key config just for that.
            app.state.chat_model = model
            app.state.transcription_port = resolved_transcription_port
            app.state.langfuse_enabled = configure_langfuse(settings)

            yield

    app = FastAPI(
        title="AI Assistant Backend",
        # Interactive docs (Swagger UI) let you execute requests straight from the browser.
        # Fine while the only endpoints are read-only health checks; worth revisiting once
        # this exposes anything that can send an email or a WhatsApp message on someone's
        # behalf — at that point "reachable interactive docs" becomes "reachable action
        # trigger" unless there's auth in front of it. Gating on environment now means the
        # decision doesn't get forgotten later.
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url=None,
        lifespan=lifespan,
    )

    app.include_router(health.router)
    app.include_router(messages.router)

    return app


# `fastapi run` / uvicorn need a module-level ASGI callable to point at (e.g.
# `fastapi run src/backend/interfaces/api/app.py`, or `uvicorn backend.interfaces.api.app:app`
# inside the container). This is the one place the factory result gets assigned to a name —
# everywhere else (tests, scripts) calls `create_app()` directly.
app = create_app()
