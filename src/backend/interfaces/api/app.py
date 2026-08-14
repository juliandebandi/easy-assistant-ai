"""FastAPI application factory.

A `create_app()` function instead of a module-level `app = FastAPI()` for one concrete reason:
tests need to build an app instance and swap dependencies (e.g. `get_settings`, `get_db_session`)
*before* anything registers against a real database. A module-level `app` is constructed the
moment the module is imported, which is too early to inject test doubles cleanly. The factory
also keeps import-time side effects (route registration, middleware) explicit and out of module
scope, which matters once this app has more than one entrypoint (uvicorn locally, `fastapi run`
in the container, and eventually a test client).
"""

from fastapi import FastAPI

from backend.config import get_settings
from backend.interfaces.api.routes import health


def create_app() -> FastAPI:
    settings = get_settings()

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
    )

    app.include_router(health.router)

    return app


# `fastapi run` / uvicorn need a module-level ASGI callable to point at (e.g.
# `fastapi run src/backend/interfaces/api/app.py`, or `uvicorn backend.interfaces.api.app:app`
# inside the container). This is the one place the factory result gets assigned to a name —
# everywhere else (tests, scripts) calls `create_app()` directly.
app = create_app()
