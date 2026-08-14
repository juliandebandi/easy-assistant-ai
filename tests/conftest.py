"""Shared pytest fixtures."""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

from backend.interfaces.api.app import create_app


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient]:
    """An HTTP client wired directly to the ASGI app, no network socket involved.

    `ASGITransport` calls the app's ASGI callable in-process instead of routing requests
    through a real TCP port the way `TestClient`-over-`uvicorn` or hitting a running container
    would. That keeps the test suite from needing a server actually listening anywhere, and
    from paying socket setup/teardown cost per test.
    """
    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
