"""Tests for the liveness/readiness endpoints.

test_ready_ok is technically an integration test, not a unit test — it needs a real, reachable
Postgres (DATABASE_URL from .env locally, a `postgres` service container in CI; see
.github/workflows/ci.yml). It's kept in the same file as the pure-unit test_live_ok anyway:
splitting "unit" vs "integration" into separate directories only earns its keep once there are
enough integration tests for the distinction to matter for run-time or tooling — one file with
one test that happens to touch the network isn't that yet.
"""

from httpx import AsyncClient


async def test_live_ok(client: AsyncClient) -> None:
    response = await client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_ready_ok(client: AsyncClient) -> None:
    response = await client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
