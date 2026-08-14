"""Async database engine and per-request session handling.

One engine per process, created once at import time (not per-request) because `AsyncEngine`
already owns and manages a connection pool internally — recreating it per request would mean
opening a fresh TCP connection (and doing TLS/auth) for every single request instead of reusing
a warm pool. Sessions, by contrast, are cheap and *must* be per-request: sharing one
`AsyncSession` across concurrent requests would let unrelated requests see each other's
uncommitted changes.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.database_url.get_secret_value(),
    echo=settings.debug,
    # Postgres (and most poolers/load balancers in front of it) will silently drop
    # connections that sit idle for a while. Without pre-ping, the *next* request to reuse
    # that dead connection fails with an opaque `ConnectionResetError` deep in asyncpg.
    # pre_ping runs a cheap `SELECT 1` before handing a pooled connection back out, so a dead
    # connection gets transparently replaced instead of surfacing as a request-level 500.
    # Worth the tiny latency cost for a service that's meant to stay up unattended on a VPS.
    pool_pre_ping=True,
)

async_session_factory = async_sessionmaker(
    engine,
    # expire_on_commit=False: by default SQLAlchemy expires all ORM objects after commit, so
    # touching an attribute afterwards triggers a fresh SELECT. That's the right default for
    # long-lived sync sessions, but in an async request handler you often want to commit and
    # then still serialize the object you just created — expiring it would force a second
    # round trip (or worse, a `MissingGreenlet` error if that lazy-load happens outside the
    # session's async context).
    expire_on_commit=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession]:
    """FastAPI dependency yielding one session per request.

    The commit/rollback happens *here*, centrally, rather than inside every route handler —
    a route that raises partway through always rolls back instead of leaving a half-applied
    transaction, and a route that returns normally always commits, without every handler
    having to remember to do it.
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
