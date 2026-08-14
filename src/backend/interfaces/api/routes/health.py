"""Liveness and readiness endpoints.

Deliberately two separate endpoints instead of one `/health`, because they answer different
questions and get used by different callers:

- `/health/live` — "is the process up and able to handle HTTP at all?" Never touches the
  database. This is what docker-compose's own healthcheck (and later, any orchestrator) should
  poll: if it's failing, the fix is "restart the container," not "check the database."
- `/health/ready` — "is this instance actually able to do its job right now?" Checks the
  database round-trip. A reverse proxy or load balancer (once there's more than one instance)
  should use *this* to decide whether to route traffic to the instance — a container can be
  alive but not ready (e.g. DB briefly unreachable), and restarting it wouldn't help.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.infra.db.session import get_db_session

router = APIRouter(prefix="/health", tags=["health"])


class HealthStatus(BaseModel):
    status: str


@router.get("/live")
async def live() -> HealthStatus:
    return HealthStatus(status="ok")


@router.get("/ready")
async def ready(session: Annotated[AsyncSession, Depends(get_db_session)]) -> HealthStatus:
    # `SELECT 1` rather than querying a real table: it exercises the exact same connection
    # path a real query would (auth, network, pool checkout) without depending on any schema
    # existing yet. Phase 1 has no application tables — this endpoint has to work before the
    # first migration is even written.
    await session.execute(text("SELECT 1"))
    return HealthStatus(status="ok")
