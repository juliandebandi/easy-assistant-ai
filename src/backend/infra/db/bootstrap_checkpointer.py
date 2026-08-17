"""One-shot setup for LangGraph's Postgres checkpointer tables.

Run via `python -m backend.infra.db.bootstrap_checkpointer` — chained after `alembic upgrade
head` in the docker-compose `migrate` service's command (see docker-compose.yml). Deliberately
not folded into an Alembic revision: LangGraph's checkpointer versions and creates its own
internal schema inside `.setup()` (idempotent — safe to call on every deploy), which Alembic
has no model for and can't diff against. Wrapping a black-box `.setup()` call in an Alembic
migration would just be theater, not real migration tracking.
"""

import asyncio

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from backend.config import get_settings
from backend.infra.db.checkpointer import get_checkpointer_dsn


async def main() -> None:
    settings = get_settings()
    dsn = get_checkpointer_dsn(settings)
    async with AsyncPostgresSaver.from_conn_string(dsn) as checkpointer:
        await checkpointer.setup()


if __name__ == "__main__":
    asyncio.run(main())
