"""DSN handling for LangGraph's Postgres checkpointer.

The checkpointer is built on psycopg (v3), not the asyncpg driver session.py's SQLAlchemy
engine uses — two different Postgres client libraries in the same app, talking to the same
database, for two different concerns (our own application data vs. LangGraph's internal
checkpoint blobs). That split isn't a workaround, it's accepting that the checkpointer we get
"for free" from `langgraph-checkpoint-postgres` is built on psycopg internally; writing a
SQLAlchemy-based one instead would mean reimplementing LangGraph's checkpoint serialization
format ourselves; not worth it to save one dependency.

The practical consequence: the two drivers spell their connection strings differently.
SQLAlchemy needs a dialect+driver prefix (`postgresql+asyncpg://`) so it knows which DBAPI to
load; psycopg expects the plain libpq-style URL (`postgresql://`). Same database, same
credentials — just two different string formats for the one DATABASE_URL setting.
"""

from backend.config import Settings


def get_checkpointer_dsn(settings: Settings) -> str:
    dsn = settings.database_url.get_secret_value()
    return dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
