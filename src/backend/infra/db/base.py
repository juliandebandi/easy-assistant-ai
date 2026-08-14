"""SQLAlchemy declarative base.

Deliberately its own module (not defined inside session.py or a models.py that doesn't exist
yet) because two very different things need to import it without pulling in the other:

- Every future ORM model (`class Document(Base): ...`) inherits from it.
- Alembic's `migrations/env.py` imports *only* `Base.metadata` to diff against the database
  for autogenerate — it must not import session.py, which creates a real async engine and
  therefore requires a live DATABASE_URL at import time. Keeping Base import-safe (no engine,
  no I/O) is what lets `alembic revision --autogenerate` run without a running app.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
